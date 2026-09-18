# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline multi-GPU image generation with FLUX.

Reads a prompt file (one prompt per line, e.g. dancegrpo_consist-id.txt),
shards the prompts across ranks, and saves one image per prompt plus a
metadata jsonl mapping each image to its prompt.

Launch with torchrun:

    torchrun --nproc_per_node=8 examples/diffusionnft_trainer/minimax_h3/gen_flux_images.py \
        --prompt_file data/ConsisID-preview-Data/dancegrpo_consist-id.txt \
        --model_path /path/to/FLUX.1-dev \
        --output_dir data/flux_images \
        --height 400 --width 640

Each rank saves images as {output_dir}/images/{global_index:06d}.jpg and
appends to {output_dir}/metadata_rank{rank}.jsonl. Re-running skips images
that already exist, so interrupted jobs can be resumed safely.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from diffusers import FluxPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--num_per_prompt", type=int, default=1, help="Images to generate per prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed; per-image seed is seed + global_index.")
    parser.add_argument("--max_prompts", type=int, default=-1, help="Limit total prompts for smoke tests (-1 = all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    with args.prompt_file.open(encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if args.max_prompts >= 0:
        prompts = prompts[: args.max_prompts]

    # Shard by global prompt index so resume logic stays stable.
    shard = [(i, p) for i, p in enumerate(prompts) if i % world_size == rank]

    image_dir = args.output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.output_dir / f"metadata_rank{rank}.jsonl"

    pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)

    done = 0
    skipped = 0
    with meta_path.open("a", encoding="utf-8") as meta_file:
        for global_index, prompt in shard:
            for sample_idx in range(args.num_per_prompt):
                name = f"{global_index:06d}" + (f"_{sample_idx}" if args.num_per_prompt > 1 else "")
                image_path = image_dir / f"{name}.jpg"
                if image_path.exists():
                    skipped += 1
                    continue
                generator = torch.Generator(device).manual_seed(
                    args.seed + global_index * args.num_per_prompt + sample_idx
                )
                image = pipe(
                    prompt,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=args.max_sequence_length,
                    generator=generator,
                ).images[0]
                image.save(image_path, quality=95)
                meta_file.write(
                    json.dumps(
                        {
                            "image": str(image_path.relative_to(args.output_dir)),
                            "prompt": prompt,
                            "prompt_index": global_index,
                            "sample_index": sample_idx,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                meta_file.flush()
                done += 1
            if (done + skipped) % 50 == 0:
                print(
                    f"[rank {rank}] progress {done + skipped}/{len(shard)} (generated {done}, skipped {skipped})",
                    flush=True,
                )

    print(f"[rank {rank}] finished: generated {done}, skipped {skipped}", flush=True)


if __name__ == "__main__":
    main()
