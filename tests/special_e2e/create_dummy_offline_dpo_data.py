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
"""Create a minimal offline DPO parquet for SD3 smoke testing.

Uses a tiny SD3 pipeline to precompute VAE latents and text-encoder outputs.
Win/lose images are solid-color PIL images (no diffusion sampling) to keep
data prep fast while preserving tensor shapes expected by OfflineDPODataset.
"""

import argparse
import io
import os

import pandas as pd
import torch
from PIL import Image

DEFAULT_SYSTEM_PROMPT = "You are a helpful image generation assistant."
DEFAULT_MODEL_PATH = os.path.expanduser("~/models/tiny-random/stable-diffusion-3-tiny-random")

USER_PROMPTS = [
    "A red circle on a white background",
    "A blue square on a black background",
    "A green triangle next to an orange rectangle",
]


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def _build_messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def _encode_image_latent(pipe, image: Image.Image, height: int, width: int, device: str) -> torch.Tensor:
    pixel_values = pipe.image_processor.preprocess([image], height=height, width=width)
    pixel_values = pixel_values.to(device=device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(pixel_values).latent_dist.sample()
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    latents = (latents - shift_factor) * scaling_factor
    return latents[0].detach().cpu()


def _normalize_sd3_tokenizer_limits(pipe) -> None:
    """Ensure CLIP tokenizers expose a sane max length for diffusers encode_prompt."""
    clip_max_length = 77
    for name in ("tokenizer", "tokenizer_2"):
        tokenizer = getattr(pipe, name, None)
        if tokenizer is not None:
            tokenizer.model_max_length = clip_max_length
    pipe.tokenizer_max_length = clip_max_length


def _encode_prompt_tensors(
    pipe,
    prompt: str,
    negative_prompt: str,
    *,
    device: str,
    guidance_scale: float,
    max_sequence_length: int,
) -> dict[str, bytes | None]:
    do_cfg = guidance_scale > 1.0
    with torch.no_grad():
        encoded = pipe.encode_prompt(
            prompt=[prompt],
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_cfg,
            negative_prompt=[negative_prompt] if do_cfg else None,
            negative_prompt_2=None,
            negative_prompt_3=None,
            max_sequence_length=max_sequence_length,
        )

    if len(encoded) == 4:
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = encoded
    elif len(encoded) == 2:
        prompt_embeds, pooled_prompt_embeds = encoded
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
    else:
        raise ValueError(f"Unexpected SD3 encode_prompt output length: {len(encoded)}")

    prompt_embeds_mask = torch.ones(
        prompt_embeds.shape[0],
        prompt_embeds.shape[1],
        dtype=torch.int32,
        device=prompt_embeds.device,
    )
    result: dict[str, bytes | None] = {
        "prompt_embeds": _tensor_to_bytes(prompt_embeds[0]),
        "prompt_embeds_mask": _tensor_to_bytes(prompt_embeds_mask[0]),
        "pooled_prompt_embeds": _tensor_to_bytes(pooled_prompt_embeds[0]),
        "negative_prompt_embeds": None,
        "negative_prompt_embeds_mask": None,
        "negative_pooled_prompt_embeds": None,
    }
    if negative_prompt_embeds is not None:
        negative_prompt_embeds_mask = torch.ones(
            negative_prompt_embeds.shape[0],
            negative_prompt_embeds.shape[1],
            dtype=torch.int32,
            device=negative_prompt_embeds.device,
        )
        result["negative_prompt_embeds"] = _tensor_to_bytes(negative_prompt_embeds[0])
        result["negative_prompt_embeds_mask"] = _tensor_to_bytes(negative_prompt_embeds_mask[0])
        result["negative_pooled_prompt_embeds"] = _tensor_to_bytes(negative_pooled_prompt_embeds[0])
    return result


def _solid_image(color: tuple[int, int, int], height: int, width: int) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def build_rows(
    *,
    num_pairs: int,
    pipe,
    image_dir: str,
    output_parent: str,
    device: str,
    height: int,
    width: int,
    guidance_scale: float,
    max_sequence_length: int,
    negative_prompt: str,
    system_prompt: str,
) -> list[dict]:
    os.makedirs(image_dir, exist_ok=True)
    rows = []
    win_color = (220, 40, 40)
    lose_color = (40, 40, 220)

    for idx in range(num_pairs):
        prompt = USER_PROMPTS[idx % len(USER_PROMPTS)]
        prompt_tensors = _encode_prompt_tensors(
            pipe,
            prompt,
            negative_prompt,
            device=device,
            guidance_scale=guidance_scale,
            max_sequence_length=max_sequence_length,
        )

        win_path = os.path.join(image_dir, f"{idx:06d}_win.png")
        lose_path = os.path.join(image_dir, f"{idx:06d}_lose.png")
        win_image = _solid_image(win_color, height, width)
        lose_image = _solid_image(lose_color, height, width)
        win_image.save(win_path)
        lose_image.save(lose_path)

        rows.append(
            {
                "data_source": "offline_dpo_smoke",
                "prompt": _build_messages(prompt, system_prompt),
                "negative_prompt": _build_messages(negative_prompt, system_prompt),
                "img_win": os.path.relpath(win_path, output_parent),
                "img_lose": os.path.relpath(lose_path, output_parent),
                "img_win_latents": _tensor_to_bytes(_encode_image_latent(pipe, win_image, height, width, device)),
                "img_lose_latents": _tensor_to_bytes(_encode_image_latent(pipe, lose_image, height, width, device)),
                **prompt_tensors,
                "win_score": 1.0,
                "lose_score": 0.0,
                "reward_model": {"style": "rule", "ground_truth": prompt},
                "extra_info": {
                    "split": "smoke",
                    "index": idx,
                    "raw_prompt": prompt,
                    "raw_negative_prompt": negative_prompt,
                },
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate offline DPO parquet for SD3 smoke tests")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_offline_dpo"),
        help="Directory to write smoke.parquet and images/",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Parquet path (default: <local_save_dir>/smoke.parquet)",
    )
    parser.add_argument("--num_pairs", type=int, default=2, help="Number of DPO pairs")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=None, help="Diffusers device (default: cuda:0 or cpu)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.local_save_dir, exist_ok=True)
    output_file = args.output_file or os.path.join(args.local_save_dir, "smoke.parquet")
    output_parent = os.path.dirname(os.path.abspath(output_file))
    image_dir = os.path.join(args.local_save_dir, "images")

    from diffusers import StableDiffusion3Pipeline

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(
            f"SD3 smoke model not found at {args.model_path!r}. "
            "Run tests/special_e2e/build_sd3_tiny_random.py first or pass --model_path."
        )
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe.to(device)
    _normalize_sd3_tokenizer_limits(pipe)

    rows = build_rows(
        num_pairs=args.num_pairs,
        pipe=pipe,
        image_dir=image_dir,
        output_parent=output_parent,
        device=device,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        max_sequence_length=args.max_sequence_length,
        negative_prompt=args.negative_prompt,
        system_prompt=args.system_prompt,
    )

    df = pd.DataFrame(rows)
    df.to_parquet(output_file)
    print(f"Wrote {len(df)} offline DPO pairs to {output_file}")


if __name__ == "__main__":
    main()
