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
"""Build FL2VA train/test JSONL splits from a prompt file and generated images.

Pairs each prompt (one per line) with its same-index condition image
(``{index:06d}.jpg``, e.g. produced by ``examples/diffusionnft_trainer/minimax_h3/gen_flux_images.py``),
shuffles with a fixed seed, and writes ``train.jsonl`` / ``test.jsonl``
in the format consumed by the MiniMax H3 FL2VA ``prepare_data.py``:

    {"prompt": "...", "images": ["images/000123.jpg"]}

Image paths are written relative to ``--output_dir``, so point
``prepare_data.py --input_dir`` at the same directory.

Example:

    python3 examples/diffusionnft_trainer/minimax_h3/build_fl2va_jsonl.py \
        --prompt_file dancegrpo_consist-id.txt \
        --image_dir data/flux_images/images \
        --output_dir data/flux_images \
        --test_size 128 --seed 42
"""

import argparse
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt_file", type=Path, required=True)
    parser.add_argument("--image_dir", type=Path, required=True, help="Directory containing {index:06d}.jpg images.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where train.jsonl / test.jsonl are written; image paths in the JSONL are relative to this directory.",
    )
    parser.add_argument(
        "--image_prefix", type=str, default="images", help="Path prefix stored in the JSONL, relative to output_dir."
    )
    parser.add_argument("--test_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.prompt_file.open(encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    missing = [i for i in range(len(prompts)) if not (args.image_dir / f"{i:06d}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images missing under {args.image_dir} (e.g. {missing[:5]}); "
            "run examples/diffusionnft_trainer/minimax_h3/gen_flux_images.py to completion first."
        )

    indices = list(range(len(prompts)))
    random.seed(args.seed)
    random.shuffle(indices)
    test_indices, train_indices = indices[: args.test_size], indices[args.test_size :]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_indices in [("train", train_indices), ("test", test_indices)]:
        out_path = args.output_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as out:
            for i in split_indices:
                out.write(
                    json.dumps(
                        {"prompt": prompts[i], "images": [f"{args.image_prefix}/{i:06d}.jpg"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"Wrote {len(split_indices)} rows to {out_path}")


if __name__ == "__main__":
    main()
