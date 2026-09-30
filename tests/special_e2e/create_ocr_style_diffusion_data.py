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
"""
Create a small OCR-style parquet dataset for the DiffusionNFT Boogu-Image
convergence check requested on PR #568: real generative-reward-model scoring
(``genrm_ocr.compute_score_ocr``) instead of the rule-based jpeg_compressibility
reward, using the same prompt/negative_prompt/reward_model schema as
``examples/flowgrpo_trainer/data_process/boogu_image_ocr.py`` so the DiffusionNFT
smoke test exercises the same reward path as the QwenImage+DiffusionNFT example.

This generates prompts synthetically rather than fetching the full flow_grpo OCR
dataset, to keep the smoke test self-contained and its data footprint tiny.
"""

import argparse
import os

import pandas as pd

# Ported verbatim from examples/flowgrpo_trainer/data_process/boogu_image_ocr.py
# so rollout/training sees the same prompt distribution as the real OCR recipe.
BOOGU_SYSTEM_PROMPT_T2I = (
    "You are a helpful assistant that generates high-quality images based on user instructions. "
    "The instructions are as follows."
)
BOOGU_SYSTEM_PROMPT_DROP = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. Generate a new image "
    "that meets the user's requirements while maintaining consistency with the original input where appropriate."
)

OCR_WORDS = [
    "HELLO",
    "WORLD",
    "OPEN",
    "SOURCE",
    "IMAGE",
    "DIFFUSION",
    "REWARD",
    "TRAINING",
    "VERL",
    "BOOGU",
    "PROMPT",
    "MODEL",
]


def build_rows(split: str, n: int):
    rows = []
    for i in range(n):
        word = OCR_WORDS[i % len(OCR_WORDS)]
        text = f'The image displays "{word}".'
        rows.append(
            {
                "data_source": "flow_grpo/ocr",
                "prompt": [
                    {"role": "system", "content": BOOGU_SYSTEM_PROMPT_T2I},
                    {"role": "user", "content": text},
                ],
                "negative_prompt": [
                    {"role": "system", "content": BOOGU_SYSTEM_PROMPT_DROP},
                    {"role": "user", "content": ""},
                ],
                "ability": "ocr",
                "reward_model": {"style": "model", "ground_truth": word},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate OCR-style diffusion parquet data for e2e testing")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/ocr_style_diffusion"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=32, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=8, help="Number of validation samples")
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)

    train_df = pd.DataFrame(build_rows("train", args.train_size))
    val_df = pd.DataFrame(build_rows("test", args.val_size))

    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")

    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)

    print(f"Wrote {len(train_df)} train samples to {train_path}")
    print(f"Wrote {len(val_df)} val samples to {val_path}")


if __name__ == "__main__":
    main()
