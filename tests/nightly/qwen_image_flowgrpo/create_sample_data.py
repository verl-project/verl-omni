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
"""Create the deterministic eight-prompt OCR dataset used by the nightly test."""

from __future__ import annotations

import argparse
import os

import pandas as pd

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
SAMPLES = (
    (
        "A minimalist white poster with the black word CI centered in large bold letters",
        "CI",
    ),
    (
        "A sunny yellow sign with the dark blue word OK in the middle and rounded corners",
        "OK",
    ),
    (
        "A blackboard-style image with the white number 42 written prominently in chalk",
        "42",
    ),
    (
        "A red delivery package label showing the bold uppercase text AB near the center",
        "AB",
    ),
    (
        "A blue book cover with the title CD printed in clean white letters at the top",
        "CD",
    ),
    (
        "A green stadium scoreboard displaying the uppercase text EF in bright digital letters",
        "EF",
    ),
    (
        "A purple concert badge with the initials GH printed in the center",
        "GH",
    ),
    (
        "A wooden workshop sign hanging on a wall with the bold letters JK painted in black",
        "JK",
    ),
)


def build_rows(split: str, size: int) -> list[dict]:
    rows = []
    for index in range(size):
        sample_index = index % len(SAMPLES)
        user_prompt, ground_truth = SAMPLES[sample_index]
        rows.append(
            {
                "data_source": "qwen_image_flowgrpo",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "negative_prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "blurry, low quality, distorted text"},
                ],
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
                "extra_info": {
                    "split": split,
                    "index": sample_index,
                    "source_index": sample_index,
                    "repeat_index": index,
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eight-prompt Qwen-Image FlowGRPO data")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/qwen_image_flowgrpo"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=16, help="Repeated train rows")
    parser.add_argument("--val_size", type=int, default=4, help="Repeated validation rows")
    args = parser.parse_args()

    os.makedirs(args.local_save_dir, exist_ok=True)
    train_path = os.path.join(args.local_save_dir, "train.parquet")
    val_path = os.path.join(args.local_save_dir, "test.parquet")

    pd.DataFrame(build_rows("train", args.train_size)).to_parquet(train_path)
    pd.DataFrame(build_rows("test", args.val_size)).to_parquet(val_path)

    print(f"Wrote eight-prompt train data to {train_path}")
    print(f"Wrote eight-prompt validation data to {val_path}")


if __name__ == "__main__":
    main()
