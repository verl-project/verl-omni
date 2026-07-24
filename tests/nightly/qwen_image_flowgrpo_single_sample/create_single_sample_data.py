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
"""Create the deterministic single-sample OCR dataset used by the nightly test."""

from __future__ import annotations

import argparse
import os

import pandas as pd

SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
USER_PROMPT = "A clean white square image with the black text CI centered in large bold letters"
GROUND_TRUTH = "CI"


def build_rows(split: str, size: int) -> list[dict]:
    row = {
        "data_source": "qwen_image_flowgrpo_single_sample",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "negative_prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "blurry, low quality, distorted text"},
        ],
        "reward_model": {"style": "rule", "ground_truth": GROUND_TRUTH},
        "extra_info": {"split": split, "index": 0, "source_index": 0},
    }
    rows = []
    for index in range(size):
        repeated = dict(row)
        repeated["extra_info"] = {**row["extra_info"], "repeat_index": index}
        rows.append(repeated)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate repeated single-sample Qwen-Image FlowGRPO data")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/qwen_image_flowgrpo_single_sample"),
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

    print(f"Wrote repeated single-sample train data to {train_path}")
    print(f"Wrote repeated single-sample validation data to {val_path}")


if __name__ == "__main__":
    main()
