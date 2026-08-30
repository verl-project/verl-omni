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
"""Create dummy LingBot Dense T2V parquet data for e2e tests."""

from __future__ import annotations

import argparse
import os

import pandas as pd

_CAPTIONS = [
    {
        "comprehensive_description": "a red cube slowly rotating on a wooden table",
        "camera_info": "static medium shot",
        "background": "soft studio lighting",
    },
    {
        "comprehensive_description": "a paper boat drifting across a calm blue pond",
        "camera_info": "slow top-down pan",
        "background": "gentle ripples",
    },
    {
        "comprehensive_description": "a yellow balloon rising past a brick wall",
        "camera_info": "vertical tilt up",
        "background": "overcast daylight",
    },
    {
        "comprehensive_description": "a green apple rolling along a marble counter",
        "camera_info": "static close-up",
        "background": "bright kitchen",
    },
]

_NEGATIVE_CAPTION = {
    "universal_negative": {
        "visual_quality": ["low quality", "blurry", "jpeg artifacts"],
        "temporal_and_motion_stability": ["flickering", "motion blur", "incoherent motion"],
    }
}


def build_rows(split: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "data_source": "jpeg_compressibility",
                "prompt": _CAPTIONS[i % len(_CAPTIONS)],
                "negative_prompt": _NEGATIVE_CAPTION,
                "ability": "t2v",
                "reward_model": {"style": "rule", "ground_truth": ""},
                "extra_info": {"split": split, "index": i},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate dummy LingBot Dense T2V parquet data")
    parser.add_argument(
        "--local_save_dir",
        default=os.path.expanduser("~/data/dummy_lingbot_video"),
        help="Directory to write train.parquet and test.parquet",
    )
    parser.add_argument("--train_size", type=int, default=4, help="Number of training samples")
    parser.add_argument("--val_size", type=int, default=4, help="Number of validation samples")
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
