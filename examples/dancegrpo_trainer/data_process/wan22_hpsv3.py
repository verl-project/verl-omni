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
Preprocess the HPSv3 video prompt dataset to parquet format (for Wan2.2 DanceGRPO training).

The pre-split train/test prompts are loaded from ``video_prompts/train.txt``
and ``video_prompts/test.txt`` respectively. Lines containing Chinese characters
are filtered out (following the original DanceGRPO preprocessing).
"""

import argparse
import os
import re

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [line for line in lines if not _contains_chinese(line)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--train_path",
        default=os.path.join(
            os.path.dirname(__file__), "video_prompts", "train.txt"
        ),
        help="Path to the train prompts file.",
    )
    parser.add_argument(
        "--test_path",
        default=os.path.join(
            os.path.dirname(__file__), "video_prompts", "test.txt"
        ),
        help="Path to the test prompts file.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/hpsv3",
        help="Directory to save the preprocessed parquet files.",
    )

    args = parser.parse_args()
    train_path = os.path.expanduser(args.train_path)
    test_path = os.path.expanduser(args.test_path)
    output_dir = os.path.expanduser(args.output_dir)

    train_prompts = _load_prompts(train_path)
    test_prompts = _load_prompts(test_path)
    print(f"Loaded {len(train_prompts)} train prompts (after filtering Chinese lines)")
    print(f"Loaded {len(test_prompts)} test prompts (after filtering Chinese lines)")

    data_source = "dance_grpo/hpsv3"

    system_prompt = ""
    negative_user_prompt = " "

    def make_record(prompt: str, split: str, idx: int) -> dict:
        return {
            "data_source": data_source,
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "negative_prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": negative_user_prompt},
            ],
            "ability": "t2v",
            "reward_model": {"style": "model", "ground_truth": prompt},
            "extra_info": {"split": split, "index": idx},
        }

    train_records = [make_record(p, "train", i) for i, p in enumerate(train_prompts)]
    test_records = [make_record(p, "test", i) for i, p in enumerate(test_prompts)]

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_parquet_path = os.path.join(output_dir, "train.parquet")
    test_parquet_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_parquet_path)
    test_df.to_parquet(test_parquet_path)

    print(f"Train: {len(train_records)} records -> {train_parquet_path}")
    print(f"Test:  {len(test_records)} records -> {test_parquet_path}")

    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=output_dir, dst=hdfs_dir)
