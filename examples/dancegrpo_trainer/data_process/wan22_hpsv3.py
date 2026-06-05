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

The raw data is a plain-text file with one prompt per line, obtained from:
  https://github.com/XueZeyue/DanceGRPO/blob/main/assets/video_prompts.txt

Lines containing Chinese characters are filtered out (following the original
DanceGRPO preprocessing), the prompts are shuffled, and then split into
train/test sets. The test set size is ``min(5% total, 1000)``.
"""

import argparse
import os
import random
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
        "--input_path",
        default="~/data/hpsv3/prompt.txt",
        help="Path to the raw prompt.txt file.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/hpsv3",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.05,
        help="Fraction of prompts reserved for the test set.",
    )

    args = parser.parse_args()
    input_path = os.path.expanduser(args.input_path)
    output_dir = os.path.expanduser(args.output_dir)

    prompts = _load_prompts(input_path)
    print(f"Loaded {len(prompts)} prompts (after filtering Chinese lines)")

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

    # Shuffle and split into train/test
    random.seed(42)
    random.shuffle(prompts)
    test_count = max(1, min(int(len(prompts) * args.test_ratio), 1000))
    test_prompts = prompts[:test_count]
    train_prompts = prompts[test_count:]

    train_records = [make_record(p, "train", i) for i, p in enumerate(train_prompts)]
    test_records = [make_record(p, "test", i) for i, p in enumerate(test_prompts)]

    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Train: {len(train_records)} records -> {train_path}")
    print(f"Test:  {len(test_records)} records -> {test_path}")

    hdfs_dir = args.hdfs_dir
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=output_dir, dst=hdfs_dir)
