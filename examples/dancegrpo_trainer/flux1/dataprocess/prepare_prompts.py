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

"""Convert explicit train/test text prompt files to verl-omni parquet data."""

import argparse
import os

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs


def _load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as prompt_file:
        prompts = [line.strip() for line in prompt_file if line.strip()]
    if not prompts:
        raise ValueError(f"No non-empty prompts found in {path}")
    return prompts


def _make_record(prompt: str, split: str, index: int) -> dict:
    return {
        "data_source": "dance_grpo/flux1",
        "prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ],
        "negative_prompt": [
            {"role": "system", "content": ""},
            {"role": "user", "content": " "},
        ],
        "ability": "t2i",
        "reward_model": {"style": "model", "ground_truth": prompt},
        "extra_info": {"split": split, "index": index},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", required=True, help="UTF-8 text file with one training prompt per line")
    parser.add_argument("--test-path", required=True, help="UTF-8 text file with one validation prompt per line")
    parser.add_argument(
        "--output-dir",
        default="~/data/hpsv2",
        help="Directory for train.parquet and test.parquet (default: ~/data/hpsv2)",
    )
    parser.add_argument("--hdfs-dir", default=None, help="Optional HDFS destination")
    args = parser.parse_args()

    train_path = os.path.abspath(os.path.expanduser(args.train_path))
    test_path = os.path.abspath(os.path.expanduser(args.test_path))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    train_prompts = _load_prompts(train_path)
    test_prompts = _load_prompts(test_path)

    os.makedirs(output_dir, exist_ok=True)
    outputs = {
        "train": (train_prompts, os.path.join(output_dir, "train.parquet")),
        "test": (test_prompts, os.path.join(output_dir, "test.parquet")),
    }
    for split, (prompts, output_path) in outputs.items():
        records = [_make_record(prompt, split, index) for index, prompt in enumerate(prompts)]
        pd.DataFrame(records).to_parquet(output_path, index=False)
        print(f"{split}: {len(records)} records -> {output_path}")

    if args.hdfs_dir:
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)


if __name__ == "__main__":
    main()
