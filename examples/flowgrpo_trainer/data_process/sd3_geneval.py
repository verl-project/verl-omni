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
"""Preprocess the Flow-GRPO GenEval dataset to parquet format for SD3.5 training.

You can obtain the raw dataset from
https://github.com/yifan123/flow_grpo/tree/main/dataset/geneval
(``train_metadata.jsonl`` and ``test_metadata.jsonl``).

Usage:
    python sd3_geneval.py --input_dir ~/data/geneval --output_dir ~/data/geneval/sd3
"""

import argparse
import json
import os
import random

import datasets
from verl.utils.hdfs_io import copy, makedirs


def build_row(metadata, split, index):
    """Build one parquet row; the full metadata rides along as a JSON string
    because the GenEval scorer needs the include/exclude object spec."""
    prompt = metadata["prompt"]
    return {
        "data_source": "flow_grpo/geneval",
        "prompt": [
            {"role": "user", "content": prompt},
        ],
        "ability": "geneval",
        "reward_model": {"style": "model", "ground_truth": json.dumps(metadata)},
        "extra_info": {"split": split, "index": index, "raw_prompt": prompt, "tag": metadata["tag"]},
    }


def load_split(input_dir, split, size, seed):
    with open(os.path.join(input_dir, f"{split}_metadata.jsonl"), encoding="utf-8") as f:
        metadatas = [json.loads(line) for line in f]
    if size is not None and size < len(metadatas):
        metadatas = random.Random(seed).sample(metadatas, size)
    return datasets.Dataset.from_list([build_row(m, split, i) for i, m in enumerate(metadatas)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_dir", default="~/data/geneval/", help="Path to the raw GenEval dataset directory.")
    parser.add_argument(
        "--output_dir", default="~/data/geneval/sd3", help="Directory to save the preprocessed parquet files."
    )
    parser.add_argument(
        "--train_size", type=int, default=None, help="Subsample the train split to this many prompts (None = all)."
    )
    parser.add_argument(
        "--val_size", type=int, default=None, help="Subsample the test split to this many prompts (None = all)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed used when subsampling.")

    args = parser.parse_args()
    input_dir = os.path.expanduser(args.input_dir)

    train_dataset = load_split(input_dir, "train", args.train_size, args.seed)
    test_dataset = load_split(input_dir, "test", args.val_size, args.seed)

    local_save_dir = os.path.expanduser(args.output_dir)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Wrote {len(train_dataset)} train / {len(test_dataset)} test samples to {local_save_dir}")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
