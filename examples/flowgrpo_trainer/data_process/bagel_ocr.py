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
Preprocess the OCR dataset for BAGEL FlowGRPO training.

BAGEL uses raw ``tokenizer.encode(user_text)`` with BOS/EOS markers
(no chat template).  This script pre-tokenizes prompts in BAGEL format
so the training adapter can read ``prompt_token_ids`` directly instead
of decoding + re-encoding chat-template tokens at every step.

Usage::

    python examples/flowgrpo_trainer/data_process/bagel_ocr.py \
        --model_path ~/models/ByteDance-Seed/BAGEL-7B-MoT \
        --input_dir ~/data/ocr \
        --output_dir ~/data/ocr/bagel

You can obtain the raw OCR dataset from:
https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr
"""

import argparse
import json
import os

import datasets
import numpy as np
from transformers import AutoTokenizer
from verl.utils.hdfs_io import copy, makedirs


def extract_ocr_solution(text: str) -> str:
    """Extract the ground-truth text from OCR solution string."""
    return text.split('"')[1]


def tokenize_bagel_prompt(
    tokenizer: AutoTokenizer,
    user_text: str,
    max_length: int = 256,
) -> list[int]:
    """Tokenize a user prompt in BAGEL native format.

    BAGEL rollout uses raw ``tokenizer.encode(prompt)`` wrapped with
    ``<|im_start|>`` / ``<|im_end|>`` markers — no chat template.
    """
    bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    raw_ids = tokenizer.encode(user_text, add_special_tokens=False)
    bagel_ids = [bos_id] + raw_ids + [eos_id]
    return bagel_ids[:max_length]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess OCR dataset for BAGEL FlowGRPO training.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to BAGEL-7B-MoT model directory (for tokenizer).",
    )
    parser.add_argument(
        "--input_dir",
        default="~/data/ocr/",
        help="Path to the raw OCR dataset directory.",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/ocr/bagel",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        help="Max token length for BAGEL prompts.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS output directory.")
    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.input_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Load explicitly as text files to avoid conflicts with any stale
    # parquet files that may exist in subdirectories (e.g. bagel/).
    train_file = os.path.join(local_dataset_path, "train.txt")
    test_file = os.path.join(local_dataset_path, "test.txt")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Expected raw text files at {train_file} and {test_file}. "
            f"Download the dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr"
        )
    dataset = datasets.load_dataset("text", data_files={"train": train_file, "test": test_file})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    data_source = "flow_grpo/ocr"

    negative_user_prompt = " "

    def make_map_fn(split: str):
        def process_fn(example, idx):
            text = example.pop("text")
            solution = extract_ocr_solution(text)

            # Official FlowGRPO feeds BAGEL the stripped OCR line as one plain prompt.
            # Keep rollout text and actor-side token IDs byte-identical.
            user_text = text.strip()

            # Pre-tokenize in BAGEL format so the training adapter
            # can read prompt_token_ids directly (no runtime conversion).
            prompt_token_ids = tokenize_bagel_prompt(tokenizer, user_text, max_length=args.max_prompt_length)

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": user_text}],
                "negative_prompt": [
                    {"role": "user", "content": negative_user_prompt},
                ],
                "prompt_token_ids": np.array(prompt_token_ids, dtype=np.int64),
                "ability": "ocr",
                "reward_model": {
                    "style": "model",
                    "ground_truth": solution,
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_save_dir = os.path.expanduser(args.output_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    # Save a metadata file with the tokenizer path for reproducibility.
    meta = {"model_path": args.model_path, "max_prompt_length": args.max_prompt_length}
    with open(os.path.join(local_save_dir, "bagel_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved BAGEL preprocessed data to {local_save_dir}")
    print(f"  train: {len(train_dataset)} samples")
    print(f"  test:  {len(test_dataset)} samples")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
