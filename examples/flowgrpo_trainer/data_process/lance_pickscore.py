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
Preprocess the PickScore dataset for Lance FlowGRPO training.

Lance inherits its prompt handling from the BAGEL lineage: raw
``tokenizer.encode(user_text)`` wrapped in ``<|im_start|>`` / ``<|im_end|>``,
no chat template.  Pre-tokenizing here lets the training adapter read
``prompt_token_ids`` directly instead of decoding and re-encoding at every
step.  ``--model_path`` accepts the bundle root or its ``Lance_3B``
subdirectory; the tokenizer ships inside the checkpoint.

Usage::

    python examples/flowgrpo_trainer/data_process/lance_pickscore.py \
        --model_path ~/models/bytedance-research/Lance \
        --input_dir ~/data/pickscore \
        --output_dir ~/data/pickscore/lance

The raw PickScore dataset (train.txt / test.txt) should come from:
https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore
"""

import argparse
import json
import os

import datasets
import numpy as np
from transformers import AutoTokenizer
from verl.utils.hdfs_io import copy, makedirs

from verl_omni.pipelines.lance_flow_grpo.lance_model import resolve_checkpoint_dir


def tokenize_lance_prompt(
    tokenizer: AutoTokenizer,
    user_text: str,
    max_length: int = 256,
) -> list[int]:
    """Tokenize a user prompt in Lance's native format.

    The rollout uses raw ``tokenizer.encode(prompt)`` wrapped with
    ``<|im_start|>`` / ``<|im_end|>`` markers, with no chat template.
    """
    bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    raw_ids = tokenizer.encode(user_text, add_special_tokens=False)
    prompt_ids = [bos_id] + raw_ids + [eos_id]
    return prompt_ids[:max_length]


def prepare_training_tokenizer(tokenizer: AutoTokenizer, output_dir: str) -> str:
    """Save a copy of the tokenizer with the markers the trainer needs designated.

    The released ``Lance_3B`` tokenizer carries ``<|im_start|>`` / ``<|im_end|>``
    in its vocabulary but designates neither an ``eos_token`` nor a
    ``pad_token``.  The rollout repairs that for itself (vllm-omni's
    ``add_special_tokens``); the trainer loads the directory as it is, so
    ``verl``'s ``set_pad_token_id`` copies a pad token from an eos token that
    is also unset, and the agent loop fails the moment it pads a batch of
    prompts.  Designate both as ``<|im_end|>``, the Qwen convention the rollout
    already resolves to, and write the result beside the parquet so the recipe
    can point ``model.tokenizer_path`` at something complete without touching
    the released checkpoint.

    Args:
        tokenizer: Tokenizer loaded from the Lance checkpoint.
        output_dir: Directory the parquet files are written to.

    Returns:
        Path of the saved tokenizer directory.
    """
    end_of_turn = "<|im_end|>"
    if tokenizer.convert_tokens_to_ids(end_of_turn) is None:
        raise ValueError(f"{end_of_turn} is not in the tokenizer vocabulary")
    if tokenizer.eos_token is None:
        tokenizer.eos_token = end_of_turn
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)
    return tokenizer_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PickScore dataset for Lance FlowGRPO training.")
    parser.add_argument(
        "--model_path",
        required=True,
        help="Lance bundle root or a Lance_3B checkpoint directory (for tokenizer).",
    )
    parser.add_argument(
        "--input_dir",
        default="~/data/pickscore/",
        help="Path to the raw PickScore dataset directory (contains train.txt / test.txt).",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/pickscore/lance",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        help="Max token length for Lance prompts.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS output directory.")
    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.input_dir)

    tokenizer_path = resolve_checkpoint_dir(os.path.expanduser(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    train_file = os.path.join(local_dataset_path, "train.txt")
    test_file = os.path.join(local_dataset_path, "test.txt")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Expected raw text files at {train_file} and {test_file}. "
            f"Download the dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/pickscore"
        )
    dataset = datasets.load_dataset("text", data_files={"train": train_file, "test": test_file})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    data_source = "flow_grpo/pickscore"

    negative_user_prompt = " "

    def make_map_fn(split: str):
        def process_fn(example, idx):
            text = example.pop("text")
            caption = text.strip()

            prompt_token_ids = tokenize_lance_prompt(tokenizer, caption, max_length=args.max_prompt_length)

            return {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": caption}],
                "negative_prompt": [
                    {"role": "user", "content": negative_user_prompt},
                ],
                "prompt_token_ids": np.array(prompt_token_ids, dtype=np.int64),
                "ability": "pickscore",
                "reward_model": {
                    "style": "model",
                    "ground_truth": caption,
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

    tokenizer_dir = prepare_training_tokenizer(tokenizer, local_save_dir)

    meta = {
        "model_path": args.model_path,
        "max_prompt_length": args.max_prompt_length,
        "tokenizer_dir": tokenizer_dir,
    }
    with open(os.path.join(local_save_dir, "lance_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved Lance preprocessed data to {local_save_dir}")
    print(f"  tokenizer with eos/pad designated: {tokenizer_dir}")
    print(f"  train: {len(train_dataset)} samples")
    print(f"  test:  {len(test_dataset)} samples")

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_save_dir, dst=args.hdfs_dir)
