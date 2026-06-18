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
Preprocess the PickScore dataset for BAGEL FlowGRPO training.

Prompts are stored in standard chat-message format.  Captions are also
pre-tokenized for BAGEL training-side ``old_log_prob`` (``prepare_prompts``
convention) — a workaround until vllm-omni ``BagelPipeline`` accepts token
IDs directly.

The official PickScore dataset is available from the flow_grpo repository.
To prepare it::

    wget -P ~/data/pickscore/ \
      https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/train.txt
    wget -P ~/data/pickscore/ \
      https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/test.txt
    python examples/flowgrpo_trainer/data_process/bagel_pickscore.py

If you have your own prompt dataset, place train.txt / test.txt in any
directory and pass ``--input_dir`` / ``--output_dir``.
"""

import argparse
import os

import datasets
from transformers import AutoTokenizer


# BAGEL text2img tokenizes captions as [bos] + encode(caption) + [eos] (vllm-omni
# prepare_prompts).  Stored in parquet for training-side old_log_prob only.
def bagel_prepare_prompt_token_ids(tokenizer, caption: str) -> list[int]:
    """Match vllm-omni BAGEL ``prepare_prompts`` tokenization."""
    bos_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text_ids = tokenizer.encode(caption.strip(), add_special_tokens=False) if caption.strip() else []
    return [bos_token_id, *text_ids, eos_token_id]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess PickScore dataset for BAGEL FlowGRPO training.")
    parser.add_argument(
        "--input_dir",
        default="~/data/pickscore/",
        help="Directory containing train.txt and test.txt (one caption per line).",
    )
    parser.add_argument(
        "--output_dir",
        default="~/data/pickscore/bagel",
        help="Directory to save the preprocessed parquet files.",
    )
    parser.add_argument(
        "--model_path",
        default="~/models/ByteDance-Seed/BAGEL-7B-MoT",
        help="BAGEL tokenizer path (must match training model).",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS output directory.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(args.model_path), trust_remote_code=True)

    local_dataset_path = os.path.expanduser(args.input_dir)

    train_file = os.path.join(local_dataset_path, "train.txt")
    test_file = os.path.join(local_dataset_path, "test.txt")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise FileNotFoundError(
            f"Expected raw text files at {train_file} and {test_file}. "
            f"Download them from the flow_grpo repository:\n"
            f"  wget -P {local_dataset_path} \\\n"
            f"    https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/train.txt\n"
            f"  wget -P {local_dataset_path} \\\n"
            f"    https://raw.githubusercontent.com/yifan123/flow_grpo/main/dataset/pickscore/test.txt"
        )
    dataset = datasets.load_dataset("text", data_files={"train": train_file, "test": test_file})
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    data_source = "flow_grpo/pickscore"

    def make_map_fn(split: str):
        def process_fn(example, idx):
            prompt_text = example.pop("text").strip()
            if not prompt_text:
                return None

            # PickScore compares the generated image against the prompt text.
            # Set ground_truth = prompt_text directly.
            return {
                "data_source": data_source,
                "prompt": [
                    {"role": "user", "content": prompt_text},
                ],
                "prompt_token_ids": bagel_prepare_prompt_token_ids(tokenizer, prompt_text),
                "negative_prompt": [
                    {"role": "user", "content": " "},
                ],
                "ability": "pickscore",
                "reward_model": {
                    "style": "model",
                    "ground_truth": prompt_text,
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

    print(f"Saved preprocessed PickScore dataset to {local_save_dir}/")
    print(f"  train: {len(train_dataset)} samples")
    print(f"  test:  {len(test_dataset)} samples")

    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs

            makedirs(args.hdfs_dir, exist_ok=True)
            copy(src=local_save_dir, dst=args.hdfs_dir)
        except ImportError:
            print("Warning: verl not installed, skipping HDFS upload.")
