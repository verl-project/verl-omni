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
Preprocess the Flow-GRPO OCR dataset to parquet format for SD3.5 training.

You can obtain the raw dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr

Unlike Qwen-Image, SD3 runs its own CLIP/T5 text encoders on the raw prompt
text, so no chat template or system prompt is applied: each sample is a single
user message whose content is the raw OCR prompt. ``--train_size`` /
``--val_size`` subsample the dataset for the fast convergence test (final
sizing is a Phase 2 tuning knob).
"""

import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    # The solution is stored in the format: 'The image displays "xxx".'
    return solution_str.split('"')[1]


def _as_list(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value if isinstance(value, list) else None


def _content_to_text(content):
    if hasattr(content, "tolist"):
        content = content.tolist()
    if isinstance(content, str):
        return content
    parts = []
    for item in _as_list(content) or []:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(part for part in parts if part)


def _messages_to_text(messages):
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        messages = [messages]
    user_parts = []
    fallback_parts = []
    for message in _as_list(messages) or []:
        if not isinstance(message, dict):
            continue
        text = _content_to_text(message.get("content"))
        if text:
            fallback_parts.append(text)
            if message.get("role") == "user":
                user_parts.append(text)
    return "\n".join(user_parts or fallback_parts)


def _extract_prompt_and_solution(example):
    if "text" in example and example["text"]:
        text = example.pop("text")
        return text, extract_solution(text)

    prompt = _messages_to_text(example.get("prompt"))
    reward_model = example.get("reward_model") or {}
    solution = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    if not prompt or solution is None:
        raise ValueError("SD3 OCR data requires either raw `text` or (`prompt`, `reward_model.ground_truth`).")
    return prompt, solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--input_dir", default="~/data/ocr/", help="Path to the raw OCR dataset directory.")
    parser.add_argument(
        "--output_dir", default="~/data/ocr/sd3", help="Directory to save the preprocessed parquet files."
    )
    parser.add_argument(
        "--train_size", type=int, default=None, help="Subsample the train split to this many prompts (None = all)."
    )
    parser.add_argument(
        "--val_size", type=int, default=None, help="Subsample the test split to this many prompts (None = all)."
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed used when subsampling.")

    args = parser.parse_args()
    local_dataset_path = os.path.expanduser(args.input_dir)

    data_source = "flow_grpo/ocr"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        raise NotImplementedError(
            "It is not existed in huggingface hub. "
            "Please get dataset from https://github.com/yifan123/flow_grpo/tree/main/dataset/ocr"
        )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    if args.train_size is not None and args.train_size < len(train_dataset):
        train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.train_size))
    if args.val_size is not None and args.val_size < len(test_dataset):
        test_dataset = test_dataset.shuffle(seed=args.seed).select(range(args.val_size))

    def make_map_fn(split):
        def process_fn(example, idx):
            text, solution = _extract_prompt_and_solution(example)
            data = {
                "data_source": data_source,
                # Raw prompt only: SD3 encodes the text itself (CLIP-L/G + T5),
                # so no system prompt / chat template is applied.
                "prompt": [
                    {"role": "user", "content": text},
                ],
                "ability": "ocr",
                "reward_model": {"style": "model", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx, "raw_prompt": text},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.output_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    print(f"Wrote {len(train_dataset)} train / {len(test_dataset)} test samples to {local_save_dir}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
