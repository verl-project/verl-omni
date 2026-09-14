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
"""Preprocess Geometry3K for Qwen3-Omni image-to-text RL.

The reward used by verl expects an explicit ``<think>...</think>\boxed{...}``
contract.  This converter states that contract directly instead of relying on
model-specific implicit thinking behavior.
"""

import argparse
import os

import datasets

DATA_SOURCE = "hiyouga/geometry3k"
ABILITY = "math_vl"

SYSTEM_PROMPT = (
    "You are a geometry reasoning assistant. Start every response with the literal tag <think>. "
    "Put all reasoning between <think> and </think>. Immediately after </think>, give only the final answer in "
    "\\boxed{}. Do not write any text or whitespace before <think>."
)

FORMAT_INSTRUCTION = (
    "Your first output characters must be <think>. Enclose the reasoning in <think> and </think>, then "
    "immediately output the final answer in \\boxed{}. Output the literal tags, not a description of them."
)


def build_rl_row(example: dict, split: str, index: int) -> dict:
    """Convert one Geometry3K example to verl's multimodal RL schema."""
    problem = str(example["problem"]).strip()
    answer = str(example["answer"]).strip()
    images = example["images"]
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{problem} {FORMAT_INSTRUCTION}"},
        ],
        "images": images,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "answer": answer,
            "question": problem,
        },
    }


def make_map_fn(split: str):
    """Return a datasets ``map`` callback for one split."""

    def process_fn(example, index):
        return build_rl_row(example, split, index)

    return process_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Geometry3K to verl image-to-text RL parquet.")
    parser.add_argument("--local_dataset_path", default=None, help="Optional local Hugging Face dataset path.")
    parser.add_argument("--local_save_dir", default="~/data/geo3k", help="Output directory for parquet files.")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of dataset map workers.")
    args = parser.parse_args()

    dataset = datasets.load_dataset(args.local_dataset_path or DATA_SOURCE)
    output_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(output_dir, exist_ok=True)

    for split in ("train", "test"):
        source = dataset[split]
        converted = source.map(
            function=make_map_fn(split),
            with_indices=True,
            num_proc=args.num_proc,
            remove_columns=source.column_names,
        )
        output_path = os.path.join(output_dir, f"{split}.parquet")
        converted.to_parquet(output_path)
        print(f"[{split}] wrote {len(converted)} rows to {output_path}")


if __name__ == "__main__":
    main()
