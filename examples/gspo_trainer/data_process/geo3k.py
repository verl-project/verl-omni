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
r"""Preprocess Geometry3K for Qwen3-Omni image-to-text RL.

The reward used by verl expects an explicit ``<think>...</think>\boxed{...}``
contract.  This converter states that contract directly instead of relying on
model-specific implicit thinking behavior.
"""

import argparse
import os

import datasets

DATA_SOURCE = "hiyouga/geometry3k"
ABILITY = "math"

SYSTEM_PROMPT = (
    r"You are a geometry reasoning assistant. Start every response with the literal tag <think>. "
    r"Put all reasoning between <think> and </think>. Immediately after </think>, give only the final answer in "
    r'\boxed{}. Do not write any text or whitespace before <think>. Do not write labels such as "Your reasoning"; '
    "output the tags themselves.\n\n"
    'A valid response to the question "What is 1+1?" is exactly:\n'
    r"<think>1+1=2.</think>\boxed{2}"
    "\n\nThe opening <think> tag is mandatory. Beginning directly with reasoning and later writing only </think> "
    "is invalid."
)

FORMAT_INSTRUCTION = (
    r"Your first output characters must be <think>. Enclose the reasoning in <think> and </think>, then "
    r"immediately output the final answer in \boxed{}. Output the literal tags, not a description of them. "
    r"An implicit opening is invalid: you must explicitly write <think> before the first reasoning word."
)


def build_rl_row(example: dict, split: str, index: int) -> dict:
    """Convert one Geometry3K example to verl's multimodal RL schema."""
    # Keep source whitespace so regenerated prompts match the dataset exactly.
    problem = str(example["problem"])
    answer = str(example["answer"]).strip()
    images = example["images"]
    if not problem.strip():
        raise ValueError("Geometry3K problem must not be empty")
    if not answer:
        raise ValueError("Geometry3K answer must not be empty")
    if not isinstance(images, list) or not images:
        raise ValueError("Geometry3K example must contain at least one image")
    placeholder_count = problem.count("<image>")
    if placeholder_count != len(images):
        raise ValueError(
            f"Geometry3K image placeholder mismatch: prompt has {placeholder_count}, payload has {len(images)}"
        )
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


def preserve_image_bytes(source: datasets.Dataset) -> datasets.Dataset:
    """Disable image decoding before mapping so parquet keeps source bytes."""
    return source.cast_column("images", datasets.Sequence(datasets.Image(decode=False)))


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
        source = preserve_image_bytes(dataset[split])
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
