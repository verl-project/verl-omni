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

"""Preprocess the GSM8K dataset to verl RL parquet format (text-in / text-out).

Raw GSM8K schema (``openai/gsm8k``, config ``main``):
    question  - the word problem
    answer    - chain of thought ending in ``#### <number>``

Output verl RL schema (one row per problem):
    data_source  = "openai/gsm8k"
    prompt       = [{"role": "user", "content": "<instruction>\\n\\n<question>"}]
    ability      = "math"
    reward_model = {"style": "rule", "ground_truth": <number after ####, commas stripped>}
    extra_info   = {split, index, dataset, raw_question, raw_answer}

GSM8K is the text-only verification task of the MiniCPM-o 4.5 simplex OPD
recipe: the thinker LLM's initial accuracy is high, so reward curves are
interpretable from the first steps. The instruction asks for a ``\\boxed{}`
final answer and nothing else — the gsm8k_reward scorer has no format
component, so the initial reward equals raw accuracy.

Pure helpers (extract_ground_truth) are importable without torch/verl so they
can be unit-tested on CPU.
"""

import argparse
import glob
import os
import re

import datasets

DATA_SOURCE = "openai/gsm8k"
ABILITY = "math"
DATASET_NAME = "GSM8K"

INSTRUCTION = (
    "You should first think about the reasoning process in the mind and then "
    "provide the user with the answer. The final answer must be wrapped using "
    "the \\boxed{} command."
)

_ANSWER_SUFFIX_RE = re.compile(r"####\s*(.+?)\s*$")


def extract_ground_truth(answer: str) -> str | None:
    """Return the numeric ground truth after GSM8K's ``####`` marker.

    Strips thousands commas and currency marks so ``math_verify`` compares plain
    numbers. Returns None when the answer has no ``####`` suffix.
    """
    match = _ANSWER_SUFFIX_RE.search(answer or "")
    if match is None:
        return None
    return match.group(1).replace(",", "").replace("$", "").strip() or None


def make_keep_fn():
    """Return a ``.filter()`` predicate that drops rows without a usable answer."""

    def keep(example) -> bool:
        question = str(example.get("question") or "").strip()
        return bool(question and extract_ground_truth(str(example.get("answer") or "")))

    return keep


def make_map_fn(split: str):
    """Return a ``.map()`` function that builds one verl RL row per example."""

    def process_fn(example, idx):
        question = str(example.get("question") or "").strip()
        raw_answer = str(example.get("answer") or "").strip()
        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "user", "content": f"{INSTRUCTION}\n\n{question}"},
            ],
            "ability": ABILITY,
            "reward_model": {"style": "rule", "ground_truth": extract_ground_truth(raw_answer)},
            "extra_info": {
                "split": split,
                "index": idx,
                "dataset": DATASET_NAME,
                "raw_question": question,
                "raw_answer": raw_answer,
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GSM8K to verl RL parquet.")
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local dir with the raw GSM8K train-*.parquet / test-*.parquet shards (falls back to the HF hub).",
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )
    args = parser.parse_args()

    if args.local_dataset_path is not None:
        data_files = {
            "train": sorted(glob.glob(os.path.join(args.local_dataset_path, "train-*.parquet"))),
            "test": sorted(glob.glob(os.path.join(args.local_dataset_path, "test-*.parquet"))),
        }
        if not data_files["train"] and not data_files["test"]:
            raise FileNotFoundError(f"No train-*.parquet / test-*.parquet shards found in {args.local_dataset_path!r}")
        dataset = datasets.load_dataset("parquet", data_files=data_files)
    else:
        dataset = datasets.load_dataset("openai/gsm8k", "main")

    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split in ("train", "test"):
        split_dataset = dataset[split]
        n_input = len(split_dataset)
        split_dataset = split_dataset.filter(make_keep_fn(), num_proc=8)
        split_dataset = split_dataset.map(
            function=make_map_fn(split),
            with_indices=True,
            num_proc=8,
            remove_columns=split_dataset.column_names,
        )
        n_kept = len(split_dataset)

        out_path = os.path.join(local_save_dir, f"{split}.parquet")
        split_dataset.to_parquet(out_path)
        print(f"[{split}] wrote {out_path}")
        print(f"  input={n_input} kept={n_kept} dropped={n_input - n_kept}")
