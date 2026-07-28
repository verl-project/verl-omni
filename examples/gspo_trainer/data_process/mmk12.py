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
"""Preprocess the MMK12 dataset to verl RL parquet format (image-in / text-out).

Raw MMK12 schema:
    id, question, answer, subject, image   # image is {"bytes": <png bytes>, "path": ...}

Output verl RL schema (one row per problem):
    data_source  = "math_dapo"   # routes through the math_dapo rule reward
    prompt       = [{"role": "system", "content": <SYSTEM_PROMPT>},
                    {"role": "user",   "content": "<image>\\n{question}"}]
    images       = [{"bytes": <png bytes>}]   # carried inline, byte-identical to the source
    ability      = "math_vl"
    reward_model = {"style": "rule", "ground_truth": <answer>}
    extra_info   = {split, index, id, dataset, subject, raw_question, raw_answer, options}

Pure helpers (is_valid_image, normalize_image, can_open_image, classify_answer,
parse_options) are importable without torch/verl so they can be unit-tested on CPU.
"""

import argparse
import glob
import io
import json
import os
import re
from typing import Any

import datasets
from PIL import Image

DATA_SOURCE = "math_dapo"
ABILITY = "math_vl"
DATASET_NAME = "MMK12"

SYSTEM_PROMPT = (
    "Solve the question. The user asks a question, and you solve it. "
    "You first think about the reasoning process in the mind and then "
    "provide the user with the answer. The answer is in latex format and "
    "wrapped in $...$. The final answer must be wrapped using the \\boxed{} "
    "command. The answer should be enclosed within <answer></answer> tags, "
    "i.e., Since $1+1=2$, so the answer is $2$. "
    "<answer>The answer is $\\boxed{2}$</answer>, which means the final "
    "answer assistant's output should start with <answer> and end with </answer>."
)

USER_PROMPT_TEMPLATE = "<image>\n{question}"

_OPTION_LINE_RE = re.compile(r"^\s*([A-E])[.)、]\s*(.+?)\s*$", re.MULTILINE)


def is_valid_image(image_field: Any) -> bool:
    """True if the field carries non-empty image bytes (directly or in a dict)."""
    if image_field is None:
        return False
    if isinstance(image_field, dict):
        return bool(image_field.get("bytes"))
    if isinstance(image_field, bytes | bytearray):
        return True
    return False


def normalize_image(image_field: Any) -> dict:
    """Return a {"bytes": ...} dict the RL dataset loader can open with PIL."""
    if isinstance(image_field, dict):
        if image_field.get("bytes"):
            return {"bytes": image_field["bytes"]}
    if isinstance(image_field, bytes | bytearray):
        return {"bytes": bytes(image_field)}
    raise TypeError(f"unsupported image type: {type(image_field)!r}")


def can_open_image(image_bytes: bytes) -> bool:
    """Verify the bytes decode as an image without fully loading them."""
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.verify()
        return True
    except Exception:
        return False


def classify_answer(answer: str) -> str:
    """Coarse answer-type bucket for conversion stats."""
    a = str(answer).strip()
    if len(a) == 1 and a.upper() in "ABCDE":
        return "option"
    stripped = a.replace("-", "", 1).replace(".", "", 1)
    if stripped.isdigit():
        return "numeric"
    return "other"


def parse_options(question: str) -> dict[str, str]:
    """Extract choice options ``{A: content, B: content, ...}`` from question text.

    Matches lines like ``A. 5.25`` / ``B) content`` / ``C、 content``.  Returns
    empty dict if no options found (e.g. fill-in-the-blank questions).
    """
    opts = {}
    for m in _OPTION_LINE_RE.finditer(question or ""):
        letter, content = m.group(1).upper(), m.group(2).strip()
        opts[letter] = content
    return opts


def make_keep_fn(verify_images: bool):
    """Return a ``.filter()`` predicate that drops empty / undecodable rows."""

    def keep(example) -> bool:
        question = str(example.get("question") or "").strip()
        answer = str(example.get("answer") or "").strip()
        image_field = example.get("image")
        if not question or not answer or not is_valid_image(image_field):
            return False
        if verify_images and not can_open_image(normalize_image(image_field)["bytes"]):
            return False
        return True

    return keep


def make_map_fn(split: str):
    """Return a ``.map()`` function that builds one verl RL row per example."""

    def process_fn(example, idx):
        question = str(example.get("question") or "").strip()
        answer = str(example.get("answer") or "").strip()
        image_field = example.get("image")
        options = parse_options(question)
        return {
            "data_source": DATA_SOURCE,
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(question=question)},
            ],
            "images": [normalize_image(image_field)],
            "ability": ABILITY,
            "reward_model": {"style": "rule", "ground_truth": answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "id": str(example.get("id") or ""),
                "dataset": DATASET_NAME,
                "subject": str(example.get("subject") or ""),
                "raw_question": question,
                "raw_answer": answer,
                "options": json.dumps(options, ensure_ascii=False),  # JSON string; parsed in reward
            },
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MMK12 to verl RL parquet.")
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local dir with the raw MMK12 train-*.parquet / test-*.parquet shards.",
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/mmk12", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--no_verify_images", action="store_true", help="Skip PIL verify of image bytes.")
    args = parser.parse_args()

    local_dataset_path = args.local_dataset_path
    if local_dataset_path is None:
        raise ValueError("dataset does not exist. Download the raw parquet shards from HuggingFace.")

    data_files = {
        "train": sorted(glob.glob(os.path.join(local_dataset_path, "train-*.parquet"))),
        "test": sorted(glob.glob(os.path.join(local_dataset_path, "test-*.parquet"))),
    }
    if not data_files["train"] and not data_files["test"]:
        raise FileNotFoundError(f"No train-*.parquet / test-*.parquet shards found in {local_dataset_path!r}")
    dataset = datasets.load_dataset("parquet", data_files=data_files)

    verify_images = not args.no_verify_images
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split in ("train", "test"):
        split_dataset = dataset[split]
        # Keep image bytes raw (skip PIL decode) so the output bytes are identical to the source.
        split_dataset = split_dataset.cast_column("image", datasets.Image(decode=False))

        n_input = len(split_dataset)
        split_dataset = split_dataset.filter(make_keep_fn(verify_images), num_proc=8)
        split_dataset = split_dataset.map(
            function=make_map_fn(split),
            with_indices=True,
            num_proc=8,
            remove_columns=split_dataset.column_names,
        )
        n_kept = len(split_dataset)

        out_path = os.path.join(local_save_dir, f"{split}.parquet")
        split_dataset.to_parquet(out_path)

        answer_types = {"option": 0, "numeric": 0, "other": 0}
        for reward_model in split_dataset["reward_model"]:
            answer_types[classify_answer(reward_model["ground_truth"])] += 1
        print(f"[{split}] wrote {out_path}")
        print(f"  input={n_input} kept={n_kept} dropped={n_input - n_kept}")
        print(f"  answer_types={answer_types}")
