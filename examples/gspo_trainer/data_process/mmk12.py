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
"""Convert MMK12 parquet to verl RL parquet (image-in / text-out).

Raw MMK12 schema:
    id, question, answer, subject, image   # image is {"bytes": <png bytes>}

Output verl RL schema (one row per problem):
    data_source  = "math_dapo"   # routes through upstream math_dapo rule reward
    prompt       = [{"role": "system", "content": <SYSTEM_PROMPT>},
                    {"role": "user", "content": <prompt_text>},]
    images       = [{"bytes": <png bytes>}]
    ability      = "math_vl"
    reward_model = {"style": "rule", "ground_truth": <answer>}
    extra_info   = {split, index, id, dataset, subject, raw_question, raw_answer, options}

Pure helpers (is_valid_image, normalize_image, classify_answer) are importable
without torch/verl so they can be unit-tested on CPU.
"""

import argparse
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
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
    if len(a) == 1 and a.upper() in "ABCDEFGH":
        return "option"
    stripped = a.replace("-", "", 1).replace(".", "", 1)
    if stripped.isdigit():
        return "numeric"
    return "other"


_OPTION_LINE_RE = re.compile(r"^\s*([A-E])[.)、]\s*(.+?)\s*$", re.MULTILINE)


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


def _build_rl_row(row: dict, split: str, index: int, verify_images: bool = True) -> tuple[dict | None, str | None]:
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()
    image_field = row.get("image")

    if not question:
        return None, "dropped_empty_question"
    if not answer:
        return None, "dropped_empty_answer"
    if not is_valid_image(image_field):
        return None, "dropped_bad_image"

    image_dict = normalize_image(image_field)
    if verify_images and not can_open_image(image_dict["bytes"]):
        return None, "dropped_bad_image"

    prompt_text = USER_PROMPT_TEMPLATE.format(question=question)
    options = parse_options(question)
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "images": [image_dict],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": split,
            "index": index,
            "id": str(row.get("id") or ""),
            "dataset": DATASET_NAME,
            "subject": str(row.get("subject") or ""),
            "raw_question": question,
            "raw_answer": answer,
            "options": json.dumps(options, ensure_ascii=False),  # JSON string; parse in reward
        },
    }, None


def convert_dataset(
    input_paths: list[str],
    output_path: str,
    split: str,
    verify_images: bool = True,
) -> dict:
    """Read input parquet shards, convert to verl RL rows, write one parquet file.

    Returns a stats dict with input/kept/dropped counts and answer-type tallies.
    """
    frames = [pd.read_parquet(p, engine="pyarrow") for p in input_paths]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    stats = {
        "input": len(df),
        "kept": 0,
        "dropped_empty_question": 0,
        "dropped_empty_answer": 0,
        "dropped_bad_image": 0,
        "answer_types": {"option": 0, "numeric": 0, "other": 0},
    }

    out_rows: list[dict] = []
    for index, row in df.iterrows():
        rl_row, drop_reason = _build_rl_row(row.to_dict(), split=split, index=int(index), verify_images=verify_images)
        if rl_row is None:
            stats[drop_reason] += 1
            continue
        stats["answer_types"][classify_answer(rl_row["reward_model"]["ground_truth"])] += 1
        out_rows.append(rl_row)

    stats["kept"] = len(out_rows)
    out_df = pd.DataFrame(out_rows)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    out_df.to_parquet(output_path, engine="pyarrow", index=False)
    return stats


def _default_input_shards(data_dir: str, split: str) -> list[str]:
    pattern = f"{split}-*.parquet"
    return sorted(str(p) for p in Path(data_dir).glob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MMK12 to verl RL parquet.")
    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--no_verify_images", action="store_true", help="Skip PIL verify of image bytes.")
    args = parser.parse_args()

    verify = not args.no_verify_images
    for split in ("train", "test"):
        shards = _default_input_shards(args.input_dir, split)
        if not shards:
            print(f"[{split}] no input shards found in {args.input_dir}, skipping")
            continue
        out_path = os.path.join(args.output_dir, f"{split}.parquet")
        stats = convert_dataset(shards, out_path, split=split, verify_images=verify)
        print(f"[{split}] wrote {out_path}")
        print(f"  input={stats['input']} kept={stats['kept']}")
        print(
            f"  dropped: empty_q={stats['dropped_empty_question']} "
            f"empty_a={stats['dropped_empty_answer']} bad_img={stats['dropped_bad_image']}"
        )
        print(f"  answer_types={stats['answer_types']}")


if __name__ == "__main__":
    main()
