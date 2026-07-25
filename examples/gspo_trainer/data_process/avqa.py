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
"""Convert AVQA-R1-6K JSON into verl RL parquet (image + audio + text -> text)."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

DATA_SOURCE = "avqa_r1_6k"
ABILITY = "audio_visual_qa"
DATASET_NAME = "AVQA-R1-6K"

SYSTEM_PROMPT = (
    "Please think about this question as if you were a human pondering deeply, carefully considering both the "
    "visual and audio information before answering, engaging in an internal dialogue using expressions such as "
    "let me think, wait, hmm, oh I see, or let's break it down, including self-reflection or verification in the "
    "reasoning process, providing the detailed reasoning between the <think> </think> tags, and finally giving "
    "only the single option letter (e.g., A, B, C, D, etc.) as the final answer within the <answer> </answer> tags."
)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-H])\s*</answer>", re.IGNORECASE)
_OPTION_RE = re.compile(r"^\s*([A-H])[.)、:]\s*(.+?)\s*$", re.IGNORECASE)


def extract_answer(solution: Any) -> str | None:
    """Extract and normalize a single option letter from the source solution."""
    match = _ANSWER_TAG_RE.fullmatch(str(solution or "").strip())
    return match.group(1).upper() if match else None


def normalize_options(options: Any) -> tuple[list[str], dict[str, str]]:
    """Validate source options and return display lines plus a letter-to-text map."""
    if not isinstance(options, list) or not options:
        return [], {}

    lines: list[str] = []
    option_map: dict[str, str] = {}
    for raw_option in options:
        option = str(raw_option or "").strip()
        match = _OPTION_RE.fullmatch(option)
        if match is None:
            return [], {}
        letter, content = match.group(1).upper(), match.group(2).strip()
        if not content or letter in option_map:
            return [], {}
        lines.append(f"{letter}. {content}")
        option_map[letter] = content
    return lines, option_map


def resolve_media_path(split_dir: Path, media_path: Any) -> Path | None:
    """Resolve a source media path and reject missing files or path traversal."""
    raw_path = str(media_path or "").strip()
    if not raw_path:
        return None

    split_dir = split_dir.resolve()
    candidate = (split_dir / raw_path).resolve()
    try:
        candidate.relative_to(split_dir)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def build_rl_row(record: dict[str, Any], split_dir: Path, split: str, index: int) -> tuple[dict | None, str | None]:
    """Convert one AVQA record, returning a row or a stable drop reason."""
    problem = str(record.get("problem") or "").strip()
    if not problem:
        return None, "empty_problem"
    if record.get("data_type") != "image_audio":
        return None, "unsupported_modality"

    option_lines, option_map = normalize_options(record.get("options"))
    if not option_lines:
        return None, "invalid_options"

    answer = extract_answer(record.get("solution"))
    if answer is None or answer not in option_map:
        return None, "invalid_solution"

    media = record.get("path") or {}
    image_path = resolve_media_path(split_dir, media.get("image"))
    audio_path = resolve_media_path(split_dir, media.get("audio"))
    if image_path is None:
        return None, "missing_image"
    if audio_path is None:
        return None, "missing_audio"

    question = f"<image><audio>{problem}\nOptions:\n" + "\n".join(option_lines)
    problem_id = record.get("problem_id", index)
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        # Media paths intentionally remain external to parquet. Every training
        # node must mount the converted dataset at the same absolute path.
        "images": [{"image": str(image_path)}],
        "audios": [str(audio_path)],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": str(record["solution"]).strip()},
        "extra_info": {
            "split": split,
            "index": index,
            "problem_id": str(problem_id),
            "dataset": DATASET_NAME,
            "problem_type": str(record.get("problem_type") or ""),
            "raw_problem": problem,
            "raw_solution": str(record.get("solution") or ""),
            "options": json.dumps(option_map, ensure_ascii=False),
        },
    }, None


def convert_split(input_json: str | Path, output_path: str | Path, split: str) -> dict[str, Any]:
    """Convert one split and write it to parquet."""
    input_json = Path(input_json).expanduser().resolve()
    records = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {input_json}")

    rows: list[dict] = []
    dropped: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            dropped["invalid_record"] += 1
            continue
        row, reason = build_rl_row(record, input_json.parent, split=split, index=index)
        if row is None:
            dropped[reason or "invalid_record"] += 1
            continue
        rows.append(row)
        label = extract_answer(row["reward_model"]["ground_truth"]) or "invalid"
        answer_counts[label] += 1

    if not rows:
        raise ValueError(f"No valid AVQA examples found in {input_json}; dropped={dict(dropped)}")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False)
    return {
        "input": len(records),
        "kept": len(rows),
        "dropped": dict(sorted(dropped.items())),
        "answers": dict(sorted(answer_counts.items())),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help="AVQA_R1 directory containing train/ and valid/.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for train.parquet and validation.parquet.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    split_specs = (
        ("train", input_dir / "train" / "omni_rl_format_train.json", output_dir / "train.parquet"),
        ("validation", input_dir / "valid" / "omni_rl_format_valid.json", output_dir / "validation.parquet"),
    )
    for split, input_json, output_path in split_specs:
        stats = convert_split(input_json, output_path, split)
        print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
