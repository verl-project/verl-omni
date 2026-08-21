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
"""Convert TinyLLaVA-Video-R1 NextQA JSONL into verl RL parquet."""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

DATA_SOURCE = "nextqa"
ABILITY = "video_qa"
DATASET_NAME = "TinyLLaVA-Video-R1-NextQA"
DEFAULT_INPUT_FILE = "nextqa_0-30s.jsonl"

SYSTEM_PROMPT = (
    "Analyze the visual and audio information in the video and the question carefully. Explain your reasoning "
    "inside <think> </think> tags, then give "
    "only the single correct option letter inside <answer> </answer> tags. Your response must end in exactly this "
    "form: <answer>A</answer>, where A is replaced by the correct option letter."
)

_ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-E])\s*</answer>", re.IGNORECASE)
_OPTION_RE = re.compile(r"^\s*([A-E])[.)]\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def extract_answer(solution: Any) -> str | None:
    """Extract a normalized option letter from an exactly tagged solution."""
    match = _ANSWER_TAG_RE.fullmatch(str(solution or "").strip())
    return match.group(1).upper() if match else None


def parse_options(problem: Any) -> dict[str, str]:
    """Extract the five NextQA options embedded in the problem text."""
    options: dict[str, str] = {}
    for match in _OPTION_RE.finditer(str(problem or "")):
        letter, content = match.group(1).upper(), match.group(2).strip()
        if not content or letter in options:
            return {}
        options[letter] = content
    return options if set(options) == set("ABCDE") else {}


def resolve_video_path(input_dir: Path, video_filename: Any) -> Path | None:
    """Resolve a video path below ``input_dir`` and reject traversal/missing files."""
    raw_path = str(video_filename or "").strip()
    if not raw_path:
        return None

    input_dir = input_dir.resolve()
    candidate = (input_dir / raw_path).resolve()
    try:
        candidate.relative_to(input_dir)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def build_rl_row(
    record: dict[str, Any],
    input_dir: Path,
    split: str,
    index: int,
    fps: float = 1.0,
    min_pixels: int = 32 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
    max_frames: int = 32,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one source record, returning a row or a stable drop reason."""
    problem = str(record.get("problem") or "").strip()
    if not problem:
        return None, "empty_problem"
    if record.get("data_type") != "video":
        return None, "unsupported_modality"

    options = parse_options(problem)
    if not options:
        return None, "invalid_options"

    answer = extract_answer(record.get("solution"))
    if answer is None or answer not in options:
        return None, "invalid_solution"

    video_path = resolve_video_path(input_dir, record.get("video_filename"))
    if video_path is None:
        return None, "missing_video"

    problem_id = record.get("problem_id", index)
    problem_type = record.get("problem_type")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<video>{problem}"},
        ],
        # Media stays outside parquet. Mount the input directory at the same
        # absolute path on every Ray worker.
        "videos": [
            {
                "video": str(video_path),
                "fps": fps,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "max_frames": max_frames,
            }
        ],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": f"<answer>{answer}</answer>"},
        "extra_info": {
            "split": split,
            "index": index,
            "problem_id": str(problem_id),
            "dataset": DATASET_NAME,
            "problem_type": json.dumps(problem_type, ensure_ascii=False),
            "raw_problem": problem,
            "raw_solution": str(record.get("solution") or ""),
            "video_filename": str(record.get("video_filename") or ""),
            "options": json.dumps(options, ensure_ascii=False),
        },
    }, None


def _validation_videos(rows: list[dict[str, Any]], validation_ratio: float, seed: int) -> set[str]:
    """Choose validation video groups deterministically to prevent media leakage."""
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if validation_ratio == 0:
        return set()

    videos = sorted({row["videos"][0]["video"] for row in rows})
    if len(videos) < 2:
        raise ValueError("At least two distinct videos are required when validation_ratio is greater than zero")
    random.Random(seed).shuffle(videos)
    validation_count = min(len(videos) - 1, max(1, round(len(videos) * validation_ratio)))
    return set(videos[:validation_count])


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Dictionary-encoded nested columns can fail in some supported PyArrow
    # versions, so match the AVQA converter and keep nested columns plain.
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False, use_dictionary=False)


def convert_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    input_file: str = DEFAULT_INPUT_FILE,
    validation_ratio: float = 0.05,
    seed: int = 42,
    fps: float = 1.0,
    min_pixels: int = 32 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
    max_frames: int = 32,
) -> dict[str, Any]:
    """Validate, group-split, and write the NextQA training parquets."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("pixel limits must satisfy 0 < min_pixels <= max_pixels")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    input_dir = Path(input_dir).expanduser().resolve()
    source_path = (input_dir / input_file).resolve()
    try:
        source_path.relative_to(input_dir)
    except ValueError as error:
        raise ValueError("input_file must resolve below input_dir") from error
    if not source_path.is_file():
        raise FileNotFoundError(f"NextQA annotation file not found: {source_path}")

    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    input_count = 0
    with source_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            input_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                dropped["invalid_json"] += 1
                continue
            if not isinstance(record, dict):
                dropped["invalid_record"] += 1
                continue
            row, reason = build_rl_row(
                record,
                input_dir,
                split="train",
                index=index,
                fps=fps,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                max_frames=max_frames,
            )
            if row is None:
                dropped[reason or "invalid_record"] += 1
                continue
            rows.append(row)
            answer_counts[extract_answer(row["reward_model"]["ground_truth"]) or "invalid"] += 1

    if not rows:
        raise ValueError(f"No valid NextQA examples found in {source_path}; dropped={dict(dropped)}")

    validation_videos = _validation_videos(rows, validation_ratio=validation_ratio, seed=seed)
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["videos"][0]["video"] in validation_videos:
            row["extra_info"]["split"] = "validation"
            validation_rows.append(row)
        else:
            train_rows.append(row)

    output_dir = Path(output_dir).expanduser().resolve()
    _write_parquet(train_rows, output_dir / "train.parquet")
    if validation_rows:
        _write_parquet(validation_rows, output_dir / "validation.parquet")

    return {
        "input": input_count,
        "kept": len(rows),
        "train": len(train_rows),
        "validation": len(validation_rows),
        "dropped": dict(sorted(dropped.items())),
        "answers": dict(sorted(answer_counts.items())),
        "train_output": str(output_dir / "train.parquet"),
        "validation_output": str(output_dir / "validation.parquet") if validation_rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True, help="Directory containing JSONL and the extracted NextQA/ tree.")
    parser.add_argument("--output_dir", required=True, help="Directory for train.parquet and validation.parquet.")
    parser.add_argument("--input_file", default=DEFAULT_INPUT_FILE, help="Annotation filename relative to input_dir.")
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.05,
        help="Fraction of distinct videos assigned to validation (default: 0.05).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic video-group splitting.")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling rate passed to qwen_omni_utils.")
    parser.add_argument("--min_pixels", type=int, default=32 * 28 * 28, help="Minimum pixels per sampled frame.")
    parser.add_argument("--max_pixels", type=int, default=128 * 28 * 28, help="Maximum pixels per sampled frame.")
    parser.add_argument("--max_frames", type=int, default=32, help="Maximum sampled frames per video.")
    args = parser.parse_args()

    stats = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_file=args.input_file,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
