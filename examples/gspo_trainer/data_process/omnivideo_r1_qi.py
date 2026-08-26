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
"""Convert OmniVideo-R1 QI JSONL into verl RL parquet."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

ABILITY = "omnivideo_qi"
DATASET_NAME = "OmniVideo-R1"

SYSTEM_PROMPT = """You are an expert tasked with solving problems based on video and audio content.
When answering questions about videos with audio, carefully observe and analyze important clues.
For each key segment that may help answer the question, first note the time range using one decimal place
for seconds (for example, 6.2-9.3), with no overlap between segments. Then describe the key visual and
audio clues using <time>start_time-end_time</time><caption>Description of key clues</caption>.
After extracting several relevant clues, integrate them and outline your overall reasoning inside
<thinking></thinking> tags. Approach the question as a human reflecting deeply within the visual and audio
context, using natural thought expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', or
'let's break it down'. Finally, present your final answer inside <answer></answer> tags.

Example:
<time>1.0-2.5</time><caption>A person dips a paintbrush into red paint, with a soft brushing sound.</caption>
<time>5.2-6.8</time><caption>The person paints a sun on a white canvas.</caption>
<thinking>Let me think. The first segment shows preparation, and the second shows a sun being painted.
The main action is therefore painting.</thinking>
<answer>Painting a sun on a canvas</answer>"""

PROMPT_SUFFIX = """First, extract helpful segments for answering the question in this format:
<time>start_time-end_time</time><caption>Key clue description</caption>. Then provide the reasoning and final
answer as <thinking>Reasoning process here</thinking><answer>Answer here</answer>."""


def parse_path_map(value: str) -> tuple[str, Path]:
    """Parse ``SOURCE=DEST`` without requiring SOURCE to exist locally."""
    try:
        source, destination = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("path maps must use SOURCE=DEST") from exc
    source = source.rstrip("/")
    if not source or not destination:
        raise argparse.ArgumentTypeError("path maps must use non-empty SOURCE=DEST")
    return source, Path(destination).expanduser().resolve()


def resolve_media_path(raw_path: Any, path_maps: list[tuple[str, Path]]) -> Path | None:
    """Resolve one annotation path through an explicit longest-prefix map."""
    raw = str(raw_path or "").strip()
    if not raw:
        return None

    direct = Path(raw).expanduser()
    if direct.is_absolute() and direct.is_file():
        return direct.resolve()

    for source, destination in sorted(path_maps, key=lambda item: len(item[0]), reverse=True):
        if raw == source or raw.startswith(f"{source}/"):
            suffix = raw[len(source) :].lstrip("/")
            candidate = (destination / suffix).resolve()
            return candidate if candidate.is_file() else None
    return None


def _single_path(record: dict[str, Any], key: str) -> Any:
    values = record.get(key)
    if not isinstance(values, list) or len(values) != 1:
        return None
    return values[0]


def _question_text(problem: Any, audio_from_video: bool) -> str:
    question = str(problem or "").strip()
    if not question:
        return ""
    if "<video>" not in question:
        question = f"<video><audio>\n{question}"
    if audio_from_video:
        question = question.replace("<audio>", "")
    elif "<audio>" not in question:
        question = question.replace("<video>", "<video><audio>", 1)
    return question


def build_rl_row(
    record: dict[str, Any],
    index: int,
    path_maps: list[tuple[str, Path]],
    *,
    audio_from_video: bool,
    fps: float,
    max_frames: int,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one QI annotation, returning a row or stable drop reason."""
    question = _question_text(record.get("problem"), audio_from_video)
    solution = str(record.get("solution") or "").strip()
    if not question:
        return None, "empty_problem"
    if not solution:
        return None, "empty_solution"
    if question.count("<video>") != 1:
        return None, "invalid_video_placeholder"

    raw_video_path = _single_path(record, "videos")
    video_path = resolve_media_path(raw_video_path, path_maps)
    if video_path is None:
        return None, "missing_video"

    row: dict[str, Any] = {
        "data_source": str(record.get("data_source") or "omnivideo_r1_qi"),
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{question}\n\n{PROMPT_SUFFIX}"},
        ],
        "videos": [{"video": str(video_path), "fps": fps, "min_frames": 4, "max_frames": max_frames}],
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {
            "id": str(record.get("id") or index),
            "index": index,
            "dataset": DATASET_NAME,
            "type": str(record.get("Type") or ""),
            "question": question.replace("<video>", "").replace("<audio>", "").strip(),
            "is_multiple_choice": "_mc_" in str(record.get("Type") or "").lower(),
            "video_path": str(video_path),
            "original_video_path": str(raw_video_path or ""),
        },
    }

    if audio_from_video:
        if "<audio>" in question:
            return None, "invalid_audio_placeholder"
    else:
        if question.count("<audio>") != 1:
            return None, "invalid_audio_placeholder"
        raw_audio_path = _single_path(record, "audios")
        audio_path = resolve_media_path(raw_audio_path, path_maps)
        if audio_path is None:
            return None, "missing_audio"
        row["audios"] = [str(audio_path)]
        row["extra_info"]["audio_path"] = str(audio_path)
    return row, None


def convert_jsonl(
    input_jsonl: str | Path,
    output_dir: str | Path,
    path_maps: list[tuple[str, Path]],
    *,
    audio_from_video: bool = True,
    fps: float = 2.0,
    max_frames: int = 64,
    max_samples: int = 0,
    val_size: int = 256,
    seed: int = 42,
) -> dict[str, Any]:
    """Convert JSONL and split by video so train/validation never leak media."""
    input_jsonl = Path(input_jsonl).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    read = 0
    with input_jsonl.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            read += 1
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
                index,
                path_maps,
                audio_from_video=audio_from_video,
                fps=fps,
                max_frames=max_frames,
            )
            if row is None:
                dropped[reason or "invalid_record"] += 1
                continue
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break

    if len(rows) < 2:
        raise ValueError(f"Need at least two valid QI samples; kept={len(rows)}, dropped={dict(dropped)}")

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[row["extra_info"]["video_path"]].append(row)
    video_paths = sorted(by_video)
    random.Random(seed).shuffle(video_paths)

    target_val_size = min(max(val_size, 1), len(rows) - 1)
    val_paths: list[str] = []
    val_count = 0
    for video_path in video_paths:
        if val_count >= target_val_size and val_paths:
            break
        if len(by_video[video_path]) >= len(rows):
            continue
        val_paths.append(video_path)
        val_count += len(by_video[video_path])

    val_path_set = set(val_paths)
    validation_rows = [row for path in val_paths for row in by_video[path]]
    train_rows = [row for path in video_paths if path not in val_path_set for row in by_video[path]]
    if not train_rows or not validation_rows:
        raise ValueError("Could not create non-empty train/validation splits from distinct videos")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "validation.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, engine="pyarrow", index=False, use_dictionary=False)
    pd.DataFrame(validation_rows).to_parquet(validation_path, engine="pyarrow", index=False, use_dictionary=False)

    return {
        "read": read,
        "kept": len(rows),
        "dropped": dict(sorted(dropped.items())),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train": str(train_path),
        "validation": str(validation_path),
        "audio_from_video": audio_from_video,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to merged_train_all_qi.jsonl.")
    parser.add_argument("--output_dir", required=True, help="Directory for train/validation parquet.")
    parser.add_argument(
        "--path_map",
        action="append",
        default=[],
        type=parse_path_map,
        metavar="SOURCE=DEST",
        help="Repeatable annotation-prefix to local-directory mapping.",
    )
    parser.add_argument(
        "--audio_from_video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read the audio stream from each video instead of resolving the audios field.",
    )
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0, help="Stop after this many valid rows; 0 keeps all.")
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.path_map:
        parser.error("at least one --path_map is required")
    if args.fps <= 0 or args.max_frames < 4 or args.val_size < 1 or args.max_samples < 0:
        parser.error("fps must be > 0, max_frames >= 4, val_size >= 1, and max_samples >= 0")

    stats = convert_jsonl(
        args.input,
        args.output_dir,
        args.path_map,
        audio_from_video=args.audio_from_video,
        fps=args.fps,
        max_frames=args.max_frames,
        max_samples=args.max_samples,
        val_size=args.val_size,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
