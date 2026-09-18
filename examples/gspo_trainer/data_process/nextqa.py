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
"""Convert the official NExT-QA annotations and videos into verl RL parquet."""

import argparse
import json
import math
import subprocess
from collections import Counter
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import pandas as pd

DATA_SOURCE = "nextqa"
ABILITY = "video_qa"
DATASET_NAME = "NExT-QA"
ANSWER_LETTERS = "ABCDE"
REQUIRED_COLUMNS = {"video", "question", "answer", "qid", "type", "a0", "a1", "a2", "a3", "a4"}
MEDIA_PROBE_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "Analyze the visual and audio information in the video and the question carefully. Explain your reasoning "
    "inside <think> </think> tags, then give "
    "only the single correct option letter inside <answer> </answer> tags. Your response must end in exactly this "
    "form: <answer>A</answer>, where A is replaced by the correct option letter."
)


def _text(value: Any) -> str | None:
    """Return stripped non-missing scalar text without turning NaN into ``"nan"``."""
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        return None
    text = str(value).strip()
    return text or None


def _identifier(value: Any) -> str | None:
    """Normalize pandas numeric identifiers to the string keys used by the map."""
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            return None
        return str(int(numeric_value))
    return _text(value)


def _answer_index(value: Any) -> int | None:
    """Parse an integer answer index without truncating fractional values."""
    if isinstance(value, Integral):
        answer = int(value)
    elif isinstance(value, Real):
        numeric_value = float(value)
        if not math.isfinite(numeric_value) or not numeric_value.is_integer():
            return None
        answer = int(numeric_value)
    else:
        text = _text(value)
        if text is None:
            return None
        try:
            answer = int(text)
        except ValueError:
            return None
    return answer if 0 <= answer < len(ANSWER_LETTERS) else None


def load_video_mapping(mapping_path: str | Path) -> dict[str, str]:
    """Load and validate the official video-ID to VidOR-path mapping."""
    mapping_path = Path(mapping_path).expanduser().resolve()
    try:
        raw_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in NExT-QA video mapping: {mapping_path}") from error
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Expected a JSON object in NExT-QA video mapping: {mapping_path}")

    mapping: dict[str, str] = {}
    for raw_video_id, raw_relative_path in raw_mapping.items():
        video_id = _identifier(raw_video_id)
        relative_path = _text(raw_relative_path)
        if video_id is None or relative_path is None:
            raise ValueError(f"Invalid entry in NExT-QA video mapping: {raw_video_id!r}: {raw_relative_path!r}")
        mapping[video_id] = relative_path
    return mapping


def resolve_video_path(video_root: Path, relative_path: Any) -> Path | None:
    """Resolve a mapped VidOR path below ``video_root`` and require an existing MP4."""
    mapped_path = _text(relative_path)
    if mapped_path is None:
        return None
    if Path(mapped_path).suffix.lower() != ".mp4":
        mapped_path += ".mp4"

    video_root = video_root.resolve()
    candidate = (video_root / mapped_path).resolve()
    try:
        candidate.relative_to(video_root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def build_options(record: dict[str, Any]) -> dict[str, str] | None:
    """Build the ordered A--E option map, rejecting missing option text."""
    options: dict[str, str] = {}
    for option_index, letter in enumerate(ANSWER_LETTERS):
        option = _text(record.get(f"a{option_index}"))
        if option is None:
            return None
        options[letter] = option
    return options


def build_rl_row(
    record: dict[str, Any],
    video_mapping: dict[str, str],
    video_root: Path,
    split: str,
    index: int,
    fps: float = 1.0,
    min_pixels: int = 32 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
    max_frames: int = 32,
) -> tuple[dict[str, Any] | None, str | None]:
    """Convert one official NExT-QA record, returning a row or drop reason."""
    question = _text(record.get("question"))
    if question is None:
        return None, "empty_question"

    options = build_options(record)
    if options is None:
        return None, "empty_option"

    answer_index = _answer_index(record.get("answer"))
    if answer_index is None:
        return None, "invalid_answer"
    answer_letter = ANSWER_LETTERS[answer_index]

    video_id = _identifier(record.get("video"))
    if video_id is None or video_id not in video_mapping:
        return None, "missing_video_mapping"
    video_path = resolve_video_path(video_root, video_mapping[video_id])
    if video_path is None:
        return None, "missing_video"

    qid = _identifier(record.get("qid"))
    problem_id = f"{video_id}_{qid}" if qid is not None else f"{video_id}_{index}"
    option_lines = "\n".join(f"{letter}. {option}" for letter, option in options.items())
    user_prompt = f"<video>{question}\n{option_lines}"
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
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
        "reward_model": {"style": "rule", "ground_truth": f"<answer>{answer_letter}</answer>"},
        "extra_info": {
            "split": split,
            "index": index,
            "problem_id": problem_id,
            "dataset": DATASET_NAME,
            "video_id": video_id,
            "qid": qid or "",
            "problem_type": _text(record.get("type")) or "",
            "raw_question": question,
            "answer_index": answer_index,
            "answer_letter": answer_letter,
            "options": json.dumps(options, ensure_ascii=False),
        },
    }, None


def _validate_csv_columns(frame: pd.DataFrame, source_path: Path) -> None:
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in {source_path}: {', '.join(missing_columns)}")


def _write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Dictionary-encoded nested columns can fail in some supported PyArrow
    # versions, so match the AVQA converter and keep nested columns plain.
    pd.DataFrame(rows).to_parquet(output_path, engine="pyarrow", index=False, use_dictionary=False)


def probe_audio_stream(video_path: str | Path) -> str | None:
    """Return a stable drop reason when the first audio stream is not decodable."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=MEDIA_PROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffprobe is required to filter NExT-QA videos without audio; install ffmpeg and ensure "
            "ffprobe is available in PATH"
        ) from error
    except subprocess.TimeoutExpired:
        return "invalid_media"

    if result.returncode != 0:
        return "invalid_media"
    if not result.stdout.strip():
        return "missing_audio_stream"

    try:
        decode_result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-map",
                "0:a:0",
                "-frames:a",
                "1",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=MEDIA_PROBE_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffmpeg is required to validate NExT-QA audio decoding; install ffmpeg and ensure "
            "ffmpeg is available in PATH"
        ) from error
    except subprocess.TimeoutExpired:
        return "invalid_media"

    return None if decode_result.returncode == 0 else "invalid_media"


def convert_split(
    source_path: str | Path,
    output_path: str | Path,
    video_mapping: dict[str, str],
    video_root: str | Path,
    split: str,
    fps: float = 1.0,
    min_pixels: int = 32 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
    max_frames: int = 32,
) -> dict[str, Any]:
    """Convert one official CSV split and write its valid rows to parquet."""
    source_path = Path(source_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve()
    frame = pd.read_csv(source_path)
    _validate_csv_columns(frame, source_path)

    rows: list[dict[str, Any]] = []
    dropped: Counter[str] = Counter()
    answer_counts: Counter[str] = Counter()
    audio_status_by_video: dict[str, str | None] = {}
    for index, record in enumerate(frame.to_dict(orient="records")):
        row, reason = build_rl_row(
            record,
            video_mapping,
            video_root,
            split=split,
            index=index,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_frames=max_frames,
        )
        if row is None:
            dropped[reason or "invalid_record"] += 1
            continue

        video_path = row["videos"][0]["video"]
        if video_path not in audio_status_by_video:
            audio_status_by_video[video_path] = probe_audio_stream(video_path)
        audio_reason = audio_status_by_video[video_path]
        if audio_reason is not None:
            dropped[audio_reason] += 1
            continue

        rows.append(row)
        answer_counts[row["extra_info"]["answer_letter"]] += 1

    if not rows:
        raise ValueError(f"No valid NExT-QA examples found in {source_path}; dropped={dict(dropped)}")

    _write_parquet(rows, output_path)
    audio_counts = Counter("with_audio" if status is None else status for status in audio_status_by_video.values())
    return {
        "input": len(frame),
        "kept": len(rows),
        "dropped": dict(sorted(dropped.items())),
        "audio": {
            "checked_videos": len(audio_status_by_video),
            "with_audio": audio_counts["with_audio"],
            "missing_audio_stream": audio_counts["missing_audio_stream"],
            "invalid_media": audio_counts["invalid_media"],
        },
        "answers": dict(sorted(answer_counts.items())),
        "output": str(output_path),
    }


def _validate_dataset_layout(input_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Validate the documented NExT-QA layout without guessing alternate roots."""
    repo_dir = input_dir / "repo"
    train_csv = repo_dir / "train.csv"
    val_csv = repo_dir / "val.csv"
    mapping_path = repo_dir / "map_vid_vidorID.json"
    video_root = input_dir / "NExTVideo"

    for required_file in (train_csv, val_csv, mapping_path):
        if not required_file.is_file():
            raise FileNotFoundError(f"Required NExT-QA file not found: {required_file}")
    if not video_root.is_dir():
        raise FileNotFoundError(
            f"Required NExT-QA video directory not found: {video_root}. Extract NExTVideo.zip from {input_dir} "
            "with `unzip NExTVideo.zip`; do not use `-d NExTVideo`."
        )
    if (video_root / "NExTVideo").is_dir():
        raise ValueError(
            f"Invalid nested NExT-QA video directory: {video_root / 'NExTVideo'}. Re-extract NExTVideo.zip from "
            f"{input_dir} with `unzip NExTVideo.zip`; the archive already contains NExTVideo/."
        )
    return train_csv, val_csv, mapping_path, video_root


def convert_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    fps: float = 1.0,
    min_pixels: int = 32 * 28 * 28,
    max_pixels: int = 128 * 28 * 28,
    max_frames: int = 32,
) -> dict[str, dict[str, Any]]:
    """Validate and convert the official train and validation splits."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("pixel limits must satisfy 0 < min_pixels <= max_pixels")
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")

    input_dir = Path(input_dir).expanduser().resolve()
    train_csv, val_csv, mapping_path, video_root = _validate_dataset_layout(input_dir)
    video_mapping = load_video_mapping(mapping_path)
    output_dir = Path(output_dir).expanduser().resolve()

    split_specs = (
        ("train", train_csv, output_dir / "train.parquet"),
        ("validation", val_csv, output_dir / "validation.parquet"),
    )
    stats: dict[str, dict[str, Any]] = {}
    for split, source_path, output_path in split_specs:
        stats[split] = convert_split(
            source_path,
            output_path,
            video_mapping,
            video_root,
            split,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_frames=max_frames,
        )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "NextQA directory containing repo/{train.csv,val.csv,map_vid_vidorID.json} and NExTVideo/. "
            "ffprobe must be available in PATH."
        ),
    )
    parser.add_argument("--output_dir", required=True, help="Directory for train.parquet and validation.parquet.")
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling rate passed to qwen_omni_utils.")
    parser.add_argument("--min_pixels", type=int, default=32 * 28 * 28, help="Minimum pixels per sampled frame.")
    parser.add_argument("--max_pixels", type=int, default=128 * 28 * 28, help="Maximum pixels per sampled frame.")
    parser.add_argument("--max_frames", type=int, default=32, help="Maximum sampled frames per video.")
    args = parser.parse_args()

    stats = convert_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_frames=args.max_frames,
    )
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
