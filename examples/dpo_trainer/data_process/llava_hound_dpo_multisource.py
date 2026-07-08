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

"""Build VeOmni-style multisource LLaVA-Hound-DPO parquet data.

The same LLaVA-Hound-DPO preference pair is converted into three offline DPO
sources:

* image + text: first video frame plus question
* text only: question without media
* video + text: original video plus question

Train/test splitting is performed at video-name granularity so validation
samples never share a source video with training samples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = "You are a helpful assistant."
VIDEO_FILE_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi")


def _content_item(item_type: str, *, text: str | None = None, image="", video="", audio="") -> dict:
    return {"type": item_type, "text": text, "image": image, "video": video, "audio": audio}


FRAME_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ANNOTATION_URL = (
    "https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction"
    "/resolve/main/video_instruction/train/dpo/sft_dpo_17k.jsonl?download=true"
)

SOURCE_SPECS = {
    "image": {
        "data_source": "llava_hound_dpo/image",
        "ability": "image_qa",
        "name": "LLaVA-Hound-DPO-Image",
    },
    "text": {
        "data_source": "llava_hound_dpo/text",
        "ability": "text_qa",
        "name": "LLaVA-Hound-DPO-Text",
    },
    "video": {
        "data_source": "llava_hound_dpo/video",
        "ability": "video_qa",
        "name": "LLaVA-Hound-DPO-Video",
    },
}

SOURCE_ORDER = ["image", "text", "video"]


def _strip_video_token(text: str) -> str:
    return text.replace("<video>\n", "").replace("<video>", "").strip()


def _video_key(video_rel: str) -> str:
    """Stable group key used to keep all rows from one video in one split."""
    return Path(video_rel.replace("\\", "/")).name or video_rel


def _frame_paths_for_dir(path: str) -> list[str]:
    frames = [
        os.path.abspath(os.path.join(path, name))
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name)) and Path(name).suffix.lower() in FRAME_IMAGE_EXTENSIONS
    ]
    return sorted(frames)


def _resolve_video(video_rel: str, video_dir: str) -> str | list[str] | None:
    for rel in (video_rel, video_rel.lstrip("/").lstrip("\\")):
        candidate = os.path.join(video_dir, rel)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if os.path.isdir(candidate):
            frames = _frame_paths_for_dir(candidate)
            if frames:
                return frames
        if not Path(rel).suffix:
            for extension in VIDEO_FILE_EXTENSIONS:
                candidate_with_extension = candidate + extension
                if os.path.isfile(candidate_with_extension):
                    return os.path.abspath(candidate_with_extension)
    return None


def _first_frame(video_media: str | list[str]) -> str | None:
    if isinstance(video_media, list):
        return video_media[0] if video_media else None
    return None


def _video_prompt_path(video_media: str | list[str]) -> str:
    if isinstance(video_media, list):
        return os.path.dirname(video_media[0])
    return video_media


def _image_path_for_video(video_rel: str, image_dir: str) -> str:
    rel = Path(video_rel.replace("\\", "/").lstrip("/")).with_suffix(".jpg")
    return os.path.abspath(os.path.join(image_dir, os.fspath(rel)))


def _extract_first_frame(video_path: str, image_path: str) -> bool:
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            image_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        return False
    try:
        return os.path.getsize(image_path) > 0
    except OSError:
        return False


def _download_annotation(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading annotation to {dest} ...", flush=True)
    try:
        subprocess.run(["wget", "-c", "-O", dest, url], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        subprocess.run(["curl", "-L", "-C", "-", "-o", dest, url], check=True)
    print("Download complete.", flush=True)


def _read_records(dpo_jsonl: str) -> list[dict]:
    records: list[dict] = []
    with open(dpo_jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _split_video_keys(records: list[dict], test_ratio: float, seed: int) -> set[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_video_key(record.get("video", ""))].append(record)

    groups = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    target_test_rows = max(1, round(len(records) * test_ratio))
    test_keys: set[str] = set()
    test_rows = 0

    for key, group_records in groups:
        if len(test_keys) and test_rows >= target_test_rows:
            break
        test_keys.add(key)
        test_rows += len(group_records)

    if len(test_keys) == len(groups) and len(groups) > 1:
        # Keep at least one video group for training when tiny subsets are used.
        test_keys.remove(groups[-1][0])
    return test_keys


def _text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return str(value.get("value", "")).strip()
    return str(value).strip()


def _question(record: dict) -> str:
    for turn in record.get("conversations", []):
        if turn.get("from") == "human":
            return _strip_video_token(turn.get("value", ""))
    prompt = _text_value(record.get("prompt"))
    if prompt:
        return _strip_video_token(prompt)
    return ""


def _base_row(record: dict, split: str, index: int, source: str, question: str) -> dict | None:
    chosen = _text_value(record.get("chosen"))
    rejected = _text_value(record.get("rejected"))
    if not question or not chosen or not rejected:
        return None

    spec = SOURCE_SPECS[source]
    video_rel = record.get("video", "")
    sample_id = str(record.get("id", f"{split}_{index}"))

    return {
        "data_source": spec["data_source"],
        "chosen": chosen,
        "rejected": rejected,
        "win_score": float(record.get("chosen_score", 1.0)),
        "lose_score": float(record.get("rejected_score", 0.0)),
        "ability": spec["ability"],
        "reward_model": {"style": "preference"},
        "extra_info": {
            "split": split,
            "index": index,
            "sample_id": sample_id,
            "question": question,
            "source_video": video_rel,
            "source_video_name": _video_key(video_rel),
        },
    }


def _build_text_row(record: dict, split: str, index: int) -> dict | None:
    question = _question(record)
    row = _base_row(record, split, index, "text", question)
    if row is None:
        return None
    row["prompt"] = [
        {"role": "system", "content": [_content_item("text", text=SYSTEM_PROMPT)]},
        {"role": "user", "content": [_content_item("text", text=question)]},
    ]
    return row


def _build_video_row(record: dict, split: str, index: int, video_dir: str) -> dict | None:
    video_media = _resolve_video(record.get("video", ""), video_dir)
    if video_media is None:
        return None

    question = _question(record)
    row = _base_row(record, split, index, "video", question)
    if row is None:
        return None
    row["prompt"] = [
        {"role": "system", "content": [_content_item("text", text=SYSTEM_PROMPT)]},
        {
            "role": "user",
            "content": [
                _content_item("video", video=_video_prompt_path(video_media)),
                _content_item("text", text=question),
            ],
        },
    ]
    if isinstance(video_media, list):
        row["extra_info"]["video_frame_count"] = len(video_media)
        row["extra_info"]["video_frame_dir"] = os.path.dirname(video_media[0])
    else:
        row["extra_info"]["video_path"] = video_media
    return row


def _build_image_row(
    record: dict,
    split: str,
    index: int,
    video_dir: str,
    image_dir: str,
    extract_images: bool,
) -> dict | None:
    video_rel = record.get("video", "")
    video_media = _resolve_video(video_rel, video_dir)
    if video_media is None:
        return None

    first_frame = _first_frame(video_media)
    if first_frame is not None:
        image_path = first_frame
    else:
        image_path = _image_path_for_video(video_rel, image_dir)
        if not os.path.isfile(image_path):
            if not extract_images:
                return None
            if not isinstance(video_media, str) or not _extract_first_frame(video_media, image_path):
                return None
        if not os.path.isfile(image_path):
            return None

    question = _question(record)
    row = _base_row(record, split, index, "image", question)
    if row is None:
        return None
    row["prompt"] = [
        {"role": "system", "content": [_content_item("text", text=SYSTEM_PROMPT)]},
        {
            "role": "user",
            "content": [
                _content_item("image", image=image_path),
                _content_item("text", text=question),
            ],
        },
    ]
    if isinstance(video_media, list):
        row["extra_info"]["video_frame_count"] = len(video_media)
        row["extra_info"]["video_frame_dir"] = os.path.dirname(video_media[0])
    else:
        row["extra_info"]["video_path"] = video_media
    row["extra_info"]["image_path"] = image_path
    return row


def _write_split(rows: list[dict], output_dir: str, source: str) -> None:
    train_rows = [row for row in rows if row["extra_info"]["split"] == "train"]
    test_rows = [row for row in rows if row["extra_info"]["split"] == "test"]

    source_dir = os.path.join(output_dir, source)
    os.makedirs(source_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(source_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(source_dir, "test.parquet"), index=False)
    print(f"  {source}: train={len(train_rows):,}, test={len(test_rows):,}", flush=True)


def _default_multisource_config_path() -> str:
    repo_example_dir = Path(__file__).resolve().parent.parent / "qwen3_omni"
    return os.fspath(repo_example_dir / "llava_hound_dpo_multisource.yaml")


def _write_multisource_config(output_dir: str, config_path: str, source_weights: list[float]) -> None:
    lines = [
        "sources:",
        *[f"- {os.path.join(output_dir, source, 'train.parquet')}" for source in SOURCE_ORDER],
        "names:",
        *[f"- {SOURCE_SPECS[source]['name']}" for source in SOURCE_ORDER],
        "schedule:",
        "- schedule_type: const",
        f"  weights: [{', '.join(str(weight) for weight in source_weights)}]",
        "val_sources:",
        *[f"- {os.path.join(output_dir, source, 'test.parquet')}" for source in SOURCE_ORDER],
        "",
    ]
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  config: {config_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert LLaVA-Hound-DPO into image/text/video offline DPO multisource parquet.",
    )
    parser.add_argument("--dpo_jsonl", required=True, help="Path to sft_dpo_17k.jsonl.")
    parser.add_argument("--video_dir", required=True, help="Root directory containing unpacked MP4 files.")
    parser.add_argument(
        "--image_dir",
        default=None,
        help="Directory for extracted image frames. Defaults to <output_dir>/../images.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/data/llava_hound_dpo/parquet"),
        help="Directory to write image/text/video parquet folders.",
    )
    parser.add_argument(
        "--source_weights",
        nargs=3,
        type=float,
        required=True,
        metavar=("IMAGE_WEIGHT", "TEXT_WEIGHT", "VIDEO_WEIGHT"),
        help="Sampling weights for image, text, and video sources in the generated multisource YAML.",
    )
    parser.add_argument(
        "--multisource_config_path",
        default=_default_multisource_config_path(),
        help="Path to write the VeOmni-style multisource YAML.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Target test row ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for video-name split.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["image", "text", "video"],
        default=["image", "text", "video"],
        help="Sources to generate.",
    )
    parser.add_argument(
        "--no_extract_images",
        action="store_true",
        help="Require existing image files instead of extracting first frames with ffmpeg.",
    )
    parser.add_argument(
        "--download_annotations",
        action="store_true",
        help="Download the DPO annotation JSONL automatically if missing.",
    )
    parser.add_argument("--max_samples", type=int, default=-1, help="Truncate input records for quick tests.")
    args = parser.parse_args()

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1")
    if any(weight < 0 for weight in args.source_weights):
        raise ValueError("--source_weights must be non-negative")
    if sum(args.source_weights) <= 0:
        raise ValueError("--source_weights must contain at least one positive value")
    if args.sources != SOURCE_ORDER:
        raise ValueError("--source_weights and multisource YAML generation require --sources image text video")

    dpo_jsonl = os.path.expanduser(args.dpo_jsonl)
    video_dir = os.path.expanduser(args.video_dir)
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    multisource_config_path = os.path.abspath(os.path.expanduser(args.multisource_config_path))
    image_dir = os.path.abspath(
        os.path.expanduser(args.image_dir) if args.image_dir else os.path.join(os.path.dirname(output_dir), "images")
    )

    if not os.path.isfile(dpo_jsonl):
        if args.download_annotations:
            _download_annotation(ANNOTATION_URL, dpo_jsonl)
        else:
            print(
                f"ERROR: {dpo_jsonl} not found. Pass --download_annotations to fetch it automatically.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loading {dpo_jsonl} ...", flush=True)
    records = _read_records(dpo_jsonl)
    if args.max_samples > 0:
        records = records[: args.max_samples]
    print(f"  Loaded {len(records):,} records.", flush=True)

    test_keys = _split_video_keys(records, test_ratio=args.test_ratio, seed=args.seed)
    train_keys = {_video_key(record.get("video", "")) for record in records} - test_keys
    print(
        f"  Split by video name: train_videos={len(train_keys):,}, test_videos={len(test_keys):,}",
        flush=True,
    )

    rows_by_source: dict[str, list[dict]] = {source: [] for source in args.sources}
    skipped: dict[str, int] = {source: 0 for source in args.sources}

    for idx, record in enumerate(records):
        split = "test" if _video_key(record.get("video", "")) in test_keys else "train"

        if "image" in rows_by_source:
            row = _build_image_row(
                record,
                split,
                idx,
                video_dir,
                image_dir,
                extract_images=not args.no_extract_images,
            )
            if row is None:
                skipped["image"] += 1
            else:
                rows_by_source["image"].append(row)

        if "text" in rows_by_source:
            row = _build_text_row(record, split, idx)
            if row is None:
                skipped["text"] += 1
            else:
                rows_by_source["text"].append(row)

        if "video" in rows_by_source:
            row = _build_video_row(record, split, idx, video_dir)
            if row is None:
                skipped["video"] += 1
            else:
                rows_by_source["video"].append(row)

    os.makedirs(output_dir, exist_ok=True)
    print("Writing parquet files:", flush=True)
    for source, rows in rows_by_source.items():
        _write_split(rows, output_dir, source)
        if skipped[source]:
            print(f"    skipped_{source}={skipped[source]:,}", flush=True)

    _write_multisource_config(output_dir, multisource_config_path, args.source_weights)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
