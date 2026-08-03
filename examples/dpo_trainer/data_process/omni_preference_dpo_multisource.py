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

"""Build offline MLLM DPO parquet data from Omni-Preference RL preference pairs.

Reads ``dataset_jsonl/{image,video,audio}/final_rl_data.jsonl`` and converts
each row into multimodal offline DPO parquet (media + question as prompt,
chosen/rejected as candidate answers).

Train/test splitting is performed at media-file granularity so validation
samples never share a source image/video/audio with training samples.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

VIDEO_FILE_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi")
AUDIO_FILE_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
CONTEXT_PATTERN = re.compile(
    r"(?:Image|Video|Audio) file:\s*(?P<media>.+?)\s+Question:\s*(?P<question>.+?)\s+"
    r"Candidate A:\s*(?P<candidate_a>.+?)\s+Candidate B:\s*(?P<candidate_b>.+?)\s*\Z",
    re.DOTALL,
)

MODALITY_CONFIGS = {
    "image": {
        "jsonl_relpath": "dataset_jsonl/image/final_rl_data.jsonl",
        "media_field": "images",
        "media_subdir": None,
        "data_source": "omni_preference/image",
        "ability": "image_qa",
    },
    "video": {
        "jsonl_relpath": "dataset_jsonl/video/final_rl_data.jsonl",
        "media_field": "videos",
        "media_subdir": "video-dataset",
        "data_source": "omni_preference/video",
        "ability": "video_qa",
    },
    "audio": {
        "jsonl_relpath": "dataset_jsonl/audio/final_rl_data.jsonl",
        "media_field": "audios",
        "media_subdir": "audio_files",
        "data_source": "omni_preference/audio",
        "ability": "audio_qa",
    },
}


def _media_key(media_rel: str) -> str:
    return Path(media_rel.replace("\\", "/")).name or media_rel


def _dataset_media_rel(media_path: str) -> str:
    normalized = media_path.replace("\\", "/").strip().rstrip("_")
    if normalized.startswith("/data/"):
        return normalized[len("/data/") :]
    return normalized.lstrip("/")


def _read_records(jsonl_path: str) -> list[dict]:
    records: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _parse_context(content: str) -> dict[str, str] | None:
    if "### Context" not in content:
        return None
    context = content.split("### Context", 1)[1].strip()
    match = CONTEXT_PATTERN.search(context)
    if match is None:
        return None
    return {
        "media": match.group("media").strip().rstrip("_"),
        "question": match.group("question").strip(),
        "candidate_a": match.group("candidate_a").strip(),
        "candidate_b": match.group("candidate_b").strip(),
    }


def _normalize_record(record: dict, modality: str, index: int) -> dict | None:
    config = MODALITY_CONFIGS[modality]
    media_items = record.get(config["media_field"]) or []
    messages = record.get("messages") or []
    if not media_items or not messages:
        return None

    try:
        solution = json.loads(record["solution"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return None

    better = str(solution.get("better", "")).strip()
    if better == "equal":
        return None

    context = _parse_context(str(messages[0].get("content", "")))
    if context is None:
        return None

    candidate_a = context["candidate_a"]
    candidate_b = context["candidate_b"]
    if not candidate_a or not candidate_b:
        return None

    try:
        score_a = float(solution.get("score_A", 0))
        score_b = float(solution.get("score_B", 0))
    except (TypeError, ValueError):
        return None

    if better == "A":
        chosen, rejected = candidate_a, candidate_b
        win_score, lose_score = score_a, score_b
    elif better == "B":
        chosen, rejected = candidate_b, candidate_a
        win_score, lose_score = score_b, score_a
    else:
        return None

    media_path = str(media_items[0]).strip()
    dataset_media_rel = _dataset_media_rel(media_path)
    question = context["question"]
    if not question or not chosen or not rejected:
        return None

    return {
        "modality": modality,
        "media_path": media_path,
        "dataset_media_rel": dataset_media_rel,
        "question": question,
        "chosen": chosen,
        "rejected": rejected,
        "win_score": win_score,
        "lose_score": lose_score,
        "better": better,
        "sample_id": f"omni_pref_{modality}_{index}",
    }


def _build_audio_basename_index(audio_dir: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for root, _, files in os.walk(audio_dir):
        for filename in files:
            if Path(filename).suffix.lower() not in AUDIO_FILE_EXTENSIONS:
                continue
            abs_path = os.path.abspath(os.path.join(root, filename))
            existing = index.get(filename)
            if existing is None or len(abs_path) < len(existing):
                index[filename] = abs_path
    return index


def _resolve_image_or_video(dataset_root: str, media_subdir: str | None, media_rel: str) -> str | None:
    candidates = []
    if media_subdir:
        candidates.append(os.path.join(dataset_root, media_subdir, media_rel))
    candidates.append(os.path.join(dataset_root, media_rel))

    rel = media_rel.lstrip("/").lstrip("\\")
    if media_subdir:
        candidates.append(os.path.join(dataset_root, media_subdir, rel))
    candidates.append(os.path.join(dataset_root, rel))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if not Path(rel).suffix:
            for extension in VIDEO_FILE_EXTENSIONS:
                with_extension = candidate + extension
                if os.path.isfile(with_extension):
                    return os.path.abspath(with_extension)
    return None


def _resolve_audio(dataset_root: str, media_subdir: str, media_rel: str, basename_index: dict[str, str]) -> str | None:
    filename = Path(_dataset_media_rel(media_rel)).name
    indexed = basename_index.get(filename)
    if indexed is not None:
        return indexed

    candidates = [
        os.path.join(dataset_root, media_subdir, media_rel),
        os.path.join(dataset_root, media_subdir, "audio", filename),
        os.path.join(dataset_root, media_subdir, "audio_files", filename),
        os.path.join(dataset_root, media_subdir, filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return None


def _resolve_media(
    record: dict,
    dataset_root: str,
    audio_basename_index: dict[str, str],
) -> str | None:
    modality = record["modality"]
    config = MODALITY_CONFIGS[modality]
    media_rel = record["dataset_media_rel"]

    if modality == "audio":
        return _resolve_audio(dataset_root, config["media_subdir"] or "audio_files", media_rel, audio_basename_index)
    return _resolve_image_or_video(dataset_root, config["media_subdir"], media_rel)


def _split_media_keys(records: list[dict], test_ratio: float, seed: int) -> set[str]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[_media_key(record["dataset_media_rel"])].append(record)

    groups = list(grouped.items())
    rng = random.Random(seed)
    rng.shuffle(groups)

    target_test_rows = max(1, round(len(records) * test_ratio))
    test_keys: set[str] = set()
    test_rows = 0

    for key, group_records in groups:
        if test_keys and test_rows >= target_test_rows:
            break
        test_keys.add(key)
        test_rows += len(group_records)

    if len(test_keys) == len(groups) and len(groups) > 1:
        test_keys.remove(groups[-1][0])
    return test_keys


def _base_row(record: dict, split: str, index: int) -> dict:
    modality = record["modality"]
    config = MODALITY_CONFIGS[modality]
    media_rel = record["dataset_media_rel"]
    question = record["question"]

    return {
        "data_source": config["data_source"],
        "chosen": {"role": "assistant", "content": record["chosen"]},
        "rejected": {"role": "assistant", "content": record["rejected"]},
        "win_score": record["win_score"],
        "lose_score": record["lose_score"],
        "ability": config["ability"],
        "reward_model": {"style": "preference"},
        "extra_info": {
            "split": split,
            "index": index,
            "sample_id": record["sample_id"],
            "question": question,
            "modality": modality,
            "source_media": media_rel,
            "source_media_name": _media_key(media_rel),
            "better": record["better"],
        },
    }


def _build_multimodal_row(record: dict, split: str, index: int, media_path: str) -> dict:
    modality = record["modality"]
    config = MODALITY_CONFIGS[modality]
    question = record["question"]
    row = _base_row(record, split, index)
    row["prompt"] = [
        {
            "role": "user",
            "content": f"<{modality}>{question}",
        },
    ]
    row[config["media_field"]] = [media_path]
    row["extra_info"][f"{modality}_path"] = media_path
    return row


def _write_split(rows: list[dict], output_dir: str) -> None:
    train_rows = [row for row in rows if row["extra_info"]["split"] == "train"]
    test_rows = [row for row in rows if row["extra_info"]["split"] == "test"]

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    pd.DataFrame(test_rows).to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    print(f"  train={len(train_rows):,}, test={len(test_rows):,}", flush=True)


def _process_modality(
    modality: str,
    dataset_root: str,
    output_dir: str,
    test_ratio: float,
    seed: int,
    max_samples: int,
    audio_basename_index: dict[str, str],
) -> None:
    config = MODALITY_CONFIGS[modality]
    jsonl_path = os.path.join(dataset_root, config["jsonl_relpath"])
    modality_output_dir = os.path.join(output_dir, modality)

    if not os.path.isfile(jsonl_path):
        print(f"WARNING: skipping {modality}; missing {jsonl_path}", flush=True)
        return

    print(f"\n=== {modality} ===", flush=True)
    print(f"Loading {jsonl_path} ...", flush=True)
    raw_records = _read_records(jsonl_path)
    if max_samples > 0:
        raw_records = raw_records[:max_samples]
    print(f"  Loaded {len(raw_records):,} raw records.", flush=True)

    records: list[dict] = []
    skipped_equal = 0
    skipped_parse = 0
    for index, raw_record in enumerate(raw_records):
        normalized = _normalize_record(raw_record, modality, index)
        if normalized is None:
            try:
                solution = json.loads(raw_record["solution"])
                if str(solution.get("better", "")).strip() == "equal":
                    skipped_equal += 1
                else:
                    skipped_parse += 1
            except (KeyError, json.JSONDecodeError, TypeError):
                skipped_parse += 1
            continue
        records.append(normalized)

    print(
        f"  Parsed {len(records):,} preference pairs "
        f"(skipped_equal={skipped_equal:,}, skipped_parse={skipped_parse:,}).",
        flush=True,
    )
    if not records:
        print(f"  WARNING: no valid {modality} preference pairs found.", flush=True)
        return

    test_keys = _split_media_keys(records, test_ratio=test_ratio, seed=seed)
    train_keys = {_media_key(record["dataset_media_rel"]) for record in records} - test_keys
    print(
        f"  Split by media name: train_media={len(train_keys):,}, test_media={len(test_keys):,}",
        flush=True,
    )

    rows: list[dict] = []
    skipped_media = 0

    for idx, record in enumerate(records):
        split = "test" if _media_key(record["dataset_media_rel"]) in test_keys else "train"
        media_path = _resolve_media(record, dataset_root, audio_basename_index)
        if media_path is None:
            skipped_media += 1
            continue
        rows.append(_build_multimodal_row(record, split, idx, media_path))

    print("  Writing parquet files:", flush=True)
    _write_split(rows, modality_output_dir)
    if skipped_media:
        print(f"  skipped_missing_media={skipped_media:,}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Omni-Preference image/video/audio RL JSONL into offline DPO parquet.",
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Omni-Preference repo root containing dataset_jsonl/ and media folders.",
    )
    parser.add_argument(
        "--output_dir",
        default=os.path.expanduser("~/data/omni_preference_dpo/parquet"),
        help="Directory to write {image,video,audio}/train.parquet.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=["image", "video", "audio"],
        default=["image", "video", "audio"],
        help="Modalities to convert.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.10, help="Target test row ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for media-name split.")
    parser.add_argument("--max_samples", type=int, default=-1, help="Truncate each modality for quick tests.")
    args = parser.parse_args()

    if not 0 < args.test_ratio < 1:
        raise ValueError("--test_ratio must be between 0 and 1")

    dataset_root = os.path.abspath(os.path.expanduser(args.dataset_root))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.isdir(dataset_root):
        print(f"ERROR: dataset_root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    audio_basename_index: dict[str, str] = {}
    if "audio" in args.modalities:
        audio_dir = os.path.join(dataset_root, "audio_files")
        if os.path.isdir(audio_dir):
            audio_basename_index = _build_audio_basename_index(audio_dir)
            print(f"Indexed {len(audio_basename_index):,} audio files under {audio_dir}.", flush=True)
        else:
            print(f"WARNING: audio_files not found under {dataset_root}; audio rows may be skipped.", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    for modality in args.modalities:
        _process_modality(
            modality=modality,
            dataset_root=dataset_root,
            output_dir=output_dir,
            test_ratio=args.test_ratio,
            seed=args.seed,
            max_samples=args.max_samples,
            audio_basename_index=audio_basename_index,
        )

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
