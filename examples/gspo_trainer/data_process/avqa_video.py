# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Convert original AVQA video annotations; AVQA-R1 image/audio is a different dataset."""

import argparse
import hashlib
import json
from collections import Counter
from numbers import Integral
from pathlib import Path

import pandas as pd

SYSTEM_PROMPT = (
    "Consider both the visual and audio information in the video. Explain your reasoning inside "
    "<think> </think> tags, then give only the correct option letter inside <answer> </answer> tags."
)


def build_rl_row(record, video_root, split, index, fps=1.0, max_frames=32):
    """Validate one original AVQA annotation and preserve its clip identity."""
    question = record.get("question_text")
    if not isinstance(question, str) or not question.strip():
        return None, "empty_question"
    choices = record.get("multi_choice")
    if not isinstance(choices, list) or len(choices) != 4:
        return None, "invalid_options"
    if any(not isinstance(x, str) or not x.strip() for x in choices):
        return None, "invalid_options"
    choices = [x.strip() for x in choices]
    if len(set(choices)) != 4:
        return None, "duplicate_options"
    answer = record.get("answer")
    if isinstance(answer, bool) or not isinstance(answer, Integral) or not 0 <= answer < 4:
        return None, "invalid_answer"
    clip = record.get("video_name")
    if not isinstance(clip, str) or not clip.strip() or Path(clip).name != clip or clip in (".", ".."):
        return None, "invalid_video_name"
    root = Path(video_root).resolve()
    video = (root / (clip + ".mp4")).resolve()
    if not video.is_relative_to(root) or not video.is_file():
        return None, "missing_video"
    qid = record.get("id")
    if isinstance(qid, bool) or not isinstance(qid, str | Integral) or not str(qid).strip():
        return None, "invalid_question_id"
    letter = "ABCD"[answer]
    options = dict(zip("ABCD", choices, strict=True))
    lines = "\n".join(f"{label}. {text}" for label, text in options.items())
    return {
        "data_source": "avqa_video",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<video>{question.strip()}\nOptions:\n{lines}"},
        ],
        "videos": [
            {
                "video": str(video),
                "fps": fps,
                "max_frames": max_frames,
                "min_pixels": 32 * 28 * 28,
                "max_pixels": 128 * 28 * 28,
            }
        ],
        "ability": "audio_visual_qa",
        "reward_model": {"style": "rule", "ground_truth": f"<answer>{letter}</answer>"},
        "extra_info": {
            "split": split,
            "index": index,
            "problem_id": str(qid),
            "video_id": str(record.get("video_id", "")),
            "video_name": clip,
            "dataset": "AVQA",
            "raw_question": question.strip(),
            "answer_index": int(answer),
            "answer_letter": letter,
            "options": json.dumps(options, ensure_ascii=False),
            "question_relation": str(record.get("question_relation", "")),
            "question_type": str(record.get("question_type", "")),
        },
    }, None


def read_annotations(input_json):
    """Reject duplicate identities before missing media can hide them."""
    records = json.loads(Path(input_json).read_text())
    if not isinstance(records, list):
        raise ValueError("Original AVQA annotations must be a JSON list")
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        qid = record.get("id")
        if isinstance(qid, bool) or not isinstance(qid, str | Integral) or not str(qid).strip():
            continue
        identity = str(qid)
        if identity in seen:
            raise ValueError(f"Duplicate question id {identity!r} in {input_json}")
        seen.add(identity)
    return records


def audit_annotation_splits(train_json, validation_json):
    """Record source hashes and clip overlap without changing official splits."""
    sources, identities, clips = {}, {}, {}
    for split, path in (("train", train_json), ("validation", validation_json)):
        records = read_annotations(path)
        identities[split], clips[split] = set(), set()
        for record in records:
            if not isinstance(record, dict):
                continue
            qid = record.get("id")
            if not isinstance(qid, bool) and isinstance(qid, str | Integral) and str(qid).strip():
                identities[split].add(str(qid))
            clip = record.get("video_name")
            if isinstance(clip, str) and clip.strip():
                clips[split].add(clip)
        sources[split] = {
            "annotation_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "questions": len(records),
            "unique_question_ids": len(identities[split]),
            "unique_video_names": len(clips[split]),
        }
    question_overlap = identities["train"] & identities["validation"]
    if question_overlap:
        raise ValueError(f"Train/validation question ids overlap: {len(question_overlap)}")
    return {
        "annotation_sources": sources,
        "source_split_overlap": {
            "question_ids": 0,
            "video_names": len(clips["train"] & clips["validation"]),
        },
    }


def convert_split(input_json, video_root, output, split, fps=1.0, max_frames=32):
    """Write valid original annotations with exact kept/dropped counts."""
    records = read_annotations(input_json)
    rows, dropped = [], Counter()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            dropped["invalid_record"] += 1
            continue
        row, reason = build_rl_row(record, video_root, split, index, fps, max_frames)
        if row is None:
            dropped[reason] += 1
            continue
        rows.append(row)
    if not rows:
        raise ValueError(f"No valid video examples: {dict(dropped)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, engine="pyarrow", index=False, use_dictionary=False)
    return {"input": len(records), "kept": len(rows), "dropped": dict(sorted(dropped.items()))}


def main():
    """Convert original AVQA train/val JSON and mounted MP4 clips to parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--validation_json", required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    import math

    if not math.isfinite(args.fps) or args.fps <= 0 or args.max_frames <= 0:
        parser.error("fps and max_frames must be positive")
    output = Path(args.output_dir).expanduser().resolve()
    if not args.overwrite and any(
        (output / name).exists() for name in ("train.parquet", "validation.parquet", "manifest.json")
    ):
        parser.error("Output exists; choose a new output directory or explicitly set --overwrite")
    manifest = {
        "dataset": "AVQA original videos",
        "source": "https://github.com/AlyssaYoung/AVQA",
        "modality": "video plus soundtrack",
        "fps": args.fps,
        "max_frames": args.max_frames,
        **audit_annotation_splits(args.train_json, args.validation_json),
    }
    for split, source, filename in (
        ("train", args.train_json, "train.parquet"),
        ("validation", args.validation_json, "validation.parquet"),
    ):
        manifest[split] = convert_split(source, args.video_root, output / filename, split, args.fps, args.max_frames)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
