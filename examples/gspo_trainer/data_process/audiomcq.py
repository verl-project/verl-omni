# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Convert a local AudioMCQ JSONL and external audio assets to verl parquet."""

import argparse
import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_SOURCE = "Harland/AudioMCQ-StrongAC-GeminiCoT"


def build_rl_row(record: dict, audio_root: Path, index: int) -> dict:
    """Preserve option order and exact labels; reject ambiguous or missing inputs."""
    question = record.get("question")
    choices = record.get("choices")
    answer = record.get("answer")
    if not isinstance(question, str) or not question.strip() or "<audio>" in question:
        raise ValueError("invalid_question")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        raise ValueError("invalid_choices")
    if any(not isinstance(c, str) or not c.strip() or "<audio>" in c for c in choices):
        raise ValueError("invalid_choices")
    # Match reward normalization when rejecting answer-equivalent options.
    punctuation = " \t\r\n\"'`“”‘’.,;:!?！？。，；："
    normalized = [" ".join(unicodedata.normalize("NFKC", c).split()).strip(punctuation).casefold() for c in choices]
    if len(set(normalized)) != len(choices) or choices.count(answer) != 1:
        raise ValueError("ambiguous_answer")
    raw_path = record.get("audio_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("invalid_audio_path")
    root = audio_root.resolve()
    audio = (root / raw_path).resolve()
    if not audio.is_relative_to(root):
        raise ValueError("audio_outside_root")
    if not audio.is_file() or audio.stat().st_size == 0:
        raise ValueError("missing_audio")
    label_index = choices.index(answer)
    options = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    prompt = (
        f"<audio>\n[Question] {question.strip()}\nPlease choose the answer from the following options:\n"
        f"{options}\nOutput the final answer in <answer> </answer>."
    )
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "audios": [str(audio)],
        "ability": "audio_mcq",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer": answer,
                "choices": choices,
                "correct_choice_index": label_index,
                "correct_choice_indices": [label_index],
            },
        },
        "extra_info": {
            "index": index,
            "source_dataset": str(record.get("source_dataset", "")),
            "source_id": str(record.get("id", index)),
        },
    }


def split_rows(rows: list[dict], validation_size: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by audio asset so repeated questions cannot leak across splits."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["audios"][0], []).append(row)
    if validation_size < 1 or validation_size >= len(groups):
        raise ValueError("validation_size must leave at least one audio asset in each split")
    paths = sorted(groups)
    random.Random(seed).shuffle(paths)
    validation_paths = set(paths[:validation_size])
    train, validation = [], []
    for row in rows:
        split = "validation" if row["audios"][0] in validation_paths else "train"
        row = {**row, "extra_info": {**row["extra_info"], "split": split}}
        (validation if split == "validation" else train).append(row)
    return train, validation


def convert(input_jsonl: Path, audio_root: Path, output_dir: Path, validation_size: int, seed: int) -> dict:
    """Audit AudioMCQ records and write a deterministic, asset-disjoint split.

    Args:
        input_jsonl: JSONL records with question, choices, answer, and audio_path.
        audio_root: Root containing the referenced nonempty audio files.
        output_dir: New directory for the generated dataset; it must not exist.
        validation_size: Number of distinct audio assets held out for validation.
        seed: Seed used to shuffle audio assets before splitting.

    Returns:
        Report with source paths, seed, split row counts, and dropped-row reasons.
        The report is also saved as ``dataset_info.json`` beside ``train.parquet``
        and ``validation.parquet`` in ``output_dir``.

    Raises:
        FileExistsError: If ``output_dir`` already exists.
        ValueError: If a JSONL row is not an object or too few valid assets
            remain to form both splits. Invalid or out-of-root audio rows are
            counted as dropped and never written to either parquet.
    """
    targets = [output_dir / name for name in ("train.parquet", "validation.parquet", "dataset_info.json")]
    if output_dir.exists():
        raise FileExistsError("Use a new output directory; existing prepared datasets are not overwritten")
    rows, dropped = [], Counter()
    with input_jsonl.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {index + 1} must contain a JSON object")
            try:
                rows.append(build_rl_row(record, audio_root, index))
            except ValueError as error:
                dropped[str(error)] += 1
    train, validation = split_rows(rows, validation_size, seed)
    # Reserve a new directory atomically, including against concurrent runs.
    output_dir.mkdir(parents=True, exist_ok=False)
    for target, split in zip(targets[:2], (train, validation), strict=True):
        pd.DataFrame(split).to_parquet(target, index=False, use_dictionary=False)
    report = {
        "input": str(input_jsonl.resolve()),
        "audio_root": str(audio_root.resolve()),
        "seed": seed,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "dropped": dict(dropped),
    }
    targets[2].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-size", type=int, default=256, help="Number of held-out audio assets")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(convert(args.input_jsonl, args.audio_root, args.output_dir, args.validation_size, args.seed)))


if __name__ == "__main__":
    main()
