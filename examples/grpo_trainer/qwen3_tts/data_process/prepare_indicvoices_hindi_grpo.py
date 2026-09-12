#!/usr/bin/env python3
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
"""Build the pinned IndicVoices-R Hindi prompt split for GRPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable
from numbers import Real
from pathlib import Path

DATASET_ID = "ai4bharat/indicvoices_r"
DATASET_CONFIG = "Hindi"
DATASET_REVISION = "5f4495c91d500742a58d1be2ab07d77f73c0acf8"
VALIDATION_COUNT = 100


def _finite_number(sample: dict, field: str) -> float | None:
    value = sample.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"IndicVoices-R {field} must be a number or null, got {type(value).__name__}.")
    value = float(value)
    return value if math.isfinite(value) else None


def _record(sample: dict, source_index: int) -> tuple[dict | None, str | None]:
    duration = _finite_number(sample, "duration")
    if duration is None or not 1.0 <= duration <= 14.0:
        return None, "duration"
    snr = _finite_number(sample, "snr")
    if snr is None or snr < 20.0:
        return None, "snr"
    raw_value, normalized_value = sample.get("text"), sample.get("normalized")
    for field, value in (("text", raw_value), ("normalized", normalized_value)):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"IndicVoices-R {field} must be a string or null, got {type(value).__name__}.")
    raw_text = (raw_value or "").strip()
    text = (normalized_value or raw_text).strip()
    if not text:
        return None, "text"
    return {"source_index": source_index, "text": text, "raw_text": raw_text}, None


def collect_read_prompt_splits(
    samples: Iterable[dict],
    *,
    raw_read_limit: int = 1_000,
    validation_count: int = VALIDATION_COUNT,
) -> tuple[list[dict], list[dict], dict]:
    """Filter the first raw Read window, then collect a held-out continuation."""
    train, validation = [], []
    rejected_train, rejected_validation = Counter(), Counter()
    raw_read_seen = 0

    for source_index, sample in enumerate(samples):
        scenario = sample.get("scenario")
        if not isinstance(scenario, str):
            raise TypeError(f"IndicVoices-R scenario must be a string, got {type(scenario).__name__}.")
        if scenario not in {"Extempore", "Read"}:
            raise ValueError(f"Unknown IndicVoices-R scenario {scenario!r}; expected 'Extempore' or 'Read'.")
        if scenario != "Read":
            continue
        raw_read_seen += 1
        record, reason = _record(sample, source_index)
        if raw_read_seen <= raw_read_limit:
            train.append(record) if reason is None else rejected_train.update([reason])
        else:
            validation.append(record) if reason is None else rejected_validation.update([reason])
            if len(validation) == validation_count:
                break

    if raw_read_seen < raw_read_limit:
        raise ValueError(f"Found only {raw_read_seen}/{raw_read_limit} raw Read rows.")
    if len(validation) != validation_count:
        raise ValueError(f"Collected only {len(validation)}/{validation_count} validation rows.")
    return (
        train,
        validation,
        {
            "rejected_train": dict(sorted(rejected_train.items())),
            "rejected_validation": dict(sorted(rejected_validation.items())),
        },
    )


def _verl_row(record: dict, split: str, index: int) -> dict:
    text = record["text"]
    return {
        "data_source": DATASET_ID,
        "prompt": [{"role": "user", "content": text}],
        "ability": "text_to_speech",
        "reward_model": {"style": "model", "ground_truth": text},
        "extra_info": {
            "split": split,
            "id": f"indicvoices-hi-{record['source_index']:08d}",
            "index": index,
            "text": text,
        },
    }


def _content_sha256(rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_splits(
    train: list[dict],
    validation: list[dict],
    *,
    expected_train_count: int | None = 863,
    expected_validation_count: int = VALIDATION_COUNT,
) -> dict[str, list[dict]]:
    if expected_train_count is not None and len(train) != expected_train_count:
        raise ValueError(f"Expected {expected_train_count} training prompts, found {len(train)}.")
    if len(validation) != expected_validation_count:
        raise ValueError(f"Expected {expected_validation_count} validation prompts, found {len(validation)}.")
    source_overlap = {row["source_index"] for row in train} & {row["source_index"] for row in validation}
    train_texts = {value for row in train for value in (row["text"], row["raw_text"]) if value}
    validation_texts = {value for row in validation for value in (row["text"], row["raw_text"]) if value}
    if source_overlap or train_texts & validation_texts:
        raise ValueError("IndicVoices-R train and validation splits overlap.")

    return {
        "train": [_verl_row(record, "train", index) for index, record in enumerate(train)],
        "validation": [_verl_row(record, "validation", index) for index, record in enumerate(validation)],
    }


def _load_source():
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split="train",
        streaming=True,
        revision=DATASET_REVISION,
        token=True,
    )
    keep = {"text", "normalized", "scenario", "duration", "snr"}
    removable = [column for column in dataset.column_names if column not in keep]
    return dataset.remove_columns(removable) if removable else dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    train, validation, collection_audit = collect_read_prompt_splits(_load_source())
    splits = build_splits(train, validation)

    import pandas as pd

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for split, rows in splits.items():
        path = args.output_dir / f"{split}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        files[split] = {
            "path": path.name,
            "rows": len(rows),
            "content_sha256": _content_sha256(rows),
            "file_sha256": _file_sha256(path),
        }
    manifest = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "selection": "first 1000 Read rows filtered to 1-14s and SNR>=20; next 100 passing rows held out",
        "files": files,
        **collection_audit,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
