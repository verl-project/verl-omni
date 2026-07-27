# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Smoke-check BAGEL example data conversion and Uni-COT SFT batching.

This script downloads nothing and trains nothing. It assumes the official
``bagel_example`` directory already exists, converts a small slice into the
local Uni-COT JSONL schema, then verifies that ``UniCOTSFTDataset`` and its
collate function return the batch contract expected by BAGEL SFT.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from verl_omni.utils.dataset.unicot_sft_dataset import IGNORE_INDEX, UniCOTSFTDataset, unicot_sft_collate_fn

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from prepare_unicot_sft_data import _write_jsonl, convert_editing, convert_t2i, convert_vlm
from torch.utils.data import DataLoader


class ByteTokenizer:
    """Tiny tokenizer for data smoke checks without model/tokenizer downloads."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [byte + 1 for byte in text.encode("utf-8")]


def _convert_example_data(
    bagel_example_dir: Path,
    output_dir: Path,
    *,
    limit_per_task: int,
) -> dict[str, int]:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    image_dir = output_dir / "images"
    rows: list[dict[str, Any]] = []
    t2i_rows = convert_t2i(bagel_example_dir / "t2i", image_dir, limit_per_task)
    editing_rows = convert_editing(bagel_example_dir / "editing" / "seedxedit_multi", image_dir, limit_per_task)
    vlm_rows = convert_vlm(
        bagel_example_dir / "vlm" / "llava_ov_si.jsonl",
        bagel_example_dir / "vlm" / "images",
        limit_per_task,
    )
    for row_idx in range(max(len(t2i_rows), len(editing_rows), len(vlm_rows))):
        for task_rows in (t2i_rows, editing_rows, vlm_rows):
            if row_idx < len(task_rows):
                rows.append(task_rows[row_idx])

    if not rows:
        raise RuntimeError(f"No rows were converted from {bagel_example_dir}.")

    _write_jsonl(rows, output_dir / "train.jsonl")
    _write_jsonl(rows[: min(len(rows), 3)], output_dir / "val.jsonl")
    return {"t2i": len(t2i_rows), "editing": len(editing_rows), "vlm_sft": len(vlm_rows)}


def _assert_image_paths_exist(batch: dict[str, Any]) -> None:
    missing_paths = []
    for key in ("context_image_paths", "generated_image_paths"):
        for sample_paths in batch[key]:
            for image_path in sample_paths:
                if not Path(image_path).exists():
                    missing_paths.append(image_path)
    if missing_paths:
        raise AssertionError(f"Batch references missing image files: {missing_paths[:5]}")


def _assert_batch_contract(batch: dict[str, Any], *, batch_size: int) -> None:
    required_keys = {
        "input_ids",
        "labels",
        "attention_mask",
        "unicot_sft_events",
        "context_image_paths",
        "generated_image_paths",
        "task_type",
        "data_source",
        "extra_info",
    }
    missing = required_keys.difference(batch)
    if missing:
        raise AssertionError(f"Missing batch keys: {sorted(missing)}")

    for key in ("input_ids", "labels", "attention_mask"):
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"{key} must be a tensor, got {type(value)!r}.")
        if value.ndim != 2 or value.shape[0] != batch_size:
            raise AssertionError(f"{key} must have shape (B, L), got {tuple(value.shape)}.")

    if batch["input_ids"].shape != batch["labels"].shape:
        raise AssertionError("input_ids and labels must have the same shape.")
    if batch["input_ids"].shape != batch["attention_mask"].shape:
        raise AssertionError("input_ids and attention_mask must have the same shape.")
    if not torch.all(batch["input_ids"][batch["attention_mask"] == 0] == 0):
        raise AssertionError("Padded input_ids must be 0 where attention_mask is 0.")
    if not torch.all(batch["labels"][batch["attention_mask"] == 0] == IGNORE_INDEX):
        raise AssertionError("Padded labels must be IGNORE_INDEX where attention_mask is 0.")
    if not torch.any(batch["labels"] != IGNORE_INDEX):
        raise AssertionError("Batch has no supervised text tokens.")

    for key in ("unicot_sft_events", "context_image_paths", "generated_image_paths", "task_type"):
        if len(batch[key]) != batch_size:
            raise AssertionError(f"{key} must contain one entry per sample.")
    if not set(batch["task_type"]).issubset({"t2i", "editing", "vlm_sft"}):
        raise AssertionError(f"Unexpected task types: {batch['task_type']}")

    _assert_image_paths_exist(batch)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BAGEL toy data can be loaded as Uni-COT SFT batches.")
    parser.add_argument("--bagel_example_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--data_config", type=Path, default=Path("examples/sft_trainer/bagel/unicot_data_config.yaml"))
    parser.add_argument("--limit_per_task", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=3)
    args = parser.parse_args()

    counts = _convert_example_data(args.bagel_example_dir, args.output_dir, limit_per_task=args.limit_per_task)
    if any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Expected at least one row per task, got {counts}.")

    dataset = UniCOTSFTDataset(
        args.output_dir / "train.jsonl",
        tokenizer=ByteTokenizer(),
        config={"dataset_config_file": str(args.data_config), "max_text_length": 2048},
        is_train=True,
    )
    if len(dataset) < args.batch_size:
        raise RuntimeError(f"Dataset has {len(dataset)} rows, smaller than batch_size={args.batch_size}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=unicot_sft_collate_fn)
    batch = next(iter(loader))
    _assert_batch_contract(batch, batch_size=args.batch_size)

    summary = {
        "converted_counts": counts,
        "dataset_len": len(dataset),
        "batch_keys": sorted(batch.keys()),
        "input_ids_shape": list(batch["input_ids"].shape),
        "labels_shape": list(batch["labels"].shape),
        "attention_mask_shape": list(batch["attention_mask"].shape),
        "task_type": batch["task_type"],
        "context_image_counts": [len(paths) for paths in batch["context_image_paths"]],
        "generated_image_counts": [len(paths) for paths in batch["generated_image_paths"]],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
