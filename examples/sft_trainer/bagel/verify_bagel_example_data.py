# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Smoke-check BAGEL example-format SFT batching.

This script downloads nothing and trains nothing. It assumes the official
``bagel_example`` directory already exists, then verifies that
``BagelExampleSFTDataset`` and its collate function return the packed batch
contract expected by BAGEL SFT.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from bagel_example_sft_dataset import BagelExampleSFTDataset, bagel_example_sft_collate_fn  # noqa: E402


class ByteTokenizer:
    """Tiny tokenizer for data smoke checks without model/tokenizer downloads."""

    def __init__(self) -> None:
        self.special_tokens_map: dict[str, str] = {}
        self._vocab: dict[str, int] = {}

    def add_tokens(self, tokens: list[str]) -> int:
        added = 0
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab) + 1
                added += 1
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = len(self._vocab) + 1
        return self._vocab[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        offset = len(self._vocab) + 1
        return [offset + byte for byte in text.encode("utf-8")]


def _require_example_layout(bagel_example_dir: Path) -> None:
    required_paths = (
        bagel_example_dir / "t2i",
        bagel_example_dir / "editing" / "seedxedit_multi",
        bagel_example_dir / "editing" / "parquet_info" / "seedxedit_multi.json",
        bagel_example_dir / "vlm" / "images",
        bagel_example_dir / "vlm" / "llava_ov_si.jsonl",
    )
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing BAGEL example data paths: {missing}")


def _assert_batch_contract(batch: dict[str, Any]) -> None:
    required_keys = {
        "sequence_length",
        "sample_lens",
        "packed_text_ids",
        "packed_text_indexes",
        "packed_position_ids",
        "batch_data_indexes",
    }
    missing = required_keys.difference(batch)
    if missing:
        raise AssertionError(f"Missing batch keys: {sorted(missing)}")

    for key in ("packed_text_ids", "packed_text_indexes", "packed_position_ids"):
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise AssertionError(f"{key} must be a tensor, got {type(value)!r}.")
        if value.ndim != 1:
            raise AssertionError(f"{key} must be rank-1, got shape {tuple(value.shape)}.")
    if batch["sequence_length"] <= 0:
        raise AssertionError("sequence_length must be positive.")
    if int(batch["packed_text_indexes"].max()) >= batch["sequence_length"]:
        raise AssertionError("packed_text_indexes must stay within sequence_length.")
    if "packed_label_ids" not in batch and "padded_images" not in batch:
        raise AssertionError("Batch should contain text or image supervision.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BAGEL example data can be loaded as SFT packed batches.")
    parser.add_argument("--bagel_example_dir", type=Path, required=True)
    parser.add_argument(
        "--data_config",
        type=Path,
        default=Path("examples/sft_trainer/bagel/bagel_example_data_config.yaml"),
    )
    args = parser.parse_args()

    _require_example_layout(args.bagel_example_dir)

    dataset = BagelExampleSFTDataset(
        tokenizer=ByteTokenizer(),
        config={
            "dataloader_num_workers": 1,
            "custom_cls": {
                "bagel_example_dir": str(args.bagel_example_dir),
                "dataset_config_file": str(args.data_config),
                "expected_num_tokens": 512,
                "max_num_tokens": 1024,
                "max_num_tokens_per_sample": 1024,
                "prefer_buffer_before": 512,
                "max_buffer_size": 4,
                "use_flex": True,
                "num_packed_batches": 1,
            },
        },
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=bagel_example_sft_collate_fn)
    batch = next(iter(loader)).to_dict()
    _assert_batch_contract(batch)

    summary = {
        "batch_keys": sorted(batch.keys()),
        "sequence_length": batch["sequence_length"],
        "sample_lens": batch["sample_lens"],
        "packed_text_ids_shape": list(batch["packed_text_ids"].shape),
        "packed_text_indexes_shape": list(batch["packed_text_indexes"].shape),
        "has_text_loss": "packed_label_ids" in batch,
        "has_image_loss": "padded_images" in batch,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
