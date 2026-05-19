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
"""Validate precomputed offline DPO tensors without launching full Ray training.

Loads offline DPO parquet rows, collates them like the trainer, runs
``OfflineDPOMaterializer.materialize()``, and prints the resulting tensor_dict
(shapes, dtypes, and short stats). The parquet is expected to already contain
SD3 VAE latents and text-encoder embeddings written by ``prepare_offline_dpo``.

Example:
    python examples/dpo_trainer/debug_offline_dpo_materialize.py \\
        --parquet data/offline_dpo/train.parquet \\
        --num-rows 1
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from verl import DataProto

from verl_omni.trainer.diffusion.offline_dpo_materializer import OfflineDPOMaterializer
from verl_omni.utils.dataset.offline_dpo_dataset import OfflineDPODataset
from verl_omni.utils.dataset.rl_dataset import collate_fn

MATERIALIZED_KEYS = (
    "image_latents",
    "prompt_embeds",
    "prompt_embeds_mask",
    "pooled_prompt_embeds",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "negative_pooled_prompt_embeds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        required=True,
        help="Offline DPO parquet with prompt/img_win/img_lose plus precomputed SD3 tensors.",
    )
    parser.add_argument("--num-rows", type=int, default=1, help="Logical DPO pairs to load (each expands to win+lose).")
    parser.add_argument("--row-index", type=int, default=0, help="Start row index inside the parquet.")
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the collated batch. Skip materializer validation.",
    )
    return parser.parse_args()


class _CollateTokenizer:
    """Minimal tokenizer for offline DPO collate (materialize ignores `prompts`)."""

    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, max_length: int):
        self.max_length = max_length

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kwargs):
        del add_generation_prompt, tokenize, kwargs
        parts = []
        for message in messages:
            if isinstance(message, dict) and message.get("role") == "user":
                parts.append(str(message.get("content", "")))
        return "\n".join(parts)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=None):
        del add_special_tokens, return_tensors, truncation
        limit = self.max_length if max_length is None else max_length
        token_ids = [min(ord(ch), 255) for ch in text][:limit]
        return {"input_ids": torch.tensor([token_ids], dtype=torch.long)}


def load_collate_tokenizer(args: argparse.Namespace):
    print("Using built-in collate tokenizer stub.")
    return _CollateTokenizer(args.max_prompt_length)


def build_data_config(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "max_prompt_length": args.max_prompt_length,
            "prompt_key": "prompt",
            "negative_prompt_key": "negative_prompt",
            "img_win_key": "img_win",
            "img_lose_key": "img_lose",
            "win_score_key": "win_score",
            "lose_score_key": "lose_score",
            "apply_chat_template_kwargs": {},
        }
    )


def build_materializer_config(args: argparse.Namespace):
    del args
    return OmegaConf.create({})


def _numpy_preview(value: Any, limit: int = 4) -> Any:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)[:limit].tolist()
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "preview": flat}
    if isinstance(value, (list, tuple)):
        return list(value)[:limit]
    return value


def print_collated_batch(batch_dict: dict[str, Any]) -> None:
    print("\n=== collated batch (trainer input) ===")
    for key in sorted(batch_dict.keys()):
        value = batch_dict[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key}: {_numpy_preview(value)}")

    if "raw_prompt" in batch_dict:
        raw = batch_dict["raw_prompt"]
        items = raw.tolist() if isinstance(raw, np.ndarray) else list(raw)
        print("\n--- raw_prompt (used when the parquet tensors were precomputed) ---")
        for i, text in enumerate(items):
            print(f"  [{i}] {text!r}")

def describe_tensor(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    info: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }
    if tensor.is_floating_point():
        finite = tensor[torch.isfinite(tensor)]
        if finite.numel() > 0:
            info["min"] = float(finite.min().item())
            info["max"] = float(finite.max().item())
            info["mean"] = float(finite.mean().item())
    return info


def extract_tensor_dict(materialized: DataProto) -> dict[str, torch.Tensor]:
    tensor_dict: dict[str, torch.Tensor] = {}
    for key in MATERIALIZED_KEYS:
        if key in materialized.batch.keys():
            tensor_dict[key] = materialized.batch[key]
    return tensor_dict


def _tensor_preview(tensor: torch.Tensor) -> torch.Tensor:
    """Small slice for logging; index count must match tensor rank."""
    if tensor.ndim == 0:
        return tensor
    if tensor.ndim == 1:
        return tensor[: min(4, tensor.shape[0])]
    if tensor.ndim == 2:
        return tensor[0, : min(4, tensor.shape[1])]
    return tensor[0, : min(2, tensor.shape[1]), : min(4, tensor.shape[2])]


def print_tensor_dict(tensor_dict: dict[str, torch.Tensor]) -> None:
    print("\n=== tensor_dict (materialize output) ===")
    summary = {name: describe_tensor(name, tensor) for name, tensor in tensor_dict.items()}
    print(json.dumps(summary, indent=2))
    for name, tensor in tensor_dict.items():
        print(f"\n--- {name} ---")
        print(f"  shape={tuple(tensor.shape)} dtype={tensor.dtype}")
        if name.endswith("_mask"):
            print(f"  unique values: {tensor.unique().tolist()}")
        elif tensor.numel() > 0:
            print(f"  preview =\n{_tensor_preview(tensor)}")


def main() -> None:
    args = parse_args()
    data_config = build_data_config(args)

    tokenizer = load_collate_tokenizer(args)
    dataset = OfflineDPODataset(args.parquet, tokenizer, config=data_config, max_samples=-1)

    features = []
    end = min(args.row_index + args.num_rows, len(dataset))
    for row_idx in range(args.row_index, end):
        features.append(dataset[row_idx])
    if not features:
        raise ValueError(f"No rows selected from parquet (row_index={args.row_index}, len={len(dataset)}).")

    batch_dict = collate_fn(features)
    print_collated_batch(batch_dict)

    batch = DataProto.from_single_dict(batch_dict)
    print("\n=== DataProto.non_tensor_batch keys ===")
    print(list(batch.non_tensor_batch.keys()))

    if args.dry_run:
        print("\n(dry-run: skipped OfflineDPOMaterializer validation)")
        return

    materializer = OfflineDPOMaterializer(build_materializer_config(args))
    materialized = materializer.materialize(batch)
    tensor_dict = extract_tensor_dict(materialized)

    print_tensor_dict(tensor_dict)
    print("\nDone.")


if __name__ == "__main__":
    main()
