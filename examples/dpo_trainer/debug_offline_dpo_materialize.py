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
"""Debug OfflineDPOMaterializer without launching full Ray training.

Loads offline DPO parquet rows, collates them like the trainer, runs
``OfflineDPOMaterializer.materialize()``, and prints the resulting tensor_dict
(shapes, dtypes, and short stats).

Example:
    python examples/dpo_trainer/debug_offline_dpo_materialize.py \\
        --parquet data/offline_dpo/train.parquet \\
        --model stabilityai/stable-diffusion-3.5-medium \\
        --num-rows 1 \\
        --device cuda:0

    # ``--model`` is the SD3 diffusers checkpoint only. Do not pass it to AutoTokenizer.
    # For batch_decode(prompts) like the trainer logs, pass a real LLM tokenizer path:
    #   --tokenizer-path /path/to/your/rollout/tokenizer --decode-prompts
"""

from __future__ import annotations

import argparse
import json
import os
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
        help="Offline DPO parquet with prompt/img_win/img_lose columns.",
    )
    parser.add_argument(
        "--model",
        default="stabilityai/stable-diffusion-3.5-medium",
        help="SD3/SD3.5 diffusers model path (for materialize only). Not a tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help=(
            "Tokenizer for dataset collate / decode-prompts (verl hf_tokenizer). "
            "SD3 weights are not tokenizers; omit to use a built-in stub (recommended)."
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--num-rows", type=int, default=1, help="Logical DPO pairs to load (each expands to win+lose).")
    parser.add_argument("--row-index", type=int, default=0, help="Start row index inside the parquet.")
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu. Default: cuda if available else cpu.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print collated non_tensor_batch (raw_prompt, image_path). Skip SD3 load.",
    )
    parser.add_argument(
        "--decode-prompts",
        action="store_true",
        help="Also batch_decode batch.batch['prompts'] for comparison with raw_prompt.",
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


def resolve_tokenizer_path(model_path: str, tokenizer_path: str | None) -> str:
    if tokenizer_path is not None:
        return os.path.expanduser(tokenizer_path)
    local_path = os.path.expanduser(model_path)
    nested = os.path.join(local_path, "tokenizer")
    return nested if os.path.isdir(nested) else local_path


def load_collate_tokenizer(args: argparse.Namespace):
    if args.tokenizer_path is None:
        print("Using built-in collate tokenizer stub (SD3 model path is not a HF tokenizer).")
        return _CollateTokenizer(args.max_prompt_length), "stub"

    path = resolve_tokenizer_path(args.model, args.tokenizer_path)
    from verl.utils import hf_tokenizer

    print(f"Loading collate tokenizer from: {path}")
    return hf_tokenizer(path, trust_remote_code=args.trust_remote_code), path


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
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return OmegaConf.create(
        {
            "materialize_device": device,
            "trainer": {"device": device},
            "actor_rollout_ref": {
                "model": {"path": args.model, "local_path": None},
                "rollout": {
                    "dtype": args.dtype,
                    "pipeline": {
                        "height": args.height,
                        "width": args.width,
                        "max_sequence_length": args.max_sequence_length,
                        "guidance_scale": args.guidance_scale,
                    },
                },
            },
        }
    )


def _numpy_preview(value: Any, limit: int = 4) -> Any:
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)[:limit].tolist()
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype), "preview": flat}
    if isinstance(value, (list, tuple)):
        return list(value)[:limit]
    return value


def print_collated_batch(batch_dict: dict[str, Any], tokenizer, decode_prompts: bool) -> None:
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
        print("\n--- raw_prompt (fed to SD3 encode_prompt) ---")
        for i, text in enumerate(items):
            print(f"  [{i}] {text!r}")

    if decode_prompts and "prompts" in batch_dict and tokenizer is not None:
        if not hasattr(tokenizer, "batch_decode"):
            print("\n--- batch_decode(prompts): skipped (stub tokenizer; pass --tokenizer-path for real decode) ---")
        else:
            decoded = tokenizer.batch_decode(batch_dict["prompts"], skip_special_tokens=True)
            print("\n--- batch_decode(prompts) (verl padding artifact, NOT SD3 input) ---")
            for i, text in enumerate(decoded):
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

    tokenizer, tokenizer_source = load_collate_tokenizer(args)
    print(f"Collate tokenizer source: {tokenizer_source}")
    dataset = OfflineDPODataset(args.parquet, tokenizer, config=data_config, max_samples=-1)

    features = []
    end = min(args.row_index + args.num_rows, len(dataset))
    for row_idx in range(args.row_index, end):
        features.append(dataset[row_idx])
    if not features:
        raise ValueError(f"No rows selected from parquet (row_index={args.row_index}, len={len(dataset)}).")

    batch_dict = collate_fn(features)
    print_collated_batch(batch_dict, tokenizer, decode_prompts=args.decode_prompts)

    batch = DataProto.from_single_dict(batch_dict)
    print("\n=== DataProto.non_tensor_batch keys ===")
    print(list(batch.non_tensor_batch.keys()))

    if args.dry_run:
        print("\n(dry-run: skipped OfflineDPOMaterializer)")
        return

    materializer = OfflineDPOMaterializer(build_materializer_config(args))
    materialized = materializer.materialize(batch)
    tensor_dict = extract_tensor_dict(materialized)

    print_tensor_dict(tensor_dict)
    print("\nDone.")


if __name__ == "__main__":
    main()
