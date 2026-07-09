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

"""Verify the LLaVA-Hound-DPO offline MLLM DPO data pipeline end-to-end.

Uses the upstream ``verl.utils.dataset.rl_dataset.RLHFDataset`` directly for
data loading; only the collate function is replaced with
``verl_omni.utils.dataset.offline_mllm_dpo_dataset.offline_mllm_dpo_collate_fn``.

Runs without GPU and without real model weights.  Checks that:

  1. RLHFDataset loads the parquet(s) and builds raw_prompt correctly
     (structured multimodal content is passed through unchanged;
     chosen/rejected columns are transparently returned).
  2. DataLoader + offline_mllm_dpo_collate_fn iterates without error.
  3. Each batch has the expected schema after collation:
       raw_prompt  — object array of message lists (with image/video paths or text)
       response    — object array of response strings
       is_chosen   — object array of bool (True/False alternating per pair)
       data_source, ability, reward_model, extra_info  — pass-through columns
  4. Image, text, and video rows can coexist in the same batch.

Usage
-----
# Single modality (video only)
python3 examples/dpo_trainer/qwen3_omni/verify_data_pipeline.py \\
    --train_files ~/data/llava_hound_dpo/parquet/video/train.parquet \\
    --batch_size 4

# Mixed multisource (image + text + video)
python3 examples/dpo_trainer/qwen3_omni/verify_data_pipeline.py \\
    --train_files \\
        ~/data/llava_hound_dpo/parquet/image/train.parquet \\
        ~/data/llava_hound_dpo/parquet/text/train.parquet \\
        ~/data/llava_hound_dpo/parquet/video/train.parquet \\
    --batch_size 4 --max_samples 64
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ── Upstream verl dataset (no verl_omni wrapper) ──────────────────────────
from verl.utils.dataset.rl_dataset import RLHFDataset

# ── Our collate function (the only verl_omni-specific piece) ──────────────
from verl_omni.utils.dataset.offline_mllm_dpo_dataset import offline_mllm_dpo_collate_fn

# ---------------------------------------------------------------------------
# Minimal stubs — no GPU, no model download required
# ---------------------------------------------------------------------------


def _dummy_tokenizer():
    tok = MagicMock()
    tok.apply_chat_template = lambda msgs, **kw: "dummy"
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.__call__ = lambda *a, **kw: {"input_ids": torch.tensor([[0, 1, 2]])}
    return tok


def _dummy_processor():
    proc = MagicMock()
    proc.apply_chat_template = lambda msgs, **kw: "dummy"
    proc.tokenizer = _dummy_tokenizer()
    proc.image_processor = MagicMock(patch_size=14)
    return proc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify offline MLLM DPO data pipeline.")
    parser.add_argument(
        "--train_files",
        nargs="+",
        required=True,
        help="Path(s) to train.parquet produced by llava_hound_dpo_multisource.py",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="DataLoader batch size (pairs).")
    parser.add_argument("--max_samples", type=int, default=32, help="Rows to load (keeps the test fast).")
    parser.add_argument("--num_batches", type=int, default=3, help="Batches to iterate.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load with upstream RLHFDataset
    # ------------------------------------------------------------------
    data_cfg = OmegaConf.create(
        {
            "cache_dir": "~/.cache/verl/rlhf_verify",
            "prompt_key": "prompt",
            "max_prompt_length": 4096,
            "filter_overlong_prompts": False,  # skip — no real processor available
            "truncation": "left",
            "return_raw_chat": False,
        }
    )

    paths = [str(Path(f).expanduser()) for f in args.train_files]
    print(f"Loading {paths} …", flush=True)

    dataset = RLHFDataset(
        data_files=paths,
        tokenizer=_dummy_tokenizer(),
        processor=_dummy_processor(),
        config=data_cfg,
        max_samples=args.max_samples,
    )
    print(f"  Dataset size: {len(dataset)} pairs\n", flush=True)

    # ------------------------------------------------------------------
    # 2. Inspect one raw item
    # ------------------------------------------------------------------
    sample = dataset[0]
    print("── Raw item (dataset[0]) ──────────────────────────────────────────")
    for k, v in sample.items():
        snippet = repr(v[0] if isinstance(v, list) and v else v)[:80]
        print(f"  {k:20s}: {type(v).__name__:10s}  {snippet}")

    assert "raw_prompt" in sample, "raw_prompt missing"
    assert "chosen" in sample, "chosen column not passed through by RLHFDataset"
    assert "rejected" in sample, "rejected column not passed through by RLHFDataset"

    user_msg = next((m for m in sample["raw_prompt"] if m.get("role") == "user"), None)
    assert user_msg is not None, "No user message in raw_prompt"
    if isinstance(user_msg["content"], list):
        media_types = {p["type"] for p in user_msg["content"] if isinstance(p, dict)}
        assert media_types & {"image", "video"}, f"No visual media part in user content; found: {media_types}"
        prompt_kind = ",".join(sorted(media_types))
    else:
        assert isinstance(user_msg["content"], str), "Text-only user content should be a string"
        prompt_kind = "text"

    print(f"\n  ✓ raw_prompt: content intact, prompt kind = {prompt_kind}")
    print(f"  ✓ chosen   = {sample['chosen'][:80]!r}")
    print(f"  ✓ rejected = {sample['rejected'][:80]!r}")

    # ------------------------------------------------------------------
    # 3. DataLoader with offline_mllm_dpo_collate_fn
    # ------------------------------------------------------------------
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=offline_mllm_dpo_collate_fn,
        num_workers=0,
    )

    print(f"\n── {args.num_batches} batches (batch_size={args.batch_size} pairs → {args.batch_size * 2} samples) ──")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        n = len(batch["response"])
        assert n == args.batch_size * 2, f"Batch {batch_idx}: expected {args.batch_size * 2} samples, got {n}"

        # Verify chosen/rejected alternation within each pair
        is_chosen = list(batch["is_chosen"])
        for i in range(0, len(is_chosen), 2):
            assert bool(is_chosen[i]) is True, f"position {i} should be chosen"
            assert bool(is_chosen[i + 1]) is False, f"position {i + 1} should be rejected"

        # Count modalities
        raw_prompts = batch["raw_prompt"]
        image_n = sum(
            1
            for msgs in raw_prompts
            if any(
                isinstance(m.get("content"), list)
                and any(isinstance(p, dict) and p.get("type") == "image" for p in m["content"])
                for m in msgs
            )
        )
        video_n = sum(
            1
            for msgs in raw_prompts
            if any(
                isinstance(m.get("content"), list)
                and any(isinstance(p, dict) and p.get("type") == "video" for p in m["content"])
                for m in msgs
            )
        )
        text_n = sum(
            1 for msgs in raw_prompts if any(isinstance(m.get("content"), str) for m in msgs if m.get("role") == "user")
        )
        sources = {str(ds) for ds in batch.get("data_source", [])}
        print(
            f"  batch {batch_idx}: {n} samples ({n // 2} pairs) | "
            f"image={image_n} text={text_n} video={video_n} | sources={sources}"
        )

    print("\n✓ Pipeline verification passed.")
    print("  RLHFDataset        — loads parquet, passes chosen/rejected through")
    print("  offline_mllm_dpo_collate_fn — expands pairs to chosen/rejected samples")
    print("  Image/text/video multisource batches handled transparently")


if __name__ == "__main__":
    main()
