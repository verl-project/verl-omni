#!/usr/bin/env python3
"""Offline reward evaluation: load saved .pt rollout tensors and compute
CLAP + ImageBind scores on the current device.

Usage:
    python eval_reward_offline.py --rollout_dir outputs/<exp>/logs/<ts>/rollout_videos/1 \
        --clap_model_path /data2/m00659926/models/larger_clap_general \
        --imagebind_model_path /data2/m00659926/models/imagebind/imagebind_huge.pth \
        --device cuda:0

Run this script on both GPU and NPU machines using the SAME .pt files
(copied from one machine to the other) to verify reward parity.
"""
import argparse
import asyncio
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from verl_omni.utils.reward_score.clap import compute_score as clap_compute_score
from verl_omni.utils.reward_score.imagebind import compute_score as imagebind_compute_score


def load_pt_sample(pt_path: str):
    data = torch.load(pt_path, weights_only=False)
    video = data["video"]  # [T, C, H, W] uint8
    audio = data["audio"]  # [T] or [C, T] float
    audio_sample_rate = data["audio_sample_rate"]
    return video, audio, audio_sample_rate


def load_prompt_from_jsonl(jsonl_path: str, sample_idx: int) -> str:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                entry = json.loads(line)
                return entry.get("gts") or entry.get("input", "")
    raise IndexError(f"Sample {sample_idx} not found in {jsonl_path}")


async def eval_single(
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_sample_rate: int,
    prompt: str,
    device: str,
    clap_model_path: str,
    imagebind_model_path: str,
) -> dict:
    extra_info = {"audio": audio, "audio_sample_rate": audio_sample_rate}

    clap_result = await clap_compute_score(
        data_source="",
        solution_image=None,
        ground_truth=prompt,
        extra_info=extra_info,
        device=device,
        model_name_or_path=clap_model_path,
    )

    imagebind_result = imagebind_compute_score(
        data_source="",
        solution_image=video,
        ground_truth=prompt,
        extra_info=extra_info,
        device=device,
        model_name_or_path=imagebind_model_path,
        mode="audio_video",
    )

    clap_score = clap_result["score"]
    ib_score = imagebind_result["score"]
    combined = clap_score + ib_score

    return {
        "clap": clap_score,
        "imagebind": ib_score,
        "combined": combined,
    }


async def main():
    parser = argparse.ArgumentParser(description="Offline reward evaluation from saved .pt tensors")
    parser.add_argument("--rollout_dir", required=True, help="Path to step rollout dir (e.g. .../rollout_videos/1)")
    parser.add_argument("--clap_model_path", required=True, help="Path to CLAP model")
    parser.add_argument("--imagebind_model_path", required=True, help="Path to ImageBind weights")
    parser.add_argument("--device", default=None, help="Device (e.g. cuda:0, npu:0). Auto-detect if omitted.")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    args = parser.parse_args()

    device = args.device
    if device is None:
        from verl.utils.device import get_device_name
        device = get_device_name()
    print(f"Using device: {device}")

    jsonl_path = os.path.join(os.path.dirname(args.rollout_dir), f"{os.path.basename(args.rollout_dir)}.jsonl")

    pt_files = sorted(
        [f for f in os.listdir(args.rollout_dir) if f.endswith(".pt")],
        key=lambda x: int(x.split(".")[0]),
    )
    if args.max_samples:
        pt_files = pt_files[: args.max_samples]

    print(f"Found {len(pt_files)} samples in {args.rollout_dir}")

    for pt_file in pt_files:
        idx = int(pt_file.split(".")[0])
        pt_path = os.path.join(args.rollout_dir, pt_file)

        video, audio, audio_sample_rate = load_pt_sample(pt_path)
        prompt = load_prompt_from_jsonl(jsonl_path, idx) if os.path.exists(jsonl_path) else ""

        result = await eval_single(
            video=video,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            prompt=prompt,
            device=device,
            clap_model_path=args.clap_model_path,
            imagebind_model_path=args.imagebind_model_path,
        )

        print(f"  sample {idx}: clap={result['clap']:.4f}  imagebind={result['imagebind']:.4f}  combined={result['combined']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
