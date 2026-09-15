#!/usr/bin/env python3
"""Quick check: create a fixed .pt with dummy data and compare reward
on GPU vs NPU without running the full pipeline.

Usage:
    python eval_reward_parity.py --clap_model_path <path> --imagebind_model_path <path>
"""
import argparse
import asyncio
import torch

from verl_omni.utils.reward_score.clap import compute_score as clap_compute_score
from verl_omni.utils.reward_score.imagebind import compute_score as imagebind_compute_score


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clap_model_path", required=True)
    parser.add_argument("--imagebind_model_path", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        from verl.utils.device import get_device_name
        device = get_device_name()
    print(f"Device: {device}")

    # Fixed dummy data — identical on all platforms
    torch.manual_seed(42)
    video = torch.randint(0, 256, (81, 3, 256, 384), dtype=torch.uint8)
    audio = torch.randn(48000) * 0.1
    prompt = "a cat playing piano in a garden"

    extra_info = {"audio": audio, "audio_sample_rate": 48000}

    clap_result = await clap_compute_score(
        data_source="", solution_image=None, ground_truth=prompt,
        extra_info=extra_info, device=device, model_name_or_path=args.clap_model_path,
    )
    ib_result = imagebind_compute_score(
        data_source="", solution_image=video, ground_truth=prompt,
        extra_info=extra_info, device=device,
        model_name_or_path=args.imagebind_model_path, mode="audio_video",
    )

    print(f"CLAP:     {clap_result['score']:.6f}")
    print(f"ImageBind: {ib_result['score']:.6f}")
    print(f"Combined: {clap_result['score'] + ib_result['score']:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
