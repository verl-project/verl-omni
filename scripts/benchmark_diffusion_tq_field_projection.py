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

"""Benchmark full and metrics-only reads through a real TransferQueue."""

import argparse
import importlib.metadata
import json
import platform
import statistics
import time

import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

from verl_omni.trainer.diffusion.v1.tq_utils import diffusion_metric_tq_fields, diffusion_tq_batch_to_dataproto
from verl_omni.workers.config.diffusion.model import DIFFUSION_TQ_DRIVER_FIELDS


def _make_sd35_payload(
    batch_size: int,
    resolution: int,
    steps: int,
    prompt_tokens: int,
    algorithm: str,
) -> TensorDict:
    latent_resolution = resolution // 8
    fields = {
        "prompts": torch.zeros(batch_size, 512, dtype=torch.int64),
        "responses": torch.zeros(batch_size, 3, resolution, resolution, dtype=torch.uint8),
        "all_latents": torch.zeros(
            batch_size,
            steps + 1,
            16,
            latent_resolution,
            latent_resolution,
            dtype=torch.float32,
        ),
        "latents_clean": torch.zeros(
            batch_size,
            16,
            latent_resolution,
            latent_resolution,
            dtype=torch.float32,
        ),
        "prompt_embeds": torch.zeros(batch_size, prompt_tokens, 4096, dtype=torch.bfloat16),
        "prompt_embeds_mask": torch.ones(batch_size, prompt_tokens, dtype=torch.int64),
        "pooled_prompt_embeds": torch.zeros(batch_size, 2048, dtype=torch.bfloat16),
        "rollout_log_probs": torch.zeros(batch_size, steps, dtype=torch.float32),
        "sample_level_rewards": torch.ones(batch_size, steps, dtype=torch.float32),
        "sample_level_scores": torch.ones(batch_size, 1, dtype=torch.float32),
        "uid": NonTensorStack(*(f"prompt-{idx // 2}" for idx in range(batch_size))),
        "extra_fields": NonTensorStack(*({"reward_extra_info": {"ocr": 1.0}} for _ in range(batch_size))),
    }
    if algorithm == "policy_gradient":
        fields.update(
            {
                "advantages": torch.ones(batch_size, steps, dtype=torch.float32),
                "returns": torch.ones(batch_size, steps, dtype=torch.float32),
                "old_log_probs": torch.full((batch_size, steps), -0.5, dtype=torch.float32),
            }
        )
    return TensorDict(fields, batch_size=[batch_size])


def _make_wan22_payload(args: argparse.Namespace) -> TensorDict:
    frames, height, width = args.frames, args.height, args.width
    batch, steps = args.batch_size, args.steps
    responses = torch.zeros(batch, 3, frames, height, width, dtype=torch.uint8)
    latents = torch.zeros(batch, steps + 1, 48, (frames - 1) // 4 + 1, height // 16, width // 16)
    embeds = torch.zeros(batch, args.prompt_tokens, 4096, dtype=torch.bfloat16)
    mask = torch.ones(batch, args.prompt_tokens, dtype=torch.int64)
    return TensorDict(
        {
            "prompts": torch.zeros(batch, 512, dtype=torch.int64),
            "uid": NonTensorStack(*(f"prompt-{idx // 2}" for idx in range(batch))),
            "rm_scores": torch.ones(batch, 1),
            "all_timesteps": torch.arange(steps, dtype=torch.float32).repeat(batch, 1),
            "rollout_log_probs": torch.zeros(batch, steps),
            "responses": responses,
            "media_artifact__video_preview": responses.permute(0, 2, 1, 3, 4).contiguous().clone(),
            "media_artifact__video_latent": latents[:, -1].clone(),
            "all_latents": latents,
            "prompt_embeds": embeds,
            "negative_prompt_embeds": embeds.clone(),
            "prompt_embeds_mask": mask,
            "negative_prompt_embeds_mask": mask.clone(),
        },
        batch_size=[batch],
    )


def _tensor_nbytes(data) -> int:
    return sum(value.numel() * value.element_size() for value in data.values() if isinstance(value, torch.Tensor))


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize(samples: list[float]) -> dict[str, float | list[float]]:
    return {
        "min_ms": min(samples),
        "p25_ms": _percentile(samples, 0.25),
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p75_ms": _percentile(samples, 0.75),
        "p95_ms": _percentile(samples, 0.95),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int)
    parser.add_argument("--read-mode", choices=["metrics", "training"], default="metrics")
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--object-store-mb", type=int, default=512)
    parser.add_argument("--ray-temp-dir", default=None)
    parser.add_argument("--storage-units", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument(
        "--algorithm",
        choices=["policy_gradient", "direct_preference"],
        default="policy_gradient",
        help="Match the metric fields persisted by the selected trainer path.",
    )
    args = parser.parse_args()
    training = args.read_mode == "training"
    if args.prompt_tokens is None:
        args.prompt_tokens = 1024 if training else 333
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")
    if args.batch_size < 1 or args.storage_units < 1:
        parser.error("batch-size and storage-units must be positive")
    if args.resolution < 8 or args.resolution % 8:
        parser.error("resolution must be a positive multiple of 8")

    if training and (args.frames < 1 or (args.frames - 1) % 4):
        parser.error("Wan2.2 frames must be 4k+1")
    if training and any(dim < 16 or dim % 16 for dim in (args.height, args.width)):
        parser.error("Wan2.2 height and width must be positive multiples of 16")

    ray.init(
        address="local",
        num_cpus=args.storage_units + 2,
        num_gpus=0,
        object_store_memory=args.object_store_mb * 2**20,
        _temp_dir=args.ray_temp_dir,
        include_dashboard=False,
        log_to_driver=False,
    )
    tq.init(
        OmegaConf.create(
            {
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {
                        "total_storage_size": max(100, args.batch_size * 4),
                        "num_data_storage_units": args.storage_units,
                    },
                }
            }
        )
    )
    keys = [f"sample_{idx}_0" for idx in range(args.batch_size)]
    payload = (
        _make_wan22_payload(args)
        if training
        else _make_sd35_payload(args.batch_size, args.resolution, args.steps, args.prompt_tokens, args.algorithm)
    )
    tags = [
        {
            "status": "success",
            "response_shape": tuple(payload["responses"].shape[1:]),
        }
        for _ in keys
    ]
    try:
        batch_meta = tq.kv_batch_put(keys=keys, partition_id="benchmark", fields=payload, tags=tags)
        projected_fields = list(DIFFUSION_TQ_DRIVER_FIELDS) if training else diffusion_metric_tq_fields(args.algorithm)
        variants = [("baseline", None), ("projected", projected_fields)]
        samples = {name: [] for name, _ in variants}
        tensor_bytes = {}
        checksums = {}
        returned_fields = {}
        for round_idx in range(args.warmup + args.iterations):
            order = variants if round_idx % 2 == 0 else list(reversed(variants))
            for name, select_fields in order:
                started = time.perf_counter_ns()
                if training:
                    data = diffusion_tq_batch_to_dataproto(batch_meta, select_fields=select_fields).batch
                else:
                    data = tq.kv_batch_get(
                        keys=batch_meta.keys, partition_id=batch_meta.partition_id, select_fields=select_fields
                    )
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                if not training:
                    checksums[name] = float(data["sample_level_rewards"][0, 0])
                tensor_bytes[name] = _tensor_nbytes(data)
                returned_fields[name] = sorted(data.keys())
                if round_idx >= args.warmup:
                    samples[name].append(elapsed_ms)
    finally:
        tq.close()
        ray.shutdown()

    baseline_median = statistics.median(samples["baseline"])
    projected_median = statistics.median(samples["projected"])
    result = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ray": ray.__version__,
            "transfer_queue": importlib.metadata.version("TransferQueue"),
            "backend": "SimpleStorage",
            "storage_units": args.storage_units,
        },
        "workload": {
            "batch_size": args.batch_size,
            "resolution": args.resolution,
            "read_mode": args.read_mode,
            "response_shape": list(payload["responses"].shape[1:]),
            "steps": args.steps,
            "prompt_tokens": args.prompt_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "algorithm": args.algorithm,
            "stored_fields": sorted(payload.keys()),
            "stored_tensor_bytes": sum(
                value.numel() * value.element_size() for value in payload.values() if isinstance(value, torch.Tensor)
            ),
        },
        "selected_fields": projected_fields,
        "returned_fields": returned_fields,
        "returned_tensor_bytes": tensor_bytes,
        "tensor_byte_reduction_ratio": 1 - tensor_bytes["projected"] / tensor_bytes["baseline"],
        "latency": {name: _summarize(values) for name, values in samples.items()},
        "median_speedup": baseline_median / projected_median,
        "limits": "Synthetic single-host TQ driver reads. Tensor bytes exclude metadata and wire headers.",
    }
    if checksums:
        result["checksums"] = checksums
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
