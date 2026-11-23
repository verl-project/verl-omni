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

"""Benchmark metrics projection or training reads through a real TransferQueue."""

import argparse
import hashlib
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

from verl_omni.trainer.diffusion.v1.tq_utils import (
    diffusion_metric_tq_fields,
    diffusion_tq_batch_to_dataproto,
)
from verl_omni.utils.diffusion_tq import materialize_diffusion_trajectory


@ray.remote(num_cpus=1)
class _TrajectoryReader:
    def read(self, data: TensorDict, fields: list[str]) -> dict[str, list[dict]]:
        """Materialize replay tensors and hash every field/row on a separate worker."""
        data = materialize_diffusion_trajectory(data)
        return {
            field: [
                {
                    "shape": list(row.shape),
                    "dtype": str(row.dtype),
                    "sha256": hashlib.sha256(row.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                }
                for row in data[field].unbind()
            ]
            for field in fields
        }


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
    parser.add_argument("--prompt-tokens", type=int, default=None)
    parser.add_argument("--read-mode", choices=["metrics", "training"], default="metrics")
    parser.add_argument("--payload", choices=["sd35", "wan22"], default="sd35")
    parser.add_argument(
        "--check-row-order",
        action="store_true",
        help="Reverse worker rows and duplicate one row to check replay handles.",
    )
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
    if args.prompt_tokens is None:
        args.prompt_tokens = 1024 if args.payload == "wan22" else 333
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")
    if args.batch_size < 1 or args.storage_units < 1:
        parser.error("batch-size and storage-units must be positive")
    if args.resolution < 8 or args.resolution % 8:
        parser.error("resolution must be a positive multiple of 8")
    if min(args.steps, args.prompt_tokens, args.object_store_mb) < 1:
        parser.error("steps, prompt-tokens and object-store-mb must be positive")
    if args.read_mode == "training" and args.algorithm != "policy_gradient":
        parser.error("worker-side training reads require policy_gradient")
    if args.check_row_order and args.read_mode != "training":
        parser.error("check-row-order requires --read-mode training")
    if args.payload == "wan22":
        if args.read_mode != "training":
            parser.error("wan22 payload requires --read-mode training")
        if min(args.height, args.width) < 16 or args.height % 16 or args.width % 16:
            parser.error("Wan2.2 height and width must be positive multiples of 16")
        if args.frames < 1 or (args.frames - 1) % 4:
            parser.error("Wan2.2 frames must be 4k+1")

    ray.init(
        address="local",
        num_cpus=args.storage_units + 3,
        num_gpus=0,
        object_store_memory=args.object_store_mb * 2**20,
        _temp_dir=args.ray_temp_dir,
        include_dashboard=False,
        log_to_driver=False,
    )
    try:
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
        payload = _make_sd35_payload(
            args.batch_size,
            args.resolution,
            args.steps,
            args.prompt_tokens,
            args.algorithm,
        )
        trajectory_fields = None
        worker = None
        if args.read_mode == "training":
            from verl_omni.pipelines.model_base import DiffusionModelBase
            from verl_omni.pipelines.rollout_media import ARTIFACT_PREFIX

            payload = payload.exclude("latents_clean")
            if args.payload == "wan22":
                payload = payload.exclude("pooled_prompt_embeds")
                payload["responses"] = torch.zeros(
                    args.batch_size, 3, args.frames, args.height, args.width, dtype=torch.uint8
                )
                payload["all_latents"] = torch.zeros(
                    args.batch_size,
                    args.steps + 1,
                    48,
                    (args.frames - 1) // 4 + 1,
                    args.height // 16,
                    args.width // 16,
                    dtype=torch.float32,
                )
                architecture, algorithm = "WanPipeline", "dance_grpo"
            else:
                payload["negative_pooled_prompt_embeds"] = payload["pooled_prompt_embeds"].clone()
                architecture, algorithm = "StableDiffusion3Pipeline", "flow_grpo"
            payload["all_timesteps"] = torch.arange(args.steps, dtype=torch.float32).repeat(args.batch_size, 1)
            payload["rm_scores"] = torch.ones(args.batch_size, 1)
            payload["negative_prompt_embeds"] = payload["prompt_embeds"].clone()
            payload["negative_prompt_embeds_mask"] = payload["prompt_embeds_mask"].clone()
            adapter = DiffusionModelBase.get_class_by_name(architecture, algorithm)
            trajectory_fields = adapter.tq_trajectory_fields(
                OmegaConf.create({"pipeline": {"guidance_scale": 4.5, "true_cfg_scale": 4.5}})
            )
            for field_idx, field in enumerate(sorted(trajectory_fields.worker_required)):
                for row_idx in range(args.batch_size):
                    row = payload[field][row_idx]
                    pattern = torch.arange(row.numel(), dtype=torch.int64).remainder_(127).reshape(row.shape)
                    row.copy_(pattern)
                    row.add_(row_idx * 128 + field_idx * 1024)
                    if field == "all_latents":
                        for step in range(row.shape[0]):
                            row[step].add_(step * 256)
            kind = "video" if args.payload == "wan22" else "image"
            preview = payload["responses"]
            if kind == "video":
                preview = preview.permute(0, 2, 1, 3, 4)
            payload[f"{ARTIFACT_PREFIX}{kind}_preview"] = preview.contiguous().clone()
            payload[f"{ARTIFACT_PREFIX}{kind}_latent"] = payload["all_latents"][:, -1].clone()
            worker = _TrajectoryReader.remote()
        tags = [
            {
                "status": "success",
                "response_shape": tuple(payload["responses"].shape[1:]),
            }
            for _ in keys
        ]
        batch_meta = tq.kv_batch_put(keys=keys, partition_id="benchmark", fields=payload, tags=tags)
        metric_fields = diffusion_metric_tq_fields(args.algorithm)
        variants = [("baseline", None), ("projected", metric_fields)]
        samples = {name: [] for name, _ in variants}
        worker_samples = {name: [] for name, _ in variants}
        tensor_bytes = {}
        checksums = {}
        returned_fields = {}
        for round_idx in range(args.warmup + args.iterations):
            order = variants if round_idx % 2 == 0 else list(reversed(variants))
            for name, select_fields in order:
                started = time.perf_counter_ns()
                if args.read_mode == "training":
                    data = diffusion_tq_batch_to_dataproto(
                        batch_meta,
                        trajectory_fields=trajectory_fields if name == "projected" else None,
                    )
                    tensors = data.batch
                else:
                    data = tq.kv_batch_get(
                        keys=batch_meta.keys,
                        partition_id=batch_meta.partition_id,
                        select_fields=select_fields,
                    )
                    tensors = data
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                tensor_bytes[name] = _tensor_nbytes(tensors)
                returned_fields[name] = sorted(tensors.keys())
                if args.read_mode == "training" and name == "projected":
                    unexpected = {
                        field for field in tensors.keys() if field == "responses" or field.startswith(ARTIFACT_PREFIX)
                    }
                    if unexpected:
                        raise RuntimeError(f"Projected driver read included media fields: {sorted(unexpected)}")
                if args.read_mode == "training":
                    from verl_omni.trainer.diffusion.diffusion_trainer_utils import _to_diffusion_worker_tensordict
                    from verl_omni.workers.utils.padding import embeds_padding_2_no_padding

                    if args.check_row_order:
                        data = data.select_idxs(list(reversed(range(len(data)))) + [0])
                    fields = sorted((trajectory_fields.worker_fields | {"all_timesteps"}) & set(payload.keys()))
                    worker_inputs = embeds_padding_2_no_padding(_to_diffusion_worker_tensordict(data))
                    checksums[name] = ray.get(worker.read.remote(worker_inputs, fields))
                    worker_ms = (time.perf_counter_ns() - started) / 1_000_000
                    if round_idx >= args.warmup:
                        worker_samples[name].append(worker_ms)
                else:
                    checksums[name] = float(data["sample_level_rewards"][0, 0])
                if round_idx >= args.warmup:
                    samples[name].append(elapsed_ms)
            if checksums["baseline"] != checksums["projected"]:
                differing_fields = (
                    [field for field in fields if checksums["baseline"][field] != checksums["projected"][field]]
                    if args.read_mode == "training"
                    else ["sample_level_rewards"]
                )
                raise RuntimeError(f"Full and projected reads differ in round {round_idx}: {differing_fields}")
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
            "worker_batch_size": args.batch_size + int(args.check_row_order) if args.read_mode == "training" else None,
            "resolution": args.resolution,
            "steps": args.steps,
            "prompt_tokens": args.prompt_tokens,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "algorithm": args.algorithm,
            "read_mode": args.read_mode,
            "payload": args.payload,
            "response_shape": list(payload["responses"].shape[1:]),
            "stored_fields": sorted(payload.keys()),
            "stored_tensor_bytes": sum(
                value.numel() * value.element_size() for value in payload.values() if isinstance(value, torch.Tensor)
            ),
        },
        "selected_fields": metric_fields if trajectory_fields is None else None,
        "trajectory_fields": None
        if trajectory_fields is None
        else {
            "required": sorted(trajectory_fields.worker_required),
            "optional": sorted(trajectory_fields.worker_optional),
        },
        "returned_fields": returned_fields,
        "returned_tensor_bytes": tensor_bytes,
        "tensor_byte_reduction_ratio": 1 - tensor_bytes["projected"] / tensor_bytes["baseline"],
        "latency": {name: _summarize(values) for name, values in samples.items()},
        "median_speedup": baseline_median / projected_median,
        "checksums": checksums,
        "verification": "Per-field, per-row SHA-256 and shape/dtype equality on every round"
        if args.read_mode == "training"
        else "Metric scalar equality on every round",
        "verified_rounds": args.warmup + args.iterations,
        "read_and_worker_latency": {name: _summarize(values) for name, values in worker_samples.items() if values},
        "limits": (
            "Synthetic tensors with real TQ/Ray transport. Bytes count returned tensors, excluding metadata and wire "
            "headers. Training payloads include named preview and latent artifacts. "
            "Read-and-worker latency includes SHA-256 verification, not a model training step."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
