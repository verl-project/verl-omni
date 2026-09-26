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
"""Paired two-GPU benchmark for native reward-replica dispatch.

This measures dispatcher behavior with ID-only transport and cached images. The
configured delay is a synthetic per-sample service-cost perturbation, not
evidence of naturally occurring replica latency or full trainer throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import os
import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODES = (("static", None), ("dynamic4", 4), ("dynamic16", 16))
SCENARIOS = (("balanced", 0.0), ("delay_5ms", 0.005), ("delay_20ms", 0.020))
PROMPTS = ("a landscape", "a colorful pattern", "a photograph of an animal")


def _source_identity():
    import hashlib
    import importlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    identities = {}
    for name in (
        "verl_omni.reward_loop.replica_dispatch",
        "verl_omni.reward_loop.reward_loop",
        "verl_omni.reward_loop.reward_model_executor",
        "verl_omni.utils.reward_score.pickscore_reward",
        "verl_omni.workers.config.reward",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"Expected {name} under {root}, got {path}")
        identities[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = Path(__file__).resolve()
    identities[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"root": str(root), "sha256": identities}


class _PickScoreReplica:
    def __init__(self, model_path: str, processor_path: str):
        import torch

        from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
        from verl_omni.workers.config.reward import RewardModelSpec

        assert torch.cuda.device_count() == 1, torch.cuda.device_count()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.executor = NativeRewardExecutor(
            RewardModelSpec(
                name="pickscore",
                backend="native",
                model_path=model_path,
                executor_config={
                    "model": "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel",
                    "kwargs": {"processor_path": processor_path, "dtype": torch.float32},
                },
            )
        )
        self.images = {}
        self.active = 0
        self.reset(0.0)

    async def wake(self):
        import os

        import torch
        import transformers

        await self.executor.wake_up()
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(0)
        return {
            "gpu_uuid": str(properties.uuid),
            "device": torch.cuda.get_device_name(0),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": torch.cuda.device_count(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "source_identity": _source_identity(),
        }

    def prepare_cache(self, sample_count: int):
        import hashlib

        import torch
        from PIL import Image

        digest = hashlib.sha256()
        for sample_id in range(sample_count):
            height = (224, 240, 256)[sample_id % 3]
            generator = torch.Generator().manual_seed(912 + sample_id)
            array = torch.randint(0, 256, (height, 224, 3), dtype=torch.uint8, generator=generator).numpy()
            image = Image.fromarray(array, mode="RGB")
            self.images[sample_id] = image
            digest.update(sample_id.to_bytes(8, "little"))
            digest.update(image.tobytes())
        return {"count": sample_count, "sha256": digest.hexdigest()}

    async def compute_score(self, data):
        from verl_omni.utils.reward_score.pickscore_reward import compute_score_pickscore_native

        sample_id = int(data.batch["sample_id"][0])
        result = await compute_score_pickscore_native(
            data_source="replica-dispatch-benchmark",
            solution_image=self.images[sample_id],
            ground_truth=PROMPTS[sample_id % len(PROMPTS)],
            extra_info={},
            reward_model=self.executor,
        )
        return {
            "reward_score": float(result["score"]),
            "reward_extra_info": {"sample_id": sample_id},
        }

    async def compute_score_batch(self, data):
        import asyncio
        import time

        from verl_omni.reward_loop.reward_loop import OmniRewardLoopWorker

        sample_ids = [int(value) for value in data.batch["sample_id"].tolist()]
        self.ids.extend(sample_ids)
        self.rpc_count += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        delay = self.delay_coefficient * len(sample_ids)
        started = time.perf_counter()
        try:
            if delay:
                await asyncio.sleep(delay)
            return await OmniRewardLoopWorker.compute_score_batch(self, data)
        finally:
            self.synthetic_delay_seconds += delay
            self.service_seconds += time.perf_counter() - started
            self.active -= 1

    def reset(self, delay_coefficient: float):
        assert self.active == 0 and self.executor._inflight == 0
        self.delay_coefficient = delay_coefficient
        self.ids = []
        self.rpc_count = 0
        self.peak = 0
        self.service_seconds = 0.0
        self.synthetic_delay_seconds = 0.0

    def stats(self):
        return {
            "ids": self.ids,
            "sample_count": len(self.ids),
            "rpc_count": self.rpc_count,
            "active": self.active,
            "peak_rpc": self.peak,
            "inflight": self.executor._inflight,
            "service_seconds": self.service_seconds,
            "synthetic_delay_seconds": self.synthetic_delay_seconds,
            "awake": self.executor._model is not None,
        }

    async def sleep(self):
        import torch

        assert self.active == 0 and self.executor._inflight == 0
        await self.executor.sleep()
        torch.cuda.synchronize()
        return self.stats()


def _append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


async def _run_arm(workers, data, mode, scenario, slow_replica, reference=None):
    import torch

    from verl_omni.reward_loop.replica_dispatch import dispatch_reward_groups

    mode_name, batch_size = mode
    scenario_name, coefficient = scenario
    await asyncio.gather(
        *(worker.reset.remote(coefficient if index == slow_replica else 0.0) for index, worker in enumerate(workers))
    )
    batch_sizes = {"pickscore": batch_size} if batch_size is not None else {}
    started = time.perf_counter()
    ordered = (await dispatch_reward_groups(data, {"pickscore": workers}, batch_sizes))["pickscore"]
    scores = [float(item["reward_score"]) for item in ordered]
    elapsed = time.perf_counter() - started
    stats = await asyncio.gather(*(worker.stats.remote() for worker in workers))

    expected_ids = [int(value) for value in data.batch["sample_id"].tolist()]
    ordered_ids = [int(item["reward_extra_info"]["sample_id"]) for item in ordered]
    dispatched_ids = [sample_id for replica in stats for sample_id in replica["ids"]]
    assert ordered_ids == expected_ids
    assert len(dispatched_ids) == len(expected_ids) and sorted(dispatched_ids) == expected_ids
    assert all(replica["peak_rpc"] == 1 for replica in stats)
    assert all(replica["active"] == replica["inflight"] == 0 for replica in stats)
    score_tensor = torch.tensor(scores, dtype=torch.float64)
    assert torch.isfinite(score_tensor).all()
    max_abs_delta = None
    if reference is not None:
        torch.testing.assert_close(score_tensor, reference, atol=1e-4, rtol=1e-4)
        max_abs_delta = (score_tensor - reference).abs().max().item()
    return {
        "mode": mode_name,
        "dispatch_batch_size": batch_size,
        "scenario": scenario_name,
        "delay_coefficient_seconds_per_sample": coefficient,
        "slow_replica": slow_replica,
        "elapsed_seconds": elapsed,
        "diagnostic_service_seconds_sum": sum(replica["service_seconds"] for replica in stats),
        "scores": scores,
        "ordered_ids": ordered_ids,
        "max_abs_delta": max_abs_delta,
        "stats": stats,
    }, score_tensor


async def _execute(args, actor_class, jsonl_path: Path, serialization: dict):
    import torch
    from verl.protocol import DataProto

    workers = [actor_class.remote(args.model_path, args.processor_path) for _ in range(2)]
    try:
        hardware = await asyncio.gather(*(worker.wake.remote() for worker in workers))
        assert all(item["cuda_device_count"] == 1 for item in hardware)
        assert len({item["gpu_uuid"] for item in hardware}) == 2
        assert all(item["source_identity"] == serialization["source_identity"] for item in hardware)
        assert all(not item["tf32_matmul"] and not item["tf32_cudnn"] for item in hardware)
        manifests = await asyncio.gather(*(worker.prepare_cache.remote(args.samples) for worker in workers))
        assert manifests[0] == manifests[1] and manifests[0]["count"] == args.samples
        data = DataProto.from_dict(tensors={"sample_id": torch.arange(args.samples, dtype=torch.int64)})

        baseline, reference = await _run_arm(workers, data, MODES[0], SCENARIOS[0], 0)
        baseline["phase"] = "baseline"
        _append_jsonl(jsonl_path, baseline)
        warms = []
        for scenario in SCENARIOS:
            for mode in MODES:
                record, _ = await _run_arm(workers, data, mode, scenario, 0, reference)
                record["phase"] = "warm"
                warms.append(record)
                _append_jsonl(jsonl_path, record)

        records = []
        mode_orders = list(itertools.permutations(MODES))
        for round_index, mode_order in enumerate(mode_orders):
            scenario_order = SCENARIOS[round_index % len(SCENARIOS) :] + SCENARIOS[: round_index % len(SCENARIOS)]
            slow_replica = round_index % 2
            for scenario in scenario_order:
                for mode in mode_order:
                    record, _ = await _run_arm(workers, data, mode, scenario, slow_replica, reference)
                    record.update(phase="measured", round=round_index, mode_order=[item[0] for item in mode_order])
                    records.append(record)
                    _append_jsonl(jsonl_path, record)
        expected_arms = args.repeats * len(MODES) * len(SCENARIOS)
        assert len(mode_orders) == args.repeats == 6
        assert len(warms) == 9 and len(records) == expected_arms == 54
    finally:
        cleanup = await asyncio.gather(*(worker.sleep.remote() for worker in workers), return_exceptions=True)
    assert all(isinstance(item, dict) and not item["awake"] for item in cleanup), cleanup
    return {
        "complete": True,
        "plan": {
            "samples": args.samples,
            "repeats": args.repeats,
            "modes": [{"name": name, "dispatch_batch_size": size} for name, size in MODES],
            "scenarios": [{"name": name, "delay_seconds_per_sample": value} for name, value in SCENARIOS],
            "mode_orders": [[item[0] for item in order] for order in mode_orders],
            "scenario_order_rule": "rotate left by round_index % 3",
            "slow_replica_rule": "round_index % 2",
            "timing_boundary": "dispatcher entry through ordered scalar result extraction",
            "excluded_from_timing": ["model load", "cache generation", "reset", "stats", "validation"],
            "interpretation": (
                "ID-only transport with synthetic sequential per-RPC service cost; not full trainer or natural latency"
            ),
            "jsonl_sidecar": str(jsonl_path),
        },
        "hardware": hardware,
        "runtime": serialization,
        "cache_manifest": manifests[0],
        "baseline": baseline,
        "warms": warms,
        "records": records,
        "cleanup": cleanup,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ray-temp-dir")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=6)
    return parser.parse_args(argv)


def run(argv=None):
    import ray
    import torch
    import transformers

    args = _parse_args(argv)
    if args.samples < 32 or args.samples % 2:
        raise ValueError("--samples must be an even integer of at least 32 so both replicas receive work")
    if args.repeats != 6:
        raise ValueError("--repeats must be 6 so every mode permutation is measured exactly once")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output.with_name(f"{args.output.name}.jsonl")
    if args.output.exists() or jsonl_path.exists():
        raise FileExistsError("Choose a new output path; existing benchmark evidence is not overwritten")
    with jsonl_path.open("x", encoding="utf-8"):
        pass

    actor_class = ray.remote(num_gpus=1, num_cpus=2)(_PickScoreReplica)
    serialized = ray.cloudpickle.dumps(actor_class.__ray_metadata__.modified_class)
    serialization = {
        "python": sys.version,
        "platform": platform.platform(),
        "ray": ray.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "modified_actor_cloudpickle_bytes": len(serialized),
        "modified_actor_cloudpickle_sha256": hashlib.sha256(serialized).hexdigest(),
        "model_path": args.model_path,
        "processor_path": args.processor_path,
        "jsonl": str(jsonl_path),
        "ray_temp_dir": args.ray_temp_dir,
        "source_identity": _source_identity(),
    }
    init_kwargs = {"num_cpus": 6, "num_gpus": 2, "include_dashboard": False, "object_store_memory": 256 * 1024**2}
    if args.ray_temp_dir:
        init_kwargs["_temp_dir"] = args.ray_temp_dir
    ray.init(**init_kwargs)
    try:
        result = asyncio.run(_execute(args, actor_class, jsonl_path, serialization))
        temporary = args.output.with_name(f".{args.output.name}.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
        return result
    finally:
        ray.shutdown()


if __name__ == "__main__":
    run()
