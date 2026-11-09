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

"""Manual real-GPU acceptance of synchronous CLIP reward replicas.

Use the production named manager, rank implementation and scorer. Test-only
subclasses expose raw features and memory/resource telemetry without changing
inference. Run with exactly two GPUs per declared logical replica.
"""

import argparse
import asyncio
import gc
import hashlib
import json
import os
import signal
import tempfile
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayResourcePool

from tests.reward_loop.validate_native_fsdp_cpu import _ADAPTERS, assert_dead, make_tiny_checkpoint
from verl_omni.reward_loop import native_fsdp
from verl_omni.reward_loop.reward_loop import OmniFSDPRewardLoopWorker, OmniRewardLoopManager


class GPUValidationRank(native_fsdp.PickScoreFSDPRank):
    """Observe actual allocation and placement on the unmodified execution path."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def status(self):
        """Return rank state with actual Ray bundle IDs and allocator peaks."""
        resources = ray._private.worker.global_worker.core_worker.resource_ids()
        return {
            **super().status(),
            "assigned_resources": {key: sum(amount for _, amount in values) for key, values in resources.items()},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }


class GPUValidationFacade(OmniFSDPRewardLoopWorker):
    """Expose the same executor's raw outputs for numerical acceptance only."""

    async def raw_features(self, prompts, images):
        """Run through the facade's existing ordered executor, preserving ownership."""
        executor = next(iter(self.native_reward_executors.values()))
        return await executor.infer(prompts, images)


def inputs(media_jsonl):
    """Use real generated media with odd/repeated/reordered and truncated input."""
    rows = [json.loads(line) for line in media_jsonl.read_text().splitlines()]
    assert rows
    images = [Image.open(rows[index % len(rows)]["output"]).convert("RGB") for index in range(17)]
    prompts = [rows[index % len(rows)]["input"] for index in range(17)]
    prompts[-1] += " with details" * 100
    return prompts, images


def data_batch(prompts, images):
    """Build the production decoded-image input contract."""
    return DataProto.from_dict(
        tensors={
            "responses": torch.stack([torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) for image in images])
        },
        non_tensors={
            "data_source": ["gpu_fsdp_acceptance"] * len(prompts),
            "reward_model": [{"ground_truth": prompt} for prompt in prompts],
            "media_kind": ["image"] * len(prompts),
        },
    )


def main():
    """Require actual sharding, output parity, lifecycle completion and actor death."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--media-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", choices=sorted(_ADAPTERS), default="clipscore")
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--resident", action="store_true")
    parser.add_argument("--fault", choices=["none", "rank_exit", "facade_exit", "facade_hang"], default="none")
    args = parser.parse_args()
    if args.replicas < 1 or args.cycles < 1 or torch.cuda.device_count() != 2 * args.replicas:
        parser.error("Expose exactly two distinct GPUs per replica; replicas/cycles must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_path = make_tiny_checkpoint(args.output / "tiny-clip") if args.tiny else args.model_path
    module, model_class, inferencer_class, score_fn = _ADAPTERS[args.adapter]
    prompts, images = inputs(args.media_jsonl)
    reference = inferencer_class(model_path=model_path, processor_path=args.processor_path, device="cuda:0")
    raw = {key: value.detach().cpu() for key, value in reference.infer(prompts, images).items()}
    expected = score_fn(**raw)
    global_numel = sum(parameter.numel() for parameter in reference.model.parameters())
    reference_memory = torch.cuda.max_memory_allocated()
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    with initialize_config_dir(config_dir=str(Path("verl_omni/trainer/config").resolve()), version_base=None):
        cfg = compose(config_name="diffusion_trainer")
    cfg.actor_rollout_ref.model.tokenizer_path = args.processor_path
    cfg.trainer.use_v1 = True
    cfg.trainer.v1.trainer_mode = "sync"
    cfg.reward.reward_model.enable = False
    cfg.reward.reward_model.enable_resource_pool = True
    cfg.reward.reward_model.nnodes = 1
    cfg.reward.reward_model.n_gpus_per_node = 2 * args.replicas
    cfg.reward.reward_manager.name = "MultiVisualRewardManager"
    groups = (
        [[1, 0]]
        if args.replicas == 1
        else [[3, 1], [0, 2]]
        if args.replicas == 2
        else [[2 * index + 1, 2 * index] for index in range(args.replicas)]
    )
    cfg.reward.models = {
        args.adapter: {
            "backend": "native",
            "model_path": model_path,
            "offload": not args.resident,
            "placement": {"replica_groups": groups},
            "executor": {"model": f"{module}:{model_class}", "kwargs": {"processor_path": args.processor_path}},
            "fsdp": {"timeout": 300, "collective_timeout": 120},
        }
    }
    cfg.reward.reward_functions = {
        args.adapter: {
            "path": f"pkg://{module}",
            "name": f"compute_score_{args.adapter}_native",
            "required": True,
        }
    }
    report = {
        "passed": False,
        "backend": "gpu:nccl",
        "adapter": args.adapter,
        "groups": groups,
        "cycles": args.cycles,
        "resident": args.resident,
        "fault": args.fault,
        "model_path": model_path,
        "global_numel": global_numel,
        "reference_peak_allocated_bytes": reference_memory,
        "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "media_sha256": [hashlib.sha256(image.tobytes()).hexdigest() for image in images],
        "media_source": str(args.media_jsonl),
    }
    ray.init(
        address="local",
        num_cpus=8 + 2 * args.replicas,
        num_gpus=2 * args.replicas,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _temp_dir=tempfile.mkdtemp(prefix="nfg-"),
    )
    pool = RayResourcePool([2 * args.replicas], max_colocate_count=1, name_prefix="gpu_reward_acceptance")
    manager = None
    primary = None
    try:
        build = partial(native_fsdp.build_replica, _rank_class=GPUValidationRank)
        remote = ray.remote

        def remote_for_validation(*args, **kwargs):
            if args and args[0] is OmniFSDPRewardLoopWorker:
                args = (GPUValidationFacade, *args[1:])
            return remote(*args, **kwargs)

        with patch.object(native_fsdp, "build_replica", build), patch.object(ray, "remote", remote_for_validation):
            manager = OmniRewardLoopManager(cfg, rm_resource_pool=pool)
        owner = manager.multi_reward_model_manager.models[args.adapter]
        asyncio.run(owner.wake_up())
        states = ray.get([rank.status.remote() for rank in owner.rank_workers], timeout=20)
        assert len(states) == 2 * args.replicas
        assert len({state["gpu_uuid"] for state in states}) == 2 * args.replicas
        for index, group in enumerate(groups):
            shards = states[2 * index : 2 * index + 2]
            assert [state["rank"] for state in shards] == [0, 1]
            assert all(state["all_parameters_sharded"] and state["device_type"] == "cuda" for state in shards)
            assert sum(state["local_numel"] for state in shards) == global_numel
            for state, bundle in zip(shards, group, strict=True):
                assert any(
                    name.startswith(f"GPU_group_{bundle}_") and amount > 0
                    for name, amount in state["assigned_resources"].items()
                )
        report["shards"] = states
        parity = []
        for replica in range(args.replicas):
            for indices in [[0], [0, 1, 2], list(range(16)), list(reversed(range(17)))]:
                output = asyncio.run(
                    owner.run_rpc(
                        "raw_features",
                        [prompts[i] for i in indices],
                        [images[i] for i in indices],
                        replica_index=replica,
                    )
                )
                error = 0.0
                for row, index in zip(output, indices, strict=True):
                    for key, value in row.items():
                        target = raw[key] if raw[key].ndim == 0 else raw[key][index]
                        torch.testing.assert_close(value, target, rtol=1e-4, atol=2e-5)
                        error = max(error, float((value - target).abs().max()))
                parity.append({"replica": replica, "indices": indices, "max_raw_abs_error": error})
        report["raw_parity"] = parity

        data = data_batch(prompts, images)
        errors, unloaded, peaks = [], [], []
        for _ in range(args.cycles):
            for validate in [False, True]:
                data.meta_info["validate"] = validate
                actual = manager.compute_rm_score(data).batch["rm_scores"].flatten()
                torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-5)
                errors.append(float((actual - expected).abs().max()))
            states = ray.get([rank.status.remote() for rank in owner.rank_workers], timeout=20)
            assert all(state["awake"] == args.resident for state in states)
            unloaded.append([state["allocated_bytes"] for state in states])
            peaks.append([state["peak_allocated_bytes"] for state in states])
        if not args.resident and args.cycles >= 3:
            for rank in range(2 * args.replicas):
                assert max(row[rank] for row in unloaded[2:]) - min(row[rank] for row in unloaded[2:]) < 2 * 1024**2
        report.update(max_score_abs_error=max(errors), allocations_after_cycles=unloaded, peak_allocated_bytes=peaks)
        if args.fault != "none":
            owner.timeout = 3
            if args.fault == "rank_exit":
                ray.kill(owner.rank_workers[-1], no_restart=True)
            elif args.fault == "facade_exit":
                ray.kill(owner._workers[-1], no_restart=True)
            else:
                row = ray._private.state.actors()[owner._workers[-1]._actor_id.hex()]
                os.kill(row["Pid"], signal.SIGSTOP)
            try:
                manager.compute_rm_score(data)
            except Exception as error:
                report["expected_fault_error"] = type(error).__name__
            else:
                raise AssertionError("Injected GPU failure returned scores")
            assert owner._closed
        report["passed"] = True
    except BaseException as error:
        primary = error
    finally:
        try:
            if manager is not None:
                owner = manager.multi_reward_model_manager.models[args.adapter]
                asyncio.run(manager.multi_reward_model_manager.close_native_models())
                assert_dead([*owner.rank_workers, *owner._workers])
                report["all_owned_actors_dead_before_ray_shutdown"] = True
        except BaseException as cleanup:
            primary = (
                cleanup
                if primary is None
                else BaseExceptionGroup("GPU acceptance and cleanup failed", [primary, cleanup])
            )
        finally:
            report["passed"] = report["passed"] and primary is None
            if primary is not None:
                report["failure"] = {"type": type(primary).__name__, "message": str(primary)}
            (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
            ray.shutdown()
    if primary is not None:
        raise primary
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
