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
"""Real-Ray construction rollback with actual two-rank CUDA/NCCL replicas."""

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import ray
import torch
from verl.single_controller.ray import RayResourcePool

from tests.reward_loop.test_cpu_native_reward_ray_on_cpu import _full_config
from tests.reward_loop.validate_native_fsdp import finish_validation
from verl_omni.reward_loop import native_fsdp, reward_loop
from verl_omni.reward_loop.reward_model import MultiRewardModelManager
from verl_omni.workers.config.reward import parse_reward_model_config


class ConstructionFacade(reward_loop.OmniFSDPRewardLoopWorker):
    """Fail or block only after actual facade/rank attachment has completed."""

    def __init__(self, *args, mode, markers):
        super().__init__(*args)
        self.mode, self.markers = mode, Path(markers)
        (self.markers / "constructed").write_text(str(os.getpid()))
        if mode.startswith("constructor_"):
            self._fault()

    def _fault(self):
        (self.markers / "fault_entered").write_text(str(os.getpid()))
        if self.mode.endswith("error"):
            raise RuntimeError(f"CONSTRUCTION_{self.mode}")
        time.sleep(120)

    def ready(self):
        """Exercise a real readiness RPC after a successful constructor."""
        if self.mode.startswith("ready_"):
            self._fault()
        return super().ready()


def wait_for_marker(path):
    """Separate worker cold start from the deliberately short readiness deadline."""
    deadline = time.monotonic() + 90
    while not path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Fault setup did not reach {path}")
        time.sleep(0.02)


def assert_dead(actors):
    """Require actor RPC death and GCS DEAD while the test cluster remains live."""
    for actor in actors:
        try:
            ray.get(actor.__ray_ready__.remote(), timeout=10)
        except ray.exceptions.RayActorError:
            continue
        raise AssertionError("Owned actor survived construction rollback")
    deadline = time.monotonic() + 10
    while True:
        rows = ray._private.state.actors()
        states = [rows[actor._actor_id.hex()]["State"] for actor in actors]
        if all(state == "DEAD" for state in states):
            return states
        assert time.monotonic() < deadline, states
        time.sleep(0.02)


def run_case(cfg, pool, mode, directory, replicas):
    """Call the production factory and inspect every created group before shutdown."""
    directory.mkdir()
    manager = object.__new__(reward_loop.OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, pool)
    owner = manager.multi_reward_model_manager.models["pickscore"]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    spec = manager.multi_reward_model_manager.reward_model_specs["pickscore"]
    facades, ranks, markers, observed, rank_states = [], [], [], [], []
    remote, get, build = ray.remote, ray.get, native_fsdp.build_replica
    first_group_alive = []

    def build_group(*args, **kwargs):
        replica = build(*args, **kwargs)
        ranks.extend(replica.worker_group.workers)
        states = get([rank.status.remote() for rank in replica.worker_group.workers], timeout=30)
        assert [row["rank"] for row in states] == [0, 1]
        assert len({row["gpu_uuid"] for row in states}) == 2
        rank_states.extend(states)
        return replica

    def submit_facade(*args):
        index = len(facades)
        marker = directory / f"group-{index}"
        marker.mkdir()
        actual_mode = mode if index == replicas - 1 else "none"
        actor = remote(ConstructionFacade).remote(*args, mode=actual_mode, markers=str(marker))
        facades.append(actor)
        markers.append(marker / ("constructed" if actual_mode == "none" else "fault_entered"))
        return actor

    def remote_class(*args, **kwargs):
        if args and args[0] is reward_loop.OmniFSDPRewardLoopWorker:
            return SimpleNamespace(remote=submit_facade)
        return remote(*args, **kwargs)

    def observe_get(refs, **kwargs):
        if len(facades) > len(observed):
            index = len(observed)
            wait_for_marker(markers[index])
            assert owner._workers == facades
            observed.append(time.monotonic())
            if index == replicas - 1 and replicas > 1:
                first_group_alive.extend(get([rank.status.remote() for rank in ranks[:2]], timeout=10))
                assert get(facades[0].ready.remote(), timeout=10) == 2
        return get(refs, **kwargs)

    primary = None
    try:
        with (
            patch.object(native_fsdp, "build_replica", build_group),
            patch.object(ray, "remote", remote_class),
            patch.object(ray, "get", observe_get),
        ):
            try:
                manager._create_fsdp_worker(cfg, spec, model)
            except BaseExceptionGroup as error:
                primary = error
            else:
                raise AssertionError("Construction fault returned a successful deployment")
        assert observed and primary is not None
        elapsed = time.monotonic() - observed[-1]
        assert owner._closed and owner._termination_error is None
        assert len(ranks) == replicas * 2 and len(facades) == replicas
        assert len({row["gpu_uuid"] for row in rank_states}) == replicas * 2
        if mode.endswith("error"):
            assert f"CONSTRUCTION_{mode}" in str(primary.exceptions[0])
        else:
            assert isinstance(primary.exceptions[0], ray.exceptions.GetTimeoutError)
        assert elapsed < cfg.reward.models.pickscore.fsdp.timeout + 25
        states = assert_dead([*facades, *ranks])
        asyncio.run(owner.close())
        return {
            "mode": mode,
            "replicas": replicas,
            "owned_ranks": len(ranks),
            "owned_facades": len(facades),
            "constructed_ranks": rank_states,
            "states_before_ray_shutdown": states,
            "readiness_and_cleanup_seconds": elapsed,
            "first_group_alive_before_second_failure": first_group_alive,
            "primary_error": str(primary.exceptions[0]),
            "passed": True,
        }
    finally:
        # Emergency test teardown is separate from the asserted production cleanup.
        for actor in [*facades, *ranks]:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass


def main():
    """Exercise actual constructor/readiness failures without production fault switches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    assert torch.cuda.device_count() == args.replicas * 2
    args.output.mkdir(parents=True)
    cfg = _full_config(args.output / "tokenizer", True)
    cfg.trainer.use_v1 = True
    cfg.trainer.v1.trainer_mode = "sync"
    cfg.reward.reward_model.enable_resource_pool = True
    cfg.reward.reward_model.nnodes = 1
    cfg.reward.models = {
        "pickscore": {
            "backend": "native",
            "model_path": args.model_path,
            "placement": {"replica_groups": [[2 * i + 1, 2 * i] for i in range(args.replicas)]},
            "executor": {
                "model": "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel",
                "kwargs": {"processor_path": args.processor_path},
            },
            "fsdp": {"timeout": 3, "collective_timeout": 60},
        }
    }
    cfg.reward.reward_functions = {
        "pickscore": {
            "path": "pkg://verl_omni.utils.reward_score.pickscore_reward",
            "name": "compute_score_pickscore_native",
            "required": True,
        }
    }
    report = {"passed": False, "cases": [], "backend": "cuda:nccl"}
    ray.init(
        address="local",
        num_cpus=12,
        num_gpus=args.replicas * 2,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _temp_dir=tempfile.mkdtemp(prefix="nf-construct-"),
    )
    pool = RayResourcePool([args.replicas * 2], max_colocate_count=1, name_prefix="construction_validation")
    primary = None
    try:
        for mode in ("constructor_error", "constructor_hang", "ready_error", "ready_hang"):
            report["cases"].append(run_case(cfg, pool, mode, args.output / mode, args.replicas))
        report["passed"] = True
    except BaseException as error:
        primary = error
    finally:
        finish_validation(None, pool, args.output, report, primary)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
