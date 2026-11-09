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
"""Actual facade exit/pause during idle/scoring, with independent owner cleanup."""

import argparse
import asyncio
import hashlib
import json
import os
import signal
import tempfile
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import ray
import torch
import transfer_queue as tq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.protocol import DataProto
from verl.single_controller.ray import RayResourcePool

from tests.reward_loop.validate_native_fsdp import finish_validation
from tests.reward_loop.validate_native_fsdp_faults import FaultRank, case_checkpoint, wait_markers
from verl_omni.reward_loop import native_fsdp
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.trainer.main_diffusion_v1 import finish_training_resources


def owned_actors():
    """Inspect GCS before Ray shutdown or Python garbage collection."""
    return [
        {key: row[key] for key in ("ActorClassName", "ActorID", "Pid", "State")}
        for row in ray._private.state.actors().values()
        if row["ActorClassName"] in {"PickScoreFSDPRank", "FaultRank", "TransferQueueController", "SimpleStorageUnit"}
    ]


async def exercise(manager, data, mode, markers):
    """Enter real scoring before killing/pausing its CPU coordinator."""
    owner = manager.multi_reward_model_manager.models["pickscore"]
    await owner.wake_up()
    owner.timeout = 2
    facade = manager._reward_worker_groups["pickscore"][0]
    pid = ray._private.state.actors()[facade._actor_id.hex()]["Pid"]
    pending = None
    if mode.startswith("scoring"):
        pending = asyncio.create_task(manager.async_compute_rm_score(data))
        await asyncio.to_thread(wait_markers, markers, 20)
        assert not pending.done()
    if mode.endswith("exit"):
        ray.kill(facade, no_restart=True)
    else:
        os.kill(pid, signal.SIGSTOP)
    started = time.monotonic()
    try:
        if pending is None:
            await owner.close()
        else:
            await pending
    except Exception as exc:
        assert owner._closed and owner._termination_error is None
        assert time.monotonic() - started < 15
        for actor in owner.rank_workers:
            try:
                await actor.status.remote()
            except ray.exceptions.RayActorError:
                continue
            raise AssertionError("rank survived independent owner cleanup") from exc
        return exc, time.monotonic() - started
    raise AssertionError("facade failure returned success")


def main():
    """Exercise production manager, dispatcher, owner, ranks and actual queue cleanup."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--processor-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(Path("verl_omni/trainer/config").resolve()), version_base=None):
        base = compose(config_name="diffusion_trainer", overrides=json.loads(args.overrides.read_text()))
    base.actor_rollout_ref.model.tokenizer_path = args.processor_path
    del base.reward.reward_functions.jpeg
    base.transfer_queue.enable = True
    ray.init(
        num_cpus=16,
        num_gpus=2,
        include_dashboard=False,
        object_store_memory=256 * 1024**2,
        _temp_dir=tempfile.mkdtemp(prefix="nf-owner-"),
        runtime_env={"env_vars": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "4"}},
    )
    pool = RayResourcePool([2], max_colocate_count=1, name_prefix="facade_validation")
    data = DataProto.from_dict(
        tensors={"responses": torch.full((1, 3, 224, 224), 127, dtype=torch.uint8)},
        non_tensors={
            "data_source": np.array(["acceptance"]),
            "reward_model": np.array([{"ground_truth": "a gray image"}]),
            "media_kind": np.array(["image"]),
        },
        meta_info={"validate": False},
    )
    report = {
        "cases": [],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).resolve().relative_to(Path.cwd()),
                Path("verl_omni/reward_loop/native_fsdp.py"),
                Path("verl_omni/reward_loop/reward_loop.py"),
                Path("verl_omni/reward_loop/reward_model.py"),
                Path("verl_omni/reward_loop/replica_dispatch.py"),
                Path("verl_omni/trainer/main_diffusion_v1.py"),
            )
        },
    }
    primary = None
    manager = None
    try:
        for mode in ("idle_exit", "scoring_exit", "idle_hang", "scoring_hang"):
            cfg = OmegaConf.create(OmegaConf.to_container(base, resolve=True))
            checkpoint = args.output / mode
            markers = case_checkpoint(args.model_path, checkpoint, "hold" if mode.startswith("scoring") else "none", 0)
            cfg.reward.models.pickscore.model_path = str(checkpoint.resolve())
            tq.init(cfg.transfer_queue)
            build = native_fsdp.build_replica

            with patch.object(native_fsdp, "build_replica", partial(build, _rank_class=FaultRank)):
                manager = OmniRewardLoopManager(cfg, rm_resource_pool=pool)
            error, elapsed = asyncio.run(exercise(manager, data, mode, markers))
            try:
                finish_training_resources(SimpleNamespace(reward_loop_manager=manager), cfg, tq.close, error)
            except Exception as exc:
                assert exc is error
            deadline = time.monotonic() + 10
            while not all(row["State"] == "DEAD" for row in owned_actors()):
                assert time.monotonic() < deadline
                time.sleep(0.1)
            report["cases"].append(
                {
                    "mode": mode,
                    "error": type(error).__name__,
                    "seconds": elapsed,
                    "all_owned_actors_dead_before_shutdown": True,
                    "actors": owned_actors(),
                }
            )
            manager = None
        report["passed"] = True
    except BaseException as exc:
        primary = exc
    finally:
        try:
            if manager is not None:
                manager.multi_reward_model_manager.models["pickscore"]._terminate_owned()
        finally:
            tq.close()
            finish_validation(None, pool, args.output, report, primary)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
