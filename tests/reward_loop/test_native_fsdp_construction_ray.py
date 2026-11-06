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
"""Explicit real-Ray CPU integration tests; excluded from the lightweight L1 selector."""

import asyncio
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray

from tests.reward_loop.test_native_fsdp_integration_on_cpu import config
from tests.reward_loop.validate_native_fsdp_construction import assert_dead, wait_for_marker
from verl_omni.reward_loop import native_fsdp, reward_loop
from verl_omni.reward_loop.reward_model import MultiRewardModelManager
from verl_omni.workers.config.reward import parse_reward_model_config


class _ControlRank:
    def status(self):
        return True


class _ControlFacade:
    def __init__(self, *args, mode, markers):
        self.mode, self.markers = mode, Path(markers)
        if mode.startswith("constructor_"):
            self._fault()

    def _fault(self):
        (self.markers / "fault_entered").write_text("injected")
        if self.mode.endswith("error"):
            raise RuntimeError(f"CONTROL_PLANE_{self.mode}")
        time.sleep(120)

    def ready(self):
        if self.mode.startswith("ready_"):
            self._fault()
        return True


@pytest.fixture(scope="module")
def construction_cluster():
    try:
        ray.init(
            address="local",
            num_cpus=6,
            num_gpus=0,
            include_dashboard=False,
            object_store_memory=128 * 1024**2,
            _temp_dir=tempfile.mkdtemp(prefix="nf-construction-cpu-"),
        )
        yield
    finally:
        ray.shutdown()


@pytest.mark.parametrize("mode", ["constructor_error", "constructor_hang", "ready_error", "ready_hang"])
def test_real_ray_construction_rollback_confirms_owned_actor_death(construction_cluster, tmp_path, monkeypatch, mode):
    cfg = config()
    cfg.reward.models.pickscore.fsdp = {"timeout": 3, "collective_timeout": 30}
    manager = object.__new__(reward_loop.OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=2))
    owner = manager.multi_reward_model_manager.models["pickscore"]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    spec = manager.multi_reward_model_manager.reward_model_specs["pickscore"]
    remote, get = ray.remote, ray.get
    ranks, facades, observed = [], [], []
    try:
        for _ in range(2):
            ranks.append(remote(_ControlRank).options(num_cpus=1, num_gpus=0).remote())
        get([rank.status.remote() for rank in ranks], timeout=90)

        def submit(*args):
            actor = (
                remote(_ControlFacade).options(num_cpus=1, num_gpus=0).remote(*args, mode=mode, markers=str(tmp_path))
            )
            facades.append(actor)
            return actor

        def observe_get(ref, **kwargs):
            if not observed:
                wait_for_marker(tmp_path / "fault_entered")
                assert owner._workers == facades
                observed.append(time.monotonic())
            return get(ref, **kwargs)

        monkeypatch.setattr(
            native_fsdp,
            "build_replica",
            lambda *args, **kwargs: SimpleNamespace(worker_group=SimpleNamespace(workers=ranks)),
        )
        monkeypatch.setattr(reward_loop.ray, "remote", lambda cls: SimpleNamespace(remote=submit))
        monkeypatch.setattr(reward_loop.ray, "get", observe_get)
        with pytest.raises(BaseExceptionGroup) as error:
            manager._create_fsdp_worker(cfg, spec, model)
        assert owner._closed and owner._termination_error is None
        if mode.endswith("error"):
            assert f"CONTROL_PLANE_{mode}" in str(error.value.exceptions[0])
        else:
            assert isinstance(error.value.exceptions[0], ray.exceptions.GetTimeoutError)
        assert time.monotonic() - observed[0] < 15
        assert assert_dead([*facades, *ranks]) == ["DEAD"] * 3
        asyncio.run(owner.close())
    finally:
        for actor in [*facades, *ranks]:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass
