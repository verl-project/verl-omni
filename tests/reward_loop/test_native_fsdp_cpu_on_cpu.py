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

"""CPU FSDP placement must preserve complete CPU replicas and avoid accelerator APIs."""

from types import SimpleNamespace

import pytest
import torch

from tests.reward_loop.test_native_fsdp_integration_on_cpu import config
from verl_omni.reward_loop import native_fsdp, native_fsdp_cpu, reward_loop
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import MultiRewardModelManager
from verl_omni.workers.config.reward import parse_reward_model_config, validate_reward_model_terms


def cpu_config():
    cfg = config()
    cfg.reward.reward_model.enable_resource_pool = False
    cfg.reward.models.pickscore.placement = {
        "resource": "cpu",
        "replica_groups": [[0, 1], [2, 3]],
        "cpus_per_worker": 2,
    }
    return cfg


def test_cpu_shards_need_no_accelerator_pool_and_flat_cpu_replicas_keep_their_contract():
    cfg = cpu_config()
    validate_reward_model_terms(cfg)
    manager = MultiRewardModelManager(cfg)
    assert manager.native_device_assignments == {}
    assert manager.native_resource_pools == {}
    model = manager.models["pickscore"]
    assert model.placement.bundle_indices == [0, 1, 2, 3]
    assert model.executor_spec.device_type == "cpu"
    flat = parse_reward_model_config(
        "cpu",
        {"backend": "native", "placement": {"resource": "cpu", "num_replicas": 3}, "executor": {"model": "test:Model"}},
    )
    assert flat.placement.num_replicas == 3 and flat.fsdp is None


def test_cpu_group_construction_selects_cpu_factory_and_reserves_cpu_capacity(monkeypatch):
    cfg = cpu_config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg)
    owner = manager.multi_reward_model_manager.models["pickscore"]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    observed = []

    def build(*args, **kwargs):
        observed.append(kwargs["cpus_per_rank"])
        return SimpleNamespace(worker_group=SimpleNamespace(workers=[object(), object()]))

    facade = SimpleNamespace(ready=SimpleNamespace(remote=lambda: True))
    monkeypatch.setattr(native_fsdp_cpu, "build_cpu_replica", build)
    monkeypatch.setattr(
        native_fsdp, "build_replica", lambda *args, **kwargs: pytest.fail("CPU requested GPU placement")
    )
    monkeypatch.setattr(reward_loop.ray, "remote", lambda cls: SimpleNamespace(remote=lambda *args: facade))
    monkeypatch.setattr(reward_loop.ray, "get", lambda ref, **kwargs: ref)
    workers = manager._create_fsdp_worker(cfg, owner.executor_spec, model)
    assert observed == [2, 2]
    assert len(workers) == 2 and len(owner.rank_workers) == 4


def test_cpu_rank_initialization_and_status_do_not_touch_accelerator_module(monkeypatch):
    calls = []
    monkeypatch.setattr(native_fsdp.Worker, "__init__", lambda self: None)
    monkeypatch.setattr(native_fsdp, "get_torch_device", lambda: pytest.fail("CPU touched the accelerator API"))
    monkeypatch.setattr(native_fsdp, "initialize_global_process_group_ray", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(native_fsdp, "init_device_mesh", lambda device, shape: (device, shape))
    monkeypatch.setattr(native_fsdp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(native_fsdp.dist, "get_rank", lambda: 0)
    rank = native_fsdp.PickScoreFSDPRank("explicit", device_type="cpu", cpu_threads=1)
    assert rank.device == torch.device("cpu")
    assert rank.mesh == ("cpu", (2,))
    assert calls == [{"timeout_second": 30, "backend": "gloo"}]
    assert rank.status()["gpu_uuid"] is None
    rank.sleep()


@pytest.mark.parametrize("cpus", [0, -1, True, 1.5])
def test_invalid_cpu_reservation_fails_before_loading_or_ray_allocation(cpus, monkeypatch):
    monkeypatch.setattr(native_fsdp_cpu.CLIPProcessor, "from_pretrained", lambda *args: pytest.fail("loaded processor"))
    with pytest.raises(ValueError, match="positive integer"):
        native_fsdp_cpu.build_cpu_replica("model", "processor", cpus_per_rank=cpus)


def test_cpu_rank_factory_clears_inherited_accelerator_visibility_overrides(monkeypatch):
    observed = []

    class Factory:
        def options(self, **kwargs):
            observed.append(kwargs)
            return self

        def remote(self, *args):
            return object()

    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(native_fsdp_cpu.CLIPProcessor, "from_pretrained", lambda *args: object())
    monkeypatch.setattr(
        native_fsdp_cpu.ray, "get_runtime_context", lambda: SimpleNamespace(get_node_id=lambda: "1" * 56)
    )
    monkeypatch.setattr(
        native_fsdp_cpu,
        "get_master_addr_port",
        SimpleNamespace(options=lambda **kwargs: SimpleNamespace(remote=lambda: ("127.0.0.1", "12345"))),
    )
    monkeypatch.setattr(native_fsdp_cpu.ray, "get", lambda value, **kwargs: value)
    monkeypatch.setattr(native_fsdp_cpu.ray, "remote", lambda cls: Factory())
    monkeypatch.setattr(native_fsdp_cpu, "RayClassWithInitArgs", lambda *args: None)
    monkeypatch.setattr(
        native_fsdp_cpu,
        "RayWorkerGroup",
        lambda **kwargs: SimpleNamespace(world_size=2, workers=kwargs["worker_handles"]),
    )
    replica = native_fsdp_cpu.build_cpu_replica("model", "processor")
    assert replica.worker_group.world_size == 2
    assert len(observed) == 2
    assert all(options["num_gpus"] == 0 for options in observed)
    assert all(
        options["runtime_env"]["env_vars"]["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "" for options in observed
    )
