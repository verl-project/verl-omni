# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Omni reward-manager wiring to the shared worker profiler lifecycle."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from verl.experimental.reward_loop.reward_loop import RewardLoopManager

from verl_omni.reward_loop import reward_loop as omni_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager


def test_accelerator_manager_initializes_profilers_after_workers(monkeypatch):
    config = OmegaConf.create({"reward": {"reward_model": {"enable": False}}})
    workers = [object()]
    events = []
    monkeypatch.setattr(omni_module.ray, "remote", Mock())
    monkeypatch.setattr(omni_module, "resolve_reward_manager_cls", Mock())
    monkeypatch.setattr(omni_module, "accelerator_workers_enabled", lambda _: True)
    monkeypatch.setattr(omni_module, "has_reward_models", lambda _: False)
    monkeypatch.setattr(omni_module, "MultiRewardModelManager", lambda *args, **kwargs: SimpleNamespace(models={}))

    def create_workers(manager):
        manager.reward_loop_workers = workers
        events.append("workers")

    def initialize_profilers(manager):
        assert manager.reward_loop_workers is workers
        events.append("profilers")

    monkeypatch.setattr(OmniRewardLoopManager, "_init_reward_loop_workers", create_workers)
    monkeypatch.setattr(RewardLoopManager, "_init_reward_loop_profilers", initialize_profilers)

    OmniRewardLoopManager(config)

    assert events == ["workers", "profilers"]


@pytest.mark.parametrize("action", ["start", "stop"])
def test_manager_delegates_to_workers_and_rollout_replicas(monkeypatch, action):
    manager = object.__new__(OmniRewardLoopManager)
    events = []
    method = f"{action}_profile"
    base = Mock(side_effect=lambda **kwargs: events.append(("workers", kwargs)))
    monkeypatch.setattr(RewardLoopManager, method, base)
    manager._run_on_replicas = Mock(side_effect=lambda *args, **kwargs: events.append(("replicas", kwargs)))

    kwargs = (
        {"profile_step": 2, "role": "reward", "profile_prefix": "reward"}
        if action == "start"
        else {"run_command": False}
    )
    getattr(manager, method)(**kwargs)

    if action == "start":
        assert events == [("workers", kwargs), ("replicas", kwargs)]
    else:
        assert events == [("replicas", {}), ("workers", kwargs)]
    manager._run_on_replicas.assert_called_once_with(method, **(kwargs if action == "start" else {}))


def test_replica_stop_failure_still_stops_workers(monkeypatch):
    manager = object.__new__(OmniRewardLoopManager)
    stop = Mock()
    monkeypatch.setattr(RewardLoopManager, "stop_profile", stop)
    manager._run_on_replicas = Mock(side_effect=RuntimeError("replica stop failed"))

    with pytest.raises(RuntimeError, match="replica stop"):
        manager.stop_profile(run_command=False)

    stop.assert_called_once_with(run_command=False)


def test_serving_profiler_ranks_use_flattened_group_offset(monkeypatch, reward_worker_class):
    from omegaconf import OmegaConf

    from verl_omni.reward_loop import reward_loop as reward_module

    config = OmegaConf.create(
        {
            "reward": {
                "num_workers": 2,
                "profiler": {
                    "tool": "nsys",
                    "enable": True,
                    "ranks": [3],
                    "ray_actor_nsight_options": {"trace": "cuda", "o": "{role}_rank_{rank}"},
                },
            }
        }
    )
    manager = object.__new__(reward_module.OmniRewardLoopManager)
    manager._reward_worker_groups = {"shared": [object(), object()]}
    manager.reward_loop_workers_class = reward_worker_class
    manager.reward_router_address = None
    monkeypatch.setattr(
        reward_module.ray, "nodes", lambda: [{"NodeID": "01" * 28, "Alive": True, "Resources": {"CPU": 1}}]
    )
    monkeypatch.setattr(
        reward_module,
        "get_platform",
        lambda: SimpleNamespace(ray_noset_envvars=lambda: ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]),
    )

    workers = manager._create_node_affinity_workers(config, {}, "reward-worker")

    assert "runtime_env" not in workers[0].options
    assert workers[1].options["runtime_env"]["nsight"]["o"] == "reward_loop_rank_3"
    assert workers[1].options["runtime_env"]["env_vars"] == {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}


@pytest.mark.parametrize(
    "method,kwargs",
    [
        ("start_profile", {"profile_step": 2}),
        (
            "start_profile",
            {
                "profile_step": 2,
                "role": "reward",
                "profile_prefix": "reward",
                "delay_iterations": 1,
                "max_iterations": 4,
                "profiler_kwargs": {"record_shapes": True},
            },
        ),
        ("stop_profile", {}),
    ],
    ids=["start-with-step", "start-with-step-and-engine-options", "stop"],
)
def test_profiler_fanout_includes_legacy_and_named_engine_models(monkeypatch, method, kwargs):
    manager = object.__new__(OmniRewardLoopManager)
    events = []

    class Replica:
        def __init__(self, name):
            self.name = name

        async def start_profile(self, **kwargs):
            events.append((self.name, kwargs))

        async def stop_profile(self):
            events.append((self.name, {}))

    manager.reward_model_manager = SimpleNamespace(rollout_replicas=[Replica("legacy")])
    manager.multi_reward_model_manager = SimpleNamespace(
        models={
            "engine": SimpleNamespace(reward_model_manager=SimpleNamespace(rollout_replicas=[Replica("engine")])),
            "native": SimpleNamespace(),
        }
    )
    monkeypatch.setattr(RewardLoopManager, method, Mock())
    getattr(manager, method)(**kwargs)
    assert events == [("legacy", kwargs), ("engine", kwargs)]


@pytest.mark.parametrize(
    "profiler,profiled_indices",
    [
        (None, []),
        (
            {
                "tool": "nsys",
                "enable": True,
                "ranks": [3],
                "ray_actor_nsight_options": {"trace": "cuda", "o": "{role}_rank_{rank}"},
            },
            [1],
        ),
    ],
    ids=["disabled", "selected-cpu-native-rank"],
)
def test_cpu_native_workers_receive_ranked_nsight_options(monkeypatch, reward_worker_class, profiler, profiled_indices):
    config = OmegaConf.create({"reward": {"profiler": profiler}})
    specs = {"cpu": "model-spec"}
    manager = object.__new__(OmniRewardLoopManager)
    manager._reward_worker_groups = {"shared": [object(), object()]}
    manager.reward_loop_workers_class = reward_worker_class
    manager.multi_reward_model_manager = SimpleNamespace(
        models={"cpu": SimpleNamespace(placement=SimpleNamespace(is_cpu=True, num_replicas=2, cpus_per_worker=4))}
    )
    monkeypatch.setattr(
        omni_module,
        "get_platform",
        lambda: SimpleNamespace(ray_noset_envvars=lambda: ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"]),
    )

    workers = manager._create_native_workers(config, specs, "cpu", "native_cpu")

    assert len(workers) == 2
    for index, worker in enumerate(workers):
        assert worker.options["num_cpus"] == 4
        assert worker.options["name"] == f"native_cpu_{index}"
        assert worker.args == (config, None, specs)
        if index in profiled_indices:
            assert worker.options["runtime_env"] == {
                "nsight": {"trace": "cuda", "o": "reward_loop_rank_3"},
                "env_vars": {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"},
            }
        else:
            assert "runtime_env" not in worker.options
