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

from types import SimpleNamespace

import pytest

from verl_omni.reward_loop import local_accelerator_reward_loop as reward_loop_module


class _FakeActorClass:
    def __init__(self):
        self.calls = []

    def options(self, **options):
        calls = self.calls

        class _ConfiguredActor:
            def remote(self, *args):
                handle = SimpleNamespace(options=options, args=args)
                calls.append(handle)
                return handle

        return _ConfiguredActor()


class _FakeResourcePool:
    store = [2, 1]
    max_colocate_count = 3

    def __init__(self):
        self.device_name = None

    def get_placement_groups(self, device_name):
        self.device_name = device_name
        return ["pg0", "pg1"]


def _manager(num_workers, resource_pool):
    manager = object.__new__(reward_loop_module.LocalAcceleratorRewardLoopManager)
    manager.config = SimpleNamespace(
        reward=SimpleNamespace(
            custom_reward_function={"use_accelerator": True},
            num_workers=num_workers,
        )
    )
    manager.reward_router_address = "router"
    manager.reward_loop_workers_class = _FakeActorClass()
    manager.accelerator_resource_pool = resource_pool
    return manager


def test_accelerator_reward_workers_use_distinct_resource_pool_bundles(monkeypatch):
    resource_pool = _FakeResourcePool()
    manager = _manager(num_workers=3, resource_pool=resource_pool)
    monkeypatch.setattr(reward_loop_module, "get_device_name", lambda: "npu")
    monkeypatch.setattr(
        reward_loop_module,
        "get_platform",
        lambda: SimpleNamespace(ray_resource_options=lambda count: {"resources": {"NPU": count}}),
    )
    monkeypatch.setattr(
        reward_loop_module,
        "PlacementGroupSchedulingStrategy",
        lambda placement_group, placement_group_bundle_index: (placement_group, placement_group_bundle_index),
    )

    manager._init_reward_loop_workers()

    assert resource_pool.device_name == "npu"
    assert len(manager.reward_loop_workers) == 3
    assert [worker.options["scheduling_strategy"] for worker in manager.reward_loop_workers] == [
        ("pg0", 0),
        ("pg1", 0),
        ("pg0", 1),
    ]
    assert all(worker.options["resources"] == {"NPU": 1 / 3} for worker in manager.reward_loop_workers)


def test_accelerator_reward_workers_require_enough_bundles():
    manager = _manager(num_workers=4, resource_pool=_FakeResourcePool())

    with pytest.raises(ValueError, match="exceeds accelerator pool size"):
        manager._init_reward_loop_workers()


def test_accelerator_reward_workers_require_colocation_capacity():
    resource_pool = _FakeResourcePool()
    resource_pool.max_colocate_count = 1
    manager = _manager(num_workers=1, resource_pool=resource_pool)

    with pytest.raises(ValueError, match="max_colocate_count >= 2"):
        manager._init_reward_loop_workers()


@pytest.mark.parametrize("use_accelerator", [False, True])
def test_v1_factory_selects_accelerator_manager_only_when_enabled(monkeypatch, use_accelerator):
    config = SimpleNamespace(
        reward=SimpleNamespace(
            custom_reward_function={"use_accelerator": use_accelerator},
        )
    )
    monkeypatch.setattr(reward_loop_module, "OmniRewardLoopManager", lambda **kwargs: ("default", kwargs))
    monkeypatch.setattr(
        reward_loop_module,
        "LocalAcceleratorRewardLoopManager",
        lambda config, accelerator_resource_pool: ("accelerator", config, accelerator_resource_pool),
    )

    manager = reward_loop_module.create_v1_reward_loop_manager(config, "rm_pool", "accelerator_pool")

    if use_accelerator:
        assert manager == ("accelerator", config, "accelerator_pool")
    else:
        assert manager == ("default", {"config": config, "rm_resource_pool": "rm_pool"})
