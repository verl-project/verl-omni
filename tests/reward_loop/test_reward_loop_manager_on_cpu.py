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
"""CPU contracts for accelerator-bound reward worker placement."""

from types import SimpleNamespace

import pytest

from verl_omni.reward_loop import accelerator_reward_workers as worker_module


class _FakeActorClass:
    def options(self, **options):
        class _ConfiguredActor:
            def remote(self, *args):
                return SimpleNamespace(options=options, args=args)

        return _ConfiguredActor()


class _FakeResourcePool:
    store = [2, 1]
    max_colocate_count = 3

    def __init__(self):
        self.device_name = None

    def get_placement_groups(self, device_name):
        self.device_name = device_name
        return ["pg0", "pg1"]


class _FakeSubResourcePool(_FakeResourcePool):
    store = [4, 4]
    start_bundle_index = 2
    world_size = 3


def _config(num_workers):
    return SimpleNamespace(reward=SimpleNamespace(num_workers=num_workers))


def _build_workers(num_workers, resource_pool):
    return worker_module.build_accelerator_reward_workers(
        config=_config(num_workers),
        reward_loop_workers_class=_FakeActorClass(),
        accelerator_resource_pool=resource_pool,
        reward_router_address="router",
        reward_executor_specs={"native": "spec"},
    )


def _build_selected_workers(bundle_indices, resource_pool, prefix="native_reward_loop_worker_pickscore"):
    return worker_module.build_accelerator_reward_workers(
        config=_config(99),
        reward_loop_workers_class=_FakeActorClass(),
        accelerator_resource_pool=resource_pool,
        reward_router_address="router",
        reward_executor_specs={"pickscore": "spec"},
        bundle_indices=bundle_indices,
        worker_name_prefix=prefix,
    )


def test_accelerator_reward_workers_use_distinct_resource_pool_bundles(monkeypatch):
    resource_pool = _FakeResourcePool()
    monkeypatch.setattr(worker_module, "get_device_name", lambda: "npu")
    monkeypatch.setattr(
        worker_module,
        "get_platform",
        lambda: SimpleNamespace(ray_resource_options=lambda count: {"resources": {"NPU": count}}),
    )
    monkeypatch.setattr(
        worker_module,
        "PlacementGroupSchedulingStrategy",
        lambda placement_group, placement_group_bundle_index: (placement_group, placement_group_bundle_index),
    )

    workers = _build_workers(3, resource_pool)

    assert resource_pool.device_name == "npu"
    assert len(workers) == 3
    assert [worker.options["scheduling_strategy"] for worker in workers] == [
        ("pg0", 0),
        ("pg1", 0),
        ("pg0", 1),
    ]
    assert all(worker.options["resources"] == {"NPU": 1 / 3} for worker in workers)
    assert all(worker.args[1:] == ("router", {"native": "spec"}) for worker in workers)


def test_accelerator_reward_workers_require_enough_bundles():
    with pytest.raises(ValueError, match="exceeds accelerator pool size"):
        _build_workers(4, _FakeResourcePool())


def test_accelerator_reward_workers_require_colocation_capacity():
    resource_pool = _FakeResourcePool()
    resource_pool.max_colocate_count = 1

    with pytest.raises(ValueError, match="max_colocate_count >= 2"):
        _build_workers(1, resource_pool)


def test_accelerator_reward_workers_respect_subpool_bundle_offset(monkeypatch):
    resource_pool = _FakeSubResourcePool()
    monkeypatch.setattr(worker_module, "get_device_name", lambda: "npu")
    monkeypatch.setattr(
        worker_module,
        "get_platform",
        lambda: SimpleNamespace(ray_resource_options=lambda count: {"resources": {"NPU": count}}),
    )
    monkeypatch.setattr(
        worker_module,
        "PlacementGroupSchedulingStrategy",
        lambda placement_group, placement_group_bundle_index: (placement_group, placement_group_bundle_index),
    )

    workers = _build_workers(3, resource_pool)

    assert [worker.options["scheduling_strategy"] for worker in workers] == [
        ("pg0", 2),
        ("pg0", 3),
        ("pg1", 0),
    ]


def test_accelerator_reward_workers_allow_explicit_native_bundle_indices(monkeypatch):
    resource_pool = _FakeSubResourcePool()
    monkeypatch.setattr(worker_module, "get_device_name", lambda: "npu")
    monkeypatch.setattr(
        worker_module,
        "get_platform",
        lambda: SimpleNamespace(ray_resource_options=lambda count: {"resources": {"NPU": count}}),
    )
    monkeypatch.setattr(
        worker_module,
        "PlacementGroupSchedulingStrategy",
        lambda placement_group, placement_group_bundle_index: (placement_group, placement_group_bundle_index),
    )

    workers = _build_selected_workers([0, 2], resource_pool)

    assert [worker.options["scheduling_strategy"] for worker in workers] == [("pg0", 2), ("pg1", 0)]
    assert [worker.options["name"] for worker in workers] == [
        "native_reward_loop_worker_pickscore_0",
        "native_reward_loop_worker_pickscore_1",
    ]
    assert all(worker.args[1:] == ("router", {"pickscore": "spec"}) for worker in workers)


def test_accelerator_reward_workers_reject_invalid_explicit_bundle_indices():
    with pytest.raises(ValueError, match="bundle index 3 exceeds"):
        _build_selected_workers([3], _FakeResourcePool())
