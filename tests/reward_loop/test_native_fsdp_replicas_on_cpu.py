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
"""Multiple logical replicas retain independent dispatch and complete ownership."""

import asyncio
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from verl import DataProto

from tests.reward_loop.test_native_fsdp_integration_on_cpu import config
from verl_omni.reward_loop import native_fsdp, reward_loop
from verl_omni.reward_loop.replica_dispatch import dispatch_reward_groups
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import MultiRewardModelManager
from verl_omni.workers.config.reward import parse_reward_model_config, validate_reward_model_terms


def multi_config():
    cfg = config()
    cfg.reward.models.pickscore.placement.replica_groups = [[3, 1], [0, 2]]
    return cfg


def test_all_groups_claim_parent_bundles_and_register_one_worker_each(monkeypatch):
    cfg = multi_config()
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    assert model.placement.bundle_indices == [3, 1, 0, 2]
    validate_reward_model_terms(cfg)
    manager = object.__new__(OmniRewardLoopManager)
    manager.config = cfg
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=4))
    observed = []

    def create(group_config, spec, parsed):
        observed.append((group_config.reward.num_workers, parsed.placement.replica_groups))
        return ["replica0", "replica1"]

    monkeypatch.setattr(manager, "_create_fsdp_worker", create)
    manager._init_reward_loop_workers()
    assert observed == [(2, [[3, 1], [0, 2]])]
    assert manager._reward_worker_groups["pickscore"] == ["replica0", "replica1"]
    assert manager.multi_reward_model_manager.native_device_assignments == {"pickscore": (3, 1, 0, 2)}


@pytest.mark.parametrize("groups", [[[0, 1], [2]], [[0, 1], [1, 2]], [[0, 1], [2, True]], [[], [0, 1]]])
def test_invalid_second_group_is_rejected(groups):
    cfg = config()
    cfg.reward.models.pickscore.placement.replica_groups = groups
    with pytest.raises(ValueError):
        parse_reward_model_config("pickscore", cfg.reward.models.pickscore)


def test_second_group_bounds_are_checked_before_allocation():
    cfg = multi_config()
    cfg.reward.models.pickscore.placement.replica_groups = [[0, 1], [2, 4]]
    with pytest.raises(ValueError, match="only 4 bundles"):
        MultiRewardModelManager(cfg, SimpleNamespace(world_size=4))


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_size", [None, 2])
async def test_dispatch_routes_each_chunk_through_its_own_facade(batch_size):
    owner = MultiRewardModelManager(multi_config(), SimpleNamespace(world_size=4)).models["pickscore"]
    calls = []

    def facade(index):
        async def score(data):
            calls.append((index, data.batch["sample"].tolist()))
            await asyncio.sleep(0)
            return [{"reward_score": float(value)} for value in data.batch["sample"]]

        async def ready():
            return True

        return SimpleNamespace(compute_score_batch=SimpleNamespace(remote=score), ready=SimpleNamespace(remote=ready))

    workers = [facade(0), facade(1)]
    owner.bind_workers(workers)
    data = DataProto(batch=TensorDict({"sample": torch.arange(7)}, batch_size=7))
    output = await dispatch_reward_groups(
        data,
        {"pickscore": workers},
        {} if batch_size is None else {"pickscore": batch_size},
        rpc_owners={"pickscore": owner},
    )
    assert [item["reward_score"] for item in output["pickscore"]] == list(range(7))
    assert {index for index, _ in calls} == {0, 1}


@pytest.mark.asyncio
async def test_failure_on_second_facade_terminates_all_owned_replica_groups(monkeypatch):
    owner = MultiRewardModelManager(multi_config(), SimpleNamespace(world_size=4)).models["pickscore"]
    calls = []

    async def healthy():
        await asyncio.Future()

    async def fail(*args):
        raise RuntimeError("second replica failed")

    workers = [
        SimpleNamespace(ready=SimpleNamespace(remote=healthy)),
        SimpleNamespace(ready=SimpleNamespace(remote=healthy), compute_score_batch=SimpleNamespace(remote=fail)),
    ]
    owner.bind_workers(workers)
    owner.bind_rank_workers(["g0r0", "g0r1"])
    owner.bind_rank_workers(["g1r0", "g1r1"])
    monkeypatch.setattr(
        native_fsdp, "terminate_ranks", lambda actors, timeout, status_method: calls.append(list(actors))
    )
    with pytest.raises(RuntimeError, match="second replica"):
        await owner.run_rpc("compute_score_batch", object(), replica_index=1)
    assert calls == [workers, ["g0r0", "g0r1", "g1r0", "g1r1"]]
    assert owner._closed
    await owner.close()


def test_second_replica_construction_failure_rolls_back_previous_facade_and_ranks(monkeypatch):
    cfg = multi_config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=4))
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    spec = manager.multi_reward_model_manager.reward_model_specs["pickscore"]
    facade = SimpleNamespace(ready=SimpleNamespace(remote=lambda: True))
    observed = []

    def build(*args, bundle_indices, **kwargs):
        observed.append(bundle_indices)
        if len(observed) == 2:
            raise ValueError("second group allocation failed")
        return SimpleNamespace(worker_group=SimpleNamespace(workers=["g0r0", "g0r1"]))

    killed = []
    monkeypatch.setattr(native_fsdp, "build_replica", build)
    monkeypatch.setattr(reward_loop.ray, "remote", lambda cls: SimpleNamespace(remote=lambda *args: facade))
    monkeypatch.setattr(reward_loop.ray, "get", lambda ref, **kwargs: ref)
    monkeypatch.setattr(
        native_fsdp, "terminate_ranks", lambda actors, timeout, status_method: killed.append(list(actors))
    )
    with pytest.raises(BaseExceptionGroup, match="construction failed") as error:
        manager._create_fsdp_worker(cfg, spec, model)
    assert isinstance(error.value.exceptions[0], ValueError)
    assert observed == [[3, 1], [0, 2]]
    assert killed == [[facade], ["g0r0", "g0r1"]]


@pytest.mark.parametrize("failure", ["submission", "ready_submission", "ready_error", "ready_timeout"])
@pytest.mark.parametrize("cleanup_failure", [None, "facades", "ranks", "both"])
@pytest.mark.asyncio
async def test_second_facade_failure_retains_and_terminates_both_groups(monkeypatch, failure, cleanup_failure):
    cfg = multi_config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=4))
    owner = manager.multi_reward_model_manager.models["pickscore"]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    spec = manager.multi_reward_model_manager.reward_model_specs["pickscore"]
    primary = (
        reward_loop.ray.exceptions.GetTimeoutError("second facade readiness")
        if failure == "ready_timeout"
        else (RuntimeError("second facade construction failed"))
    )
    ranks = [["g0r0", "g0r1"], ["g1r0", "g1r1"]]
    facades, built, terminated = [], [], []

    def build(*args, **kwargs):
        index = len(built)
        built.append(index)
        return SimpleNamespace(worker_group=SimpleNamespace(workers=ranks[index]))

    def submit(*args):
        index = len(facades)
        if index == 1 and failure == "submission":
            assert owner.rank_workers == ranks[0] + ranks[1]
            raise primary

        def ready():
            assert owner._workers == facades
            if index == 1 and failure == "ready_submission":
                raise primary
            return index

        actor = SimpleNamespace(ready=SimpleNamespace(remote=ready))
        facades.append(actor)
        return actor

    def get(ref, **kwargs):
        if ref == 1:
            assert owner._workers == facades and owner.rank_workers == ranks[0] + ranks[1]
            raise primary
        return True

    def terminate(actors, timeout, status_method):
        terminated.append(list(actors))
        role = "facades" if status_method == "ready" else "ranks"
        if cleanup_failure in {role, "both"}:
            raise TimeoutError(f"{role} death unconfirmed")

    monkeypatch.setattr(native_fsdp, "build_replica", build)
    monkeypatch.setattr(reward_loop.ray, "remote", lambda cls: SimpleNamespace(remote=submit))
    monkeypatch.setattr(reward_loop.ray, "get", get)
    monkeypatch.setattr(native_fsdp, "terminate_ranks", terminate)
    with pytest.raises(BaseExceptionGroup) as error:
        manager._create_fsdp_worker(cfg, spec, model)
    assert error.value.exceptions[0] is primary
    assert built == [0, 1] and terminated == [facades, ranks[0] + ranks[1]]
    assert owner._closed
    if cleanup_failure is not None:
        assert error.value.exceptions[1] is owner._termination_error
        with pytest.raises(BaseExceptionGroup) as repeated:
            await owner.close()
        assert repeated.value is owner._termination_error
    else:
        await owner.close()


@pytest.mark.asyncio
async def test_lifecycle_reaches_every_facade_and_resident_wake_is_idempotent():
    cfg = multi_config()
    cfg.reward.models.pickscore.offload = False
    owner = MultiRewardModelManager(cfg, SimpleNamespace(world_size=4)).models["pickscore"]
    calls = []

    def worker(index):
        async def wake(name):
            calls.append((index, "wake", name))

        async def ready():
            return True

        return SimpleNamespace(wake_up_reward_model=SimpleNamespace(remote=wake), ready=SimpleNamespace(remote=ready))

    owner.bind_workers([worker(0), worker(1)])
    await owner.wake_up()
    await owner.wake_up()
    await owner.sleep()
    assert sorted(calls) == [(0, "wake", "pickscore"), (1, "wake", "pickscore")]
