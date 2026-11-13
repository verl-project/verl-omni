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
"""Opt-in configuration, construction rollback, and trainer cleanup contracts."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.reward_loop import native_fsdp
from verl_omni.reward_loop import reward_loop as reward_loop_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import MultiRewardModelManager
from verl_omni.trainer.main_diffusion_v1 import finish_training_resources
from verl_omni.workers.config.reward import (
    NativeRewardFSDPConfig,
    NativeRewardModelConfig,
    parse_reward_model_config,
    validate_reward_model_terms,
)


def config():
    """The smallest supported synchronous, dedicated two-rank deployment."""
    return OmegaConf.create(
        {
            "trainer": {"use_v1": True, "v1": {"trainer_mode": "sync"}},
            "reward": {
                "num_workers": 1,
                "reward_model": {"enable": False, "enable_resource_pool": True, "nnodes": 1},
                "models": {
                    "pickscore": {
                        "backend": "native",
                        "model_path": "immutable-checkpoint",
                        "placement": {"replica_groups": [[0, 1]]},
                        "executor": {"model": "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel"},
                    }
                },
                "reward_functions": {
                    "pickscore": {
                        "path": "pkg://verl_omni.utils.reward_score.pickscore_reward",
                        "name": "compute_score_pickscore_native",
                        "required": True,
                    }
                },
            },
        }
    )


def test_rank_order_is_explicit_and_flat_devices_remain_complete_replicas():
    cfg = config()
    cfg.reward.models.pickscore.placement.replica_groups = [[3, 1]]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    assert model.placement.bundle_indices == [3, 1] and model.fsdp.timeout == 120
    flat = OmegaConf.to_container(cfg.reward.models.pickscore)
    flat["placement"] = {"devices": [3, 1]}
    ordinary = parse_reward_model_config("pickscore", flat)
    assert ordinary.placement.devices == [3, 1] and ordinary.fsdp is None


@pytest.mark.parametrize(
    ("group", "fields", "valid"),
    [
        (False, {}, True),
        (False, {"fsdp": None}, True),
        (False, {"fsdp": {}}, False),
        (True, {}, True),
        (True, {"fsdp": None}, True),
        (True, {"fsdp": {"timeout": 9}}, True),
        (True, {"fsdp": {"timeout": 0}}, False),
    ],
)
def test_mapping_and_direct_fsdp_defaults_agree(group, fields, valid):
    model = OmegaConf.to_container(config().reward.models.pickscore)
    if not group:
        model["placement"] = {"devices": [0, 1]}
    model.update(fields)

    def construct(direct):
        if not direct:
            return parse_reward_model_config("pickscore", model)
        parsed = parse_reward_model_config("pickscore", {key: value for key, value in model.items() if key != "fsdp"})
        values = dict(parsed)
        values.pop("fsdp")
        if "fsdp" in fields:
            values["fsdp"] = None if fields["fsdp"] is None else NativeRewardFSDPConfig(**fields["fsdp"])
        return NativeRewardModelConfig(**values)

    for direct in (False, True):
        if not valid:
            with pytest.raises(ValueError):
                construct(direct)
        else:
            parsed = construct(direct)
            if group:
                assert parsed.fsdp.timeout == (fields.get("fsdp") or {}).get("timeout", 120)
                assert parsed.fsdp.collective_timeout == 30
            else:
                assert parsed.fsdp is None


@pytest.mark.parametrize(
    "placement",
    [
        {"devices": [0], "replica_groups": [[0, 1]]},
        {"replica_groups": [[0, 0]]},
        {"replica_groups": [[0]]},
        {"replica_groups": [[0, 1], [2, 3]]},
    ],
)
def test_invalid_placement_is_rejected(placement):
    model = OmegaConf.to_container(config().reward.models.pickscore)
    model["placement"] = placement
    with pytest.raises(ValueError):
        parse_reward_model_config("pickscore", model)


def test_optional_terms_are_rejected_before_resource_allocation():
    cfg = config()
    cfg.reward.reward_functions.pickscore.required = False
    with pytest.raises(ValueError, match="required=true"):
        validate_reward_model_terms(cfg)


@pytest.mark.parametrize("mode", ["shared", "streaming", "async", "v0", "multinode"])
def test_unsupported_modes_fail_before_allocation(mode):
    cfg = config()
    if mode == "shared":
        cfg.reward.reward_model.enable_resource_pool = False
    elif mode == "streaming":
        cfg.reward.streaming = {"enabled": True}
    elif mode == "async":
        cfg.trainer.v1.trainer_mode = "separate_async"
    elif mode == "v0":
        cfg.trainer.use_v1 = False
    else:
        cfg.reward.reward_model.nnodes = 2
    with pytest.raises(ValueError):
        validate_reward_model_terms(cfg)


def test_dispatcher_registers_one_logical_worker_for_two_ranks(monkeypatch):
    cfg = config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.config = cfg
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=2))
    observed = []

    def create(group_config, spec, model):
        observed.append(group_config.reward.num_workers)
        return ["complete_replica"]

    monkeypatch.setattr(manager, "_create_fsdp_worker", create)
    manager._init_reward_loop_workers()
    assert observed == [1]
    assert manager._reward_worker_groups["pickscore"] == ["complete_replica"]


def test_partial_worker_group_construction_rolls_back_accepted_ranks(monkeypatch):
    calls = []

    def init(group, *args, **kwargs):
        group._workers = ["rank0"]
        raise ValueError("rank1 submission failed")

    monkeypatch.setattr(native_fsdp.RayWorkerGroup, "__init__", init)
    monkeypatch.setattr(native_fsdp, "terminate_ranks", lambda workers, timeout: calls.extend(workers))
    with pytest.raises(ValueError, match="rank1"):
        native_fsdp._RankWorkerGroup(cleanup_timeout=1)
    assert calls == ["rank0"]


def test_trainer_cleanup_closes_fsdp_and_queue_even_on_fit_or_close_failure():
    calls = []

    async def close():
        calls.append("models")
        raise RuntimeError("model close failed")

    def queue():
        calls.append("queue")

    trainer = SimpleNamespace(
        reward_loop_manager=SimpleNamespace(multi_reward_model_manager=SimpleNamespace(close_native_models=close))
    )
    with pytest.raises(BaseExceptionGroup) as error:
        finish_training_resources(trainer, config(), queue, ValueError("fit failed"))
    assert [str(exc) for exc in error.value.exceptions] == ["fit failed", "model close failed"]
    assert calls == ["models", "queue"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exit", "hang"])
async def test_facade_failure_terminates_owned_ranks_before_return(failure, monkeypatch):
    cfg = config()
    owner = MultiRewardModelManager(cfg, SimpleNamespace(world_size=2)).models["pickscore"]
    owner.timeout = 0.01
    calls = []

    async def ready():
        await asyncio.Future()

    def compute(*args):
        async def result():
            if failure == "exit":
                raise RuntimeError("facade died")
            await asyncio.Future()

        return result()

    facade = SimpleNamespace(compute_score_batch=SimpleNamespace(remote=compute), ready=SimpleNamespace(remote=ready))
    owner.bind_workers([facade])
    ranks = [object(), object()]
    owner.bind_rank_workers(ranks)
    monkeypatch.setattr(native_fsdp, "terminate_ranks", lambda workers, timeout, status_method: calls.append(workers))
    with pytest.raises((RuntimeError, TimeoutError)):
        await owner.run_rpc("compute_score_batch", object())
    assert calls == [[facade], ranks] and owner._closed
    await owner.close()


@pytest.mark.asyncio
async def test_owner_cleanup_attempts_ranks_after_facade_termination_error(monkeypatch):
    owner = MultiRewardModelManager(config(), SimpleNamespace(world_size=2)).models["pickscore"]
    calls = []

    def broken_submission(*args):
        raise ValueError("primary")

    facade = SimpleNamespace(close_reward_model=SimpleNamespace(remote=broken_submission))
    owner.bind_workers([facade])
    ranks = [object(), object()]
    owner.bind_rank_workers(ranks)

    def terminate(workers, timeout, status_method):
        calls.append(workers)
        if workers == [facade]:
            raise RuntimeError("unconfirmed facade termination")

    monkeypatch.setattr(native_fsdp, "terminate_ranks", terminate)
    with pytest.raises(BaseExceptionGroup) as error:
        await owner.close()
    assert str(error.value.exceptions[0]) == "primary" and calls == [[facade], ranks]
    with pytest.raises(BaseExceptionGroup, match="owner cleanup failed"):
        await owner.close()


@pytest.fixture
def construction(monkeypatch):
    cfg = config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=2))
    owner = manager.multi_reward_model_manager.models["pickscore"]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    state = SimpleNamespace(
        cfg=cfg,
        manager=manager,
        owner=owner,
        model=model,
        spec=manager.multi_reward_model_manager.reward_model_specs["pickscore"],
        failure=None,
        cleanup_failure=None,
        primary=RuntimeError("construction failed"),
        events=[],
        bound_at_ready=[],
        dead=set(),
    )

    def ready():
        if not owner._closed:
            state.bound_at_ready.append(owner._workers == [state.facade])
            if state.failure == "ready_submission":
                raise state.primary
        return "facade"

    state.facade = SimpleNamespace(ready=SimpleNamespace(remote=ready))
    state.ranks = [
        SimpleNamespace(status=SimpleNamespace(remote=lambda rank=rank: rank)) for rank in ("rank0", "rank1")
    ]

    def submit(*args):
        if state.failure == "facade_submission":
            raise state.primary
        return state.facade

    def get(ref, **kwargs):
        if not owner._closed:
            if state.failure in {"ready_error", "ready_timeout"}:
                raise state.primary
            return True
        state.events.append(("confirm", ref))
        if ref in {"facade", "rank1"} and state.cleanup_failure:
            raise TimeoutError(f"{ref} death unconfirmed")
        assert ref in state.dead
        raise native_fsdp.ray.exceptions.ActorDiedError()

    def kill(actor, **kwargs):
        name = "facade" if actor is state.facade else "rank0" if actor is state.ranks[0] else "rank1"
        state.events.append(("kill", name))
        state.dead.add(name)

    monkeypatch.setattr(
        native_fsdp,
        "build_replica",
        lambda *args, **kwargs: SimpleNamespace(worker_group=SimpleNamespace(workers=state.ranks)),
    )
    monkeypatch.setattr(reward_loop_module.ray, "remote", lambda cls: SimpleNamespace(remote=submit))
    monkeypatch.setattr(reward_loop_module.ray, "get", get)
    monkeypatch.setattr(reward_loop_module.ray, "kill", kill)
    return state


def test_facade_is_owned_before_successful_readiness(construction):
    state = construction
    assert state.manager._create_fsdp_worker(state.cfg, state.spec, state.model) == [state.facade]
    assert state.bound_at_ready == [True]
    assert state.owner.rank_workers == state.ranks and state.owner._workers == [state.facade]
    assert not state.events and not state.owner._closed


@pytest.mark.parametrize(
    ("failure", "cleanup_failure"),
    [("facade_submission", False), ("ready_submission", False), ("ready_error", False), ("ready_timeout", True)],
)
@pytest.mark.asyncio
async def test_construction_failure_confirms_every_owned_actor(construction, failure, cleanup_failure):
    state = construction
    state.failure, state.cleanup_failure = failure, cleanup_failure
    if failure == "ready_timeout":
        state.primary = native_fsdp.ray.exceptions.GetTimeoutError("readiness deadline")
    with pytest.raises(BaseExceptionGroup) as error:
        state.manager._create_fsdp_worker(state.cfg, state.spec, state.model)
    assert error.value.exceptions[0] is state.primary
    expected = ["rank0", "rank1"] if failure == "facade_submission" else ["facade", "rank0", "rank1"]
    assert [name for event, name in state.events if event == "kill"] == expected
    assert [name for event, name in state.events if event == "confirm"] == expected
    assert state.owner._closed
    if failure != "facade_submission":
        assert state.bound_at_ready == [True]
        assert state.owner._workers == [state.facade]
    if cleanup_failure:
        assert error.value.exceptions[1] is state.owner._termination_error
        assert len(state.owner._termination_error.exceptions) == 2
        with pytest.raises(BaseExceptionGroup) as repeated:
            await state.owner.close()
        assert repeated.value is state.owner._termination_error
    else:
        assert len(error.value.exceptions) == 1 and state.owner._termination_error is None
        await state.owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(("failure", "cancel"), [(False, False), (True, False), (False, True), (True, True)])
async def test_concurrent_close_waits_for_scoring_cleanup(monkeypatch, failure, cancel):
    owner = MultiRewardModelManager(config(), SimpleNamespace(world_size=2)).models["pickscore"]
    entered, release = threading.Event(), threading.Event()
    calls = []

    async def compute(*args):
        raise RuntimeError("scoring failed")

    owner.bind_workers([SimpleNamespace(compute_score_batch=SimpleNamespace(remote=compute))])
    owner.bind_rank_workers([object(), object()])

    def terminate(workers, timeout, status_method):
        calls.append(status_method)
        if status_method == "ready":
            entered.set()
            assert release.wait(5)
            if failure:
                raise TimeoutError("facade death unconfirmed")

    monkeypatch.setattr(native_fsdp, "terminate_ranks", terminate)
    scoring = asyncio.create_task(owner.run_rpc("compute_score_batch", object()))
    closing = []
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        closing = [asyncio.create_task(owner.close()) for _ in range(2)]
        await asyncio.sleep(0.02)
        assert all(not task.done() for task in closing)
        if cancel:
            closing[0].cancel()
            closing[0].cancel()
            await asyncio.sleep(0.02)
            assert not closing[0].done()
    finally:
        release.set()
        results = await asyncio.gather(scoring, *closing, return_exceptions=True)
    assert calls == ["ready", "status"]
    assert isinstance(results[0], BaseExceptionGroup if failure else RuntimeError)
    if cancel:
        assert isinstance(results[1], asyncio.CancelledError)
    for result in results[2:] if cancel else results[1:]:
        assert result is owner._termination_error
    if failure:
        with pytest.raises(BaseExceptionGroup) as repeated:
            await owner.close()
        assert repeated.value is owner._termination_error
    else:
        await owner.close()
