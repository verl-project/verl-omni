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
import json
import os
import sys
import threading
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from tests.special_e2e import validate_native_fsdp_training_failure as failure_validation
from tests.special_e2e.verify_native_fsdp_training import optimizer_steps, validate_training_log
from verl_omni.reward_loop import native_fsdp
from verl_omni.reward_loop import reward_loop as reward_loop_module
from verl_omni.reward_loop.native_fsdp import (
    FSDPRewardReplica,
    NativeFSDPExecutor,
    NativeFSDPGroupError,
    terminate_ranks,
)
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
@pytest.mark.parametrize(("failure", "cancel"), [(False, False), (True, True)])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["success", "submit", "ready"])
async def test_construction_records_facade_before_readiness_and_rolls_back(monkeypatch, phase):
    cfg = config()
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = MultiRewardModelManager(cfg, SimpleNamespace(world_size=2))
    owner = manager.multi_reward_model_manager.models["pickscore"]
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    spec = manager.multi_reward_model_manager.reward_model_specs["pickscore"]
    ranks, calls = [object(), object()], []
    facade = SimpleNamespace(ready=SimpleNamespace(remote=lambda: "ready"))
    primary = RuntimeError("construction failed")

    def submit(*args):
        if phase == "submit":
            raise primary
        return facade

    def ready(*args, **kwargs):
        assert owner._workers == [facade] and owner.rank_workers == ranks
        if phase == "ready":
            raise primary

    def terminate(workers, timeout, status_method):
        calls.append(workers)
        if phase == "ready" and status_method == "ready":
            raise TimeoutError("facade death unconfirmed")

    monkeypatch.setattr(
        native_fsdp,
        "build_replica",
        lambda *args, **kwargs: SimpleNamespace(worker_group=SimpleNamespace(workers=ranks)),
    )
    monkeypatch.setattr(reward_loop_module.ray, "remote", lambda cls: SimpleNamespace(remote=submit))
    monkeypatch.setattr(reward_loop_module.ray, "get", ready)
    monkeypatch.setattr(native_fsdp, "terminate_ranks", terminate)
    if phase == "success":
        assert manager._create_fsdp_worker(cfg, spec, model) == [facade]
        assert not calls and not owner._closed
    else:
        with pytest.raises(BaseExceptionGroup) as error:
            manager._create_fsdp_worker(cfg, spec, model)
        assert error.value.exceptions[0] is primary and owner._closed
        assert calls == ([[], ranks] if phase == "submit" else [[facade], ranks])
        if phase == "ready":
            assert error.value.exceptions[1] is owner._termination_error
            with pytest.raises(BaseExceptionGroup) as repeated:
                await owner.close()
            assert repeated.value is owner._termination_error
        else:
            await owner.close()


def test_acceptance_rejects_adjacent_metrics_and_wrong_optimizer_counters(tmp_path):
    for value in ("nan", "inf", "", "0"):
        with pytest.raises((ValueError, AssertionError)):
            validate_training_log(f"actor/grad_norm: {value}\nactor/lr: 0.001", 1)
    for prefix in ("Found checkpoint:", "Resuming diffusion from"):
        assert validate_training_log(f"{prefix} /run/global_step_2\nactor/grad_norm: 0.5", 3, 2) == [0.5]
    path = tmp_path / "optimizer.pt"
    for step in (2.0, 2.5):
        torch.save(
            {"state": {0: {"step": torch.tensor(step), "exp_avg": torch.ones(1), "exp_avg_sq": torch.ones(1)}}}, path
        )
        if step == 2:
            assert optimizer_steps(path, 2) == [2.0]
        else:
            with pytest.raises(AssertionError):
                optimizer_steps(path, 2)


def test_acceptance_uses_ray_job_id_string_before_actor_and_checkpoint_checks(tmp_path, monkeypatch):
    overrides, output = tmp_path / "overrides.json", tmp_path / "result.json"
    overrides.write_text("[]")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("0")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify",
            "--overrides",
            str(overrides),
            "--checkpoint-root",
            str(tmp_path),
            "--expected-step",
            "0",
            "--output",
            str(output),
        ],
    )

    def fail(*args, **kwargs):
        assert os.environ["HYDRA_FULL_ERROR"] == "1"
        raise RuntimeError("fit: FSDP wake_up failed: model.safetensors missing")

    roles = [
        "PickScoreFSDPRank",
        "PickScoreFSDPRank",
        "OmniFSDPRewardLoopWorker",
        "TransferQueueController",
        "SimpleStorageUnit",
    ]
    rows = {
        str(i): {"ActorID": str(i), "ActorClassName": role, "Name": "", "Pid": 1, "State": "DEAD", "JobID": "job"}
        for i, role in enumerate(roles)
    }
    rows["other"] = dict(rows["0"], ActorID="other", JobID="other", State="ALIVE")
    monkeypatch.setenv("HYDRA_FULL_ERROR", "0")
    monkeypatch.setattr(failure_validation.runpy, "run_module", fail)
    monkeypatch.setattr(failure_validation.ray._private.state, "actors", lambda: rows)
    monkeypatch.setattr(failure_validation.ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        failure_validation.ray, "get_runtime_context", lambda: SimpleNamespace(get_job_id=lambda: "job")
    )
    monkeypatch.setattr(failure_validation.ray, "shutdown", lambda: None)
    failure_validation.main()
    report = json.loads(output.read_text())
    assert report["passed"] and report["latest_checkpoint"] == 0
    assert len(report["actors_before_ray_shutdown"]) == 5


def test_partial_submission_and_rank_failure_terminate_whole_group(monkeypatch):
    calls = []
    ranks = [object(), object()]

    def partial(*args):
        raise RuntimeError("rank failed")

    group = SimpleNamespace(world_size=2, workers=ranks, wake_up=lambda: [None, None], infer=partial)
    model = FSDPRewardReplica(group, lambda **kwargs: {})
    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(
        "verl_omni.reward_loop.native_fsdp.terminate_ranks", lambda workers, timeout: calls.extend(workers)
    )
    model.wake_up()
    with pytest.raises(NativeFSDPGroupError):
        model.infer(["prompt"], [object()])
    assert calls == ranks
    with pytest.raises(RuntimeError, match="closed"):
        model.wake_up()
    model.close()


@pytest.mark.parametrize(
    "probe_error",
    [
        native_fsdp.ray.exceptions.ActorDiedError(),
        native_fsdp.ray.exceptions.ActorUnavailableError("unavailable", None),
        TimeoutError("rank did not exit"),
    ],
)
def test_only_confirmed_death_succeeds_and_every_rank_is_attempted(monkeypatch, probe_error):
    calls = []
    probes = []
    actors = [SimpleNamespace(status=SimpleNamespace(remote=lambda index=index: index)) for index in range(2)]
    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.kill", lambda actor, **kwargs: calls.append(actor))

    def probe(index, **kwargs):
        probes.append(index)
        raise probe_error if index == 0 else native_fsdp.ray.exceptions.ActorDiedError()

    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.get", probe)
    if isinstance(probe_error, native_fsdp.ray.exceptions.ActorDiedError):
        terminate_ranks(actors, 0.1)
    else:
        with pytest.raises(BaseExceptionGroup, match="job termination") as error:
            terminate_ranks(actors, 0.1)
        assert error.value.exceptions == (probe_error,)
    assert calls == actors and probes == [0, 1]


@pytest.mark.asyncio
async def test_cancelled_native_handle_retains_accepted_work_until_settlement():
    entered, release = threading.Event(), threading.Event()
    finished = threading.Event()

    def infer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {
            "image_embeddings": torch.ones(1, 2),
            "text_embeddings": torch.ones(1, 2),
            "logit_scale": torch.ones(()),
        }

    handle = NativeFSDPExecutor(SimpleNamespace(infer=infer))
    task = asyncio.create_task(handle.infer(["prompt"], [object()]))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert finished.is_set()
