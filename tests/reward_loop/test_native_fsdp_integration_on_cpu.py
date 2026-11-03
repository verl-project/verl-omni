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
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.reward_loop import native_fsdp
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model import MultiRewardModelManager
from verl_omni.trainer.main_diffusion_v1 import finish_training_resources
from verl_omni.workers.config.reward import parse_reward_model_config, validate_reward_model_terms


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
    "placement",
    [
        {"devices": [0], "replica_groups": [[0, 1]]},
        {"replica_groups": [[0, 0]]},
        {"replica_groups": [[0, True]]},
        {"replica_groups": [[0]]},
        {"replica_groups": [[0, 1], [1, 2]]},
        {"resource": "cpu", "num_replicas": 1, "replica_groups": [[0, 1]]},
    ],
)
def test_invalid_placement_is_rejected(placement):
    model = OmegaConf.to_container(config().reward.models.pickscore)
    model["placement"] = placement
    with pytest.raises(ValueError):
        parse_reward_model_config("pickscore", model)


@pytest.mark.parametrize("required", [False, None, "true"])
def test_optional_or_ambiguous_terms_are_rejected_before_resource_allocation(required):
    cfg = config()
    cfg.reward.reward_functions.pickscore.required = required
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


def test_group_bundles_reuse_parent_overlap_and_bounds_validation():
    cfg = config()
    cfg.reward.models.pickscore.placement.replica_groups = [[0, 2]]
    with pytest.raises(ValueError, match="only 2 bundles"):
        MultiRewardModelManager(cfg, SimpleNamespace(world_size=2))
    model = parse_reward_model_config("pickscore", cfg.reward.models.pickscore)
    other = parse_reward_model_config(
        "other",
        {
            "backend": "native",
            "placement": {"devices": [2]},
            "executor": {"model": "other:Model"},
        },
    )
    with pytest.raises(ValueError, match="overlaps"):
        MultiRewardModelManager._validate_native_device_assignments([("pickscore", model), ("other", other)])


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


@pytest.mark.parametrize("failure", [False, True])
def test_trainer_cleanup_closes_fsdp_and_queue_even_on_fit_or_close_failure(failure):
    calls = []

    async def close():
        calls.append("models")
        if failure:
            raise RuntimeError("model close failed")

    def queue():
        calls.append("queue")

    trainer = SimpleNamespace(
        reward_loop_manager=SimpleNamespace(multi_reward_model_manager=SimpleNamespace(close_native_models=close))
    )
    if failure:
        with pytest.raises(BaseExceptionGroup) as error:
            finish_training_resources(trainer, config(), queue, ValueError("fit failed"))
        assert [str(exc) for exc in error.value.exceptions] == ["fit failed", "model close failed"]
    else:
        finish_training_resources(trainer, config(), queue)
    assert calls == ["models", "queue"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exit", "hang", "submission"])
async def test_facade_failure_terminates_owned_ranks_before_return(failure, monkeypatch):
    cfg = config()
    owner = MultiRewardModelManager(cfg, SimpleNamespace(world_size=2)).models["pickscore"]
    owner.timeout = 0.01
    calls = []

    async def ready():
        await asyncio.Future()

    def compute(*args):
        if failure == "submission":
            raise ValueError("submission failed")

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
    with pytest.raises((RuntimeError, TimeoutError, ValueError)):
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
