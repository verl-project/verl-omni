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
"""V1 sync validation hands the colocated accelerator back to rollout."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from verl import DataProto
from verl.trainer.ppo.utils import Role

from tests.trainer.diffusion.test_v1_colocate_reward_on_cpu import compose_cfg
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.trainer.diffusion.v1 import trainer_base
from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync
from verl_omni.workers.config.reward import reward_role_required


@pytest.fixture
def validation(monkeypatch):
    def setup(*, options=(), shared=True, named=True):
        config = compose_cfg(
            (
                [
                    "+reward.models.quality.backend=native",
                    "+reward.models.quality.placement.devices=[0]",
                    "+reward.models.quality.executor.model=example:Model",
                ]
                if named
                else ["reward.reward_model.enable=true"]
            )
            + list(options)
        )
        trainer = PolicyGradientDiffusionTrainerV1Sync(config)
        calls = []
        actor_pool = object()
        reward_pool = actor_pool if shared else object()
        trainer.mapping = (
            {Role.RewardModel: "global_pool" if shared else "reward_pool"} if reward_role_required(config) else {}
        )
        trainer.resource_pool_manager = SimpleNamespace(get_resource_pool=lambda role: reward_pool)

        async def wake():
            calls.append("reward wake")

        async def sleep():
            calls.append("reward sleep")

        async def score(batch, *, preserve_scoring_error_on_cancel=False):
            calls.append("score")
            return DataProto.from_dict(tensors={"rm_scores": batch.batch["prompts"].float()})

        reward_manager = object.__new__(OmniRewardLoopManager)
        reward_manager.config = config
        reward_manager._score_lock = asyncio.Lock()
        reward_manager.multi_reward_model_manager = SimpleNamespace(
            models={"quality": object()}, wake_up=wake, sleep=sleep
        )
        reward_manager._compute_named_model_scores = score
        reward_manager.reward_loop_workers = [object()]
        monkeypatch.setattr("verl_omni.reward_loop.OmniRewardLoopManager", lambda **kwargs: reward_manager)
        checkpoint = SimpleNamespace(
            sleep_replicas=lambda: calls.append("rollout sleep"),
            update_weights=lambda step: calls.append(("restore", step)),
            wait_for_weight_update=lambda: calls.append("commit"),
        )
        monkeypatch.setattr(trainer_base, "CheckpointEngineManager", lambda **kwargs: checkpoint)
        monkeypatch.setattr(
            trainer_base.LLMServerManager,
            "create",
            lambda **kwargs: SimpleNamespace(get_replicas=lambda: [object()]),
        )
        trainer.actor_rollout_wg = object()
        trainer._init_online_rollout_stack(actor_pool)
        trainer.global_steps = 4
        trainer._gen_batch_seq = 0
        trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kwargs: str(ids.item()))
        trainer.val_dataloader = [{"raw_prompt": ["second", "first"]}]
        trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: calls.append("generate"))
        meta = SimpleNamespace(keys=["0", "1"], partition_id="val")
        trainer.replay_buffer = SimpleNamespace(sample=lambda **kwargs: (meta, {}))
        trainer._maybe_log_val_generations = Mock()
        trainer._val_metrics_update = Mock(return_value={})
        monkeypatch.setattr(trainer_base.tq, "kv_batch_put", Mock())
        monkeypatch.setattr(trainer_base.tq, "kv_clear", Mock())
        monkeypatch.setattr(trainer_base, "canonicalize_diffusion_tq_meta", lambda meta: meta)
        monkeypatch.setattr(
            trainer_base,
            "diffusion_tq_batch_to_dataproto",
            lambda *args, **kwargs: DataProto.from_dict(
                tensors={"prompts": torch.tensor([[2], [1]]), "responses": torch.ones(2, 3, 2, 2, dtype=torch.uint8)}
            ),
        )
        return trainer, reward_manager, calls

    return setup


def test_validation_restores_after_scoring_cancellation(validation):
    trainer, reward_manager, calls = validation()
    cancellation = asyncio.CancelledError("scoring cancelled")

    async def cancel(batch, *, preserve_scoring_error_on_cancel=False):
        raise cancellation

    reward_manager._compute_named_model_scores = cancel
    with pytest.raises(asyncio.CancelledError) as caught:
        trainer._validate()
    assert caught.value is cancellation
    assert calls == ["generate", "rollout sleep", "reward wake", "reward sleep", ("restore", 4), "commit"]


def test_validation_commits_weights_before_merging_each_batch(validation, monkeypatch):
    trainer, _, calls = validation()
    original_union = DataProto.union

    def record_union(data, reward):
        calls.append("union")
        return original_union(data, reward)

    monkeypatch.setattr(DataProto, "union", record_union)
    trainer.val_dataloader *= 2
    trainer._validate()
    batch_calls = [
        "generate",
        "rollout sleep",
        "reward wake",
        "score",
        "reward sleep",
        ("restore", 4),
        "commit",
        "union",
    ]
    assert calls == batch_calls * 2
    assert trainer._val_metrics_update.call_args.args[2]["reward"] == [2.0, 1.0, 2.0, 1.0]


@pytest.mark.parametrize(
    ("shared", "named", "options"),
    [
        (
            True,
            True,
            (
                "reward.models.quality.placement.devices=[]",
                "+reward.models.quality.placement.resource=cpu",
                "+reward.models.quality.placement.num_replicas=1",
                "+reward.models.quality.offload=false",
            ),
        ),
        (False, True, ("reward.reward_model.enable_resource_pool=true", "+reward.models.quality.offload=false")),
        (True, False, ()),
    ],
    ids=["cpu", "dedicated", "legacy"],
)
def test_other_topologies_keep_existing_validation_order(validation, monkeypatch, shared, named, options):
    trainer, _, calls = validation(shared=shared, named=named, options=options)

    # A positional-only stub rejects any attempt to invoke release-aware scoring.
    def score(data, /):
        calls.append("score")
        return DataProto.from_dict(tensors={"rm_scores": data.batch["prompts"].float()})

    monkeypatch.setattr(trainer, "_compute_reward_colocate", score)
    trainer._validate()
    assert calls == ["generate", "rollout sleep", "score", ("restore", 4)]


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("+reward.models.quality.offload=false", "sharing the rollout pool requires offload=true"),
        ("actor_rollout_ref.rollout.free_cache_engine=false", "requires rollout.free_cache_engine=true"),
    ],
)
def test_shared_validation_rejects_disabled_release_at_startup(validation, option, message):
    with pytest.raises(ValueError, match=message):
        validation(options=(option,))


def test_validation_stops_when_rollout_sleep_fails(validation):
    trainer, _, calls = validation()
    failure = RuntimeError("rollout sleep failed")

    def fail():
        calls.append("rollout sleep failed")
        raise failure

    trainer.checkpoint_manager.sleep_replicas = fail
    with pytest.raises(RuntimeError) as caught:
        trainer._validate()
    assert caught.value is failure
    assert calls == ["generate", "rollout sleep failed"]


def test_validation_does_not_restore_after_unconfirmed_wake(validation):
    trainer, manager, calls = validation()
    failure = RuntimeError("wake failed")

    async def fail():
        raise failure

    manager.multi_reward_model_manager.wake_up = fail
    with pytest.raises(RuntimeError) as caught:
        trainer._validate()
    assert caught.value is failure
    assert "Validation reward release unconfirmed" in failure.__notes__
    assert calls == ["generate", "rollout sleep", "reward sleep"]


def test_validation_does_not_restore_if_scoring_returns_without_release(validation, monkeypatch):
    trainer, manager, calls = validation()
    rewards = DataProto.from_dict(tensors={"rm_scores": torch.ones(2, 1)})
    monkeypatch.setattr(manager, "compute_rm_score", Mock(return_value=rewards))

    with pytest.raises(RuntimeError, match="^Validation reward release unconfirmed$"):
        trainer._validate()

    assert calls == ["generate", "rollout sleep"]


def test_validation_preserves_scoring_and_weight_commit_failures(validation):
    trainer, manager, calls = validation()
    scoring = ValueError("scoring failed")
    restore = RuntimeError("weight sync failed")

    async def fail(batch, *, preserve_scoring_error_on_cancel=False):
        calls.append("score")
        raise scoring

    def fail_restore():
        calls.append("commit failed")
        raise restore

    manager._compute_named_model_scores = fail
    trainer.checkpoint_manager.wait_for_weight_update = fail_restore
    with pytest.raises(BaseExceptionGroup) as caught:
        trainer._validate()
    assert caught.value.exceptions == (scoring, restore)
    assert "Validation rollout weight restore failed" in restore.__notes__
    assert calls == [
        "generate",
        "rollout sleep",
        "reward wake",
        "score",
        "reward sleep",
        ("restore", 4),
        "commit failed",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_manager_cancellation_waits_for_cleanup_and_retains_errors(validation, cleanup_fails):
    _, manager, calls = validation()
    entered = asyncio.Event()
    release = asyncio.Event()
    cleanup_failure = RuntimeError("cleanup failed")
    score_failure = ValueError("scoring failed")

    async def fail_score(batch, *, preserve_scoring_error_on_cancel=False):
        calls.append("score")
        raise score_failure

    async def delayed_sleep():
        entered.set()
        await release.wait()
        if cleanup_fails:
            raise cleanup_failure
        calls.append("reward sleep")

    manager._compute_named_model_scores = fail_score
    manager.multi_reward_model_manager.sleep = delayed_sleep
    data = DataProto.from_dict(tensors={"prompts": torch.arange(2).view(2, 1)})
    scoring = asyncio.create_task(manager.async_compute_rm_score(data, on_release=lambda: calls.append("released")))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        for message in ("first cancellation", "second cancellation"):
            scoring.cancel(message)
            checkpoint = asyncio.Event()
            asyncio.get_running_loop().call_soon(checkpoint.set)
            await checkpoint.wait()
            assert not scoring.done()
            assert calls == ["reward wake", "score"]
    finally:
        release.set()
        await asyncio.wait({scoring}, timeout=5)
    with pytest.raises(BaseExceptionGroup) as caught:
        scoring.result()
    error, cancellation, *cleanup_errors = caught.value.exceptions
    assert error is score_failure
    assert isinstance(cancellation, asyncio.CancelledError)
    assert cancellation.args == ("first cancellation",)
    assert cleanup_errors == ([cleanup_failure] if cleanup_fails else [])
    assert calls == (
        ["reward wake", "score"] if cleanup_fails else ["reward wake", "score", "reward sleep", "released"]
    )
