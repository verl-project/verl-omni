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
"""CPU test for the colocated reward-model step of the v1 diffusion trainer."""

import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto
from verl.utils import tensordict_utils as tu

import verl_omni

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class _Stop(Exception):
    pass


def test_colocate_reward_keeps_trajectory_fields(monkeypatch):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    assert trainer.use_rm

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    checkpoint_calls = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: checkpoint_calls.append("sleep"),
        update_weights=lambda step: checkpoint_calls.append(("update_weights", step)),
    )
    trainer.global_steps = 1
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda d: reward)
    captured = {}

    def stop_at_balance(d, metrics):
        captured["data"] = d
        raise _Stop

    monkeypatch.setattr(trainer, "_balance_batch", stop_at_balance)

    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())

    assert "all_timesteps" in captured["data"].batch
    assert "rm_scores" in captured["data"].batch
    assert checkpoint_calls == []


def test_colocate_reward_keeps_rollout_asleep_through_actor_update(monkeypatch):
    """In sync mode the colocated-RM block must not wake the rollout engine.

    ``update_weights`` resumes the rollout replicas' GPU weights (~55GB for
    Qwen-Image at rollout TP=1). If it runs between the reward phase and the
    actor update, every training phase (old log-prob, ref, advantage, actor
    update) executes next to the resident rollout weights and OOMs. In sync
    mode, waking and weight-syncing belongs to ``on_step_end``, after the
    actor update.
    """
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    assert trainer.use_rm

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    calls: list[str] = []
    trainer.checkpoint_manager = SimpleNamespace(
        sleep_replicas=lambda: calls.append("sleep"),
        update_weights=lambda step: calls.append("update_weights"),
    )
    trainer.global_steps = 1
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda d: (calls.append("reward"), reward)[1])
    monkeypatch.setattr(trainer, "_balance_batch", lambda d, metrics: d)
    old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": torch.zeros(2, 4)}))
    monkeypatch.setattr(trainer, "_compute_old_log_prob", lambda d: (calls.append("old_log_prob"), old_log_prob)[1])
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )
    monkeypatch.setattr(trainer, "_compute_advantage", lambda d: (calls.append("advantage"), d)[1])

    def stop_at_update_actor(data):
        calls.append("update_actor")
        raise _Stop

    monkeypatch.setattr(trainer, "_update_actor", stop_at_update_actor)

    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())

    assert calls == ["reward", "old_log_prob", "advantage", "update_actor"]


SEPARATE_ASYNC = [
    "reward.reward_model.enable=true",
    "trainer.use_v1=true",
    "trainer.v1.trainer_mode=separate_async",
    "trainer.v1.separate_async.parameter_sync_step=1",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.n_gpus_per_node=1",
    "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    "data.train_batch_size=8",
    "actor_rollout_ref.actor.ppo_mini_batch_size=8",
]


class _ColocatedReplicas:
    """Records the colocated checkpoint-manager calls and whether the replicas are asleep."""

    def __init__(self, calls: list):
        self.calls = calls
        self.asleep = False

    def sleep_replicas(self):
        self.calls.append("sleep_while_asleep" if self.asleep else "sleep")
        self.asleep = True

    def update_weights(self, step):
        self.calls.append("update_weights")
        self.asleep = False

    def abort_replicas(self, reject_request=False):
        pass

    def resume_generation_replicas(self):
        pass


def make_separate_async_trainer(monkeypatch, calls: list, overrides=()):
    from verl_omni.trainer.diffusion.v1 import trainer_separate_async
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import PolicyGradientDiffusionTrainerV1SeparateAsync

    trainer = PolicyGradientDiffusionTrainerV1SeparateAsync(compose_cfg(SEPARATE_ASYNC + list(overrides)))
    assert trainer.use_rm
    replicas = _ColocatedReplicas(calls)
    trainer.checkpoint_manager = replicas
    trainer.standalone_checkpoint_manager = SimpleNamespace(
        wait_for_weight_update=lambda: None, update_weights=lambda step: {}
    )
    trainer.add_replicas_to_balancer = lambda: None
    trainer.remove_replicas_from_balancer = lambda: None
    trainer.sync_compatible = False  # set in _setup
    trainer.timing_raw = {}
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    trainer.global_steps = 1

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": torch.zeros(2, 4)}))
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )
    monkeypatch.setattr(trainer_separate_async, "marked_timer", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda d: (calls.append("reward"), reward)[1])
    monkeypatch.setattr(trainer, "_balance_batch", lambda d, metrics: d)
    monkeypatch.setattr(trainer, "_compute_old_log_prob", lambda d: old_log_prob)
    monkeypatch.setattr(trainer, "_compute_advantage", lambda d: d)

    def stop_at_update_actor(data):
        calls.append("update_actor_asleep" if replicas.asleep else "update_actor_awake")
        raise _Stop

    monkeypatch.setattr(trainer, "_update_actor", stop_at_update_actor)
    return trainer


def train_one_batch(trainer):
    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())


def test_separate_async_colocate_reward_leaves_replicas_asleep(monkeypatch):
    """In trainer mode the replicas are already asleep; the colocated RM leaves them so."""
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import HybridEngineMode

    calls: list[str] = []
    trainer = make_separate_async_trainer(monkeypatch, calls)
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer.checkpoint_manager.asleep = True

    train_one_batch(trainer)

    assert calls == ["reward", "update_actor_asleep"]


def test_separate_async_colocate_reward_keeps_replicas_asleep_across_steps(monkeypatch):
    """Through the real hooks, no sleep hits a sleeping replica and no actor update runs next to awake ones.

    switch disabled: on_train_begin reclaims the replicas, validation lends them
    to generation, and the next on_step_begin reclaims them again.
    """
    from verl_omni.trainer.diffusion.v1 import trainer_separate_async
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import HybridEngineMode

    calls: list[str] = []
    trainer = make_separate_async_trainer(monkeypatch, calls)
    assert not trainer.hybrid_rollout_config.enable_switch
    trainer.current_mode = HybridEngineMode.ROLLOUT  # as _setup leaves it
    trainer._add_batch_to_generate = lambda: None
    trainer._add_prompts_to_generate = lambda n: None
    monkeypatch.setattr(trainer_separate_async.tq, "kv_list", lambda partition: {})

    trainer.on_train_begin()
    for _ in range(2):
        trainer.on_step_begin()
        train_one_batch(trainer)
    trainer.on_validate_begin()
    trainer.on_step_begin()
    train_one_batch(trainer)

    assert calls == [
        "sleep",  # on_train_begin
        "reward",
        "update_actor_asleep",
        "reward",
        "update_actor_asleep",
        "update_weights",  # on_validate_begin -> switch_to_rollout
        "sleep",  # on_step_begin -> switch_to_trainer
        "reward",
        "update_actor_asleep",
    ]


def test_separate_async_colocate_reward_keeps_replicas_asleep_with_switching(monkeypatch):
    """With switching on, on_step_end lends the replicas to generation and the next step reclaims them asleep."""
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import HybridEngineMode

    calls: list[str] = []
    trainer = make_separate_async_trainer(
        monkeypatch,
        calls,
        [
            "trainer.v1.separate_async.hybrid_rollout.enable_switch=true",
            "trainer.v1.separate_async.sync_compatible=false",
        ],
    )
    assert trainer.hybrid_rollout_config.enable_switch
    sampleable = {"count": trainer.config.data.train_batch_size}
    trainer.replay_buffer = SimpleNamespace(
        get_sampleable_count=lambda step, partition: sampleable["count"],
        wait_for_sampleable=lambda step, partition, threshold: (None, {}),
    )
    trainer.clear_sticky_cache = lambda: {}
    trainer.prev_step_profile = False
    trainer.total_training_steps = 3
    trainer.current_mode = HybridEngineMode.ROLLOUT  # as _setup leaves it

    trainer.on_step_begin()  # enough sampleable groups: reclaim now
    trainer._wait_for_sampleable_and_switch()
    train_one_batch(trainer)
    sampleable["count"] = 0
    trainer.on_step_end()  # the next step's groups are missing: lend the replicas
    assert trainer.current_mode == HybridEngineMode.ROLLOUT
    trainer.global_steps = 2
    trainer.on_step_begin()
    trainer._wait_for_sampleable_and_switch()  # reclaim once the groups are sampleable
    train_one_batch(trainer)

    assert calls == [
        "sleep",  # on_step_begin -> switch_to_trainer
        "reward",
        "update_actor_asleep",
        "update_weights",  # on_step_end -> switch_to_rollout
        "sleep",  # _wait_for_sampleable_and_switch -> switch_to_trainer
        "reward",
        "update_actor_asleep",
    ]
