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


NAMED_REWARD = [
    "reward.reward_model.enable=false",
    "+reward.models.quality.backend=native",
    "+reward.models.quality.placement.devices=[0]",
    "+reward.models.quality.executor.model=example:Model",
]


@pytest.mark.parametrize("named", [False, True], ids=["legacy", "named"])
def test_colocate_reward_keeps_rollout_asleep_through_actor_update(monkeypatch, named):
    """The sync sample and step-end hooks bracket the entire training batch."""
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(
        compose_cfg(NAMED_REWARD if named else ["reward.reward_model.enable=true"])
    )
    assert trainer.use_rm

    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(2, 4)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(2, 1)}))
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    calls: list[str] = []
    trainer.reward_loop_manager = SimpleNamespace(
        reward_loop_worker_handles=None, compute_rm_score=lambda d: (calls.append("reward"), reward)[1]
    )
    trainer.checkpoint_manager = _ColocatedReplicas(calls)
    trainer.global_steps = 1
    trainer.curr_step_profile = False
    trainer.timing_raw = {}
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(trainer, "_balance_batch", lambda d, metrics: d)
    old_log_prob = DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": torch.zeros(2, 4)}))
    monkeypatch.setattr(trainer, "_compute_old_log_prob", lambda d: (calls.append("old_log_prob"), old_log_prob)[1])
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )
    monkeypatch.setattr(trainer, "_compute_advantage", lambda d: (calls.append("advantage"), d)[1])

    def update_actor(data):
        assert trainer.checkpoint_manager.asleep
        calls.append("update_actor")
        return DataProto.from_dict(meta_info={"metrics": {}})

    monkeypatch.setattr(trainer, "_update_actor", update_actor)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.put_dataproto_fields_to_tq", lambda *args, **kwargs: None
    )
    trainer.on_sample_end()
    trainer._train_sampled_batch({}, {}, SimpleNamespace(keys=["0", "1"]))
    trainer.on_step_end()
    assert calls == ["sleep", "reward", "old_log_prob", "advantage", "update_actor", "update_weights"]


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


def make_separate_async_trainer(monkeypatch, calls: list, overrides=(), *, complete_update=False):
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

    def update_actor(data):
        calls.append("update_actor_asleep" if replicas.asleep else "update_actor_awake")
        if not complete_update:
            raise _Stop
        return DataProto.from_dict(meta_info={"metrics": {}})

    monkeypatch.setattr(trainer, "_update_actor", update_actor)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.put_dataproto_fields_to_tq", lambda *args, **kwargs: None
    )
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


def test_separate_async_sync_compatible_only_resumes_standalone_after_actor_update(monkeypatch):
    from verl_omni.trainer.diffusion.v1 import trainer_separate_async
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import HybridEngineMode

    calls: list[str] = []
    trainer = make_separate_async_trainer(
        monkeypatch,
        calls,
        [
            *NAMED_REWARD,
            "trainer.v1.separate_async.sync_compatible=true",
            "trainer.v1.separate_async.num_warmup_batches=0",
        ],
        complete_update=True,
    )
    trainer.sync_compatible = True
    trainer._standalone_paused = False
    trainer.prev_step_profile = False
    trainer.current_mode = HybridEngineMode.ROLLOUT
    standalone_server = object()

    def remove_standalone(addresses):
        assert addresses == ["standalone"]
        calls.append("remove standalone")

    def add_standalone(servers):
        assert servers == {"standalone": standalone_server}
        calls.append("add standalone")

    def update_standalone(step):
        assert step == trainer.global_steps
        assert trainer.checkpoint_manager.asleep
        calls.append("update standalone")
        return {}

    trainer.standalone_server_manager = SimpleNamespace(
        global_load_balancer=SimpleNamespace(
            remove_servers=SimpleNamespace(remote=remove_standalone),
            add_servers=SimpleNamespace(remote=add_standalone),
        ),
        server_addresses=["standalone"],
        server_handles=[standalone_server],
    )
    trainer.standalone_checkpoint_manager = SimpleNamespace(
        abort_replicas=lambda: calls.append("abort standalone"),
        update_weights=update_standalone,
        resume_generation_replicas=lambda: calls.append("resume standalone"),
    )
    monkeypatch.setattr(trainer_separate_async.ray, "get", lambda value: value)

    trainer.on_step_begin()
    trainer.on_sample_begin()
    trainer.on_sample_end()
    trainer._train_sampled_batch({}, {}, SimpleNamespace(keys=["0", "1"]))
    trainer.on_step_end()

    assert calls == [
        "sleep",
        "abort standalone",
        "remove standalone",
        "reward",
        "update_actor_asleep",
        "update standalone",
        "add standalone",
        "resume standalone",
    ]
    assert trainer.checkpoint_manager.asleep
    assert not trainer._standalone_paused


def test_separate_async_validation_reopens_hybrid_routing_before_standalone_weight_wait(monkeypatch):
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import HybridEngineMode

    calls: list[str] = []
    trainer = make_separate_async_trainer(monkeypatch, calls)
    assert not trainer.hybrid_rollout_config.enable_switch
    trainer.current_mode = HybridEngineMode.TRAINER
    trainer.checkpoint_manager.asleep = True
    trainer.checkpoint_manager.resume_generation_replicas = lambda: calls.append("resume hybrid")
    trainer.add_replicas_to_balancer = lambda: calls.append("add hybrid routes")
    trainer.standalone_checkpoint_manager.wait_for_weight_update = lambda: calls.append("wait standalone")

    trainer.on_validate_begin()

    assert calls == ["update_weights", "resume hybrid", "add hybrid routes", "wait standalone"]
    assert trainer.current_mode == HybridEngineMode.ROLLOUT
