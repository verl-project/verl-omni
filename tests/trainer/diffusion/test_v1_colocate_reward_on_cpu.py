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
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.utils import tensordict_utils as tu

import verl_omni
from verl_omni.trainer.diffusion.v1 import trainer_base as base_module
from verl_omni.trainer.diffusion.v1 import trainer_colocate_async as module
from verl_omni.trainer.diffusion.v1.trainer_colocate_async import PolicyGradientDiffusionTrainerV1ColocateAsync

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


@pytest.mark.parametrize("trainer_mode", ["sync", "colocate_async"])
def test_colocate_reward_keeps_rollout_asleep_through_actor_update(monkeypatch, trainer_mode):
    """Both colocated modes keep rollout asleep through the actor update.

    ``update_weights`` resumes the rollout replicas' GPU weights (~55GB for
    Qwen-Image at rollout TP=1). If it runs between the reward phase and the
    actor update, every training phase (old log-prob, ref, advantage, actor
    update) executes next to the resident rollout weights and OOMs. In both
    colocated modes, waking and weight-syncing belongs to ``on_step_end``, after the
    actor update.
    """
    from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

    trainer = get_diffusion_trainer_cls(trainer_mode)(
        compose_cfg(["reward.reward_model.enable=true", f"trainer.v1.trainer_mode={trainer_mode}"])
    )
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


def test_colocate_reward_wakes_rollout_mid_cycle_in_async_mode(monkeypatch):
    """Separate-async keeps the mid-cycle weight sync after the colocated RM.

    separate_async's ``on_step_end`` only weight-syncs the standalone rollout;
    the colocated replicas can be woken by ``switch_to_rollout``, but hybrid
    switching is optional. The shared reward block preserves the existing
    colocated weight sync independently of that switch policy.
    The sync subclass is used as a lightweight CPU harness for the shared
    base; only ``trainer_mode`` differs.
    """
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    trainer = PolicyGradientDiffusionTrainerV1Sync(compose_cfg(["reward.reward_model.enable=true"]))
    assert trainer.use_rm
    trainer.trainer_mode = "separate_async"

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

    def stop_at_balance(d, metrics):
        calls.append("balance")
        raise _Stop

    monkeypatch.setattr(trainer, "_balance_batch", stop_at_balance)

    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object())

    assert calls == ["sleep", "reward", "update_weights", "balance"]


@pytest.mark.parametrize("dp_size", [1, 3])
def test_dedicated_reward_overlaps_old_log_prob_and_preserves_rows(monkeypatch, dp_size):
    """A barrier proves overlap; padded rows retain their original reward correspondence."""
    from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

    trainer = get_diffusion_trainer_cls("colocate_async")(
        compose_cfg(
            [
                "trainer.v1.trainer_mode=colocate_async",
                "reward.reward_model.enable=true",
                "reward.reward_model.enable_resource_pool=true",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                "actor_rollout_ref.rollout.n=1",
            ]
        )
    )
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    trainer.actor_rollout_wg = SimpleNamespace(_query_dispatch_info=lambda mesh: {i: i for i in range(dp_size)})
    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.arange(4).unsqueeze(1)}))
    scores = torch.arange(1, 5, dtype=torch.float32).unsqueeze(1)
    reward = DataProto.from_dict(
        tensors={"rm_scores": scores},
        non_tensors={"ocr": np.array(["a", "b", "c", "d"])},
        meta_info={"reward_extra_keys": ["ocr"]},
    )
    rendezvous = threading.Barrier(2)
    reward_done = threading.Event()
    log_prob_done = threading.Event()

    def compute_reward(original):
        assert original is data and len(original) == 4
        rendezvous.wait(timeout=5)
        assert log_prob_done.wait(timeout=5)
        assert "old_log_probs" not in original.batch
        reward_done.set()
        return reward

    def compute_log_prob(balanced):
        rendezvous.wait(timeout=5)
        values = balanced.batch["all_timesteps"].float()
        log_prob_done.set()
        return DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": values}))

    monkeypatch.setattr(trainer, "_compute_reward_colocate", compute_reward)
    monkeypatch.setattr(trainer, "_compute_old_log_prob", compute_log_prob)
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )

    def advantage(joined):
        assert reward_done.is_set() and log_prob_done.is_set()
        joined.batch["advantages"] = torch.ones(len(joined), 1)
        return joined

    captured = {}
    monkeypatch.setattr(trainer, "_compute_advantage", advantage)

    def update(joined):
        captured["data"] = joined
        raise _Stop

    monkeypatch.setattr(trainer, "_update_actor", update)
    timing = {}
    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, timing, object(), data=data)

    expected = [0, 1, 2, 3] if dp_size == 1 else [0, 1, 2, 3, 0, 1]
    joined = captured["data"]
    assert joined.batch["old_log_probs"].flatten().tolist() == expected
    assert joined.batch["rm_scores"].flatten().tolist() == [i + 1 for i in expected]
    assert joined.non_tensor_batch["ocr"].tolist() == ["abcd"[i] for i in expected]
    assert joined.meta_info["reward_extra_keys"] == ["ocr"]
    assert joined.batch["advantages"].flatten().tolist() == [1.0] * 4 + [0.0] * (len(expected) - 4)
    assert {"reward", "old_log_prob"} <= timing.keys()


@pytest.mark.parametrize("failed_phase", ["reward", "old_log_prob"])
def test_dedicated_reward_overlap_drains_work_before_raising(monkeypatch, failed_phase):
    """Failures in either concurrent phase wait for the reward worker to settle."""
    from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

    trainer = get_diffusion_trainer_cls("colocate_async")(
        compose_cfg(
            [
                "trainer.v1.trainer_mode=colocate_async",
                "reward.reward_model.enable=true",
                "reward.reward_model.enable_resource_pool=true",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
            ]
        )
    )
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    data = DataProto.from_tensordict(tu.get_tensordict({"all_timesteps": torch.zeros(4, 1)}))
    monkeypatch.setattr(trainer, "_balance_batch", lambda value, metrics: value)
    rendezvous = threading.Barrier(2)
    reward_settled = threading.Event()

    def reward(value):
        try:
            rendezvous.wait(timeout=5)
            if failed_phase == "reward":
                raise RuntimeError("reward failed")
            return DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(4, 1)}))
        finally:
            reward_settled.set()

    def old_log_prob(value):
        rendezvous.wait(timeout=5)
        if failed_phase == "old_log_prob":
            raise RuntimeError("old_log_prob failed")
        return DataProto.from_tensordict(tu.get_tensordict({"old_log_probs": torch.zeros(4, 1)}))

    monkeypatch.setattr(trainer, "_compute_reward_colocate", reward)
    monkeypatch.setattr(trainer, "_compute_old_log_prob", old_log_prob)
    with pytest.raises(RuntimeError, match=f"{failed_phase} failed"):
        trainer._train_sampled_batch({}, {}, object(), data=data)
    assert reward_settled.is_set()


def test_dedicated_reward_stays_serial_when_old_log_prob_is_bypassed(monkeypatch):
    from verl_omni.trainer.diffusion.v1 import get_diffusion_trainer_cls

    trainer = get_diffusion_trainer_cls("colocate_async")(
        compose_cfg(
            [
                "trainer.v1.trainer_mode=colocate_async",
                "reward.reward_model.enable=true",
                "reward.reward_model.enable_resource_pool=true",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                "actor_rollout_ref.rollout.calculate_log_probs=true",
                "algorithm.rollout_correction.bypass_mode=true",
            ]
        )
    )
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    data = DataProto.from_tensordict(tu.get_tensordict({"rollout_log_probs": torch.ones(4, 1)}))
    reward = DataProto.from_tensordict(tu.get_tensordict({"rm_scores": torch.ones(4, 1)}))
    monkeypatch.setattr(trainer, "_compute_reward_colocate", lambda value: reward)

    def balance(scored, metrics):
        assert "rm_scores" in scored.batch
        assert "old_log_probs" not in scored.batch
        return scored

    monkeypatch.setattr(trainer, "_balance_batch", balance)
    monkeypatch.setattr(trainer, "_compute_old_log_prob", lambda value: pytest.fail("bypass recomputed log-probs"))
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.compute_rollout_corr_metrics_from_batch",
        lambda data, bypass_mode: {},
    )
    monkeypatch.setattr(trainer, "_compute_advantage", lambda value: value)

    def update(prepared):
        torch.testing.assert_close(prepared.batch["old_log_probs"], prepared.batch["rollout_log_probs"])
        raise _Stop

    monkeypatch.setattr(trainer, "_update_actor", update)
    with pytest.raises(_Stop):
        trainer._train_sampled_batch({}, {}, object(), data=data)


def _remote(fn):
    return SimpleNamespace(remote=fn)


@pytest.mark.parametrize("profile", [False, True])
@pytest.mark.parametrize("free_cache", [False, True])
def test_routing_stays_closed_from_abort_through_weight_commit(monkeypatch, profile, free_cache):
    """Failed weight commits never expose rollout replicas to new requests."""
    trainer = PolicyGradientDiffusionTrainerV1ColocateAsync(
        compose_cfg(
            [
                "trainer.v1.trainer_mode=colocate_async",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                f"actor_rollout_ref.rollout.free_cache_engine={str(free_cache).lower()}",
            ]
        )
    )
    events = []
    trainer.global_steps = 3
    trainer.curr_step_profile = profile
    trainer._stop_rollout_profiling = lambda: events.append("stop profile")
    trainer.timing_raw = {}
    trainer.llm_server_manager = SimpleNamespace(
        server_addresses=["server"],
        server_handles=["handle"],
        global_load_balancer=SimpleNamespace(
            remove_servers=_remote(lambda servers: events.append(("remove", servers))),
            add_servers=_remote(lambda servers: events.append(("add", servers))),
        ),
    )
    trainer.checkpoint_manager = SimpleNamespace(
        abort_replicas=lambda: events.append("abort"),
        sleep_replicas=lambda: events.append("sleep"),
        update_weights=lambda step: events.append(("commit", step)),
        resume_generation_replicas=lambda: events.append("resume"),
    )
    monkeypatch.setattr(module.ray, "get", lambda result: result)
    trainer.on_sample_end()
    events.append("actor update")
    trainer.on_step_end()
    assert events == [
        ("remove", ["server"]),
        "abort",
        "sleep",
        *(["stop profile"] if profile else []),
        "actor update",
        ("commit", 3),
        "resume",
        ("add", {"server": "handle"}),
    ]

    # A failed sync never reopens routing.
    events.clear()
    trainer.checkpoint_manager.update_weights = lambda step: (_ for _ in ()).throw(RuntimeError("sync failed"))
    with pytest.raises(RuntimeError, match="sync failed"):
        trainer.on_step_end()
    assert events == []


@pytest.mark.parametrize("separate_reward", [False, True])
def test_validation_reclaims_rollout_only_for_colocated_reward(monkeypatch, separate_reward):
    """Dedicated reward GPUs do not require a rollout ownership switch during validation."""
    trainer = PolicyGradientDiffusionTrainerV1ColocateAsync(
        compose_cfg(
            [
                "trainer.v1.trainer_mode=colocate_async",
                "data.train_batch_size=4",
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                "reward.reward_model.enable=true",
                f"reward.reward_model.enable_resource_pool={str(separate_reward).lower()}",
            ]
        )
    )
    trainer.global_steps = 0
    trainer._gen_batch_seq = 0
    trainer.val_dataloader = [{"raw_prompt": [[{"role": "user", "content": "prompt"}]]}]
    trainer.agent_loop_manager = SimpleNamespace(generate_sequences=lambda batch: None)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=None)
    batch = KVBatchMeta(partition_id="val", keys=["sample_0_0"], tags=[{}])
    trainer.replay_buffer.sample = lambda **kwargs: (batch, {})
    events = []
    trainer.on_sample_end = lambda: events.append("abort/sleep")
    trainer.on_step_end = lambda: events.append("commit/resume")
    data = DataProto.from_dict(tensors={"prompts": torch.ones(1, 2), "responses": torch.ones(1, 3, 2, 2)})
    trainer._compute_reward_colocate = lambda value: (events.append("reward"), value)[1]
    monkeypatch.setattr(base_module, "canonicalize_diffusion_tq_meta", lambda value: value)
    monkeypatch.setattr(base_module, "diffusion_tq_batch_to_dataproto", lambda *args, **kwargs: data)
    monkeypatch.setattr(base_module.tq, "kv_batch_put", lambda **kwargs: None)

    class StopAfterReward(Exception):
        pass

    def decode(*args, **kwargs):
        raise StopAfterReward

    trainer.tokenizer = SimpleNamespace(pad_token_id=0, decode=decode)
    with pytest.raises(StopAfterReward):
        trainer._validate()
    assert events == (["reward"] if separate_reward else ["abort/sleep", "reward", "commit/resume"])
