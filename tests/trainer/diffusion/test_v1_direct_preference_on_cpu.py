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
"""CPU tests for online DPO on the v1 diffusion trainer.

Necessity: the v1 PG loop recomputes ``old_log_probs`` via ``infer_actor_batch``.
DPO engines return ``noise_pred`` (log_probs=None), which crashed
``run_qwen_image_online_dpo_lora_v1.sh``. These tests lock the direct-preference
branch that pairs rewards and uses ref noise preds instead.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.utils import tensordict_utils as tu

import verl_omni

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")

DPO_OVERRIDES = [
    "algorithm.trainer_type=direct_preference",
    "algorithm.sample_source=online",
    "algorithm.paired_preference=true",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo",
]


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


def make_dpo_trainer(overrides=None):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    return PolicyGradientDiffusionTrainerV1Sync(compose_cfg(DPO_OVERRIDES + list(overrides or [])))


def test_offline_direct_preference_rejected_on_v1():
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    with pytest.raises(NotImplementedError, match="offline DPO stays on the v0 trainer"):
        PolicyGradientDiffusionTrainerV1Sync(
            compose_cfg(
                [
                    "algorithm.trainer_type=direct_preference",
                    "algorithm.sample_source=offline",
                    "actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo",
                ]
            )
        )


def test_dpo_enables_reference_policy_without_kl():
    trainer = make_dpo_trainer()
    assert trainer._is_direct_preference
    assert trainer.use_reference_policy
    assert trainer._has_old_adapter is False


def test_dpo_train_step_skips_old_log_prob_and_pairs_batch(monkeypatch):
    trainer = make_dpo_trainer()
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
    trainer.global_steps = 1

    uid = np.array(["p0", "p0", "p1", "p1"], dtype=object)
    data = DataProto.from_dict(
        tensors={
            "rm_scores": torch.tensor([[1.0], [0.0], [0.2], [0.8]]),
            "latents_clean": torch.zeros(4, 2, 4, 4),
        },
        non_tensors={"uid": uid},
    )
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
        lambda meta, pad_token_id: data,
    )
    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.extract_reward",
        lambda batch: (batch.batch["rm_scores"], {}),
    )
    tq_writes: list[list[str]] = []

    def capture_tq(batch_meta, batch, fields):
        del batch_meta, batch
        tq_writes.append(list(fields))

    monkeypatch.setattr(
        "verl_omni.trainer.diffusion.v1.trainer_base.put_dataproto_fields_to_tq",
        capture_tq,
    )

    def fail_old_log_prob(_data):
        raise AssertionError("DPO must not recompute old_log_probs")

    monkeypatch.setattr(trainer, "_compute_old_log_prob", fail_old_log_prob)

    def fail_balance(d, metrics):
        raise AssertionError("DPO must not DP-pad before pairing")

    monkeypatch.setattr(trainer, "_balance_batch", fail_balance)

    captured = {}

    def capture_ref(batch):
        captured["ref"] = batch
        return DataProto.from_tensordict(tu.get_tensordict({"ref_noise_pred": torch.ones(len(batch), 2, 4, 4)}))

    def capture_update(batch):
        captured["update"] = batch
        return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/dpo_loss": 0.1}})

    monkeypatch.setattr(trainer, "_compute_ref_noise_pred", capture_ref)
    monkeypatch.setattr(trainer, "_update_actor", capture_update)

    batch_meta = KVBatchMeta(
        partition_id="train",
        keys=["p0_0_0", "p0_1_0", "p1_0_0", "p1_1_0"],
        tags=[{"is_padding": False}] * 4,
    )
    result = trainer._train_sampled_batch({}, {}, batch_meta)

    assert result is batch_meta
    assert "old_log_probs" not in tq_writes[0]
    assert "sample_level_scores" in tq_writes[0]
    paired = captured["update"]
    assert len(paired) == 4
    assert list(paired.non_tensor_batch["uid"]) == ["p0", "p0", "p1", "p1"]
    scores = paired.batch["sample_level_scores"].reshape(-1)
    assert scores[0] >= scores[1]
    assert scores[2] >= scores[3]
    assert "ref_noise_pred" in captured["update"].batch
    assert "old_log_probs" not in captured["update"].batch


def test_dpo_update_actor_uses_paired_mini_batch_size():
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    actor = MagicMock()
    actor.update_actor.return_value = tu.get_tensordict({}, non_tensor_dict={"metrics": {}})
    trainer = SimpleNamespace(
        config=compose_cfg(
            DPO_OVERRIDES
            + [
                "actor_rollout_ref.actor.ppo_mini_batch_size=2",
                "actor_rollout_ref.actor.ppo_epochs=1",
                "actor_rollout_ref.actor.data_loader_seed=0",
                "actor_rollout_ref.actor.shuffle=true",
                "actor_rollout_ref.rollout.n=16",
            ]
        ),
        _is_direct_preference=True,
        actor_rollout_wg=actor,
    )
    batch = DataProto.from_dict(tensors={"latents_clean": torch.zeros(4, 2, 2, 2)})

    PolicyGradientDiffusionTrainerV1._update_actor(trainer, batch)

    sent = actor.update_actor.call_args.args[0]
    assert tu.get_non_tensor_data(sent, "mini_batch_size", None) == 4
    assert tu.get_non_tensor_data(sent, "global_batch_size", None) == 4
    assert tu.get_non_tensor_data(sent, "dataloader_kwargs", {})["shuffle"] is False


def test_compute_old_log_prob_fails_closed_when_log_probs_missing():
    from verl_omni.trainer.diffusion.v1.trainer_base import PolicyGradientDiffusionTrainerV1

    actor = MagicMock()
    actor.infer_actor_batch.return_value = tu.get_tensordict(
        {"noise_pred": torch.zeros(2, 1)},
        non_tensor_dict={"metrics": {}},
    )
    trainer = SimpleNamespace(
        config=compose_cfg([]),
        actor_rollout_wg=actor,
    )
    batch = DataProto.from_dict(tensors={"all_latents": torch.zeros(2, 1, 2, 2, 2)})

    with pytest.raises(RuntimeError, match="log_probs=None"):
        PolicyGradientDiffusionTrainerV1._compute_old_log_prob(trainer, batch)
