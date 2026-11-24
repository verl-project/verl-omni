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
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass

import verl_omni
from verl_omni.trainer.diffusion.diffusion_trainer_utils import skip_old_log_prob
from verl_omni.workers.utils.losses import diffusion_loss

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class TestSkipOldLogProb:
    def test_flow_grpo_recomputes(self):
        assert skip_old_log_prob(compose_cfg([])) is False

    def test_pure_distill_kl_skips(self):
        cfg = compose_cfg(["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"])
        assert skip_old_log_prob(cfg) is True

    def test_auxiliary_distill_on_flow_grpo_recomputes(self):
        cfg = compose_cfg(["actor_rollout_ref.actor.use_distill_loss=true"])
        assert skip_old_log_prob(cfg) is False

    def test_distill_kl_with_rollout_correction_recomputes(self):
        cfg = compose_cfg(
            [
                "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
                "algorithm.rollout_correction.rollout_is=sequence",
            ]
        )
        assert skip_old_log_prob(cfg) is False

    def test_kl_penalty_does_not_force_recompute(self):
        cfg = compose_cfg(
            ["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl", "actor_rollout_ref.actor.use_kl_loss=true"]
        )
        assert skip_old_log_prob(cfg) is True

    @pytest.mark.parametrize("loss_mode", ["dgpo", "dmd2"])
    def test_other_log_prob_free_losses_skip(self, loss_mode):
        cfg = compose_cfg([f"actor_rollout_ref.actor.diffusion_loss.loss_mode={loss_mode}"])
        assert skip_old_log_prob(cfg) is True

    def test_direct_preference_never_sets_the_flag(self):
        cfg = compose_cfg(
            ["algorithm.trainer_type=direct_preference", "actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo"]
        )
        assert skip_old_log_prob(cfg) is False


def _actor_config(overrides):
    with initialize_config_dir(config_dir=os.path.join(CONFIG_DIR, "diffusion", "actor"), version_base=None):
        cfg = compose(
            config_name="dp_diffusion_actor", overrides=["strategy=fsdp", "ppo_micro_batch_size_per_gpu=4", *overrides]
        )
    return omega_conf_to_dataclass(cfg)


def _step_inputs(with_old_log_probs):
    torch.manual_seed(0)
    model_output = {
        "log_probs": torch.randn(4),
        "prev_sample_mean": torch.randn(4, 16, 3),
        "std_dev_t": torch.ones(4, 1, 1),
    }
    tensors = {"advantages": torch.randn(4), "teacher_prev_sample_mean": torch.randn(4, 16, 3)}
    if with_old_log_probs:
        tensors["old_log_probs"] = torch.randn(4)
    data = tu.get_tensordict(tensors)
    tu.assign_non_tensor(data, gradient_accumulation_steps=1, sp_size=1)
    return model_output, data


class TestLossWithoutOldLogProbs:
    def test_distill_kl_trains_without_them(self):
        loss, _ = diffusion_loss(_actor_config(["diffusion_loss.loss_mode=distill_kl"]), *_step_inputs(False))
        assert torch.isfinite(loss)

    def test_log_prob_loss_names_the_missing_key(self):
        with pytest.raises(KeyError, match="old_log_probs"):
            diffusion_loss(_actor_config([]), *_step_inputs(False))


DISTILL_OVERRIDES = [
    "distillation.enabled=true",
    "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
    "actor_rollout_ref.actor.ppo_mini_batch_size=2",
    "actor_rollout_ref.rollout.n=2",
]


def _make_v1_trainer(overrides):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    return PolicyGradientDiffusionTrainerV1Sync(compose_cfg(overrides))


class TestV1TrainerSkipBranch:
    def test_dpo_trainer_does_not_set_the_flag(self):
        trainer = _make_v1_trainer(
            ["algorithm.trainer_type=direct_preference", "actor_rollout_ref.actor.diffusion_loss.loss_mode=dpo"]
        )
        assert trainer.skip_old_log_prob is False

    def test_distill_kl_step_skips_the_recompute_and_updates(self, monkeypatch):
        trainer = _make_v1_trainer(DISTILL_OVERRIDES)
        assert trainer.skip_old_log_prob is True
        trainer.tokenizer = SimpleNamespace(pad_token_id=0)
        trainer.reward_loop_manager = SimpleNamespace(reward_loop_worker_handles=object())
        trainer.global_steps = 1

        def fail_infer(_batch):
            raise AssertionError("distill_kl must not recompute old_log_probs")

        trainer.actor_rollout_wg = SimpleNamespace(infer_actor_batch=fail_infer)
        trainer.teacher_model_manager = SimpleNamespace(
            compute_prev_sample_mean=lambda batch: DataProto.from_tensordict(
                tu.get_tensordict({"teacher_prev_sample_mean": torch.zeros(len(batch), 2, 16, 3)})
            )
        )

        data = DataProto.from_dict(
            tensors={
                "rm_scores": torch.tensor([[1.0], [0.0], [0.2], [0.8]]),
                "all_timesteps": torch.zeros(4, 2, dtype=torch.long),
                "latents": torch.zeros(4, 3, 16, 3),
            },
            non_tensors={"uid": np.array(["p0", "p0", "p1", "p1"], dtype=object)},
        )
        monkeypatch.setattr(
            "verl_omni.trainer.diffusion.v1.trainer_base.diffusion_tq_batch_to_dataproto",
            lambda meta, pad_token_id: data,
        )
        monkeypatch.setattr(
            "verl_omni.trainer.diffusion.v1.trainer_base.extract_reward",
            lambda batch: (batch.batch["rm_scores"], {}),
        )
        monkeypatch.setattr(
            "verl_omni.trainer.diffusion.v1.trainer_base.put_dataproto_fields_to_tq",
            lambda batch_meta, batch, fields: None,
        )
        captured = {}

        def capture_update(batch):
            captured["update"] = batch
            return DataProto.from_single_dict(data={}, meta_info={"metrics": {"actor/distill_kl_loss": 0.1}})

        monkeypatch.setattr(trainer, "_update_actor", capture_update)

        batch_meta = KVBatchMeta(
            partition_id="train",
            keys=["p0_0_0", "p0_1_0", "p1_0_0", "p1_1_0"],
            tags=[{"is_padding": False}] * 4,
        )
        metrics = {}
        assert trainer._train_sampled_batch(metrics, {}, batch_meta) is batch_meta

        updated = captured["update"].batch
        assert "old_log_probs" not in updated
        assert "teacher_prev_sample_mean" in updated
        assert updated["advantages"].shape == (4, 2)
        assert metrics["actor/distill_kl_loss"] == 0.1
