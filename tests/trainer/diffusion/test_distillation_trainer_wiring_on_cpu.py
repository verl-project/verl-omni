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
"""CPU tests for the diffusion distillation trainer wiring."""

import os
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

import verl_omni
from verl_omni.trainer.diffusion.diffusion_trainer_utils import validate_distillation_config

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")

ENABLE = [
    "distillation.enabled=true",
    "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
]


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


class TestValidateDistillationConfig:
    def test_pure_distill_kl_passes(self):
        validate_distillation_config(
            compose_cfg(ENABLE + ["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"])
        )

    def test_auxiliary_distill_loss_passes(self):
        validate_distillation_config(compose_cfg(ENABLE + ["actor_rollout_ref.actor.use_distill_loss=true"]))

    def test_default_config_passes(self):
        validate_distillation_config(compose_cfg([]))

    def test_enabled_but_no_distill_loss_raises(self):
        with pytest.raises(ValueError, match="distill"):
            validate_distillation_config(compose_cfg(ENABLE))

    def test_distill_loss_but_no_teacher_raises(self):
        with pytest.raises(ValueError, match="teacher"):
            validate_distillation_config(compose_cfg(["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"]))

    def test_distill_fm_mse_rejected(self):
        with pytest.raises(NotImplementedError, match="distill_fm_mse"):
            validate_distillation_config(
                compose_cfg(
                    ENABLE
                    + [
                        "actor_rollout_ref.actor.use_distill_loss=true",
                        "actor_rollout_ref.actor.distill_loss_mode=distill_fm_mse",
                    ]
                )
            )

    def test_direct_preference_trainer_rejected(self):
        with pytest.raises(NotImplementedError, match="policy_gradient"):
            validate_distillation_config(
                compose_cfg(
                    ENABLE
                    + [
                        "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
                        "algorithm.trainer_type=direct_preference",
                    ]
                )
            )


class TestComputeTeacherPrevSampleMean:
    def test_returns_float32_teacher_key(self):
        from verl import DataProto
        from verl.utils import tensordict_utils as tu

        from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

        prev = torch.randn(2, 4, 8, 3, dtype=torch.bfloat16)
        fake_wg = SimpleNamespace(infer_teacher_batch=lambda td: tu.get_tensordict({"prev_sample_mean": prev}))
        cfg = compose_cfg(ENABLE + ["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"])
        fake_self = SimpleNamespace(config=cfg, actor_rollout_wg=fake_wg)
        batch = DataProto.from_tensordict(
            tu.get_tensordict({"all_latents": torch.randn(2, 5, 8, 3), "all_timesteps": torch.zeros(2, 4)})
        )
        out = PolicyGradientRayTrainer._compute_teacher_prev_sample_mean(fake_self, batch)
        assert set(out.batch.keys()) == {"teacher_prev_sample_mean"}
        assert out.batch["teacher_prev_sample_mean"].dtype == torch.float32
        torch.testing.assert_close(out.batch["teacher_prev_sample_mean"], prev.float())
