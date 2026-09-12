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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config.optimizer import FSDPOptimizerConfig

import verl_omni
from verl_omni.workers.config import DiffusionDistillationConfig, DiffusionDMDConfig

CONFIG_DIR = str(Path(verl_omni.__file__).parent / "trainer" / "config")


def compose_config(overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=list(overrides))


class TestDMDConfig:
    def test_defaults_are_dmd2_without_opd_activation(self):
        config = DiffusionDMDConfig()
        assert config.fake_update_ratio == 2
        assert config.export_role == "student"
        assert isinstance(config.fake_score_optim, FSDPOptimizerConfig)
        assert config.fake_score_optim.lr == pytest.approx(2e-5)
        assert not DiffusionDistillationConfig().enabled
        assert not hasattr(DiffusionDistillationConfig(), "distribution_matching")

    @pytest.mark.parametrize(
        "field", ["fake_update_ratio", "student_micro_batch_size_per_gpu", "fake_score_micro_batch_size_per_gpu"]
    )
    @pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, "2"])
    def test_positive_counts_are_not_silently_coerced(self, field, value):
        with pytest.raises(ValueError, match="positive integer"):
            DiffusionDMDConfig(**{field: value})

    @pytest.mark.parametrize("field", ["score_discrete_steps", "ema_start_step"])
    @pytest.mark.parametrize("value", [True, -1, 0.5])
    def test_nonnegative_counts_are_validated(self, field, value):
        with pytest.raises(ValueError, match="non-negative integer"):
            DiffusionDMDConfig(**{field: value})

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"teacher_guidance_scale": 0},
            {"teacher_guidance_scale": float("inf")},
            {"negative_prompt": None},
            {"normalization_epsilon": 0},
            {"normalization_epsilon": float("nan")},
            {"cfg_norm": "legacy"},
            {"rollout_timestep_shift": 0.5},
            {"score_timestep_shift": float("nan")},
            {"score_sigma_min": 0},
            {"score_sigma_min": 0.9, "score_sigma_max": 0.2},
            {"score_sigma_max": 1.1},
            {"ema_decay": -0.1},
            {"ema_decay": float("nan")},
            {"export_role": "fake_score"},
        ],
    )
    def test_invalid_values_fail_closed(self, kwargs):
        with pytest.raises(ValueError):
            DiffusionDMDConfig(**kwargs)

    def test_optimizer_defaults_are_not_shared(self):
        first, second = DiffusionDMDConfig(), DiffusionDMDConfig()
        first.fake_score_optim.total_training_steps = 4
        assert second.fake_score_optim.total_training_steps == -1

    def test_typed_hydra_group_matches_dataclass_defaults(self):
        config = compose_config()
        dmd = omega_conf_to_dataclass(config.dmd)
        assert isinstance(dmd, DiffusionDMDConfig)
        assert isinstance(dmd.fake_score_optim, FSDPOptimizerConfig)
        assert OmegaConf.structured(dmd) == OmegaConf.structured(DiffusionDMDConfig())
        assert omega_conf_to_dataclass(config.distillation).enabled is False

    def test_dmd_loss_can_be_composed_without_model_loading(self):
        config = compose_config(
            [
                "actor_rollout_ref.model.algorithm=dmd2",
                "dmd.fake_update_ratio=3",
                "dmd.fake_score_optim.lr=1e-5",
            ]
        )
        loss = omega_conf_to_dataclass(config.actor_rollout_ref.actor.diffusion_loss)
        assert loss.loss_mode == "dmd2"
        assert omega_conf_to_dataclass(config.dmd).fake_update_ratio == 3

    @pytest.mark.parametrize("schedule", ["inline", "one_step_off"])
    def test_opd_group_is_unchanged(self, schedule):
        config = compose_config(
            [
                "distillation.enabled=true",
                f"distillation.scheduler={schedule}",
                "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
            ]
        )
        opd = omega_conf_to_dataclass(config.distillation)
        assert opd.scheduler == schedule
        assert opd.teacher_models["default"].model_path == "/ckpt/teacher"
        assert "distribution_matching" not in OmegaConf.to_container(config.distillation)
        assert isinstance(omega_conf_to_dataclass(config.dmd), DiffusionDMDConfig)

    @pytest.mark.parametrize("field", ["recipe", "profile", "adversarial", "rollout_strategy", "teacher_models"])
    def test_out_of_scope_options_are_not_accepted(self, field):
        with pytest.raises(ConfigCompositionException):
            compose_config([f"dmd.{field}=unsupported"])
