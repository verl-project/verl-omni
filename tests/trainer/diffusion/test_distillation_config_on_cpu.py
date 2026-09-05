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
"""CPU tests for the diffusion on-policy distillation config."""

import dataclasses
import os

import pytest
from hydra import compose, initialize_config_dir

import verl_omni
from verl_omni.workers.config.diffusion import (
    DiffusionDistillationConfig,
    DiffusionDistillationTeacherModelConfig,
    DiffusionDistributionMatchingConfig,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


class TestDistributionMatchingConfig:
    def test_defaults_select_dmd2_without_enabling_opd(self):
        config = DiffusionDistillationConfig()
        assert config.enabled is False
        assert config.distribution_matching.recipe == "dmd2"
        assert config.distribution_matching.profile is None
        assert config.distribution_matching.fake_update_ratio is None
        assert config.distribution_matching.role_storage == "shared_base_adapters"
        assert config.distribution_matching.student_micro_batch_size_per_gpu == 1
        assert config.distribution_matching.fake_score_micro_batch_size_per_gpu == 1
        assert config.distribution_matching.ema_decay == pytest.approx(0.999)
        assert config.distribution_matching.ema_start_step == 0
        assert config.distribution_matching.fake_score_optim.lr == pytest.approx(2e-5)

    @pytest.mark.parametrize(
        "kwargs,error",
        [
            ({"recipe": "typo"}, "Invalid recipe"),
            ({"profile": "typo"}, "Invalid profile"),
            ({"fake_update_ratio": 0}, "greater than 0"),
            ({"fake_warmup_cycles": -1}, "non-negative"),
            ({"rollout_strategy": "typo"}, "Invalid rollout_strategy"),
            ({"data_mode": "typo"}, "Invalid data_mode"),
            ({"export_role": "teacher_score"}, "Invalid export_role"),
            ({"role_storage": "remote"}, "Invalid role_storage"),
            ({"student_micro_batch_size_per_gpu": 0}, "greater than 0"),
            ({"fake_score_micro_batch_size_per_gpu": 0}, "greater than 0"),
            ({"ema_decay": -0.1}, "ema_decay"),
            ({"ema_decay": 1.1}, "ema_decay"),
            ({"ema_start_step": -1}, "non-negative"),
        ],
    )
    def test_invalid_values_fail_closed(self, kwargs, error):
        with pytest.raises(ValueError, match=error):
            DiffusionDistributionMatchingConfig(**kwargs)

    def test_null_fields_use_recipe_specific_defaults(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        config = dataclasses.asdict(DiffusionDistributionMatchingConfig(recipe="dmd"))
        plan = build_plan(
            "dmd",
            {**config, "model_path": "/m"},
            frozenset({"distribution_matching"}),
        )
        assert plan.objective["profile"] == "paper"
        assert plan.update_schedule.phases[1].repeats == 1


class TestDiffusionDistillationConfig:
    def test_disabled_by_default_skips_validation(self):
        config = DiffusionDistillationConfig()
        assert config.enabled is False
        assert config.teacher_models == {}

    def test_enabled_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path"):
            DiffusionDistillationConfig(
                enabled=True,
                teacher_models={"teacher_model": DiffusionDistillationTeacherModelConfig()},
            )

    def test_single_teacher_is_keyed_default(self):
        config = DiffusionDistillationConfig(
            enabled=True,
            teacher_models={"teacher_model": DiffusionDistillationTeacherModelConfig(model_path="/ckpt/teacher")},
        )
        assert set(config.teacher_models) == {"default"}
        assert config.teacher_models["default"].key == "default"
        assert config.teacher_models["default"].model_path == "/ckpt/teacher"
        assert config.teacher_models["default"].world_size == 0

    def test_single_teacher_fills_standalone_pool(self):
        config = DiffusionDistillationConfig(
            enabled=True,
            n_gpus_per_node=2,
            nnodes=2,
            teacher_models={"teacher_model": DiffusionDistillationTeacherModelConfig(model_path="/ckpt/teacher")},
        )
        assert config.teacher_models["default"].world_size == 4

    def test_multi_teacher_pops_default_entry_and_indexes_by_key(self):
        config = DiffusionDistillationConfig(
            enabled=True,
            teacher_models={
                "teacher_model": DiffusionDistillationTeacherModelConfig(),
                "a": DiffusionDistillationTeacherModelConfig(key="ocr", model_path="/ckpt/a"),
                "b": DiffusionDistillationTeacherModelConfig(key="aes", model_path="/ckpt/b"),
            },
        )
        assert set(config.teacher_models) == {"ocr", "aes"}
        assert config.teacher_models["ocr"].model_path == "/ckpt/a"
        assert config.teacher_models["aes"].model_path == "/ckpt/b"

    def test_multi_teacher_requires_key(self):
        with pytest.raises(ValueError, match="key must be specified"):
            DiffusionDistillationConfig(
                enabled=True,
                teacher_models={
                    "teacher_model": DiffusionDistillationTeacherModelConfig(),
                    "a": DiffusionDistillationTeacherModelConfig(model_path="/ckpt/a"),
                    "b": DiffusionDistillationTeacherModelConfig(key="aes", model_path="/ckpt/b"),
                },
            )

    def test_multi_teacher_rejects_duplicate_key(self):
        with pytest.raises(ValueError, match="Duplicate teacher key"):
            DiffusionDistillationConfig(
                enabled=True,
                teacher_models={
                    "teacher_model": DiffusionDistillationTeacherModelConfig(),
                    "a": DiffusionDistillationTeacherModelConfig(key="ocr", model_path="/ckpt/a"),
                    "b": DiffusionDistillationTeacherModelConfig(key="ocr", model_path="/ckpt/b"),
                },
            )

    def test_standalone_pool_requires_matching_world_size_sum(self):
        with pytest.raises(ValueError, match="must match the distillation resource pool size"):
            DiffusionDistillationConfig(
                enabled=True,
                n_gpus_per_node=4,
                nnodes=1,
                teacher_models={
                    "teacher_model": DiffusionDistillationTeacherModelConfig(),
                    "a": DiffusionDistillationTeacherModelConfig(key="ocr", model_path="/ckpt/a", world_size=1),
                    "b": DiffusionDistillationTeacherModelConfig(key="aes", model_path="/ckpt/b", world_size=1),
                },
            )

    def test_colocated_ignores_world_size(self):
        config = DiffusionDistillationConfig(
            enabled=True,
            teacher_models={
                "teacher_model": DiffusionDistillationTeacherModelConfig(),
                "a": DiffusionDistillationTeacherModelConfig(key="ocr", model_path="/ckpt/a"),
                "b": DiffusionDistillationTeacherModelConfig(key="aes", model_path="/ckpt/b"),
            },
        )
        assert set(config.teacher_models) == {"ocr", "aes"}


class TestDistillationConfigComposition:
    def _compose(self, overrides):
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(config_name="diffusion_trainer", overrides=overrides)

    def test_default_composition_is_disabled(self):
        from verl.utils.config import omega_conf_to_dataclass

        cfg = self._compose([])
        config = omega_conf_to_dataclass(cfg.distillation)
        assert isinstance(config, DiffusionDistillationConfig)
        assert config.enabled is False
        assert isinstance(config.distribution_matching, DiffusionDistributionMatchingConfig)
        assert config.distribution_matching.recipe == "dmd2"

    def test_cli_enable_with_teacher_path(self):
        from verl.utils.config import omega_conf_to_dataclass

        cfg = self._compose(
            [
                "distillation.enabled=true",
                "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
            ]
        )
        config = omega_conf_to_dataclass(cfg.distillation)
        assert config.teacher_models["default"].model_path == "/ckpt/teacher"
        assert config.teacher_key == "data_source"
        assert config.n_gpus_per_node == 0
        assert config.nnodes == 0

    def test_cli_distribution_matching_overrides_do_not_enable_opd(self):
        from verl.utils.config import omega_conf_to_dataclass

        cfg = self._compose(
            [
                "algorithm.trainer_type=distillation",
                "algorithm.sample_source=offline",
                "distillation.distribution_matching.recipe=dmd2",
                "distillation.distribution_matching.fake_update_ratio=2",
                "distillation.distribution_matching.rollout_strategy=consistency_renoise",
            ]
        )
        config = omega_conf_to_dataclass(cfg.distillation)
        assert config.enabled is False
        assert config.distribution_matching.fake_update_ratio == 2
        assert config.distribution_matching.rollout_strategy == "consistency_renoise"
        assert config.distribution_matching.fake_score_optim.lr == pytest.approx(2e-5)

    def test_composed_config_builds_validated_plan(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan_from_config

        cfg = self._compose(
            [
                "algorithm.trainer_type=distillation",
                "algorithm.sample_source=offline",
                "actor_rollout_ref.model.path=/m",
                "distillation.distribution_matching.fake_update_ratio=2",
            ]
        )
        plan = build_plan_from_config(cfg, frozenset({"distribution_matching"}))
        assert plan.name == "dmd2"
        assert plan.role_layout.groups[0].model_ref == "/m"
        assert plan.update_schedule.phases[1].repeats == 2

    def test_null_overrides_use_each_recipe_default(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan_from_config

        cfg = self._compose(
            [
                "algorithm.trainer_type=distillation",
                "algorithm.sample_source=offline",
                "actor_rollout_ref.model.path=/m",
                "distillation.distribution_matching.recipe=dmd",
            ]
        )
        plan = build_plan_from_config(cfg, frozenset({"distribution_matching"}))
        assert plan.objective["profile"] == "paper"
        assert plan.update_schedule.phases[1].repeats == 1

    def test_cli_multi_teacher_entries(self):
        from verl.utils.config import omega_conf_to_dataclass

        cfg = self._compose(
            [
                "distillation.enabled=true",
                "+distillation.teacher_models.a.key=ocr",
                "+distillation.teacher_models.a.model_path=/ckpt/a",
                "+distillation.teacher_models.a.world_size=1",
                "+distillation.teacher_models.b.key=aes",
                "+distillation.teacher_models.b.model_path=/ckpt/b",
                "+distillation.teacher_models.b.world_size=1",
                "distillation.n_gpus_per_node=2",
                "distillation.nnodes=1",
            ]
        )
        config = omega_conf_to_dataclass(cfg.distillation)
        assert set(config.teacher_models) == {"ocr", "aes"}
        assert all(isinstance(t, DiffusionDistillationTeacherModelConfig) for t in config.teacher_models.values())
        assert config.teacher_models["aes"].world_size == 1
