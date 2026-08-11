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

import os

import pytest
from hydra import compose, initialize_config_dir

import verl_omni
from verl_omni.workers.config.diffusion import (
    DiffusionDistillationConfig,
    DiffusionDistillationTeacherModelConfig,
)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")


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

    def test_enabled_requires_single_teacher_model_entry(self):
        with pytest.raises(NotImplementedError, match="teacher_model"):
            DiffusionDistillationConfig(
                enabled=True,
                teacher_models={
                    "teacher_model": DiffusionDistillationTeacherModelConfig(model_path="/ckpt/a"),
                    "second": DiffusionDistillationTeacherModelConfig(model_path="/ckpt/b"),
                },
            )

    def test_enabled_with_model_path_passes(self):
        config = DiffusionDistillationConfig(
            enabled=True,
            teacher_models={"teacher_model": DiffusionDistillationTeacherModelConfig(model_path="/ckpt/teacher")},
        )
        assert config.teacher_models["teacher_model"].model_path == "/ckpt/teacher"


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

    def test_cli_enable_with_teacher_path(self):
        from verl.utils.config import omega_conf_to_dataclass

        cfg = self._compose(
            [
                "distillation.enabled=true",
                "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
            ]
        )
        config = omega_conf_to_dataclass(cfg.distillation)
        assert config.teacher_models["teacher_model"].model_path == "/ckpt/teacher"
