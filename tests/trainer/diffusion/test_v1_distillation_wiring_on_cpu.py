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
"""CPU tests for multi-teacher distillation wiring in the v1 diffusion trainer."""

import os

import pytest
from hydra import compose, initialize_config_dir

import verl_omni

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(verl_omni.__file__)), "trainer", "config")

ENABLE = [
    "distillation.enabled=true",
    "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
    "actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl",
]


def compose_cfg(overrides):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=overrides)


def make_trainer(overrides):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    return PolicyGradientDiffusionTrainerV1Sync(compose_cfg(overrides))


class TestV1DistillationInit:
    def test_teacher_policy_enabled(self):
        trainer = make_trainer(ENABLE)
        assert trainer.use_teacher_policy
        assert trainer.distillation_config is not None

    def test_default_config_has_no_teacher(self):
        trainer = make_trainer([])
        assert not trainer.use_teacher_policy
        assert trainer.distillation_config is None

    def test_enabled_but_no_distill_loss_raises(self):
        with pytest.raises(ValueError, match="distill"):
            make_trainer(
                [
                    "distillation.enabled=true",
                    "distillation.teacher_models.teacher_model.model_path=/ckpt/teacher",
                ]
            )

    def test_separate_async_mode_rejected(self):
        with pytest.raises(NotImplementedError, match="sync"):
            make_trainer(ENABLE + ["trainer.v1.trainer_mode=separate_async"])


class TestV1TeacherPoolRegistration:
    def test_standalone_registers_teacher_pool(self):
        from verl.trainer.ppo.utils import Role

        trainer = make_trainer(ENABLE + ["distillation.n_gpus_per_node=2", "distillation.nnodes=1"])
        trainer._init_resource_pool_mgr()
        assert trainer.resource_pool_manager.resource_pool_spec["teacher_pool"] == [2]
        assert trainer.mapping[Role.TeacherModel] == "teacher_pool"
        assert Role.TeacherModel in trainer.role_worker_mapping

    def test_colocated_registers_nothing(self):
        from verl.trainer.ppo.utils import Role

        trainer = make_trainer(ENABLE)
        trainer._init_resource_pool_mgr()
        assert "teacher_pool" not in trainer.resource_pool_manager.resource_pool_spec
        assert Role.TeacherModel not in trainer.mapping
        assert Role.TeacherModel not in trainer.role_worker_mapping

    def test_standalone_requires_gpus_per_node(self):
        trainer = make_trainer(ENABLE + ["distillation.n_gpus_per_node=0", "distillation.nnodes=1"])
        with pytest.raises(ValueError, match="config.distillation.n_gpus_per_node must be greater than 0"):
            trainer._init_resource_pool_mgr()
