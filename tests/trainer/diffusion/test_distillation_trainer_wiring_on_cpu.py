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

import pytest
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


class TestTeacherPoolRegistration:
    def _runner(self, overrides):
        from verl_omni.trainer.main_diffusion import TaskRunner

        cfg = compose_cfg(ENABLE + ["actor_rollout_ref.actor.diffusion_loss.loss_mode=distill_kl"] + overrides)
        runner = TaskRunner()
        actor_rollout_cls, _ = runner.add_actor_rollout_worker(cfg)
        runner.add_teacher_model_worker(cfg, actor_rollout_cls)
        return cfg, runner

    def test_standalone_registers_teacher_pool(self):
        from verl.trainer.ppo.utils import Role

        cfg, runner = self._runner(["distillation.n_gpus_per_node=2", "distillation.nnodes=1"])
        manager = runner.init_resource_pool_mgr(cfg)
        assert manager.resource_pool_spec["teacher_pool"] == [2]
        assert runner.mapping[Role.TeacherModel] == "teacher_pool"
        assert Role.TeacherModel in runner.role_worker_mapping

    def test_colocated_registers_nothing(self):
        from verl.trainer.ppo.utils import Role

        cfg, runner = self._runner([])
        manager = runner.init_resource_pool_mgr(cfg)
        assert "teacher_pool" not in manager.resource_pool_spec
        assert Role.TeacherModel not in runner.mapping
        assert Role.TeacherModel not in runner.role_worker_mapping

    def test_standalone_requires_gpus_per_node(self):
        cfg, runner = self._runner(["distillation.n_gpus_per_node=0", "distillation.nnodes=1"])
        with pytest.raises(ValueError, match="config.distillation.n_gpus_per_node must be greater than 0"):
            runner.init_resource_pool_mgr(cfg)
