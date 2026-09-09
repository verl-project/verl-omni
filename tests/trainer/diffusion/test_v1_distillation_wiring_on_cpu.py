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


SEPARATE_ASYNC = [
    "trainer.use_v1=true",
    "trainer.v1.trainer_mode=separate_async",
    "trainer.v1.separate_async.parameter_sync_step=1",
    "actor_rollout_ref.rollout.nnodes=1",
    "actor_rollout_ref.rollout.n_gpus_per_node=1",
    "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    "data.train_batch_size=8",
    "actor_rollout_ref.actor.ppo_mini_batch_size=8",
]


def make_trainer(overrides):
    from verl_omni.trainer.diffusion.v1.trainer_sync import PolicyGradientDiffusionTrainerV1Sync

    return PolicyGradientDiffusionTrainerV1Sync(compose_cfg(overrides))


def make_separate_async_trainer(overrides):
    from verl_omni.trainer.diffusion.v1.trainer_separate_async import (
        PolicyGradientDiffusionTrainerV1SeparateAsync,
    )

    return PolicyGradientDiffusionTrainerV1SeparateAsync(compose_cfg(ENABLE + SEPARATE_ASYNC + overrides))


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

    def test_separate_async_supports_distillation(self):
        from verl.trainer.ppo.utils import Role

        trainer = make_separate_async_trainer(["distillation.n_gpus_per_node=1", "distillation.nnodes=1"])
        assert trainer.use_teacher_policy
        trainer._init_resource_pool_mgr()
        assert trainer.resource_pool_manager.resource_pool_spec["teacher_pool"] == [1]
        assert Role.TeacherModel in trainer.role_worker_mapping


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


class TestOneStepOffScheduler:
    def test_scheduler_defaults_to_inline(self):
        trainer = make_trainer(ENABLE)
        assert trainer.distillation_config.scheduler == "inline"

    def test_one_step_off_requires_separate_async(self):
        with pytest.raises(ValueError, match="one_step_off"):
            make_trainer(ENABLE + ["distillation.scheduler=one_step_off"])

    def test_one_step_off_requires_standalone_teachers(self):
        with pytest.raises(ValueError, match="one_step_off"):
            make_separate_async_trainer(["distillation.scheduler=one_step_off"])

    def test_one_step_off_accepted_on_separate_async(self):
        trainer = make_separate_async_trainer(
            [
                "distillation.scheduler=one_step_off",
                "distillation.n_gpus_per_node=1",
                "distillation.nnodes=1",
                "trainer.v1.separate_async.num_warmup_batches=2",
            ]
        )
        assert trainer.distillation_config.scheduler == "one_step_off"

    def test_one_step_off_rejects_sync_compatible(self):
        with pytest.raises(ValueError, match="sync_compatible"):
            make_separate_async_trainer(
                [
                    "distillation.scheduler=one_step_off",
                    "distillation.n_gpus_per_node=1",
                    "distillation.nnodes=1",
                    "trainer.v1.separate_async.sync_compatible=true",
                    "trainer.v1.separate_async.num_warmup_batches=0",
                ]
            )

    def test_one_step_off_trains_previous_batch(self, monkeypatch):
        trainer = make_separate_async_trainer(
            [
                "distillation.scheduler=one_step_off",
                "distillation.n_gpus_per_node=1",
                "distillation.nnodes=1",
                "trainer.v1.separate_async.num_warmup_batches=2",
            ]
        )
        sampled = iter(["b1", "b2", "b3"])
        monkeypatch.setattr(trainer, "_sample_batch", lambda metrics, timing_raw, n: next(sampled))
        monkeypatch.setattr(trainer, "_convert_and_dispatch", lambda meta: (meta, f"data_{meta}", f"handle_{meta}"))
        trained = []

        def fake_train(metrics, timing_raw, batch_meta, data=None, teacher_handle=None):
            trained.append((batch_meta, data, teacher_handle))
            return batch_meta

        monkeypatch.setattr(trainer, "_train_sampled_batch", fake_train)

        trainer._step_once({}, {}, 8)
        trainer._step_once({}, {}, 8)

        assert trained == [("b1", "data_b1", "handle_b1"), ("b2", "data_b2", "handle_b2")]
        assert trainer._pending_teacher_batch[0] == "b3"

    def test_one_step_off_requires_two_warmup_batches(self):
        with pytest.raises(ValueError, match="num_warmup_batches"):
            make_separate_async_trainer(
                [
                    "distillation.scheduler=one_step_off",
                    "distillation.n_gpus_per_node=1",
                    "distillation.nnodes=1",
                    "trainer.v1.separate_async.num_warmup_batches=1",
                ]
            )
