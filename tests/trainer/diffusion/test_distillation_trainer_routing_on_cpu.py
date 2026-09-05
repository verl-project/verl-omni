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
"""CPU tests for ``algorithm.trainer_type=distillation`` routing.

The DMD-family trainer is routed by ``algorithm.trainer_type``, which is
orthogonal to the existing on-policy distillation (OPD) path. OPD keeps using
``distillation.enabled=true`` together with ``trainer_type=policy_gradient``.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationRayTrainer
from verl_omni.trainer.main_diffusion import TaskRunner, _get_trainer_cls


class FakeAlgorithmConfig:
    def __init__(self, trainer_type):
        self.trainer_type = trainer_type
        self.sample_source = "offline" if trainer_type == "distillation" else "online"


class FakeTrainerConfig:
    def __init__(self, trainer_type):
        self.algorithm = FakeAlgorithmConfig(trainer_type)


def initialize_fake_base_trainer(self, **kwargs):
    for name, value in kwargs.items():
        setattr(self, name, value)
    self.config = kwargs["config"]
    self.total_training_steps = 4
    self.train_dataloader = []
    self.device_name = "cuda"


class TestTrainerRouting:
    def test_distillation_routes_to_distillation_trainer(self):
        assert _get_trainer_cls(FakeTrainerConfig("distillation")) is DistillationRayTrainer

    def test_policy_gradient_is_unchanged(self):
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import PolicyGradientRayTrainer

        assert _get_trainer_cls(FakeTrainerConfig("policy_gradient")) is PolicyGradientRayTrainer

    def test_direct_preference_is_unchanged(self):
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer

        assert _get_trainer_cls(FakeTrainerConfig("direct_preference")) is DirectPreferenceRayTrainer

    def test_unknown_trainer_type_lists_distillation(self):
        with pytest.raises(ValueError, match="distillation"):
            _get_trainer_cls(FakeTrainerConfig("bogus"))

    def test_task_runner_rejects_external_rollout_for_distillation(self):
        config = FakeTrainerConfig("distillation")
        config.algorithm.sample_source = "online"
        with pytest.raises(ValueError, match="sample_source=offline"):
            TaskRunner().add_actor_rollout_worker(config)

    def test_task_runner_registers_the_distillation_worker(self, monkeypatch):
        import ray
        from verl.trainer.ppo.ray_trainer import Role

        from verl_omni.workers.diffusion_distillation_worker import DiffusionDistillationWorker

        monkeypatch.setattr(ray, "remote", lambda worker_cls: worker_cls)
        runner = TaskRunner()
        worker_cls, _ = runner.add_actor_rollout_worker(FakeTrainerConfig("distillation"))
        assert worker_cls is DiffusionDistillationWorker
        assert runner.role_worker_mapping[Role.Actor] is DiffusionDistillationWorker
        assert runner.mapping[Role.Actor] == "global_pool"


class TestAlgorithmConfig:
    def test_distillation_is_a_valid_trainer_type(self):
        config = DiffusionAlgoConfig(trainer_type="distillation", sample_source="offline")
        assert config.trainer_type == "distillation"

    def test_distillation_rejects_external_rollout_sampling(self):
        with pytest.raises(ValueError, match="sample_source='offline'"):
            DiffusionAlgoConfig(trainer_type="distillation", sample_source="online")

    def test_existing_trainer_types_still_valid(self):
        assert DiffusionAlgoConfig(trainer_type="policy_gradient").trainer_type == "policy_gradient"
        assert DiffusionAlgoConfig(trainer_type="direct_preference").trainer_type == "direct_preference"

    def test_default_is_policy_gradient(self):
        assert DiffusionAlgoConfig().trainer_type == "policy_gradient"

    def test_invalid_trainer_type_raises(self):
        with pytest.raises(ValueError, match="Invalid trainer_type"):
            DiffusionAlgoConfig(trainer_type="bogus")


def runtime_config():
    return OmegaConf.create(
        {
            "algorithm": {"trainer_type": "distillation", "sample_source": "offline"},
            "actor_rollout_ref": {
                "actor": {
                    "diffusion_loss": {"loss_mode": "flow_grpo"},
                    "use_distill_loss": False,
                },
                "model": {"path": "/m"},
            },
            "distillation": {
                "enabled": False,
                "distribution_matching": {
                    "recipe": "dmd2",
                    "profile": "distribution_only",
                    "fake_update_ratio": 2,
                    "fake_warmup_cycles": 0,
                    "rollout_strategy": None,
                    "data_mode": None,
                    "export_role": "student_ema",
                },
            },
        }
    )


class TestRuntimeValidation:
    @staticmethod
    def trainer_config(*, role_storage="shared_base_adapters", strategy="fsdp2", lora_rank=8, use_orig=True):
        return OmegaConf.create(
            {
                "distillation": {"distribution_matching": {"role_storage": role_storage}},
                "actor_rollout_ref": {
                    "model": {"lora_rank": lora_rank},
                    "actor": {"strategy": strategy, "fsdp_config": {"use_orig_params": use_orig}},
                },
            }
        )

    def test_shared_base_requires_lora(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        trainer = object.__new__(DistillationRayTrainer)
        trainer.plan = build_plan("dmd2", {"model_path": "/m"}, frozenset({"distribution_matching"}))
        trainer.config = self.trainer_config(lora_rank=0)
        with pytest.raises(ValueError, match="lora_rank > 0"):
            trainer.validate_runtime_config()

    def test_fsdp1_shared_base_requires_orig_params(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        trainer = object.__new__(DistillationRayTrainer)
        trainer.plan = build_plan("dmd2", {"model_path": "/m"}, frozenset({"distribution_matching"}))
        trainer.config = self.trainer_config(strategy="fsdp", use_orig=False)
        with pytest.raises(ValueError, match="use_orig_params=true"):
            trainer.validate_runtime_config()


class TestDataPlaneBoundary:
    def test_production_constructor_reaches_explicit_pr2_boundary(self):
        config = runtime_config()
        trainer = DistillationRayTrainer(
            config=config,
            tokenizer=object(),
            processor=object(),
            role_worker_mapping={},
            resource_pool_manager=object(),
            ray_worker_group_cls=object,
            train_dataset=object(),
            val_dataset=object(),
            collate_fn=object(),
            train_sampler=object(),
        )
        assert trainer.config is config
        with pytest.raises(NotImplementedError, match="composed diffusion trainer config"):
            trainer.init_workers()

    def test_constructor_rejects_opd_switch(self):
        config = runtime_config()
        config.distillation.enabled = True
        with pytest.raises(ValueError, match="must keep the OPD"):
            DistillationRayTrainer(config=config)

    def test_config_and_capabilities_build_a_plan(self):
        config = runtime_config()
        trainer = DistillationRayTrainer(config=config, capabilities=frozenset({"distribution_matching"}))
        assert trainer.plan is not None
        assert trainer.plan.name == "dmd2"
        assert trainer.plan.role_layout.groups[0].model_ref == "/m"

    def test_fit_without_executor_reports_pr2_boundary(self):
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        plan = build_plan("dmd2", {"model_path": "/m"}, frozenset({"distribution_matching"}))
        trainer = DistillationRayTrainer(plan=plan)
        with pytest.raises(NotImplementedError, match="composed diffusion trainer config"):
            trainer.fit(num_cycles=1)

    def test_production_constructor_and_worker_lifecycle_are_wired(self, monkeypatch):
        from verl.trainer.ppo.ray_trainer import Role

        from verl_omni.trainer.diffusion.distillation.recipes import build_plan
        from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer
        from verl_omni.workers.diffusion_distillation_worker import DiffusionDistillationWorkerGroup

        config = runtime_config()
        config.trainer = {
            "device": "cuda",
            "ray_master_port_range": None,
            "n_gpus_per_node": 1,
            "nnodes": 1,
        }
        config.data = {"train_batch_size": 1}
        config.actor_rollout_ref.model.lora_rank = 8
        config.actor_rollout_ref.actor.strategy = "fsdp2"
        config.actor_rollout_ref.actor.fsdp_config = {
            "use_orig_params": False,
            "ulysses_sequence_parallel_size": 1,
        }
        config.distillation.distribution_matching.role_storage = "shared_base_adapters"
        config.distillation.distribution_matching.fake_score_optim = {"total_training_steps": -1}
        plan = build_plan(
            "dmd2",
            {"model_path": "/m", "fake_update_ratio": 2},
            frozenset({"distribution_matching"}),
        )

        monkeypatch.setattr(BaseRayDiffusionTrainer, "__init__", initialize_fake_base_trainer)
        resource_manager = SimpleNamespace(create_resource_pool=Mock(), get_resource_pool=Mock(return_value="pool"))
        worker_group = SimpleNamespace(init_model=Mock())
        worker_group_factory = Mock(return_value=worker_group)
        trainer = DistillationRayTrainer(
            config=config,
            plan=plan,
            role_worker_mapping={Role.Actor: object},
            resource_pool_manager=resource_manager,
            ray_worker_group_cls=worker_group_factory,
        )
        trainer.init_workers()
        resource_manager.create_resource_pool.assert_called_once()
        resource_manager.get_resource_pool.assert_called_once_with(Role.Actor)
        worker_group.init_model.assert_called_once()
        assert isinstance(trainer.executor, DiffusionDistillationWorkerGroup)
        assert config.distillation.distribution_matching.fake_score_optim.total_training_steps == 8

    def test_control_plane_binds_when_collaborators_are_supplied(self):
        from verl_omni.trainer.diffusion.distillation.control_plane import (
            FakeBatchProvider,
            FakePhaseExecutor,
        )
        from verl_omni.trainer.diffusion.distillation.recipes import build_plan

        plan = build_plan("dmd2", {"fake_update_ratio": 1, "model_path": "/m"}, frozenset({"distribution_matching"}))
        trainer = DistillationRayTrainer(
            plan=plan,
            executor=FakePhaseExecutor(),
            batch_provider=FakeBatchProvider(num_batches=100),
        )
        trainer.fit(num_cycles=2)
        assert trainer.control_plane.counters.global_step == 2
