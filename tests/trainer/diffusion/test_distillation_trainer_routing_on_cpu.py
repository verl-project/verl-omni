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

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.trainer.diffusion.distillation.ray_trainer import DistillationRayTrainer
from verl_omni.trainer.main_diffusion import _get_trainer_cls


class FakeAlgorithmConfig:
    def __init__(self, trainer_type):
        self.trainer_type = trainer_type


class FakeTrainerConfig:
    def __init__(self, trainer_type):
        self.algorithm = FakeAlgorithmConfig(trainer_type)


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


class TestAlgorithmConfig:
    def test_distillation_is_a_valid_trainer_type(self):
        config = DiffusionAlgoConfig(trainer_type="distillation")
        assert config.trainer_type == "distillation"

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
            "algorithm": {"trainer_type": "distillation"},
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


class TestPR1DataPlaneBoundary:
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
        with pytest.raises(NotImplementedError, match="PR 2"):
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
        with pytest.raises(NotImplementedError, match="PR 2"):
            trainer.fit(num_cycles=1)

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
