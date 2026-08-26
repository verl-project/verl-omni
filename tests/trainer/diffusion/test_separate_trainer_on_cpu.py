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
"""Focused CPU tests for synchronous separate diffusion configuration."""

import copy
import os
import sys
import types

import pytest
import verl.trainer.ppo.ray_trainer as upstream_ray_trainer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import Role

from verl_omni.trainer import main_diffusion
from verl_omni.trainer.diffusion import ray_diffusion_trainer


def _compose_config(*overrides):
    config_dir = os.path.abspath("verl_omni/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="diffusion_trainer", overrides=list(overrides))


@pytest.fixture
def separate_config():
    return _compose_config(
        "actor_rollout_ref.separate=true",
        "actor_rollout_ref.hybrid_engine=false",
        "actor_rollout_ref.rollout.nnodes=1",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
    )


def test_separate_config_defaults_to_colocated(separate_config):
    config = _compose_config("algorithm.trainer_type=direct_preference", "algorithm.sample_source=offline")
    assert config.actor_rollout_ref.separate is False
    assert separate_config.actor_rollout_ref.separate is True
    ray_diffusion_trainer.validate_separate_config(config)
    ray_diffusion_trainer.validate_separate_config(separate_config)


def test_config_without_separate_field_uses_colocated_mapping(monkeypatch):
    class FakeWorker:
        pass

    config = _compose_config()
    del config.actor_rollout_ref.separate

    engine_workers = types.ModuleType("verl_omni.workers.engine_workers")
    engine_workers.ActorRolloutRefWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "verl_omni.workers.engine_workers", engine_workers)
    monkeypatch.setattr(main_diffusion.ray, "remote", lambda cls: cls)

    ray_diffusion_trainer.validate_separate_config(config)
    runner = main_diffusion.TaskRunner()
    runner.add_actor_rollout_worker(config)

    assert set(runner.role_worker_mapping) == {Role.ActorRollout}
    assert runner.mapping == {Role.ActorRollout: "global_pool"}


def test_run_diffusion_validates_before_ray_initialization(monkeypatch, separate_config):
    separate_config.actor_rollout_ref.hybrid_engine = True
    monkeypatch.setattr(
        main_diffusion.ray,
        "is_initialized",
        lambda: pytest.fail("Ray must not be inspected before separate config validation"),
    )

    with pytest.raises(ValueError, match="actor_rollout_ref.hybrid_engine=false"):
        main_diffusion.run_diffusion(separate_config)


def test_task_runner_rejects_missing_actor_role_support(monkeypatch, separate_config):
    class FakeWorker:
        pass

    class RoleWithoutActor:
        pass

    engine_workers = types.ModuleType("verl_omni.workers.engine_workers")
    engine_workers.ActorRolloutRefWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "verl_omni.workers.engine_workers", engine_workers)
    monkeypatch.setattr(upstream_ray_trainer, "Role", RoleWithoutActor)

    with pytest.raises(ValueError, match="Separate training without colocated rollout requires verl Role.Actor"):
        main_diffusion.TaskRunner().add_actor_rollout_worker(separate_config)


def test_separate_trainer_rejects_missing_actor_mapping(separate_config):
    trainer = object.__new__(ray_diffusion_trainer.PolicyGradientRayTrainer)

    with pytest.raises(ValueError, match="role_worker_mapping to contain Role.Actor"):
        ray_diffusion_trainer.BaseRayDiffusionTrainer.__init__(
            trainer,
            config=separate_config,
            tokenizer=None,
            role_worker_mapping={},
            resource_pool_manager=None,
        )


def test_separate_trainer_rejects_missing_reference_mapping(separate_config):
    config = copy.deepcopy(separate_config)
    config.actor_rollout_ref.actor.use_kl_loss = True
    trainer = object.__new__(ray_diffusion_trainer.PolicyGradientRayTrainer)

    with pytest.raises(ValueError, match="role_worker_mapping to contain Role.RefPolicy"):
        ray_diffusion_trainer.BaseRayDiffusionTrainer.__init__(
            trainer,
            config=config,
            tokenizer=None,
            role_worker_mapping={Role.Actor: object()},
            resource_pool_manager=None,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("actor_rollout_ref.model.lora.rank", 8),
        ("actor_rollout_ref.model.lora_rank", 8),
        ("actor_rollout_ref.model.lora_adapter_path", "/tmp/adapter"),
    ],
)
def test_lora_separate_fails_before_ray_initialization(monkeypatch, separate_config, path, value):
    config = copy.deepcopy(separate_config)
    OmegaConf.update(config, path, value, force_add=True)
    monkeypatch.setattr(
        main_diffusion.ray,
        "is_initialized",
        lambda: pytest.fail("Ray must not be inspected before separate LoRA validation"),
    )

    with pytest.raises(ValueError, match="Separate mode currently supports full finetuning only"):
        main_diffusion.run_diffusion(config)


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("actor_rollout_ref.separate", "not-a-bool", "actor_rollout_ref.separate must be a bool"),
        ("algorithm.trainer_type", "direct_preference", "algorithm.trainer_type='policy_gradient'"),
        ("algorithm.sample_source", "offline", "algorithm.sample_source='online'"),
        ("actor_rollout_ref.hybrid_engine", True, "actor_rollout_ref.hybrid_engine=false"),
        ("actor_rollout_ref.rollout.nnodes", 0, "positive rollout nnodes and n_gpus_per_node"),
        ("actor_rollout_ref.rollout.n_gpus_per_node", 0, "positive rollout nnodes and n_gpus_per_node"),
        (
            "actor_rollout_ref.rollout.checkpoint_engine.backend",
            "naive",
            "non-naive checkpoint engine backend",
        ),
    ],
)
def test_invalid_separate_config_fails_with_named_path(separate_config, path, value, match):
    config = copy.deepcopy(separate_config)
    OmegaConf.update(config, path, value)
    with pytest.raises(ValueError, match=match):
        ray_diffusion_trainer.validate_separate_config(config)


def test_task_runner_maps_separate_actor_ref_and_reward_to_global_pool(monkeypatch, separate_config):
    class FakeWorker:
        pass

    engine_workers = types.ModuleType("verl_omni.workers.engine_workers")
    engine_workers.ActorRolloutRefWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "verl_omni.workers.engine_workers", engine_workers)
    monkeypatch.setattr(main_diffusion.ray, "remote", lambda cls: cls)

    separate_config.actor_rollout_ref.actor.use_kl_loss = True
    separate_config.reward.reward_model.enable = True
    separate_config.reward.reward_model.enable_resource_pool = False

    runner = main_diffusion.TaskRunner()
    actor_cls, _ = runner.add_actor_rollout_worker(separate_config)
    runner.add_reward_model_resource_pool(separate_config)
    runner.add_ref_policy_worker(separate_config, actor_cls)
    pool_manager = runner.init_resource_pool_mgr(separate_config)

    assert set(runner.role_worker_mapping) == {Role.Actor, Role.RefPolicy}
    assert runner.mapping == {
        Role.Actor: "global_pool",
        Role.RefPolicy: "global_pool",
        Role.RewardModel: "global_pool",
    }
    assert pool_manager.resource_pool_spec == {"global_pool": [separate_config.trainer.n_gpus_per_node]}


@pytest.mark.parametrize(
    ("separate", "expected_worker_group", "expected_pool", "expected_sleep_count"),
    [
        (False, "actor_wg", "actor_pool", 1),
        (True, None, None, 0),
    ],
)
def test_rollout_stack_selects_manager_arguments_and_sleep(
    monkeypatch, separate_config, separate, expected_worker_group, expected_pool, expected_sleep_count
):
    events = []
    replicas = [object()]

    class FakeRewardLoopManager:
        def __init__(self, **kwargs):
            events.append(("reward", kwargs))
            self.reward_loop_workers = []

    class FakeLLMServerManager:
        @classmethod
        def create(cls, **kwargs):
            events.append(("llm", kwargs))
            return cls()

        def get_client(self):
            return "client"

        def get_replicas(self):
            return replicas

    class FakeAgentLoopManager:
        @classmethod
        def create(cls, **kwargs):
            events.append(("agent", kwargs))
            return cls()

    class FakeCheckpointManager:
        def __init__(self, **kwargs):
            events.append(("checkpoint", kwargs))
            self.sleep_count = 0

        def sleep_replicas(self):
            self.sleep_count += 1

    reward_loop = types.ModuleType("verl_omni.reward_loop")
    reward_loop.OmniRewardLoopManager = FakeRewardLoopManager
    monkeypatch.setitem(sys.modules, "verl_omni.reward_loop", reward_loop)
    monkeypatch.setattr(ray_diffusion_trainer, "LLMServerManager", FakeLLMServerManager)
    monkeypatch.setattr(ray_diffusion_trainer, "CheckpointEngineManager", FakeCheckpointManager)
    monkeypatch.setattr(ray_diffusion_trainer, "load_class_from_fqn", lambda *_args: FakeAgentLoopManager)
    monkeypatch.setattr(ray_diffusion_trainer, "omega_conf_to_dataclass", lambda value: value)

    config = copy.deepcopy(separate_config) if separate else _compose_config()
    OmegaConf.update(
        config,
        "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
        "fake.AgentLoopManager",
        force_add=True,
    )
    trainer = object.__new__(ray_diffusion_trainer.PolicyGradientRayTrainer)
    trainer.config = config
    trainer.separate = separate
    trainer.use_rm = False
    trainer.actor_rollout_wg = "actor_wg"

    trainer._init_online_rollout_stack("actor_pool")

    assert [event[0] for event in events[:4]] == ["reward", "llm", "agent", "checkpoint"]
    assert events[1][1]["worker_group"] == expected_worker_group
    assert events[1][1]["rollout_resource_pool"] == expected_pool
    assert trainer.checkpoint_manager.sleep_count == expected_sleep_count
    assert events[3][1]["replicas"] is replicas
