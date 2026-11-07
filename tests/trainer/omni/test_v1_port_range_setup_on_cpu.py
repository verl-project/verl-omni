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
"""CPU tests for the omni v1 trainers' rendezvous port-range wiring (OmniPPOTrainer)."""

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from verl.trainer.ppo.utils import Role
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync

from verl_omni.trainer.omni import ray_omni_trainer_separate_async
from verl_omni.trainer.omni import trainer_base as omni_trainer_base
from verl_omni.trainer.omni.ray_omni_trainer import OmniPPOTrainerSync
from verl_omni.trainer.omni.ray_omni_trainer_separate_async import OmniPPOTrainerSeparateAsync
from verl_omni.trainer.omni.trainer_base import OmniPPOTrainer


class _RecordingWorkerGroup:
    created: list["_RecordingWorkerGroup"] = []

    def __init__(self, *, resource_pool, ray_cls_with_init, **kwargs):
        self.resource_pool = resource_pool
        self.kwargs = kwargs
        _RecordingWorkerGroup.created.append(self)

    def spawn(self, prefix_set):
        return {role: self for role in prefix_set}

    def init_model(self):
        pass

    def reset(self):
        pass

    def set_loss_fn(self, loss_fn):
        pass


class _FakeServerSide:
    rollout_replicas: list = []

    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def create(cls, *args, **kwargs):
        return cls()

    def get_replicas(self):
        return []

    def sleep_replicas(self):
        pass


class _FakePoolManager:
    def __init__(self):
        self.resource_pool_dict = {"actor_group": "pool_actor", "empty_group": "pool_empty", "critic": "pool_critic"}
        self.created = False

    def create_resource_pool(self):
        self.created = True

    def get_resource_pool(self, role):
        return "pool_critic" if role == Role.Critic else "pool_actor"


class _SetupHarness(OmniPPOTrainer):
    def __init__(self, config, use_critic=False):
        self.config = config
        self.resource_pool_manager = _FakePoolManager()
        self.role_worker_mapping = {Role.ActorRolloutRef: object}
        if use_critic:
            self.role_worker_mapping[Role.Critic] = object
        self.use_critic = use_critic
        self.use_reference_policy = False
        self.use_teacher_policy = False
        self.checkpoint_loaded = False

    def _init_tokenizer(self):
        pass

    def _init_dataloader(self):
        pass

    def _init_dump_executor(self):
        pass

    def _init_resource_pool_mgr(self):
        pass

    def _load_checkpoint(self):
        self.checkpoint_loaded = True

    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


def _config(port_range=None):
    config = OmegaConf.create(
        {
            "trainer": {"device": "cpu"},
            "actor_rollout_ref": {
                "model": {},
                "rollout": {"checkpoint_engine": {"backend": "naive"}, "prometheus": {"enable": False}},
            },
            "reward": {"reward_model": {"enable": False}},
            "global_profiler": {"steps": None},
            "critic": {
                "model": {},
                "engine": {},
                "optim": {},
                "checkpoint": {},
                "ppo_infer_max_token_len_per_gpu": 8,
                "ppo_max_token_len_per_gpu": 8,
            },
        }
    )
    if port_range is not None:
        config.trainer.ray_master_port_range = port_range
    return config


@pytest.fixture()
def stubbed_setup(monkeypatch):
    _RecordingWorkerGroup.created = []
    monkeypatch.setattr(omni_trainer_base, "RayWorkerGroup", _RecordingWorkerGroup)
    monkeypatch.setattr(omni_trainer_base, "create_colocated_worker_cls", lambda class_dict: class_dict)
    monkeypatch.setattr(omni_trainer_base, "RayClassWithInitArgs", lambda **kwargs: kwargs)
    monkeypatch.setattr(omni_trainer_base, "RewardLoopManager", _FakeServerSide)
    monkeypatch.setattr(omni_trainer_base, "LLMServerManager", _FakeServerSide)
    monkeypatch.setattr(omni_trainer_base, "CheckpointEngineManager", _FakeServerSide)
    monkeypatch.setattr(omni_trainer_base, "omega_conf_to_dataclass", lambda cfg: cfg)
    monkeypatch.setattr(omni_trainer_base, "value_loss", lambda *args, **kwargs: None)
    return _RecordingWorkerGroup


def test_setup_slices_disjoint_ranges_across_non_empty_groups(stubbed_setup):
    harness = _SetupHarness(_config(port_range=[21000, 22000]), use_critic=True)
    OmniPPOTrainer._setup(harness)

    assert [wg.resource_pool for wg in stubbed_setup.created] == ["pool_actor", "pool_critic"]
    assert [wg.kwargs["master_port_range"] for wg in stubbed_setup.created] == [[21000, 21500], [21500, 22000]]
    assert harness.checkpoint_loaded


def test_setup_without_range_passes_no_master_port_range(stubbed_setup):
    harness = _SetupHarness(_config(), use_critic=True)
    OmniPPOTrainer._setup(harness)

    assert len(stubbed_setup.created) == 2
    assert all("master_port_range" not in wg.kwargs for wg in stubbed_setup.created)


def test_separate_async_setup_gives_groups_sub_ranges_and_lora_manager(stubbed_setup, monkeypatch):
    monkeypatch.setattr(ray_omni_trainer_separate_async, "LLMServerManager", _FakeServerSide)
    monkeypatch.setattr(ray_omni_trainer_separate_async, "OmniCheckpointEngineManager", MagicMock())
    monkeypatch.setattr(ray_omni_trainer_separate_async, "omega_conf_to_dataclass", lambda cfg: cfg)

    class _SeparateAsyncHarness(_SetupHarness, OmniPPOTrainerSeparateAsync):
        def __init__(self, config):
            super().__init__(config)
            self.llm_server_manager = MagicMock()
            self.llm_server_manager.rollout_replicas = []

        def add_replicas_to_balancer(self):
            pass

    harness = _SeparateAsyncHarness(_config(port_range=[21000, 22000]))
    harness._setup()

    [wg] = stubbed_setup.created
    lo, hi = wg.kwargs["master_port_range"]
    assert 21000 <= lo < hi <= 22000
    assert harness.standalone_checkpoint_manager is not None
    assert harness.current_mode is not None


def test_omni_v1_trainers_route_setup_through_the_base():
    assert OmniPPOTrainerSync._setup is OmniPPOTrainer._setup
    assert "_setup" in OmniPPOTrainerSeparateAsync.__dict__  # copied standalone tail
    for cls in (OmniPPOTrainerSync, OmniPPOTrainerSeparateAsync):
        mro = cls.__mro__
        assert mro.index(OmniPPOTrainer) < mro.index(PPOTrainer), cls
    sep_mro = OmniPPOTrainerSeparateAsync.__mro__
    assert sep_mro.index(OmniPPOTrainer) < sep_mro.index(PPOTrainerSeparateAsync)
