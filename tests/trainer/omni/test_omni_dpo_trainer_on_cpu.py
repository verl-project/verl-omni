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
"""CPU tests for OmniDirectPreferenceRayTrainer guardrails and helpers."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.diffusion.ray_diffusion_trainer import BaseRayDiffusionTrainer
from verl_omni.trainer.omni.ray_omni_dpo_trainer import OmniDirectPreferenceRayTrainer


def _make_config(**overrides):
    config = OmegaConf.create(
        {
            "algorithm": {
                "sample_source": "offline",
                "trainer_type": "direct_preference",
                "paired_preference": False,
            },
            "actor_rollout_ref": {
                "model": {"model_type": "omni", "policy_state_adapters": ("default",)},
                "actor": {
                    "omni_loss": {"loss_mode": "dpo", "average_log_prob": False},
                    "ppo_mini_batch_size": 2,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": True,
                },
                "rollout": {"multi_turn": {"enable": False}},
            },
        }
    )
    OmegaConf.set_struct(config, False)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and OmegaConf.is_dict(config[key]):
            config[key] = OmegaConf.merge(config[key], value)
        else:
            config[key] = value
    return config


def _make_trainer(config):
    with patch.object(BaseRayDiffusionTrainer, "__init__", return_value=None):
        trainer = OmniDirectPreferenceRayTrainer(config)
    trainer.config = config
    return trainer


class TestOmniDirectPreferenceRayTrainerInit:
    def test_rejects_online_sample_source(self):
        config = _make_config(algorithm={"sample_source": "online"})
        with patch.object(BaseRayDiffusionTrainer, "__init__", return_value=None):
            with pytest.raises(NotImplementedError, match="sample_source=offline"):
                OmniDirectPreferenceRayTrainer(config)

    def test_requires_omni_model_type(self):
        config = _make_config(actor_rollout_ref={"model": {"model_type": "language_model"}})
        with patch.object(BaseRayDiffusionTrainer, "__init__", return_value=None):
            with pytest.raises(ValueError, match="model_type=omni"):
                OmniDirectPreferenceRayTrainer(config)

    def test_requires_dpo_loss_mode(self):
        config = _make_config(
            actor_rollout_ref={"actor": {"omni_loss": {"loss_mode": "other", "average_log_prob": False}}}
        )
        with patch.object(BaseRayDiffusionTrainer, "__init__", return_value=None):
            with pytest.raises(NotImplementedError, match="loss_mode=dpo"):
                OmniDirectPreferenceRayTrainer(config)

    def test_rejects_old_policy_adapter(self):
        config = _make_config(actor_rollout_ref={"model": {"policy_state_adapters": ("default", "old")}})
        with patch.object(BaseRayDiffusionTrainer, "__init__", return_value=None):
            with pytest.raises(NotImplementedError, match="old-policy adapters"):
                OmniDirectPreferenceRayTrainer(config)


class TestOmniDirectPreferenceRayTrainerHelpers:
    def test_infer_reference_policy_maps_chosen_and_rejected_logps(self):
        config = _make_config()
        trainer = _make_trainer(config)
        trainer.ref_in_actor = False
        trainer.ref_policy_wg = MagicMock()
        trainer.ref_policy_wg.infer_ref_batch.return_value = TensorDict(
            {
                "chosen_logps": torch.tensor([1.0, 2.0]),
                "rejected_logps": torch.tensor([0.5, 1.5]),
            },
            batch_size=2,
        )
        batch = DataProto.from_tensordict(TensorDict({"input_ids": torch.zeros(2, 4, dtype=torch.long)}, batch_size=2))

        result = trainer._infer_reference_policy(batch)

        assert result is not None
        assert result.batch["reference_chosen_logps"].tolist() == [1.0, 2.0]
        assert result.batch["reference_rejected_logps"].tolist() == [0.5, 1.5]

    def test_update_actor_disables_shuffle_for_paired_preference(self):
        config = _make_config(
            algorithm={"paired_preference": True},
            actor_rollout_ref={"actor": {"shuffle": True}},
        )
        trainer = _make_trainer(config)
        trainer.actor_rollout_wg = MagicMock()
        trainer.actor_rollout_wg.update_actor.return_value = TensorDict({"metrics": {}}, batch_size=0)
        batch = DataProto.from_single_dict(
            data={"input_ids": torch.zeros(2, 4, dtype=torch.long)},
            meta_info={},
        )

        with pytest.warns(UserWarning, match="Shuffle is not supported"):
            trainer._update_actor(batch)

        sent_batch = trainer.actor_rollout_wg.update_actor.call_args.args[0]
        dataloader_kwargs = tu.get_non_tensor_data(sent_batch, "dataloader_kwargs", default={})
        assert dataloader_kwargs["shuffle"] is False
