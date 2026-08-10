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

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.utils import tensordict_utils as tu

from verl_omni.trainer.omni.ray_omni_trainer import OmniDirectPreferenceRayTrainer


def _make_config(**overrides):
    config = OmegaConf.create(
        {
            "algorithm": {
                "sample_source": "offline",
                "trainer_type": "direct_preference",
                "paired_preference": False,
            },
            "actor_rollout_ref": {
                "model": {
                    "model_type": "omni_model",
                    "policy_state_adapters": ("default",),
                    "use_remove_padding": False,
                    "pad_token_id": 0,
                },
                "actor": {
                    "omni_loss": {
                        "loss_mode": "dpo",
                        "average_log_prob": False,
                        "beta": 0.1,
                        "label_smoothing": 0.0,
                        "loss_type": "sigmoid",
                    },
                    "ppo_mini_batch_size": 2,
                    "ppo_micro_batch_size_per_gpu": 1,
                    "ppo_epochs": 1,
                    "data_loader_seed": 0,
                    "shuffle": True,
                },
                "rollout": {"multi_turn": {"enable": False}},
            },
            "data": {
                "train_batch_size": 2,
                "val_batch_size": 1,
                "pad_mode": "right",
                "use_dynamic_bsz": False,
                "max_token_len_per_gpu": None,
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
    return OmniDirectPreferenceRayTrainer(config)


class TestOmniDirectPreferenceRayTrainerInit:
    def test_rejects_online_sample_source(self):
        config = _make_config(algorithm={"sample_source": "online"})
        with pytest.raises(NotImplementedError, match="sample_source=offline"):
            OmniDirectPreferenceRayTrainer(config)

    def test_requires_omni_model_type(self):
        config = _make_config(actor_rollout_ref={"model": {"model_type": "language_model"}})
        with pytest.raises(ValueError, match="model_type=omni_model"):
            OmniDirectPreferenceRayTrainer(config)

    def test_requires_dpo_loss_mode(self):
        config = _make_config(
            actor_rollout_ref={"actor": {"omni_loss": {"loss_mode": "other", "average_log_prob": False}}}
        )
        with pytest.raises(NotImplementedError, match="loss_mode=dpo"):
            OmniDirectPreferenceRayTrainer(config)

    def test_rejects_old_policy_adapter(self):
        config = _make_config(actor_rollout_ref={"model": {"policy_state_adapters": ("default", "old")}})
        with pytest.raises(NotImplementedError, match="old-policy adapters"):
            OmniDirectPreferenceRayTrainer(config)


class TestOmniDirectPreferenceRayTrainerHelpers:
    def test_infer_reference_policy_maps_row_level_logps(self):
        config = _make_config()
        trainer = _make_trainer(config)
        trainer.ref_in_actor = False
        trainer.ref_policy_wg = MagicMock()
        trainer.ref_policy_wg.infer_ref_batch.return_value = TensorDict(
            {
                "log_probs": torch.tensor([[1.0, 0.5], [2.0, 1.5]]),
            },
            batch_size=2,
        )
        batch = DataProto.from_tensordict(TensorDict({"input_ids": torch.zeros(2, 4, dtype=torch.long)}, batch_size=2))

        result = trainer._infer_reference_policy(batch)

        assert result is not None
        assert result.batch["ref_log_prob"].tolist() == [[1.0, 0.5], [2.0, 1.5]]

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

    def test_validate_computes_offline_dpo_metrics_by_modality(self):
        config = _make_config(algorithm={"paired_preference": True})
        trainer = _make_trainer(config)
        trainer.ref_in_actor = False
        trainer.ref_policy_wg = MagicMock()
        trainer.actor_rollout_wg = MagicMock()
        trainer.val_dataloader = [
            _batch_dict("image", 10, 11),
            _batch_dict("video", 12, 13),
            _batch_dict("image", 14, 15),
        ]

        trainer.ref_policy_wg.infer_ref_batch.side_effect = _infer_ref_log_probs
        trainer.actor_rollout_wg.infer_actor_batch.side_effect = _infer_actor_log_probs

        metrics = trainer._validate()

        assert metrics["val/num_samples"] == 3
        assert metrics["val/reward_accuracy"] == pytest.approx(2 / 3)
        assert metrics["val/reward_margin"] == pytest.approx(1 / 30)
        assert metrics["val/image/num_samples"] == 2
        assert metrics["val/image/reward_accuracy"] == pytest.approx(1.0)
        assert metrics["val/video/num_samples"] == 1
        assert metrics["val/video/reward_accuracy"] == pytest.approx(0.0)


def _batch_dict(modality: str, chosen_token: int, rejected_token: int):
    return {
        "modality": np.array([modality, modality], dtype=object),
        "extra_info": np.array([{"modality": modality}, {"modality": modality}], dtype=object),
        "input_ids": torch.tensor([[1, chosen_token], [1, rejected_token]]),
        "labels": torch.tensor([[-100, chosen_token], [-100, rejected_token]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "sample_level_scores": torch.tensor([[1.0], [0.0]]),
    }


def _infer_ref_log_probs(data):
    labels = data["labels"]
    return TensorDict({"log_probs": torch.zeros_like(labels, dtype=torch.float32)}, batch_size=[labels.shape[0]])


def _infer_actor_log_probs(data):
    labels = data["labels"]
    log_probs = torch.zeros_like(labels, dtype=torch.float32)
    for row, answer_token in enumerate(labels[:, 1].tolist()):
        if answer_token in {10, 14}:
            log_probs[row, 0] = 1.0
        elif answer_token == 11:
            log_probs[row, 0] = 0.0
        elif answer_token == 12:
            log_probs[row, 0] = 0.0
        elif answer_token == 13:
            log_probs[row, 0] = 1.0
    return TensorDict({"log_probs": log_probs}, batch_size=[labels.shape[0]])
