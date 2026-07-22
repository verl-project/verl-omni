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
"""CPU tests for omni training entrypoint routing."""

import pytest
from omegaconf import OmegaConf

from verl_omni.trainer.diffusion.ray_diffusion_trainer import DirectPreferenceRayTrainer, PolicyGradientRayTrainer
from verl_omni.trainer.main_omni import get_ray_trainer_cls, uses_v1_trainer
from verl_omni.trainer.omni.ray_omni_dpo_trainer import OmniDirectPreferenceRayTrainer


def _make_config(**overrides):
    config = OmegaConf.create(
        {
            "algorithm": {
                "sample_source": "online",
                "trainer_type": "policy_gradient",
            },
            "actor_rollout_ref": {
                "model": {"model_type": "omni"},
            },
            "trainer": {"use_v1": None},
        }
    )
    OmegaConf.set_struct(config, False)
    for key, value in overrides.items():
        if isinstance(value, dict) and key in config and OmegaConf.is_dict(config[key]):
            config[key] = OmegaConf.merge(config[key], value)
        else:
            config[key] = value
    return config


class TestUsesV1Trainer:
    @pytest.mark.parametrize(
        ("sample_source", "trainer_type", "use_v1", "expected"),
        [
            ("online", "policy_gradient", None, True),
            ("offline", "direct_preference", None, False),
            ("offline", "direct_preference", False, False),
            ("online", "direct_preference", None, True),
        ],
    )
    def test_routes_online_rl_to_v1_and_offline_preference_to_legacy(
        self, sample_source, trainer_type, use_v1, expected
    ):
        config = _make_config(
            algorithm={"sample_source": sample_source, "trainer_type": trainer_type},
            trainer={"use_v1": use_v1},
        )
        assert uses_v1_trainer(config) is expected


class TestGetRayTrainerCls:
    def test_policy_gradient_returns_policy_gradient_trainer(self):
        config = _make_config(algorithm={"trainer_type": "policy_gradient"})
        assert get_ray_trainer_cls(config) is PolicyGradientRayTrainer

    def test_direct_preference_omni_returns_omni_trainer(self):
        config = _make_config(
            algorithm={"trainer_type": "direct_preference"},
            actor_rollout_ref={"model": {"model_type": "omni"}},
        )
        assert get_ray_trainer_cls(config) is OmniDirectPreferenceRayTrainer

    def test_direct_preference_non_omni_returns_diffusion_trainer(self):
        config = _make_config(
            algorithm={"trainer_type": "direct_preference"},
            actor_rollout_ref={"model": {"model_type": "diffusion_model"}},
        )
        assert get_ray_trainer_cls(config) is DirectPreferenceRayTrainer

    def test_unsupported_trainer_type_raises(self):
        config = _make_config(algorithm={"trainer_type": "unknown"})
        with pytest.raises(ValueError, match="Unsupported trainer_type"):
            get_ray_trainer_cls(config)
