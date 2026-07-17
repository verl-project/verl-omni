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
"""CPU tests for omni trainer config dataclasses."""

import pytest

from verl_omni.trainer.config.algorithm import OmniAlgoConfig
from verl_omni.workers.config.omni import OmniActorConfig, OmniLossConfig


class TestOmniAlgoConfig:
    def test_defaults(self):
        cfg = OmniAlgoConfig()
        assert cfg.trainer_type == "direct_preference"
        assert cfg.sample_source == "offline"
        assert cfg.paired_preference is True
        assert cfg.adv_estimator == "dpo"
        assert cfg.norm_adv_by_std_in_grpo is True
        assert cfg.global_std is True

    @pytest.mark.parametrize(
        "field_name, value",
        [
            ("trainer_type", "invalid"),
            ("sample_source", "invalid"),
        ],
    )
    def test_invalid_values_raise(self, field_name, value):
        with pytest.raises(ValueError):
            OmniAlgoConfig(**{field_name: value})


class TestOmniLossConfig:
    def test_defaults(self):
        cfg = OmniLossConfig()
        assert cfg.loss_mode == "dpo"
        assert cfg.beta == pytest.approx(0.1)
        assert cfg.label_smoothing == pytest.approx(0.0)
        assert cfg.loss_type == "sigmoid"
        assert cfg.average_log_prob is False
        assert cfg.refer_model_precision == "bfloat16"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"loss_mode": "flow_grpo"},
            {"loss_type": "invalid"},
            {"beta": 0.0},
        ],
    )
    def test_invalid_values_raise(self, kwargs):
        with pytest.raises(ValueError):
            OmniLossConfig(**kwargs)


class TestOmniActorConfig:
    def test_includes_omni_loss(self):
        cfg = OmniActorConfig(
            strategy="fsdp",
            rollout_n=1,
            ppo_micro_batch_size_per_gpu=1,
        )
        assert isinstance(cfg.omni_loss, OmniLossConfig)
        assert cfg.omni_loss.loss_mode == "dpo"
        assert cfg.trainer_type == "direct_preference"

    def test_rejects_policy_gradient_trainer_type(self):
        with pytest.raises(ValueError, match="OmniActorConfig is only valid"):
            OmniActorConfig(
                strategy="fsdp",
                rollout_n=1,
                ppo_micro_batch_size_per_gpu=1,
                trainer_type="policy_gradient",
            )

    def test_instantiate_via_hydra(self):
        import os

        from hydra import compose, initialize_config_dir
        from verl.utils.config import omega_conf_to_dataclass

        import verl_omni

        config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="omni_trainer",
                overrides=[
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                    "actor_rollout_ref.actor.omni_loss.beta=0.2",
                ],
            )
        actor_cfg: OmniActorConfig = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)

        assert isinstance(actor_cfg, OmniActorConfig)
        assert isinstance(actor_cfg.omni_loss, OmniLossConfig)
        assert actor_cfg.omni_loss.beta == pytest.approx(0.2)
