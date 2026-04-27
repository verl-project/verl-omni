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
"""CPU tests for verl_omni worker config dataclasses."""

import pytest

from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig
from verl_omni.workers.config.diffusion.actor import (
    DiffusionLossConfig,
    FSDPDiffusionActorConfig,
)
from verl_omni.workers.config.diffusion.rollout import (
    DiffusionRolloutAlgoConfig,
    DiffusionSamplingConfig,
)

# ---------------------------------------------------------------------------
# DiffusionLossConfig
# ---------------------------------------------------------------------------


class TestDiffusionLossConfig:
    def test_defaults(self):
        cfg = DiffusionLossConfig()
        assert cfg.loss_mode == "flow_grpo"
        assert cfg.clip_ratio == pytest.approx(0.0001)
        assert cfg.adv_clip_max == pytest.approx(5.0)

    def test_custom_values(self):
        cfg = DiffusionLossConfig(loss_mode="flow_grpo", clip_ratio=0.01, adv_clip_max=10.0)
        assert cfg.clip_ratio == pytest.approx(0.01)
        assert cfg.adv_clip_max == pytest.approx(10.0)

    def test_invalid_loss_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid diffusion loss_mode"):
            DiffusionLossConfig(loss_mode="not_a_valid_mode")


# ---------------------------------------------------------------------------
# DiffusionAlgoConfig
# ---------------------------------------------------------------------------


class TestDiffusionAlgoConfig:
    def test_defaults(self):
        cfg = DiffusionAlgoConfig()
        assert cfg.adv_estimator == "flow_grpo"
        assert cfg.norm_adv_by_std_in_grpo is True
        assert cfg.bypass_mode is False
        assert cfg.global_std is True

    def test_override(self):
        cfg = DiffusionAlgoConfig(norm_adv_by_std_in_grpo=False, global_std=False)
        assert cfg.norm_adv_by_std_in_grpo is False
        assert cfg.global_std is False


# ---------------------------------------------------------------------------
# DiffusionRolloutAlgoConfig / DiffusionSamplingConfig
# ---------------------------------------------------------------------------


class TestDiffusionRolloutAlgoConfig:
    def test_defaults(self):
        cfg = DiffusionRolloutAlgoConfig()
        assert cfg.noise_level == pytest.approx(1.0)
        assert cfg.sde_type == "sde"
        assert cfg.sde_window_size is None

    def test_sde_window_range_default(self):
        cfg = DiffusionRolloutAlgoConfig()
        assert cfg.sde_window_range == [0, 5]


class TestDiffusionSamplingConfig:
    def test_defaults(self):
        cfg = DiffusionSamplingConfig()
        assert cfg.num_inference_steps == 40
        assert cfg.seed == 42
        assert isinstance(cfg.algo, DiffusionRolloutAlgoConfig)


# ---------------------------------------------------------------------------
# FSDPDiffusionActorConfig (instantiation via Hydra / omega_conf)
# ---------------------------------------------------------------------------


class TestFSDPDiffusionActorConfig:
    def test_instantiate_via_hydra(self):
        import os

        from hydra import compose, initialize_config_dir
        from verl.utils.config import omega_conf_to_dataclass

        import verl_omni

        config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="dp_diffusion_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=4",
                ],
            )
        actor_cfg: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)

        assert actor_cfg.strategy == "fsdp"
        assert actor_cfg.ppo_micro_batch_size_per_gpu == 4
        assert isinstance(actor_cfg.diffusion_loss, DiffusionLossConfig)

    def test_engine_strategy_synced(self):
        """After __post_init__, engine.strategy must mirror actor.strategy."""
        import os

        from hydra import compose, initialize_config_dir
        from verl.utils.config import omega_conf_to_dataclass

        import verl_omni

        config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="dp_diffusion_actor",
                overrides=[
                    "strategy=fsdp2",
                    "ppo_micro_batch_size_per_gpu=4",
                ],
            )
        actor_cfg: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)
        assert actor_cfg.engine.strategy == "fsdp2"

    def test_loss_config_clip_ratio_respected(self):
        import os

        from hydra import compose, initialize_config_dir
        from verl.utils.config import omega_conf_to_dataclass

        import verl_omni

        config_dir = os.path.join(os.path.dirname(verl_omni.__file__), "trainer/config/diffusion/actor")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(
                config_name="dp_diffusion_actor",
                overrides=[
                    "strategy=fsdp",
                    "ppo_micro_batch_size_per_gpu=4",
                    "diffusion_loss.clip_ratio=0.05",
                ],
            )
        actor_cfg: FSDPDiffusionActorConfig = omega_conf_to_dataclass(cfg)
        assert actor_cfg.diffusion_loss.clip_ratio == pytest.approx(0.05)
