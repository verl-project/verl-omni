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

"""Diffusion-specific algorithm config additions for verl_omni."""

from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig

__all__ = ["DiffusionNFTAlgoConfig", "DiffusionAlgoConfig"]


@dataclass
class DiffusionNFTAlgoConfig(BaseConfig):
    """DiffusionNFT-specific algorithm controls."""

    mix_beta: float = 0.5
    ref_kl_coef: float = 0.0
    old_policy_decay_type: int = 0
    old_policy_decay: Optional[float] = None
    old_policy_update_interval: int = 1
    timestep_fraction: float = 1.0
    adv_clip_max: float = 5.0
    adv_mode: str = "continuous"
    adaptive_weight_min: float = 1e-5
    rollout_adapter: str = "old"
    collect_mode: str = "final_latent"

    def __post_init__(self):
        valid_adv_modes = {"continuous", "positive_only", "negative_only", "one_only", "binary"}
        if self.adv_mode not in valid_adv_modes:
            raise ValueError(
                f"Invalid DiffusionNFT adv_mode: {self.adv_mode}. Must be one of {sorted(valid_adv_modes)}"
            )
        if self.mix_beta <= 0:
            raise ValueError(f"DiffusionNFT mix_beta must be positive, got {self.mix_beta}.")
        if self.adv_clip_max <= 0:
            raise ValueError(f"DiffusionNFT adv_clip_max must be positive, got {self.adv_clip_max}.")
        if self.old_policy_update_interval <= 0:
            raise ValueError(
                f"DiffusionNFT old_policy_update_interval must be positive, got {self.old_policy_update_interval}."
            )
        if not 0 < self.timestep_fraction <= 1:
            raise ValueError(f"DiffusionNFT timestep_fraction must be in (0, 1], got {self.timestep_fraction}.")


@dataclass
class DiffusionAlgoConfig(BaseConfig):
    """Diffusion-specific algorithm config."""

    trainer_type: str = "policy_gradient"
    sample_source: str = "online"
    adv_estimator: str = "flow_grpo"
    norm_adv_by_std_in_grpo: bool = True
    bypass_mode: bool = False
    global_std: bool = True
    diffusion_nft: DiffusionNFTAlgoConfig = field(default_factory=DiffusionNFTAlgoConfig)
