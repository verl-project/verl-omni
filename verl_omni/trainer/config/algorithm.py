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
from verl.trainer.config.algorithm import RolloutCorrectionConfig

__all__ = ["DiffusionAlgoConfig", "RolloutCorrectionConfig"]


@dataclass
class DiffusionAlgoConfig(BaseConfig):
    """Diffusion-specific algorithm config."""

    trainer_type: str = "policy_gradient"
    sample_source: str = "online"
    adv_estimator: str = "flow_grpo"
    norm_adv_by_std_in_grpo: bool = True
    global_std: bool = True
    old_policy_decay_type: int = 0
    old_policy_decay: Optional[float] = None
    old_policy_update_interval: int = 1
    timestep_fraction: float = 1.0
    adv_mode: str = "continuous"
    paired_preference: bool = False  # True for pair-based algorithms (e.g. DPO)
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    def __post_init__(self):
        valid_adv_modes = {"continuous", "positive_only", "negative_only", "one_only", "binary"}
        if self.adv_mode not in valid_adv_modes:
            raise ValueError(f"Invalid adv_mode: {self.adv_mode}. Must be one of {sorted(valid_adv_modes)}")
        if self.old_policy_decay is not None and not 0 <= self.old_policy_decay <= 1:
            raise ValueError(f"old_policy_decay must be in [0, 1], got {self.old_policy_decay}.")
        if self.old_policy_update_interval <= 0:
            raise ValueError(
                f"old_policy_update_interval must be positive, got {self.old_policy_update_interval}."
            )
        if not 0 < self.timestep_fraction <= 1:
            raise ValueError(f"timestep_fraction must be in (0, 1], got {self.timestep_fraction}.")
