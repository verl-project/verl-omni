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
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)

    # NFT-specific config
    nft_beta: float = 1.0
    nft_off_policy: bool = False
    nft_num_train_timesteps: int = 0  # 0 = derive from num_inference_steps * timestep_range
    nft_time_sampling_strategy: str = "discrete"
    # ^ Valid: "uniform", "logit_normal",
    #          "discrete", "discrete_with_init", "discrete_wo_init"
    nft_time_shift: float = 3.0
    nft_timestep_range: list[float] | None = None  # [0.0, 0.9] — fraction of denoise axis
    nft_adv_clip_range: list[float] | None = None  # [-5.0, 5.0]
    # ^ Reserved for parity with flow-factory's range-based advantage
    # clipping. The verl-omni NFT loss currently consumes a single float
    # (``actor.diffusion_loss.adv_clip_max``) rather than a range; this
    # field is wired through the config dataclass for future use and
    # for keeping example sh files self-documenting, but it is NOT
    # read by any code path today. To change effective clipping, edit
    # ``actor.diffusion_loss.adv_clip_max`` instead.
    nft_kl_beta: float = 0.0

    def __post_init__(self):
        if self.nft_timestep_range is None:
            object.__setattr__(self, "nft_timestep_range", [0.0, 0.9])
        if self.nft_adv_clip_range is None:
            object.__setattr__(self, "nft_adv_clip_range", [-5.0, 5.0])
