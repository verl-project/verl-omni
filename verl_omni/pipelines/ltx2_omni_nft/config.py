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

"""LTX-2.3 OmniNFT-specific actor and pipeline configuration."""

from dataclasses import dataclass
from typing import Optional

from verl_omni.workers.config.diffusion.actor import DiffusionLossConfig
from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig


@dataclass
class OmniNFTLossConfig(DiffusionLossConfig):
    """Combine modality losses independently of reward-to-modality routing."""

    loss_mode: str = "omni_nft"
    video_weight: float = 1.0
    audio_weight: float = 1.0
    video_ref_kl_coef: float = 0.0
    audio_ref_kl_coef: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.loss_mode != "omni_nft":
            raise ValueError(f"OmniNFT loss_mode must be 'omni_nft', got {self.loss_mode!r}.")
        for name in ("video_weight", "audio_weight", "video_ref_kl_coef", "audio_ref_kl_coef"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        if self.video_weight == 0 and self.audio_weight == 0:
            raise ValueError("At least one of video_weight or audio_weight must be positive.")


@dataclass
class LTXDiffusionPipelineConfig(DiffusionPipelineConfig):
    """LTX guidance overrides for training and vLLM-Omni rollout."""

    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    video_modality_scale: Optional[float] = None
    audio_modality_scale: Optional[float] = None
    video_rescale_scale: Optional[float] = None
    audio_rescale_scale: Optional[float] = None


__all__ = ["LTXDiffusionPipelineConfig", "OmniNFTLossConfig"]
