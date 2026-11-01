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
"""LTX-2.3 pipeline configuration with per-modality guidance for DiffusionART."""

from dataclasses import dataclass
from typing import Optional

from verl_omni.pipelines.ltx2_omni_nft.config import LTXDiffusionPipelineConfig

__all__ = ["LTXARTPipelineConfig", "ltx_guidance_scales"]


@dataclass
class LTXARTPipelineConfig(LTXDiffusionPipelineConfig):
    """OmniNFT's LTX guidance knobs plus STG, which the rollout adapter defaults to neutral when unset.

    DiffusionART recomputes ``v_old`` with the CFG-only training forward, so the training rollout keeps STG,
    modality and rescale guidance neutral and ``actor_rollout_ref.model.pipeline`` /
    ``actor_rollout_ref.rollout.pipeline`` carry equal CFG; ``val_kwargs`` may set any of them.
    """

    video_stg_scale: Optional[float] = None
    audio_stg_scale: Optional[float] = None


def ltx_guidance_scales(pipeline_config) -> tuple[float, float]:
    """Per-modality CFG scales: modality override, else the common ``guidance_scale``; ``None``/``0`` mean 1."""
    common = getattr(pipeline_config, "guidance_scale", None)
    video = getattr(pipeline_config, "video_cfg_scale", None)
    audio = getattr(pipeline_config, "audio_cfg_scale", None)
    video = video if video is not None else common
    audio = audio if audio is not None else common
    return float(video or 1.0), float(audio or 1.0)
