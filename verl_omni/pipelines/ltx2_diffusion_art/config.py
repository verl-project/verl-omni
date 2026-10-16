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

from verl_omni.workers.config.diffusion.rollout import DiffusionPipelineConfig

__all__ = ["LTXARTPipelineConfig", "ltx_guidance_scales"]


@dataclass
class LTXARTPipelineConfig(DiffusionPipelineConfig):
    """LTX guidance knobs shared by the vLLM-Omni rollout and the training forward.

    DiffusionART recomputes ``v_old`` with the training forward, so the same instance (or equal
    values) must be used for ``actor_rollout_ref.model.pipeline`` and ``actor_rollout_ref.rollout.pipeline``.
    """

    video_cfg_scale: Optional[float] = None
    audio_cfg_scale: Optional[float] = None
    video_modality_scale: Optional[float] = None
    audio_modality_scale: Optional[float] = None
    video_rescale_scale: Optional[float] = None
    audio_rescale_scale: Optional[float] = None


def ltx_guidance_scales(pipeline_config) -> tuple[float, float]:
    """Per-modality CFG scales: modality override, else the common ``guidance_scale``; ``None``/``0`` mean 1."""
    common = getattr(pipeline_config, "guidance_scale", None)
    video = getattr(pipeline_config, "video_cfg_scale", None)
    audio = getattr(pipeline_config, "audio_cfg_scale", None)
    video = video if video is not None else common
    audio = audio if audio is not None else common
    return float(video or 1.0), float(audio or 1.0)
