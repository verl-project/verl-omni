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
"""Stable Diffusion 3 training-side adapter for DGPO."""

from typing import Optional

import torch
from diffusers import ModelMixin

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.sd3_dpo.diffusers_training_adapter import StableDiffusion3DPO
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["StableDiffusion3DGPO"]


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="dgpo")
class StableDiffusion3DGPO(StableDiffusion3DPO):
    """Forward-process SD3 adapter used by DGPO.

    Inputs are built exactly as for Diffusion-DPO (noised latents at a sampled timestep).
    DGPO scores the conditional velocity, so the forward pass never applies CFG; the
    rollout guidance scale only shapes the sampled images.
    """

    @classmethod
    def forward(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        del model_config, negative_model_inputs
        return module(**model_inputs)[0]
