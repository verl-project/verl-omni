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
"""Qwen-Image training adapter for DiffusionNFT."""

from typing import Optional

import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import apply_true_cfg, build_img_shapes
from verl_omni.pipelines.qwen_image_flow_grpo.diffusers_training_adapter import QwenImage
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["QwenImageDiffusionNFT"]


@DiffusionModelBase.register("QwenImagePipeline", algorithm="diffusion_nft")
class QwenImageDiffusionNFT(QwenImage):
    """Forward-process Qwen-Image adapter used by DiffusionNFT."""
    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        del step
        xt = latents
        timestep = timesteps
        height = tu.get_non_tensor_data(data=micro_batch, key="height", default=None)
        width = tu.get_non_tensor_data(data=micro_batch, key="width", default=None)
        vae_scale_factor = tu.get_non_tensor_data(data=micro_batch, key="vae_scale_factor", default=None)
        img_shapes = build_img_shapes(height, width, xt.shape[0], vae_scale_factor)

        guidance_scale = model_config.pipeline.guidance_scale
        if getattr(module.config, "guidance_embeds", False):
            guidance = torch.full([1], guidance_scale, device=timestep.device, dtype=torch.float32)
        else:
            guidance = None

        model_inputs = {
            "hidden_states": xt,
            "timestep": timestep / 1000.0,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }
        negative_model_inputs = None
        if negative_prompt_embeds is not None:
            negative_model_inputs = {
                **model_inputs,
                "encoder_hidden_states_mask": negative_prompt_embeds_mask,
                "encoder_hidden_states": negative_prompt_embeds,
            }
        return model_inputs, negative_model_inputs

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        prediction = module(**model_inputs)[0]
        if model_config.pipeline.true_cfg_scale > 1.0 and negative_model_inputs is not None:
            negative_prediction = module(**negative_model_inputs)[0]
            prediction = apply_true_cfg(prediction, negative_prediction, model_config.pipeline.true_cfg_scale)
        return prediction

