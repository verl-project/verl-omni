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
"""Boogu-Image training adapter for DiffusionNFT."""

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.boogu_image_flow_grpo.common import (
    apply_boogu_text_cfg,
    boogu_timestep_from_scheduler,
    get_boogu_freqs_cis,
    resolve_text_guidance_scale,
)
from verl_omni.pipelines.boogu_image_flow_grpo.diffusers_training_adapter import (
    BooguImage,
    _scheduler_num_train_timesteps,
)
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["BooguImageDiffusionNFT"]


@DiffusionModelBase.register("BooguImagePipeline", algorithm="diffusion_nft")
class BooguImageDiffusionNFT(BooguImage):
    """Forward-process Boogu-Image adapter used by DiffusionNFT."""

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
        """Use the NFT engine's single-step latents (B, C, H, W) and timesteps (B,)."""
        hidden_states = latents
        num_train_timesteps = _scheduler_num_train_timesteps(model_config.local_path)
        timestep = boogu_timestep_from_scheduler(timesteps, num_train_timesteps).to(hidden_states.dtype)
        freqs_cis = get_boogu_freqs_cis(module.config.axes_dim_rope, module.config.axes_lens)
        image_latents = micro_batch.get("condition_image_latents", None)
        if image_latents is not None:
            if image_latents.dim() != 4 or image_latents.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "condition_image_latents must be (B, C, H, W) with the micro-batch batch size; "
                    f"got {tuple(image_latents.shape)} vs hidden_states {tuple(hidden_states.shape)}."
                )
            image_latents = image_latents.to(device=hidden_states.device, dtype=hidden_states.dtype)
        ref_image_hidden_states = None if image_latents is None else [[image] for image in image_latents]
        shared_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "freqs_cis": freqs_cis,
            "ref_image_hidden_states": ref_image_hidden_states,
            "return_dict": False,
        }
        model_inputs = {
            **shared_inputs,
            "instruction_hidden_states": prompt_embeds,
            "instruction_attention_mask": prompt_embeds_mask,
        }

        if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
            return model_inputs, None
        negative_model_inputs = {
            **shared_inputs,
            "instruction_hidden_states": negative_prompt_embeds,
            "instruction_attention_mask": negative_prompt_embeds_mask,
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
        """Apply text CFG and return velocity in the Diffusers convention."""
        prediction = super().forward(module, model_config, model_inputs)
        guidance_scale = resolve_text_guidance_scale(model_config.pipeline.guidance_scale)
        if guidance_scale > 1.0 and negative_model_inputs is not None:
            negative_prediction = super().forward(module, model_config, negative_model_inputs)
            prediction = apply_boogu_text_cfg(prediction, negative_prediction, guidance_scale)
        return prediction.neg()
