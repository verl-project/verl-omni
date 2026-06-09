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

"""Qwen-Image training-side adapter for offline diffusion DPO."""

from typing import Any, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, SchedulerMixin
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["QwenImageDPO"]


QWEN_IMAGE_VAE_SCALE_FACTOR = 8


def _build_img_shapes(
    height: int, width: int, batch_size: int, vae_scale_factor: int
) -> list[list[tuple[int, int, int]]]:
    latent_height = height // vae_scale_factor // 2
    latent_width = width // vae_scale_factor // 2
    return [[(1, latent_height, latent_width)]] * batch_size


def _apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    comb_pred = negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


@DiffusionModelBase.register("QwenImagePipeline", algorithm="dpo")
class QwenImageDPO(DiffusionModelBase):
    """Training adapter for Qwen-Image Diffusion-DPO."""

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_config.local_path,
            subfolder="scheduler",
        )

        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module: QwenImageTransformer2DModel,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        del step
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask is required for Qwen-Image DPO training.")

        height = model_config.pipeline.height
        width = model_config.pipeline.width
        vae_scale_factor = model_config.get("vae_scale_factor", QWEN_IMAGE_VAE_SCALE_FACTOR)
        img_shapes = _build_img_shapes(height, width, latents.shape[0], vae_scale_factor)

        guidance_scale = model_config.pipeline.guidance_scale
        if getattr(module.config, "guidance_embeds", False):
            guidance = torch.full((latents.shape[0],), guidance_scale, device=timesteps.device, dtype=torch.float32)
        else:
            guidance = None

        model_inputs = cls.build_transformer_inputs(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            guidance=guidance,
        )

        negative_model_inputs = None
        true_cfg_scale = getattr(model_config.pipeline, "true_cfg_scale", 1.0)
        if true_cfg_scale > 1.0:
            if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
                raise ValueError("Qwen-Image DPO True-CFG requires negative prompt inputs.")
            negative_model_inputs = cls.build_transformer_inputs(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                img_shapes=img_shapes,
                guidance=guidance,
            )

        return model_inputs, negative_model_inputs

    @staticmethod
    def build_transformer_inputs(
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        img_shapes: list[list[tuple[int, int, int]]],
        guidance: torch.Tensor | None,
    ) -> dict[str, Any]:
        return {
            "hidden_states": latents,
            "timestep": timesteps / 1000.0,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: QwenImageTransformer2DModel,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ) -> torch.Tensor:
        del scheduler, scheduler_inputs, step

        noise_pred = module(**model_inputs)[0]
        true_cfg_scale = getattr(model_config.pipeline, "true_cfg_scale", 1.0)
        if true_cfg_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("Qwen-Image DPO True-CFG requires negative prompt inputs when true_cfg_scale > 1.")
            neg_noise_pred = module(**negative_model_inputs)[0]
            noise_pred = _apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)
        return noise_pred
