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

"""
FLUX training-side adapter for diffusers-based FlowGRPO.
"""

from typing import Optional

import numpy as np
import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import apply_true_cfg, calculate_shift, packed_latent_seq_len, squeeze_batch_position_ids

__all__ = ["Flux"]


def _build_flux_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_flux_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    device: str,
) -> None:
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        packed_latent_seq_len(height, width),
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)


def _guidance_tensor(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    timesteps: torch.Tensor,
) -> torch.Tensor | None:
    guidance_scale = model_config.pipeline.guidance_scale
    if guidance_scale is None:
        guidance_scale = 3.5
    while hasattr(module, "module"):
        module = module.module
    if getattr(module, "guidance_embeds", False) or getattr(getattr(module, "config", None), "guidance_embeds", False):
        return torch.full([timesteps.shape[0]], guidance_scale, device=timesteps.device, dtype=torch.float32)
    return None


@DiffusionModelBase.register("FluxPipeline", algorithm="flow_grpo")
class Flux(DiffusionModelBase):
    """Training adapter for FLUX FlowGRPO.

    FLUX uses packed latent patches plus two text-conditioning paths.  Rollout
    therefore returns the T5 prompt embeds, pooled CLIP embeds, text position
    ids, and latent image ids; this adapter rebuilds the transformer inputs
    from those tensors during policy log-prob recomputation.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = _build_flux_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        _configure_flux_scheduler(
            scheduler,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def build_transformer_inputs(
        cls,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance: torch.Tensor | None,
        prompt_embeds_mask: torch.Tensor | None = None,
        joint_attention_kwargs: dict | None = None,
    ) -> dict:
        return {
            "hidden_states": latents,
            "timestep": timesteps / 1000.0,
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "encoder_attention_mask": prompt_embeds_mask,
            "txt_ids": squeeze_batch_position_ids(text_ids),
            "img_ids": squeeze_batch_position_ids(latent_image_ids),
            "joint_attention_kwargs": joint_attention_kwargs or {},
            "return_dict": False,
        }

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        negative_prompt_embeds_mask: torch.Tensor,
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        if "pooled_prompt_embeds" not in micro_batch:
            raise KeyError("FLUX FlowGRPO requires `pooled_prompt_embeds` from rollout.")
        if "text_ids" not in micro_batch:
            raise KeyError("FLUX FlowGRPO requires `text_ids` from rollout.")
        if "latent_image_ids" not in micro_batch:
            raise KeyError("FLUX FlowGRPO requires `latent_image_ids` from rollout.")

        selected_latents = latents[:, step]
        selected_timesteps = timesteps[:, step]
        guidance = _guidance_tensor(module, model_config, selected_timesteps)

        model_inputs = cls.build_transformer_inputs(
            latents=selected_latents,
            timesteps=selected_timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=micro_batch["pooled_prompt_embeds"],
            text_ids=micro_batch["text_ids"],
            latent_image_ids=micro_batch["latent_image_ids"],
            guidance=guidance,
            prompt_embeds_mask=prompt_embeds_mask,
        )

        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            if negative_prompt_embeds is None:
                raise ValueError("FLUX true CFG requires negative prompt embeds when true_cfg_scale > 1.")
            if "negative_pooled_prompt_embeds" not in micro_batch:
                raise KeyError("FLUX true CFG requires `negative_pooled_prompt_embeds` from rollout.")
            if "negative_text_ids" not in micro_batch:
                raise KeyError("FLUX true CFG requires `negative_text_ids` from rollout.")
            negative_model_inputs = cls.build_transformer_inputs(
                latents=selected_latents,
                timesteps=selected_timesteps,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=micro_batch["negative_pooled_prompt_embeds"],
                text_ids=micro_batch["negative_text_ids"],
                latent_image_ids=micro_batch["latent_image_ids"],
                guidance=guidance,
                prompt_embeds_mask=negative_prompt_embeds_mask,
            )
        else:
            negative_model_inputs = None

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs)
        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("FLUX true CFG requires negative model inputs when true_cfg_scale > 1.")
            neg_noise_pred = cls.forward(module, model_config, negative_model_inputs)
            noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
