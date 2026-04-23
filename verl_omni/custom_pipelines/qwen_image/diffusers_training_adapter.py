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
Qwen-Image training-side adapter for diffusers-based diffusion RL.
"""

from typing import Optional

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.custom_pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.models.diffusion_model import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .common import QWEN_IMAGE_VAE_SCALE_FACTOR, apply_true_cfg, build_img_shapes

__all__ = ["QwenImage"]

def _build_qwen_image_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_qwen_image_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    device: str,
) -> None:
    latent_height = height // QWEN_IMAGE_VAE_SCALE_FACTOR // 2
    latent_width = width // QWEN_IMAGE_VAE_SCALE_FACTOR // 2
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        latent_height * latent_width,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)


@DiffusionModelBase.register("QwenImagePipeline")
class QwenImage(DiffusionModelBase):
    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = _build_qwen_image_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        _configure_qwen_image_scheduler(
            scheduler,
            height=model_config.height,
            width=model_config.width,
            num_inference_steps=model_config.num_inference_steps,
            device=device,
        )

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
    ) -> tuple[dict, dict]:
        height = tu.get_non_tensor_data(data=micro_batch, key="height", default=None)
        width = tu.get_non_tensor_data(data=micro_batch, key="width", default=None)
        vae_scale_factor = tu.get_non_tensor_data(data=micro_batch, key="vae_scale_factor", default=None)
        img_shapes = build_img_shapes(height, width, latents.shape[0], vae_scale_factor)

        guidance_scale = model_config.guidance_scale
        if getattr(module.config, "guidance_embeds", False):
            guidance = torch.full([1], guidance_scale, device=timesteps.device, dtype=torch.float32)
        else:
            guidance = None

        hidden_states = latents[:, step]
        timestep = timesteps[:, step] / 1000.0

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }

        negative_model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states_mask": negative_prompt_embeds_mask,
            "encoder_hidden_states": negative_prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: QwenImageTransformer2DModel,
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

        noise_pred = module(**model_inputs)[0]
        true_cfg_scale = model_config.true_cfg_scale
        if true_cfg_scale > 1.0:
            assert negative_model_inputs is not None
            neg_noise_pred = module(**negative_model_inputs)[0]
            noise_pred = apply_true_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

        _, log_prob, prev_sample_mean, std_dev_t = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
        )
        return log_prob, prev_sample_mean, std_dev_t

