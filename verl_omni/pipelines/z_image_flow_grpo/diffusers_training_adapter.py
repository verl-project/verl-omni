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
Z-Image training-side adapter for diffusers-based diffusion RL.
"""

from typing import Optional

import numpy as np
import torch
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    Z_IMAGE_VAE_SCALE_FACTOR,
    apply_standard_cfg,
    pad_and_unpad_prompt_embeds,
)

__all__ = ["ZImage"]


def _build_z_image_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_z_image_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    height: int,
    width: int,
    num_inference_steps: int,
    device: str,
) -> None:
    latent_height = height // Z_IMAGE_VAE_SCALE_FACTOR // 2
    latent_width = width // Z_IMAGE_VAE_SCALE_FACTOR // 2
    image_seq_len = latent_height * latent_width
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)


@DiffusionModelBase.register("ZImagePipeline", algorithm="flow_grpo")
class ZImage(DiffusionModelBase):
    """Training adapter for the Z-Image diffusion model.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the ``ZImagePipeline`` architecture, providing scheduler
    configuration, model-input construction, and the forward/sampling step
    used during RL training (e.g. FlowGRPO).

    Registered under ``"ZImagePipeline"`` so it is automatically selected
    when ``DiffusionModelConfig.architecture`` matches that name.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the SDE scheduler for the Z-Image model."""
        scheduler = _build_z_image_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        """Configure timesteps and sigmas on the scheduler for Z-Image."""
        _configure_z_image_scheduler(
            scheduler,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ZImageTransformer2DModel,
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
        """Build Z-Image-specific inputs for the transformer forward pass.

        Z-Image transformer expects:
        - ``x``: list of per-sample latents, each (C, 1, H, W)
        - ``t``: (B,) timestep tensor (normalized by (1000-t)/1000 convention)
        - ``cap_feats``: list of per-sample text embeddings, each (L_i, D)

        Args:
            module: The Z-Image transformer module.
            model_config: Configuration providing pipeline settings.
            latents: Full latent tensor of shape ``(B, T, C, H, W)``.
            timesteps: Full timestep tensor of shape ``(B, T)``.
            prompt_embeds: Positive prompt embeddings of shape ``(B, L, D)``.
            prompt_embeds_mask: Attention mask for prompt_embeds of shape ``(B, L)``.
            negative_prompt_embeds: Negative prompt embeddings of shape ``(B, L, D)``.
            negative_prompt_embeds_mask: Attention mask for negative_prompt_embeds.
            micro_batch: Micro-batch containing metadata.
            step: Current denoising step index.

        Returns:
            tuple[dict, dict]: ``(model_inputs, negative_model_inputs)`` dicts.
        """
        # Slice to current step: (B, C, H, W)
        latent_step = latents[:, step]

        # Z-Image timestep convention: (1000 - t) / 1000
        timestep = (1000.0 - timesteps[:, step]) / 1000.0

        # Convert latent to list format: each (C, 1, H, W)
        latent_list = list(latent_step.unsqueeze(2).unbind(dim=0))

        # Convert padded prompt embeds to list format: each (L_i, D)
        prompt_embeds_list = pad_and_unpad_prompt_embeds(prompt_embeds, prompt_embeds_mask)

        model_inputs = {
            "x": latent_list,
            "t": timestep,
            "cap_feats": prompt_embeds_list,
            "return_dict": False,
            "patch_size": 2,
            "f_patch_size": 1,
        }

        negative_model_inputs = {
            "x": latent_list,
            "t": timestep,
            "cap_feats": pad_and_unpad_prompt_embeds(negative_prompt_embeds, negative_prompt_embeds_mask),
            "return_dict": False,
            "patch_size": 2,
            "f_patch_size": 1,
        }

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ZImageTransformer2DModel,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Run the Z-Image transformer and sample the previous denoising step.

        Z-Image predicts velocity (v-prediction), so the output is negated before
        being passed to the scheduler.  Applies standard CFG when
        ``model_config.true_cfg_scale > 1.0``.

        Args:
            module: The Z-Image transformer module.
            scheduler: SDE scheduler for sample_previous_step.
            model_config: Configuration with true_cfg_scale, noise_level, sde_type.
            model_inputs: Positive-prompt inputs for the transformer.
            negative_model_inputs: Negative-prompt inputs for CFG; may be None.
            scheduler_inputs: Must contain ``"all_latents"`` and ``"all_timesteps"``.
            step: Current denoising step index.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        # Forward for positive prompt
        noise_pred_list = module(**model_inputs)[0]
        noise_pred = torch.stack([t.float() for t in noise_pred_list], dim=0)

        # Z-Image predicts velocity → negate for scheduler
        noise_pred = -noise_pred

        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            assert negative_model_inputs is not None
            neg_noise_pred_list = module(**negative_model_inputs)[0]
            neg_noise_pred = torch.stack([t.float() for t in neg_noise_pred_list], dim=0)
            neg_noise_pred = -neg_noise_pred
            noise_pred = apply_standard_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

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
