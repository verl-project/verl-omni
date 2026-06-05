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
Wan2.2 training-side adapter for diffusers-based diffusion RL (DanceGRPO).
"""

from typing import Optional

import numpy as np
import torch
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    apply_cfg,
    sd3_time_shift,
)

__all__ = ["Wan22DanceGRPO"]


def _build_wan_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    """Build the SDE scheduler from the model's scheduler config."""
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_wan_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    num_inference_steps: int,
    shift: float,
    device: str,
) -> None:
    """Configure timesteps with DanceGRPO-style shifted sigmas.

    Args:
        scheduler: The scheduler to configure.
        num_inference_steps: Number of denoising steps.
        shift: Time shift factor (e.g. 3.0 for Wan2.1, 5.0 for Wan2.2).
        device: Target device.
    """
    sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)
    sigmas = sd3_time_shift(shift, torch.from_numpy(sigmas).float()).numpy()
    # Trim the last element (sigma=0) so sigmas has num_inference_steps elements
    sigmas = sigmas[:num_inference_steps]
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)


@DiffusionModelBase.register("WanPipeline", algorithm="dance_grpo")
class Wan22DanceGRPO(DiffusionModelBase):
    """Training adapter for the Wan2.2 diffusion model with DanceGRPO.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the ``WanPipeline`` architecture, providing scheduler
    configuration, model-input construction, and the forward/sampling step
    used during DanceGRPO RL training.

    Registered under ``("WanPipeline", "dance_grpo")``.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the SDE scheduler for Wan2.2.

        Args:
            model_config: Configuration for the diffusion model.

        Returns:
            FlowMatchSDEDiscreteScheduler: Scheduler with timesteps set.
        """
        scheduler = _build_wan_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Configure timesteps and sigmas on the scheduler for Wan2.2.

        Args:
            scheduler: The scheduler whose timesteps will be set.
            model_config: Configuration providing pipeline parameters.
            device: Target device.
        """
        shift = model_config.pipeline.get("shift", 5.0)
        _configure_wan_scheduler(
            scheduler,
            num_inference_steps=model_config.pipeline.num_inference_steps,
            shift=shift,
            device=device,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module: WanTransformer3DModel,
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
        """Build Wan2.2-specific inputs for the transformer forward pass.

        Args:
            module: The Wan2.2 transformer module.
            model_config: Configuration providing guidance scale and other settings.
            latents: Full latent tensor of shape ``(B, T, C, F, H, W)``.
            timesteps: Full timestep tensor of shape ``(B, T)``.
            prompt_embeds: Positive prompt embeddings of shape ``(B, L, D)``.
            prompt_embeds_mask: Attention mask for *prompt_embeds*.
            negative_prompt_embeds: Negative prompt embeddings.
            negative_prompt_embeds_mask: Attention mask for *negative_prompt_embeds*.
            micro_batch: Micro-batch metadata (height, width, num_frames, etc.).
            step: Current denoising step index.

        Returns:
            tuple[dict, dict]: ``(model_inputs, negative_model_inputs)`` dicts
                ready to be unpacked into the transformer forward call.
        """
        guidance_scale = model_config.pipeline.get("guidance_scale", 1.0)
        true_cfg_scale = model_config.pipeline.get("true_cfg_scale", 1.0)
        do_true_cfg = true_cfg_scale > 1.0 and guidance_scale > 1.0

        # Slice to current denoising step
        hidden_states = latents[:, step]  # (B, C, F, H, W)

        # Wan2.2 transformer expects integer timesteps in [0, 1000] range.
        # Scheduler timesteps are already in that range, pass directly as long.
        timestep = timesteps[:, step].long()

        # Apply mask to encoder_hidden_states: zero out padded positions.
        # WanTransformer3DModel does not accept an attention_mask parameter,
        # so we mask the embeddings directly to prevent cross-attention from
        # attending to padding tokens.
        if prompt_embeds_mask is not None:
            prompt_embeds = prompt_embeds * prompt_embeds_mask.unsqueeze(-1).float()
        if do_true_cfg and negative_prompt_embeds_mask is not None:
            negative_prompt_embeds = negative_prompt_embeds * negative_prompt_embeds_mask.unsqueeze(-1).float()

        # Build positive model inputs
        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "encoder_hidden_states_image": None,
            "return_dict": False,
        }

        # Build negative model inputs for CFG
        if do_true_cfg:
            negative_model_inputs = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": negative_prompt_embeds,
                "encoder_hidden_states_image": None,
                "return_dict": False,
            }
        else:
            negative_model_inputs = {}

        return model_inputs, negative_model_inputs

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: WanTransformer3DModel,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Run the Wan2.2 transformer and sample the previous denoising step.

        Used by DanceGRPO which requires log-probabilities for reversed-sampling.
        Applies standard CFG when ``guidance_scale > 1.0``.

        Args:
            module: The Wan2.2 transformer module.
            scheduler: Scheduler used to sample the previous step and compute
                log-probabilities.
            model_config: Configuration providing guidance scale, noise level,
                and SDE type.
            model_inputs: Positive-prompt inputs for the transformer forward.
            negative_model_inputs: Negative-prompt inputs for CFG.
            scheduler_inputs: Must contain ``"all_latents"`` and ``"all_timesteps"``.
            step: Current denoising step index.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        guidance_scale = model_config.pipeline.get("guidance_scale", 1.0)
        true_cfg_scale = model_config.pipeline.get("true_cfg_scale", 1.0)
        do_true_cfg = true_cfg_scale > 1.0 and guidance_scale > 1.0

        # Positive forward
        noise_pred = module(**model_inputs)[0]

        # CFG forward (if enabled)
        if do_true_cfg and negative_model_inputs:
            neg_noise_pred = module(**negative_model_inputs)[0]
            noise_pred = apply_cfg(noise_pred, neg_noise_pred, guidance_scale)

        # Sample previous step via SDE scheduler
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
