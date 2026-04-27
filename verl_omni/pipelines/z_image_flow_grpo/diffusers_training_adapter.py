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
    apply_z_image_cfg,
    latents_to_transformer_input,
    split_padded_embeds_to_list,
    stack_transformer_output,
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
    # Effective pixel-to-token stride is ``vae_scale_factor * 2`` because the
    # transformer uses patch_size=2 internally.
    image_seq_len = (height // (Z_IMAGE_VAE_SCALE_FACTOR * 2)) * (width // (Z_IMAGE_VAE_SCALE_FACTOR * 2))
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas, mu=mu)


@DiffusionModelBase.register("ZImagePipeline")
class ZImage(DiffusionModelBase):
    """Training adapter for the Z-Image diffusion model.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for the ``ZImagePipeline`` architecture. Compared with the
    Qwen-Image adapter, this adapter handles three Z-Image specific quirks:

    1. Latents are 4-D ``(B, C, H, W)`` (not packed sequences).
    2. The transformer consumes ``list[Tensor]`` for both ``x`` and ``cap_feats``
       (per-sample variable-length); we unpad with the prompt mask.
    3. The model output is negated and the timestep is flipped via
       ``(1000 - t) / 1000`` before being passed to the scheduler.

    Registered under ``"ZImagePipeline"`` so it is automatically selected when
    ``DiffusionModelConfig.architecture`` matches that name.
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
            height=model_config.height,
            width=model_config.width,
            num_inference_steps=model_config.num_inference_steps,
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

        ``latents`` are sliced to ``(B, C, H_lat, W_lat)`` and wrapped into a
        per-sample list of 4-D tensors. Prompt embeddings are unpadded back to
        the per-sample variable-length list expected by the transformer.
        """
        del micro_batch  # Z-Image does not need height / width / vae_scale_factor metadata here.

        hidden_states = latents[:, step]
        timestep = (1000.0 - timesteps[:, step]) / 1000.0

        cap_feats = split_padded_embeds_to_list(prompt_embeds, prompt_embeds_mask)
        x = latents_to_transformer_input(hidden_states)

        model_inputs = {
            "x": x,
            "t": timestep,
            "cap_feats": cap_feats,
            "return_dict": False,
        }

        negative_model_inputs = None
        if (model_config.guidance_scale or 0.0) > 0.0 and negative_prompt_embeds is not None:
            neg_cap_feats = split_padded_embeds_to_list(negative_prompt_embeds, negative_prompt_embeds_mask)
            negative_model_inputs = {
                "x": x,
                "t": timestep,
                "cap_feats": neg_cap_feats,
                "return_dict": False,
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
        """Run the Z-Image transformer and sample the previous denoising step."""
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = stack_transformer_output(module(**model_inputs)[0])
        guidance_scale = model_config.guidance_scale or 0.0
        cfg_normalization = bool(getattr(model_config, "cfg_normalization", False))
        if guidance_scale > 0.0 and negative_model_inputs is not None:
            neg_noise_pred = stack_transformer_output(module(**negative_model_inputs)[0])
            noise_pred = apply_z_image_cfg(
                noise_pred,
                neg_noise_pred,
                guidance_scale,
                cfg_normalization=cfg_normalization,
            )

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
