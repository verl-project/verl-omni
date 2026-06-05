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
Qwen-Image-Edit-Plus training-side adapter for diffusers-based diffusion RL.

This adapter handles image-to-image editing models where condition images are
concatenated with noise latents during the forward pass.
"""

from typing import Optional

import numpy as np
import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import (
    QWEN_IMAGE_VAE_SCALE_FACTOR,
    build_img_shapes,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["QwenImageEditPlus"]


def _build_qwen_image_edit_scheduler(model_path: str) -> FlowMatchSDEDiscreteScheduler:
    return FlowMatchSDEDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


def _configure_qwen_image_edit_scheduler(
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


@DiffusionModelBase.register("QwenImageEditPlusPipeline", algorithm="flow_grpo")
class QwenImageEditPlus(DiffusionModelBase):
    """Training adapter for the Qwen-Image-Edit-Plus diffusion model.

    This adapter handles image-to-image editing where condition image latents
    are concatenated with noise latents before the transformer forward pass.

    Registered under ``"QwenImageEditPlusPipeline"`` with algorithm ``"flow_grpo"``.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = _build_qwen_image_edit_scheduler(model_config.local_path)
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        _configure_qwen_image_edit_scheduler(
            scheduler,
            height=model_config.pipeline.height,
            width=model_config.pipeline.width,
            num_inference_steps=model_config.pipeline.num_inference_steps,
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
        """Build Qwen-Image-Edit-Plus-specific inputs.

        Key difference from QwenImage: concatenate condition image_latents
        with the noise latents before passing to the transformer.
        """
        height = tu.get_non_tensor_data(data=micro_batch, key="height", default=None)
        width = tu.get_non_tensor_data(data=micro_batch, key="width", default=None)
        vae_scale_factor = tu.get_non_tensor_data(data=micro_batch, key="vae_scale_factor", default=None)

        # Get condition image latents (pre-encoded during rollout)
        image_latents = micro_batch.get("image_latents", None)

        # Build img_shapes - for edit models, includes both target and condition image shapes.
        # When persisted via the agent loop and re-stacked into the micro-batch, per-sample
        # shapes come back as a tensordict NonTensorStack (or numpy object array). The
        # diffusers QwenEmbedRope.forward requires a list[list[tuple]], so coerce back to
        # a plain Python list.
        img_shapes = tu.get_non_tensor_data(data=micro_batch, key="img_shapes", default=None)
        if img_shapes is None:
            img_shapes = build_img_shapes(height, width, latents.shape[0], vae_scale_factor)
        elif hasattr(img_shapes, "tolist"):
            img_shapes = img_shapes.tolist()

        # QwenImageEditPlus does not use guidance_embeds (always None)
        guidance = None

        hidden_states = latents[:, step]
        timestep = timesteps[:, step] / 1000.0

        # Concatenate condition image latents if available
        if image_latents is not None:
            latent_model_input = torch.cat([hidden_states, image_latents], dim=1)
        else:
            latent_model_input = hidden_states

        model_inputs = {
            "hidden_states": latent_model_input,
            "timestep": timestep,
            "guidance": guidance,
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "encoder_hidden_states": prompt_embeds,
            "img_shapes": img_shapes,
            "return_dict": False,
        }

        negative_model_inputs = {
            "hidden_states": latent_model_input,
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
        """Run the transformer and sample the previous denoising step.

        For edit models, the noise_pred output is sliced to only include the
        target latent portion (excluding condition image latent tokens).
        """
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        # Get the target latent sequence length for slicing
        target_seq_len = latents[:, step].shape[1]

        noise_pred = module(**model_inputs)[0]
        # Slice to target latent tokens only (exclude condition image tokens)
        noise_pred = noise_pred[:, :target_seq_len]

        true_cfg_scale = model_config.pipeline.true_cfg_scale
        if true_cfg_scale > 1.0:
            assert negative_model_inputs is not None
            neg_noise_pred = module(**negative_model_inputs)[0]
            neg_noise_pred = neg_noise_pred[:, :target_seq_len]

            # QwenImageEditPlus uses rescaled CFG (norm-preserving).
            # Clamp the denominator to avoid div-by-near-zero when comb_pred
            # collapses (rare but observed under bf16 + tiny-random + few
            # inference steps). Without the floor a single noise_norm≈0 token
            # poisons every downstream tensor and shows up as NaN log_prob /
            # grad_norm with no clear callsite.
            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True).clamp_min(1e-6)
            noise_pred = comb_pred * (cond_norm / noise_norm)

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
