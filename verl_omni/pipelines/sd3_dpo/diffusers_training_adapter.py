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

"""Stable Diffusion 3 training-side adapter for diffusion DPO."""

from typing import Any, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler, ModelMixin, SchedulerMixin
from tensordict import TensorDict

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

__all__ = ["StableDiffusion3DPO"]


def _build_sd3_scheduler(model_path: str) -> FlowMatchEulerDiscreteScheduler:
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_name_or_path=model_path,
        subfolder="scheduler",
    )


@DiffusionModelBase.register("StableDiffusion3Pipeline", algorithm="dpo")
class StableDiffusion3DPO(DiffusionModelBase):
    """Training adapter for SD3 Diffusion-DPO.

    This adapter is intentionally limited to SD3-specific tensor preparation and
    transformer forwarding. The pairwise DPO objective itself belongs in
    ``verl_omni.workers.utils.losses``.
    """

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the SD3 flow-matching scheduler."""
        scheduler = _build_sd3_scheduler(model_config.local_path)
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: SchedulerMixin, model_config: DiffusionModelConfig, device: str):
        """No-op for SD3.5 DPO training.

        DPO flow-matching samples timesteps from the full ``num_train_timesteps``
        schedule (logit-normal over ~1000 steps).
        Rollout / offline data prep use separate diffusers pipelines with their
        own inference schedulers; they are not configured through this hook.
        """
        pass

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
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        """Build SD3 transformer inputs.

        For DPO-specific training, callers should normally pass already-noised
        latents and the sampled training timesteps.
        latents: (B, C, H, W)  # already-noised latents
        timesteps: (B,)
        """
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask is required for DPO training.")
        assert isinstance(prompt_embeds_mask, torch.Tensor)

        pooled_prompt_embeds = micro_batch.get("pooled_prompt_embeds", None)
        negative_pooled_prompt_embeds = micro_batch.get("negative_pooled_prompt_embeds", None)
        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale is None:
            guidance_scale = 1.0
        do_cfg = (
            guidance_scale > 1.0 and negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )

        model_inputs = cls.build_transformer_inputs(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )
        negative_model_inputs = None
        if do_cfg:
            negative_model_inputs = cls.build_transformer_inputs(
                latents=latents,
                timesteps=timesteps,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
            )

        # Keep a lightweight sanity check near the adapter boundary; SD3 uses
        # pooled prompt projections in addition to sequence prompt embeddings.
        if model_inputs["pooled_projections"] is None:
            raise KeyError("SD3 DPO training requires `pooled_projections` in the micro batch.")

        return model_inputs, negative_model_inputs

    @staticmethod
    def build_transformer_inputs(
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
    ) -> dict[str, Any]:
        """Create the SD3Transformer2DModel keyword arguments."""
        return {
            "hidden_states": latents,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "timestep": timesteps,
            "joint_attention_kwargs": {
                "attention_mask": prompt_embeds_mask,
            },
        }

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: SchedulerMixin,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ) -> torch.Tensor:
        """Run a single SD3 DPO transformer forward and return predicted noise."""

        noise_pred = module(**model_inputs)[0]
        guidance_scale = model_config.pipeline.guidance_scale
        if guidance_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("SD3 DPO CFG requires negative prompt inputs when guidance_scale > 1.")
            negative_noise_pred = module(**negative_model_inputs)[0]
            noise_pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)
        return noise_pred
