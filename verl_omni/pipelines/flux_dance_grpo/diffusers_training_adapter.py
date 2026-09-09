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
FLUX training-side adapter for diffusers-based diffusion RL (DanceGRPO).

Implements the :class:`DiffusionModelBase` interface for FLUX.1-dev,
handling packed 2D latents, dual T5+CLIP prompt encoding, text/image position
IDs, guidance embedding, and SDE-based log-probability computation.
"""

from typing import Optional

import numpy as np
import torch
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import sd3_time_shift

__all__ = ["FluxDanceGRPO"]


@DiffusionModelBase.register("FluxPipeline", algorithm="dance_grpo")
class FluxDanceGRPO(DiffusionModelBase):
    """Training adapter for the FLUX.1-dev diffusion model with DanceGRPO.

    Implements the :class:`~verl_omni.pipelines.model_base.DiffusionModelBase`
    interface for ``FluxPipeline`` architecture, providing scheduler
    configuration, model-input construction, and the forward/sampling step
    used during DanceGRPO RL training.

    Registered under ``("FluxPipeline", "dance_grpo")``.

    At 720x720 resolution, packed latents have shape ``(B, 2025, 64)``.
    """

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build and configure the SDE scheduler for FLUX DanceGRPO.

        Loads the scheduler configuration from the model's ``scheduler``
        subfolder via ``from_pretrained``.

        Uses ``FlowMatchSDEDiscreteScheduler`` with the ``dance_sde`` SDE variant.

        Args:
            model_config: Configuration for the diffusion model.

        Returns:
            Configured scheduler with timesteps set.
        """
        model_path = model_config.local_path
        if model_path is None:
            raise ValueError("FLUX DanceGRPO requires model_config.local_path")
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_path,
            subfolder="scheduler",
        )
        # Explicit shifted sigmas are supplied below, so neutralise the
        # scheduler's own shifting to prevent a second transformation.
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            scheduler.register_to_config(use_dynamic_shifting=False)
        if getattr(scheduler, "shift", 1.0) != 1.0:
            scheduler.register_to_config(shift=1.0)
            scheduler.set_shift(1.0)
        if scheduler.shift != 1.0:
            raise RuntimeError(f"FLUX actor scheduler shift must be 1.0, got {scheduler.shift}")

        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(
        cls,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        device: str,
    ):
        """Configure timesteps with DanceGRPO-style shifted sigmas.

        Produces a sigma schedule via ``sd3_time_shift(shift, linspace(1,0,N+1))``
        and trims to *N* entries for denoising.

        Args:
            scheduler: The scheduler whose timesteps will be set.
            model_config: Configuration providing ``num_inference_steps``.
            device: Target device string.
        """
        num_inference_steps = model_config.pipeline.num_inference_steps
        shift = model_config.pipeline.get("shift", 3.0)

        sigmas_np = np.linspace(1.0, 0.0, num_inference_steps + 1)
        sigmas_t = sd3_time_shift(shift, torch.from_numpy(sigmas_np).float())
        sigmas = sigmas_t[:num_inference_steps].numpy()
        scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas)

    # ------------------------------------------------------------------
    # Model inputs
    # ------------------------------------------------------------------

    @classmethod
    def prepare_model_inputs(
        cls,
        module: FluxTransformer2DModel,
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
        """Build FLUX-specific inputs for the transformer forward pass.

        FLUX requires the following inputs beyond the standard latent/timestep:
            - ``txt_ids``: text position IDs of shape ``(B, seq_len, 3)``.
            - ``img_ids``: image position IDs of shape ``(patches, 3)``.
            - ``pooled_projections``: CLIP pooled text embedding ``(B, 768)``.
            - ``guidance``: guidance scale embedding broadcast to ``(B,)``.
            - ``joint_attention_kwargs``: optional cross-attention kwargs.
            - ``return_dict``: always ``False`` (returns tuple).

        Timesteps are divided by 1000 before being passed to the transformer,
        matching the standard FLUX convention.

        Args:
            module: The FLUX transformer module.
            model_config: Configuration providing guidance scale and other settings.
            latents: Full latent trajectory ``(B, T, patches, C)``.
            timesteps: Full timestep tensor ``(B, T)`` (integer, 0-1000 range).
            prompt_embeds: T5 encoder_hidden_states ``(B, L, 4096)``.
            prompt_embeds_mask: Attention mask for *prompt_embeds*.
            negative_prompt_embeds: Unused; FLUX.1-dev uses distilled guidance embeddings.
            negative_prompt_embeds_mask: Unused; kept for the shared adapter interface.
            micro_batch: Micro-batch metadata containing:
                - ``text_ids``: text position IDs.
                - ``image_ids``: image position IDs.
                - ``pooled_prompt_embeds``: CLIP pooled embedding.
            step: Current denoising step index.

        Returns:
            tuple[dict, dict]: ``(model_inputs, negative_model_inputs)`` dicts
                ready to be unpacked into the FLUX transformer ``__call__``.
        """
        del prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask

        # Slice to current denoising step (step index within the trajectory)
        model_dtype = prompt_embeds.dtype
        try:
            model_dtype = next(module.parameters()).dtype
        except (AttributeError, StopIteration):
            pass
        hidden_states = latents[:, step].to(dtype=model_dtype)  # (B, patches, C)

        # FLUX transformer expects timesteps in [0, 1] range (divided by 1000)
        timestep = timesteps[:, step].float() / 1000.0

        device = hidden_states.device
        dtype = hidden_states.dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)

        def _shared_position_ids(name: str, expected_length: int) -> torch.Tensor:
            ids = micro_batch.get(name, None)
            if ids is None:
                raise KeyError(f"FLUX actor requires rollout field {name!r}")
            if ids.ndim == 2:
                shared = ids
            elif ids.ndim == 3:
                if ids.shape[0] != hidden_states.shape[0]:
                    raise ValueError(f"{name} batch {ids.shape[0]} != latent batch {hidden_states.shape[0]}")
                shared = ids[0]
                if ids.shape[0] > 1 and not torch.equal(ids, shared.unsqueeze(0).expand_as(ids)):
                    raise ValueError(f"{name} must be identical across a FLUX micro-batch")
            else:
                raise ValueError(f"{name} must be [L,3] or [B,L,3], got {tuple(ids.shape)}")
            if shared.shape != (expected_length, 3):
                raise ValueError(f"{name} expected {(expected_length, 3)}, got {tuple(shared.shape)}")
            return shared.to(device=device, dtype=dtype)

        text_ids = _shared_position_ids("text_ids", prompt_embeds.shape[1])
        image_ids = _shared_position_ids("image_ids", hidden_states.shape[1])

        pooled_prompt_embeds = micro_batch.get("pooled_prompt_embeds", None)
        if pooled_prompt_embeds is None:
            raise KeyError("FLUX actor requires pooled_prompt_embeds from rollout CLIP encoding")
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

        # FLUX.1-dev encodes distilled guidance directly in the transformer.
        guidance_scale = model_config.pipeline.get("guidance_scale", None)
        if guidance_scale is None:
            guidance_scale = 1.0
        guidance = None
        if getattr(module.config, "guidance_embeds", False):
            guidance = torch.full(
                (hidden_states.shape[0],),
                float(guidance_scale),
                device=hidden_states.device,
                dtype=torch.float32,
            )

        # Canonical FLUX prompt encoding does not pass a T5 attention mask.
        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": image_ids,
            "guidance": guidance,
            "joint_attention_kwargs": None,
            "return_dict": False,
        }

        return model_inputs, None

    # ------------------------------------------------------------------
    # Forward + sample previous step
    # ------------------------------------------------------------------

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: FluxTransformer2DModel,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        """Run the FLUX transformer and sample the previous denoising step.

        Used by DanceGRPO which requires log-probabilities for reversed sampling.

        Args:
            module: The FLUX transformer module.
            scheduler: Scheduler used to sample the previous step and compute
                log-probabilities.
            model_config: Configuration providing guidance scale, noise level,
                and SDE type.
            model_inputs: Positive-prompt inputs for the transformer forward.
            negative_model_inputs: Unused; kept for the shared adapter interface.
            scheduler_inputs: Must contain ``"all_latents"`` and ``"all_timesteps"``.
            step: Current denoising step index.

        Returns:
            tuple: ``(log_prob, prev_sample_mean, std_dev_t, sqrt_dt)``.
        """
        del negative_model_inputs
        if scheduler_inputs is None:
            raise ValueError("FLUX DanceGRPO requires scheduler inputs from rollout")
        latents = scheduler_inputs["all_latents"]
        next_latents = scheduler_inputs.get("all_next_latents", None)
        if next_latents is None:
            raise KeyError("FLUX DanceGRPO requires all_next_latents for explicit transition pairing")
        if next_latents.shape != latents.shape:
            raise ValueError(
                f"all_next_latents shape {tuple(next_latents.shape)} != all_latents {tuple(latents.shape)}"
            )
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = module(**model_inputs)[0]

        # Sample previous step via SDE scheduler
        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=next_latents[:, step].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )

        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
