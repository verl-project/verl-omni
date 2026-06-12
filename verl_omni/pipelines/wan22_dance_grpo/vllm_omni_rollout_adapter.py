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
Wan2.2 rollout-side adapter for DanceGRPO.

Subclasses the vllm-omni Wan22Pipeline and adds per-step SDE log-probability
collection required for RL training (DanceGRPO).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import PIL.Image
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline, retrieve_latents
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.platforms import current_omni_platform

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import sd3_time_shift, seed_from_prompt_ids

logger = logging.getLogger(__name__)
__all__ = ["Wan22DanceGRPOPipelineWithLogProb"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _coalesce_not_none(value, default):
    return default if value is None else value


@VllmOmniPipelineBase.register("WanPipeline", algorithm="dance_grpo")
class Wan22DanceGRPOPipelineWithLogProb(Wan22Pipeline):
    """Rollout pipeline for Wan2.2 that captures per-step log-probabilities.

    Extends :class:`~vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2.Wan22Pipeline`
    with a custom SDE-based scheduler and additional output fields required for
    DanceGRPO RL training.

    Supports both single-transformer (TI2V-5B) and dual-transformer (MoE A14B)
    models, with expand_timesteps mode for I2V conditioning.

    Registered under ``("WanPipeline", "dance_grpo")``.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        self._interrupt = False
        self.vae.use_slicing = True
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Replace the UniPC/Euler scheduler with the SDE scheduler for logprob computation.
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value: bool):
        self._interrupt = value

    def diffuse(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor,
        guidance_low: float,
        guidance_high: float,
        boundary_timestep: float | None,
        dtype: torch.dtype,
        attention_kwargs: dict[str, Any] | None,
        latent_condition: torch.Tensor | None,
        first_frame_mask: torch.Tensor | None,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Run the full SDE diffusion loop and collect per-step rollout data.

        Args:
            latents: Initial noisy latents.
            timesteps: Scheduler timestep sequence.
            prompt_embeds: Positive prompt embeddings.
            negative_prompt_embeds: Negative prompt embeddings.
            guidance_low: CFG scale for high-noise stage.
            guidance_high: CFG scale for low-noise stage.
            boundary_timestep: Boundary between high/low noise stages for
                dual-transformer models. ``None`` for single-transformer models.
            dtype: Data type for model input.
            attention_kwargs: Extra attention kwargs.
            latent_condition: Image-condition latents (I2V mode).
            first_frame_mask: Mask for first-frame conditioning (I2V mode).
            noise_level: SDE noise injection magnitude within the window.
            sde_window: ``(start, end)`` step indices for SDE data collection.
            sde_type: SDE variant (``"sde"``, ``"cps"``, or ``"dance_sde"``).
            generator: Optional random generator.
            logprobs: Whether to compute per-step log-probabilities.

        Returns:
            tuple: ``(latents, all_latents, all_log_probs, all_timesteps)``.
        """
        if attention_kwargs is None:
            attention_kwargs = {}
        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps_list: list[torch.Tensor] = []
        self.scheduler.set_begin_index(0)

        for step_idx, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            set_forward_context_denoise_step_idx(step_idx)

            # Determine noise level for SDE window
            if step_idx < sde_window[0]:
                cur_noise_level = 0.0
            elif step_idx == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents)
            elif step_idx < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            # Select model based on timestep and boundary_ratio
            # High noise stage (t >= boundary_timestep): use transformer
            # Low noise stage (t < boundary_timestep): use transformer_2
            if boundary_timestep is not None and t < boundary_timestep:
                current_guidance_scale = guidance_high
                if self.transformer_2 is not None:
                    current_model = self.transformer_2
                elif self.transformer is not None:
                    current_model = self.transformer
                else:
                    raise RuntimeError("No transformer available for low-noise stage")
            else:
                current_guidance_scale = guidance_low
                if self.transformer is not None:
                    current_model = self.transformer
                elif self.transformer_2 is not None:
                    current_model = self.transformer_2
                else:
                    raise RuntimeError("No transformer available for high-noise stage")

            if self.expand_timesteps and latent_condition is not None:
                # I2V mode: blend condition with latents using mask
                latent_model_input = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
                latent_model_input = latent_model_input.to(dtype)

                # Expand timesteps per patch - use patch_size to match patch embedding
                patch_size = self.transformer_config.patch_size
                patch_height = latents.shape[3] // patch_size[1]
                patch_width = latents.shape[4] // patch_size[2]

                # Create mask at patch resolution (same as hidden states sequence length)
                patch_mask = first_frame_mask[:, :, :, :: patch_size[1], :: patch_size[2]]
                patch_mask = patch_mask[:, :, :, :patch_height, :patch_width]
                temp_ts = (patch_mask[0][0] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                # T2V mode: standard forward
                latent_model_input = latents.to(dtype)
                timestep = t.expand(latents.shape[0])

            do_true_cfg = current_guidance_scale > 1.0 and negative_prompt_embeds is not None
            positive_kwargs = {
                "hidden_states": latent_model_input,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "attention_kwargs": attention_kwargs,
                "return_dict": False,
                "current_model": current_model,
            }
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latent_model_input,
                    "timestep": timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                    "current_model": current_model,
                }
            else:
                negative_kwargs = None
            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=do_true_cfg,
                true_cfg_scale=current_guidance_scale,
                positive_kwargs=positive_kwargs,
                negative_kwargs=negative_kwargs,
                cfg_normalize=False,
            )
            # SDE step
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                t,
                latents.float(),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if step_idx >= sde_window[0] and step_idx < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps_list.append(t)

        all_latents_tensor = torch.stack(all_latents, dim=1) if all_latents else None
        all_log_probs_tensor = (
            torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        )
        all_timesteps_tensor = (
            torch.stack(all_timesteps_list).unsqueeze(0).expand(latents.shape[0], -1) if all_timesteps_list else None
        )
        return latents, all_latents_tensor, all_log_probs_tensor, all_timesteps_tensor

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ):
        """Encode pre-tokenized prompt IDs into dense embeddings via T5.

        Unlike the parent class which accepts string prompts, this method
        accepts pre-tokenized IDs as required by the DanceGRPO training loop.

        Args:
            prompt_ids: Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask: Attention mask for *prompt_ids*; inferred as
                all-ones when ``None``.
            num_videos_per_prompt: Repeat embeddings this many times.
            prompt_embeds: Pre-computed embeddings (bypasses T5).
            max_sequence_length: Truncation/padding length.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(prompt_embeds, prompt_embeds_mask)``.
        """
        if prompt_embeds is not None:
            prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.long)
            return prompt_embeds, prompt_embeds_mask

        device = self.device
        dtype = self.text_encoder.dtype

        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        batch_size = prompt_ids.shape[0]

        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        ids, mask = prompt_ids, attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long().clamp(max=max_sequence_length)

        # Encode through T5 text encoder
        prompt_embeds = self.text_encoder(ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens, strict=False)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], device=device, dtype=torch.long)

        return prompt_embeds, prompt_embeds_mask

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        image: Any = None,
        height: int = 480,
        width: int = 832,
        num_inference_steps: int = 40,
        guidance_scale: float = 5.0,
        frame_num: int = 81,
        output_type: str | None = "np",
        generator: torch.Generator | list[torch.Generator] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        init_same_noise: bool = True,
        **kwargs,
    ) -> DiffusionOutput:
        """End-to-end video generation with DanceGRPO rollout data collection.

        Sampling parameters in *req* take precedence over keyword arguments.
        Returns raw latents (``output_type="latent"``) with full rollout data in
        ``custom_output``.

        Args:
            req: Rollout request containing prompts and sampling params.
            prompt_ids: Pre-tokenized prompt token IDs.
            prompt_mask: Attention mask for *prompt_ids*.
            negative_prompt_ids: Pre-tokenized negative prompt token IDs.
            negative_prompt_mask: Attention mask for *negative_prompt_ids*.
            image: Input image for I2V mode (unused in T2V mode).
            height: Output video height.
            width: Output video width.
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG scale for classifier-free guidance.
            frame_num: Number of frames to generate.
            output_type: ``"latent"`` or ``"np"``.
            generator: Random generator.
            prompt_embeds: Pre-computed positive prompt embeddings.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            attention_kwargs: Extra attention kwargs.
            init_same_noise: Whether to initialize the noise with the same prompt_ids.
            **kwargs: Additional arguments.

        Returns:
            DiffusionOutput: Contains the output video and a *custom_output* dict
                with ``"all_latents"``, ``"all_log_probs"``, ``"all_timesteps"``,
                ``"prompt_embeds"``, ``"prompt_embeds_mask"``,
                ``"negative_prompt_embeds"``, and ``"negative_prompt_embeds_mask"``.
        """
        # --- Extract parameters from request ---
        if len(req.prompts) > 1:
            raise ValueError(
                """This model only supports a single prompt, not a batched request.""",
                """Please pass in a single prompt object or string, or a single-item list.""",
            )
        if len(req.prompts) == 1:
            custom_prompt = req.prompts[0] if isinstance(req.prompts[0], dict) else {}
            if isinstance(custom_prompt, dict):
                prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
                prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
                negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
                negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)
                if image is None:
                    # Check both top-level and extra_args for multi_modal_data
                    multi_modal_data = custom_prompt.get("multi_modal_data", None)
                    if multi_modal_data is None:
                        extra_args = custom_prompt.get("extra_args", {})
                        multi_modal_data = (
                            extra_args.get("multi_modal_data", {}) if isinstance(extra_args, dict) else {}
                        )
                    raw_image = multi_modal_data.get("image", None) if multi_modal_data else None
                    image = raw_image if raw_image is not None else image

                # --- Handle warmup / dummy run (both prompt_ids and prompt_embeds are None) ---
                if custom_prompt.get("prompt", None) == "dummy run":
                    return DiffusionOutput(output=None, custom_output={})

        # Default dimensions
        sampling_params = req.sampling_params
        height = sampling_params.height or height
        width = sampling_params.width or width
        num_frames = sampling_params.num_frames or frame_num

        # Ensure dimensions are compatible with VAE and patch size
        # For expand_timesteps mode, we need latent dims to be even (divisible by patch_size)
        patch_size = self.transformer_config.patch_size
        mod_value = self.vae_scale_factor_spatial * patch_size[1]  # 16*2=32 for TI2V, 8*2=16 for I2V
        height = (height // mod_value) * mod_value
        width = (width // mod_value) * mod_value
        num_steps = sampling_params.num_inference_steps or num_inference_steps

        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale

        # Resolve guidance_low / guidance_high for dual-transformer models
        guidance_low = guidance_scale if isinstance(guidance_scale, int | float) else guidance_scale[0]
        guidance_high = (
            req.sampling_params.guidance_scale_2
            if req.sampling_params.guidance_scale_2 is not None
            else (
                guidance_scale[1]
                if isinstance(guidance_scale, list | tuple) and len(guidance_scale) > 1
                else guidance_low
            )
        )
        self._guidance_scale = guidance_low
        self._guidance_scale_2 = guidance_high

        # Resolve boundary_ratio for dual-transformer models
        boundary_ratio = self.boundary_ratio if self.boundary_ratio is not None else sampling_params.boundary_ratio
        if boundary_ratio is None:
            boundary_ratio = 0.875

        noise_level = _coalesce_not_none(
            sampling_params.extra_args.get("noise_level", None), kwargs.get("noise_level", 1.2)
        )
        sde_window_size = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_size", None), kwargs.get("sde_window_size")
        )
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), kwargs.get("sde_window_range", [0, 5])
        )
        sde_type = _coalesce_not_none(
            sampling_params.extra_args.get("sde_type", None), kwargs.get("sde_type", "dance_sde")
        )
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), kwargs.get("logprobs", True))
        shift = _coalesce_not_none(sampling_params.extra_args.get("shift", None), kwargs.get("shift", 5.0))

        # Validate inputs
        self.check_inputs(
            prompt_ids=prompt_ids,
            negative_prompt_ids=negative_prompt_ids,
            image=image,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # Adjust num_frames to be compatible with VAE temporal scaling
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)

        # --- Encode prompts from token IDs ---
        device = self.device
        # Get dtype from whichever transformer is loaded
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.transformer_2 is not None:
            dtype = self.transformer_2.dtype
        else:
            dtype = self.text_encoder.dtype

        # Generator setup
        if generator is None:
            generator = sampling_params.generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(sampling_params.seed)
        if init_same_noise:
            latents_seed = seed_from_prompt_ids(
                prompt_ids=prompt_ids,
                global_steps=sampling_params.extra_args.get("global_steps")
            )
            latents_generator = torch.Generator(device=device).manual_seed(latents_seed)
        else:
            latents_generator = generator

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=device, dtype=torch.long)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=device, dtype=torch.long)
        if negative_prompt_ids is not None and negative_prompt_ids.numel() == 0:
            negative_prompt_ids = torch.zeros_like(prompt_ids, device=device, dtype=torch.long)
            negative_prompt_mask = torch.zeros_like(negative_prompt_ids, device=device, dtype=torch.bool)

        num_videos_per_prompt = sampling_params.num_outputs_per_prompt or 1
        max_sequence_length = sampling_params.max_sequence_length or 512
        do_cfg = guidance_low > 1.0 or guidance_high > 1.0

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            max_sequence_length=max_sequence_length,
        )
        if do_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=negative_prompt_embeds,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds_mask = None

        # --- Prepare SDE timesteps with shifted sigmas ---
        sigmas_np = np.linspace(1.0, 0.0, num_steps + 1)
        sigmas_t = sd3_time_shift(shift, torch.from_numpy(sigmas_np).float())
        self.scheduler.set_timesteps(num_steps, device=device, sigmas=sigmas_t[:num_steps].numpy())
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        # Compute boundary_timestep for dual-transformer models
        boundary_timestep = None
        if boundary_ratio is not None:
            boundary_timestep = boundary_ratio * self.scheduler.config.num_train_timesteps

        # --- Prepare latents ---
        latent_condition = None
        first_frame_mask = None
        batch_size = prompt_embeds.shape[0]

        if self.expand_timesteps and image is not None:
            # I2V mode: encode image and prepare condition
            from diffusers.video_processor import VideoProcessor

            video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

            if isinstance(image, PIL.Image.Image):
                image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
                image_tensor = video_processor.preprocess(image, height=height, width=width)
            else:
                image_tensor = image

            # Use out_channels for noise latents (not in_channels which includes condition)
            num_channels_latents = self.transformer_config.out_channels

            # Prepare noise latents
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=latents_generator,
                latents=sampling_params.latents,
            )

            # Encode image condition
            num_latent_frames = latents.shape[2]
            latent_height = latents.shape[3]
            latent_width = latents.shape[4]

            image_tensor = image_tensor.unsqueeze(2)  # [B, C, 1, H, W]
            image_tensor = image_tensor.to(device=device, dtype=self.vae.dtype)
            latent_condition = retrieve_latents(self.vae.encode(image_tensor), sample_mode="argmax")
            latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            # Normalize condition latents
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latent_condition.device, latent_condition.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latent_condition.device, latent_condition.dtype
            )
            latent_condition = (latent_condition - latents_mean) * latents_std
            latent_condition = latent_condition.to(torch.float32)

            # Create mask: 0 for first frame (condition), 1 for rest (to denoise)
            first_frame_mask = torch.ones(
                1, 1, num_latent_frames, latent_height, latent_width, dtype=torch.float32, device=device
            )
            first_frame_mask[:, :, 0] = 0
        else:
            # T2V mode: standard latent preparation
            num_channels_latents = self.transformer_config.in_channels
            latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                height=height,
                width=width,
                num_frames=num_frames,
                dtype=torch.float32,
                device=device,
                generator=latents_generator,
                latents=sampling_params.latents,
            )

        if attention_kwargs is None:
            attention_kwargs = {}

        # --- SDE window ---
        if sde_window_size is not None:
            window_start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=device,
            ).item()
            sde_window = (window_start, window_start + sde_window_size)
        else:
            sde_window = (0, len(timesteps) - 1)

        latents, all_latents, all_log_probs, all_timesteps_tensor = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
            boundary_timestep=boundary_timestep,
            dtype=dtype,
            attention_kwargs=attention_kwargs,
            latent_condition=latent_condition,
            first_frame_mask=first_frame_mask,
            noise_level=noise_level,
            sde_window=sde_window,
            sde_type=sde_type,
            generator=generator,
            logprobs=logprobs,
        )

        # empty the cache here to avoid OOM before vae decoding.
        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()
        self._current_timestep = None

        # For I2V mode, blend final latents with condition
        if self.expand_timesteps and latent_condition is not None:
            latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents

        # Decode latents
        if output_type == "latent":
            output = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            output = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=_maybe_to_cpu(output),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps_tensor),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
            },
        )

    def check_inputs(
        self,
        prompt_ids,
        negative_prompt_ids,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if prompt_ids is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`. Please provide only one.")

        if negative_prompt_ids is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot forward both `negative_prompt` and `negative_prompt_embeds`. Please provide only one."
            )

        if prompt_ids is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
