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

import os
from typing import Any, Literal

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.z_image.pipeline_z_image import (
    ZImagePipeline,
    calculate_shift,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import apply_standard_cfg

__all__ = ["ZImagePipelineWithLogProb"]


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _coalesce_not_none(value, default):
    return default if value is None else value


@VllmOmniPipelineBase.register("ZImagePipeline", algorithm="flow_grpo")
class ZImagePipelineWithLogProb(ZImagePipeline):
    """Rollout pipeline for Z-Image that captures per-step log-probabilities.

    Extends :class:`~vllm_omni.diffusion.models.z_image.pipeline_z_image.ZImagePipeline`
    with a custom SDE-based scheduler and additional output fields required
    for RL training (e.g. FlowGRPO).  In addition to the final generated image,
    the pipeline returns all intermediate latents, their log-probabilities,
    and the corresponding timesteps.

    Registered under ``"ZImagePipeline"`` for vllm-omni rollout dispatch.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        # Replace the Euler scheduler with the SDE scheduler
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ):
        """Encode text prompt token IDs into dense padded embeddings.

        Z-Image uses Qwen3 text encoder with ``enable_thinking=True`` chat template.
        Returns padded ``(B, L, D)`` embeddings and ``(B, L)`` mask.

        Args:
            prompt_ids: Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask: Boolean mask of shape ``(B, L)`` for prompt_ids.
                Inferred as all-ones when None.
            num_images_per_prompt: Number of images to generate per prompt.
            prompt_embeds: Pre-computed embeddings; when provided, prompt_ids is ignored.
            prompt_embeds_mask: Attention mask for pre-computed prompt_embeds.
            max_sequence_length: Maximum sequence length; embeddings are truncated.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(prompt_embeds, prompt_embeds_mask)``
                of shape ``(B * num_images_per_prompt, L, D)`` and
                ``(B * num_images_per_prompt, L)`` respectively.
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )

        if prompt_embeds is not None:
            prompt_embeds = prompt_embeds[:, :max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(prompt_ids)

            batch_size = prompt_ids.shape[0]
            prompt_embeds_list = []
            prompt_embeds_mask_list = []

            for i in range(batch_size):
                # Get the actual non-padded token IDs
                mask_i = attention_mask[i].bool()
                ids_i = prompt_ids[i][mask_i]

                # Apply chat template (matching Z-Image's _encode_prompt)
                prompt_str = self.tokenizer.decode(ids_i, skip_special_tokens=False)
                messages = [{"role": "user", "content": prompt_str}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                encoded = self.tokenizer(
                    formatted,
                    padding="max_length",
                    max_length=max_sequence_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = encoded.input_ids.to(self.device)
                attn_mask = encoded.attention_mask.to(self.device).bool()

                # Encode with text encoder (use second-to-last hidden states)
                hidden_states = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                ).hidden_states[-2]

                # Extract non-padded embeddings
                non_padded = hidden_states[0][attn_mask[0]]
                prompt_embeds_list.append(non_padded)
                prompt_embeds_mask_list.append(torch.ones(len(non_padded), dtype=torch.long, device=self.device))

            # Pad to batch
            max_seq = max(e.size(0) for e in prompt_embeds_list)
            dim = prompt_embeds_list[0].size(-1)

            padded_embeds = torch.zeros(batch_size, max_seq, dim, device=self.device, dtype=prompt_embeds_list[0].dtype)
            padded_mask = torch.zeros(batch_size, max_seq, dtype=torch.long, device=self.device)
            for i in range(batch_size):
                seq_len = prompt_embeds_list[i].size(0)
                padded_embeds[i, :seq_len] = prompt_embeds_list[i]
                padded_mask[i, :seq_len] = 1

            prompt_embeds = padded_embeds[:, :max_sequence_length]
            prompt_embeds_mask = padded_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        timesteps,
        do_true_cfg,
        guidance_scale,
        true_cfg_scale,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
    ):
        """Run the full SDE diffusion loop and collect per-step rollout data.

        Iterates over all timesteps, optionally applying standard CFG guidance,
        and collects latents and log-probabilities within the SDE window.

        Args:
            prompt_embeds: Positive prompt embeddings (B, L, D).
            prompt_embeds_mask: Attention mask for prompt_embeds (B, L).
            negative_prompt_embeds: Negative prompt embeddings for CFG (B, L, D).
            negative_prompt_embeds_mask: Attention mask for negative prompt_embeds.
            latents: Initial noisy latents (B, C, H, W).
            timesteps: Scheduler timestep sequence.
            do_true_cfg: Whether to apply CFG guidance.
            guidance_scale: Distilled guidance scale (guidance_embeds mode).
            true_cfg_scale: CFG scale for standard CFG.
            noise_level: SDE noise injection magnitude within the window.
            sde_window: ``(start, end)`` step indices for SDE window.
            sde_type: SDE variant; ``"sde"`` or ``"cps"``.
            generator: Optional random generator.
            logprobs: Whether to compute per-step log-probabilities.

        Returns:
            tuple: ``(latents, all_latents, all_log_probs, all_timesteps)``.
        """
        all_latents = []
        all_log_probs = []
        all_timesteps_list = []

        self.scheduler.set_begin_index(0)

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents)
            elif sde_window[0] < i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            # Broadcast timestep to batch: (B,)
            timestep_b = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            # Z-Image timestep convention: (1000 - t) / 1000
            t_norm = (1000.0 - timestep_b) / 1000.0

            # Prepare latent: (B, C, 1, H, W) → list of (C, 1, H, W)
            latent_model_input = latents.unsqueeze(2)
            latent_list = list(latent_model_input.unbind(dim=0))

            # Convert padded prompt embeds to list format for Z-Image transformer
            prompt_embeds_list = self._pad_to_list(prompt_embeds, prompt_embeds_mask)

            # Forward pass
            noise_pred_list = self.transformer(
                latent_list,
                t_norm,
                prompt_embeds_list,
            )[0]

            # Standard CFG
            if do_true_cfg:
                neg_prompt_list = self._pad_to_list(negative_prompt_embeds, negative_prompt_embeds_mask)
                neg_noise_pred_list = self.transformer(
                    latent_list,
                    t_norm,
                    neg_prompt_list,
                )[0]
                pos_pred = torch.stack([t.float() for t in noise_pred_list], dim=0)
                neg_pred = torch.stack([t.float() for t in neg_noise_pred_list], dim=0)
                noise_pred = apply_standard_cfg(pos_pred, neg_pred, true_cfg_scale)
            else:
                noise_pred = torch.stack([t.float() for t in noise_pred_list], dim=0)

            # Z-Image predicts velocity → negate for scheduler
            noise_pred = -noise_pred

            # SDE step: x_t → x_{t-1}
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred,
                timestep_value,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if sde_window[0] <= i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps_list.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps_list).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    @staticmethod
    def _pad_to_list(prompt_embeds, prompt_embeds_mask):
        """Convert padded (B, L, D) + (B, L) mask to list of (L_i, D) tensors."""
        result = []
        for i in range(prompt_embeds.shape[0]):
            mask_i = prompt_embeds_mask[i].bool()
            result.append(prompt_embeds[i][mask_i])
        return result

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 10,
        sigmas: list[float] | None = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        """End-to-end image generation with rollout data collection.

        Encodes the prompt, prepares latents, runs the SDE diffusion loop via
        :meth:`diffuse`, and decodes the final latents through the VAE.

        Args:
            req: Rollout request containing prompts and sampling params.
            prompt_ids: Token IDs for the positive prompt.
            prompt_mask: Attention mask for prompt_ids.
            negative_prompt_ids: Token IDs for the negative prompt (CFG).
            negative_prompt_mask: Attention mask for negative_prompt_ids.
            true_cfg_scale: CFG scale; CFG disabled when <= 1.
            height: Output image height in pixels.
            width: Output image width in pixels.
            num_inference_steps: Number of denoising steps.
            sigmas: Custom sigmas for the scheduler.
            guidance_scale: Distilled guidance scale (guidance_embeds mode).
            num_images_per_prompt: Number of images per prompt.
            generator: Random generator(s).
            latents: Pre-generated initial latents.
            prompt_embeds: Pre-computed positive prompt embeddings.
            prompt_embeds_mask: Mask for pre-computed prompt_embeds.
            negative_prompt_embeds: Pre-computed negative prompt embeddings.
            negative_prompt_embeds_mask: Mask for negative_prompt_embeds.
            output_type: ``"latent"`` returns raw latents.
            attention_kwargs: Extra kwargs for attention layers.
            callback_on_step_end_tensor_inputs: Tensors for step-end callback.
            max_sequence_length: Maximum prompt embedding sequence length.
            noise_level: SDE noise injection magnitude within the window.
            sde_window_size: Number of SDE steps; None = full timestep range.
            sde_window_range: ``(start, end)`` range for SDE window position.
            sde_type: SDE variant; ``"sde"`` or ``"cps"``.
            logprobs: Whether to compute per-step log-probabilities.

        Returns:
            DiffusionOutput: Contains decoded output image and custom_output dict
                with keys ``"all_latents"``, ``"all_log_probs"``, ``"all_timesteps"``,
                ``"prompt_embeds"``, ``"prompt_embeds_mask"``,
                ``"negative_prompt_embeds"``, ``"negative_prompt_embeds_mask"``.
        """
        # Extract from request
        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or height or 1024
        width = sampling_params.width or width or 1024
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), sde_window_range
        )
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = _coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        if prompt_ids is not None:
            if isinstance(prompt_ids, list):
                prompt_ids = torch.tensor(prompt_ids, device=self.device)
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            return DiffusionOutput(output=None, custom_output={})

        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # Encode prompts
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_ids,
            attention_mask=prompt_mask,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                prompt_ids=negative_prompt_ids,
                attention_mask=negative_prompt_mask,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

        # Prepare latents
        # Z-Image uses in_channels directly (16 for 6B model)
        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # Prepare timesteps with shift
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        sigmas_in = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.scheduler.set_timesteps(num_inference_steps, device=self.device, sigmas=sigmas_in, mu=mu)
        timesteps = self.scheduler.timesteps.to(self.device)
        self._num_timesteps = len(timesteps)

        # Determine SDE window
        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps) - 1)

        # Run SDE diffusion
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            timesteps,
            do_true_cfg,
            guidance_scale,
            true_cfg_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        self._current_timestep = None

        # Decode latents
        if output_type == "latent":
            image = latents
        else:
            latents_dec = latents.to(self.vae.dtype)
            latents_dec = (latents_dec / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents_dec, return_dict=False)[0]

        return DiffusionOutput(
            output=_maybe_to_cpu(image),
            custom_output={
                "all_latents": _maybe_to_cpu(all_latents),
                "all_log_probs": _maybe_to_cpu(all_log_probs),
                "all_timesteps": _maybe_to_cpu(all_timesteps),
                "prompt_embeds": _maybe_to_cpu(prompt_embeds),
                "prompt_embeds_mask": _maybe_to_cpu(prompt_embeds_mask),
                "negative_prompt_embeds": _maybe_to_cpu(negative_prompt_embeds),
                "negative_prompt_embeds_mask": _maybe_to_cpu(negative_prompt_embeds_mask),
            },
        )
