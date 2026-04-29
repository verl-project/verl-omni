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

import logging
import os
from typing import Any, Literal

import numpy as np
import torch
from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps
from transformers import AutoModel, AutoTokenizer
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl import DistributedAutoencoderKL
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.tf_utils import get_transformer_config_kwargs

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import (
    apply_z_image_cfg,
    latents_to_transformer_input,
    split_padded_embeds_to_list,
    stack_transformer_output,
)

__all__ = ["ZImagePipelineWithLogProb"]

logger = logging.getLogger(__name__)


def _maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _coalesce_not_none(value, default):
    return default if value is None else value


@VllmOmniPipelineBase.register("ZImagePipeline")
class ZImagePipelineWithLogProb(ZImagePipeline):
    """Rollout pipeline for Z-Image that captures per-step log-probabilities.

    Extends :class:`~vllm_omni.diffusion.models.z_image.ZImagePipeline` with a
    custom SDE-based scheduler and additional output fields required for RL
    training (e.g. FlowGRPO). In addition to the final generated image the
    pipeline returns all intermediate latents, their log-probabilities, and
    the corresponding timesteps.

    Registered under ``"ZImagePipeline"`` for vllm-omni rollout dispatch.
    """

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Bypass ZImagePipeline.__init__ to avoid constructing the transformer
        # twice: the upstream init unconditionally uses hardcoded default params
        # (dim=3840 etc.), which would immediately be discarded.  Instead, call
        # nn.Module and the profiler mixin directly, then replicate the upstream
        # setup with the correct architecture derived from od_config.tf_model_config.
        torch.nn.Module.__init__(self)
        DiffusionPipelineProfilerMixin.__init__(self)

        self.od_config = od_config
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]
        self._execution_device = get_local_device()
        self.device = self._execution_device
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.text_encoder = AutoModel.from_pretrained(
            model, subfolder="text_encoder", local_files_only=local_files_only
        )
        self.vae = DistributedAutoencoderKL.from_pretrained(
            model, subfolder="vae", local_files_only=local_files_only
        ).to(self._execution_device)

        # Initialize transformer from config.json so that the architecture
        # matches the checkpoint (e.g. tiny-random models with dim != 3840).
        # This is the same pattern used by qwen_image, flux, hunyuan_video, etc.
        transformer_kwargs = get_transformer_config_kwargs(od_config.tf_model_config, ZImageTransformer2DModel)
        self.transformer = ZImageTransformer2DModel(quant_config=od_config.quantization_config, **transformer_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=local_files_only)
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

        # Replace the upstream Euler scheduler with the SDE variant required by
        # FlowGRPO-style log-probability collection.
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

        # Z-Image does not expose ``default_sample_size`` upstream; pick a
        # reasonable default that yields the canonical 1024x1024 image when
        # combined with vae_scale_factor=8.
        self.default_sample_size = 64

    # ------------------------------------------------------------------
    # Prompt encoding
    # ------------------------------------------------------------------

    def _encode_prompt_ids(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode pre-tokenized ``prompt_ids`` into Z-Image text features.

        Z-Image picks ``hidden_states[-2]`` from the text encoder and extracts
        per-sample variable-length features via the attention mask.
        """
        dtype = dtype or self.text_encoder.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)

        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        encoder_output = self.text_encoder(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device).bool(),
            output_hidden_states=True,
        )
        hidden_states = encoder_output.hidden_states[-2]

        mask = attention_mask.to(self.device).long()
        # Trim trailing padding columns to keep the transported tensor compact.
        # If the input batch is entirely padding (lengths == 0), raise instead
        # of silently keeping a padding token in the prompt embedding -- the
        # downstream ``split_padded_embeds_to_list`` would otherwise produce
        # zero-length tensors that the transformer cannot consume.
        lengths = mask.sum(dim=1)
        if lengths.numel() == 0 or int(lengths.max().item()) == 0:
            raise ValueError("encode_prompt received an entirely-padding batch (no valid tokens).")
        max_len = int(lengths.max().item())
        hidden_states = hidden_states[:, :max_len]
        mask = mask[:, :max_len]

        return hidden_states.to(dtype=dtype), mask

    def encode_prompt(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        max_sequence_length: int = 512,
    ):
        """Encode text prompt token IDs into a padded ``(B, L, D)`` tensor and
        a ``(B, L)`` mask, ready for transport across the agent loop.

        Args:
            prompt_ids (torch.Tensor): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (torch.Tensor, *optional*): Attention mask aligned with
                *prompt_ids*; defaults to all-ones.
            num_images_per_prompt (int): Number of images per prompt; embeddings
                are repeated accordingly.
            prompt_embeds (torch.Tensor, *optional*): Pre-computed embeddings;
                bypasses the text encoder when provided.
            prompt_embeds_mask (torch.Tensor, *optional*): Attention mask for
                pre-computed *prompt_embeds*.
            max_sequence_length (int): Maximum kept sequence length.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(prompt_embeds, prompt_embeds_mask)``.
        """
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = (
            attention_mask.unsqueeze(0) if attention_mask is not None and attention_mask.ndim == 1 else attention_mask
        )

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._encode_prompt_ids(prompt_ids, attention_mask=attention_mask)

        prompt_embeds = prompt_embeds[:, :max_sequence_length]
        prompt_embeds_mask = prompt_embeds_mask[:, :max_sequence_length]

        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat_interleave(num_images_per_prompt, dim=0)
            prompt_embeds_mask = prompt_embeds_mask.repeat_interleave(num_images_per_prompt, dim=0)

        return prompt_embeds, prompt_embeds_mask

    # ------------------------------------------------------------------
    # Timestep helpers
    # ------------------------------------------------------------------

    def prepare_timesteps(self, num_inference_steps, sigmas, image_seq_len):
        """Pre-compute SDE timesteps using ZImage's flow-shift policy."""
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            sigmas=sigmas,
            mu=mu,
        )
        return timesteps, num_inference_steps

    # ------------------------------------------------------------------
    # SDE rollout loop
    # ------------------------------------------------------------------

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
        cfg_normalization,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
    ):
        """Run the full SDE diffusion loop and collect per-step rollout data.

        Mirrors :meth:`QwenImagePipelineWithLogProb.diffuse` but with the
        Z-Image specific conventions: latents stay 4-D, prompt features are
        passed as per-sample lists, the timestep is flipped to ``(1000-t)/1000``,
        and the model output is negated before the scheduler step.
        """
        all_latents: list[torch.Tensor] = []
        all_log_probs: list[torch.Tensor] = []
        all_timesteps: list[torch.Tensor] = []
        self.scheduler.set_begin_index(0)

        cap_feats = split_padded_embeds_to_list(prompt_embeds, prompt_embeds_mask)
        neg_cap_feats = (
            split_padded_embeds_to_list(negative_prompt_embeds, negative_prompt_embeds_mask) if do_true_cfg else None
        )

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents)
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)
            timestep = (1000 - timestep) / 1000

            x = latents_to_transformer_input(latents.to(self.od_config.dtype))

            noise_pred = stack_transformer_output(self.transformer(x, timestep, cap_feats)[0])
            if do_true_cfg:
                neg_noise_pred = stack_transformer_output(self.transformer(x, timestep, neg_cap_feats)[0])
                noise_pred = apply_z_image_cfg(noise_pred, neg_noise_pred, guidance_scale, cfg_normalization)

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.to(torch.float32),
                timestep_value,
                latents.to(torch.float32),
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        attention_kwargs: dict[str, Any] | None = None,
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        """End-to-end image generation with rollout data collection.

        Encodes the (already chat-template-tokenized) prompt, prepares latents,
        runs the SDE diffusion loop via :meth:`diffuse`, and decodes the final
        latents through the VAE. Sampling parameters in *req* take precedence
        over the keyword arguments.
        """
        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or height or self.default_sample_size * self.vae_scale_factor * 2
        width = sampling_params.width or width or self.default_sample_size * self.vae_scale_factor * 2
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), sde_window_range
        )
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)
        cfg_normalization = _coalesce_not_none(
            sampling_params.extra_args.get("cfg_normalization", None), cfg_normalization
        )
        cfg_truncation = _coalesce_not_none(sampling_params.extra_args.get("cfg_truncation", None), 1.0)
        if float(cfg_truncation) != 1.0:
            # The diffusers Z-Image pipeline supports a time-aware ``cfg_truncation``
            # knob. The FlowGRPO training adapter does not yet mirror that schedule,
            # so allowing it here would silently desync rollout vs training log-probs.
            raise NotImplementedError(
                "cfg_truncation != 1.0 is not supported by the Z-Image FlowGRPO rollout adapter "
                "(would desync from training-side log-probabilities)."
            )

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        if getattr(sampling_params, "guidance_scale_provided", False):
            guidance_scale = sampling_params.guidance_scale
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0 or width % vae_scale != 0:
            raise ValueError(f"Height/width must be divisible by {vae_scale} (got {height}x{width}).")

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
        if guidance_scale > 0 and not has_neg_prompt:
            # Upstream diffusers ZImagePipeline auto-fills an empty-string negative
            # prompt when CFG is active. We require it to be supplied explicitly so
            # that training can replay the same trajectory; warn the user when CFG
            # is silently disabled because of a missing negative prompt.
            logger.warning(
                "guidance_scale=%s > 0 but no negative prompt provided; "
                "CFG will be disabled. Pass `negative_prompt_ids` (or use the "
                "data preprocessor's empty-string default) to enable CFG.",
                guidance_scale,
            )
        do_true_cfg = guidance_scale > 0 and has_neg_prompt

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

        num_channels_latents = self.transformer.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            self.device,
            generator,
            latents,
        )

        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, image_seq_len)
        self._num_timesteps = len(timesteps)

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

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            latents=latents,
            timesteps=timesteps,
            do_true_cfg=do_true_cfg,
            guidance_scale=guidance_scale,
            cfg_normalization=cfg_normalization,
            noise_level=noise_level,
            sde_window=sde_window,
            sde_type=sde_type,
            generator=generator,
            logprobs=logprobs,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]

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

    # ------------------------------------------------------------------
    # Properties used by the upstream ZImagePipeline parent.
    # ------------------------------------------------------------------

    @property
    def do_classifier_free_guidance(self):
        return getattr(self, "_guidance_scale", 0.0) > 0
