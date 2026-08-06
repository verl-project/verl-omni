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
"""Qwen-Image rollout adapter for DiffusionNFT."""

import copy
from dataclasses import replace
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.qwen_image import QwenImagePipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.size_utils import normalize_min_aligned_size
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import (
    QwenImageTokenIdPromptMixin,
    build_img_shapes,
    coalesce_not_none,
)

__all__ = ["QwenImageDiffusionNFTPipeline"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="diffusion_nft")
class QwenImageDiffusionNFTPipeline(QwenImageTokenIdPromptMixin, QwenImagePipeline):
    """Rollout pipeline for Qwen-Image used by DiffusionNFT.

    DiffusionNFT trains from the final clean latent with a forward-process
    objective, so the rollout side does not collect reverse-SDE trajectories or
    log-probabilities.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_progress_bar_config(disable=True)

    def _extract_step_prompt_ids(self, prompt):
        """Extract tokenized prompts, with a raw-text warm-up fallback."""
        prompt_ids = None
        prompt_mask = None
        negative_prompt_ids = None
        negative_prompt_mask = None
        if isinstance(prompt, dict):
            prompt_ids = prompt.get("prompt_token_ids")
            prompt_mask = prompt.get("prompt_mask")
            negative_prompt_ids = prompt.get("negative_prompt_ids")
            negative_prompt_mask = prompt.get("negative_prompt_mask")
            if prompt_ids is None and prompt.get("prompt"):
                prompt_ids, prompt_mask = self._tokenize_step_prompt(prompt["prompt"])
            if negative_prompt_ids is None and prompt.get("negative_prompt"):
                negative_prompt_ids, negative_prompt_mask = self._tokenize_step_prompt(prompt["negative_prompt"])
        elif isinstance(prompt, str):
            prompt_ids, prompt_mask = self._tokenize_step_prompt(prompt)
        return prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask

    def _tokenize_step_prompt(self, text: str | list[str]):
        """Tokenize raw text for the step-execution warm-up request."""
        prompt = [text] if isinstance(text, str) else text
        formatted = [self.prompt_template_encode.format(item) for item in prompt]
        tokens = self.tokenizer(
            formatted,
            max_length=self.tokenizer_max_length + self.prompt_template_encode_start_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        return tokens.input_ids, tokens.attention_mask

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """Initialize request-local state for step execution."""
        sampling = state.sampling
        prompt_ids, prompt_mask, negative_prompt_ids, negative_prompt_mask = self._extract_step_prompt_ids(state.prompt)
        if prompt_ids is None:
            raise ValueError(
                f"{self.__class__.__name__}.prepare_encode requires either "
                "'prompt_token_ids' or a text 'prompt' on state.prompt."
            )

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_inference_steps = sampling.num_inference_steps or 50
        sigmas = sampling.sigmas or None
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 1.0
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        true_cfg_scale = coalesce_not_none(sampling.true_cfg_scale, 4.0)
        max_sequence_length = sampling.max_sequence_length or 512

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

        ctx = self._prepare_token_id_generation_context(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_mask=negative_prompt_mask,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=None,
            prompt_embeds=None,
            prompt_embeds_mask=None,
            negative_prompt_embeds=None,
            negative_prompt_embeds_mask=None,
            attention_kwargs=kwargs.get("attention_kwargs"),
            max_sequence_length=max_sequence_length,
        )

        request_scheduler = copy.deepcopy(self.scheduler)
        request_scheduler.set_begin_index(0)

        state.prompt_embeds = ctx["prompt_embeds"]
        state.prompt_embeds_mask = ctx["prompt_embeds_mask"]
        state.negative_prompt_embeds = ctx["negative_prompt_embeds"]
        state.negative_prompt_embeds_mask = ctx["negative_prompt_embeds_mask"]
        state.latents = ctx["latents"]
        state.timesteps = ctx["timesteps"]
        state.step_index = 0
        state.scheduler = request_scheduler
        state.do_true_cfg = ctx["do_true_cfg"]
        state.guidance = ctx["guidance"]
        state.img_shapes = ctx["img_shapes"]
        state.txt_seq_lens = ctx["txt_seq_lens"]
        state.negative_txt_seq_lens = ctx["negative_txt_seq_lens"]
        state.sampling.cfg_normalize = True
        state.extra["height"] = height
        state.extra["width"] = width
        return state

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode step execution output with DiffusionNFT training tensors."""
        self._current_timestep = None
        height = state.extra["height"]
        width = state.extra["width"]
        output = self._decode_latents(state.latents, height, width, kwargs.get("output_type") or "pil")

        latents_clean = state.latents.float()
        return replace(
            output,
            custom_output={
                "latents_clean": latents_clean,
                "train_timesteps": state.timesteps.unsqueeze(0).expand(latents_clean.shape[0], -1),
                "prompt_embeds": state.prompt_embeds,
                "prompt_embeds_mask": state.prompt_embeds_mask,
                "negative_prompt_embeds": state.negative_prompt_embeds,
                "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
            },
            to_cpu=True,
        )

    def _prepare_token_id_generation_context(
        self,
        *,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor | None,
        negative_prompt_ids: torch.Tensor | None,
        negative_prompt_mask: torch.Tensor | None,
        true_cfg_scale: float,
        height: int,
        width: int,
        num_inference_steps: int,
        sigmas: list[float] | None,
        guidance_scale: float,
        num_images_per_prompt: int,
        generator: torch.Generator | list[torch.Generator] | None,
        latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor | None,
        prompt_embeds_mask: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds_mask: torch.Tensor | None,
        attention_kwargs: dict[str, Any] | None,
        max_sequence_length: int,
    ) -> dict[str, Any]:
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._current_timestep = None
        self._interrupt = False

        if isinstance(prompt_ids, list):
            prompt_ids = torch.tensor(prompt_ids, device=self.device)
        if isinstance(negative_prompt_ids, list):
            negative_prompt_ids = torch.tensor(negative_prompt_ids, device=self.device)

        if prompt_ids is not None:
            batch_size = prompt_ids.shape[0] if prompt_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            raise ValueError("DiffusionNFT rollout requires either `prompt_ids` or `prompt_embeds`.")

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

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

        num_channels_latents = self.transformer.in_channels // 4
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
        img_shapes = build_img_shapes(height, width, batch_size, self.vae_scale_factor)

        timesteps, num_inference_steps = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "latents": latents,
            "img_shapes": img_shapes,
            "timesteps": timesteps,
            "do_true_cfg": do_true_cfg,
            "guidance": guidance,
            "txt_seq_lens": txt_seq_lens,
            "negative_txt_seq_lens": negative_txt_seq_lens,
        }

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt_ids: torch.Tensor | list[int] | None = None,
        prompt_mask: torch.Tensor | None = None,
        negative_prompt_ids: torch.Tensor | list[int] | None = None,
        negative_prompt_mask: torch.Tensor | None = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 1.0,
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
    ) -> DiffusionOutput:
        """Generate an image and return DiffusionNFT rollout metadata."""
        del callback_on_step_end_tensor_inputs

        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt_ids = custom_prompt.get("prompt_token_ids", prompt_ids)
            prompt_mask = custom_prompt.get("prompt_mask", prompt_mask)
            negative_prompt_ids = custom_prompt.get("negative_prompt_ids", negative_prompt_ids)
            negative_prompt_mask = custom_prompt.get("negative_prompt_mask", negative_prompt_mask)

        sampling_params = req.sampling_params
        height = sampling_params.height or height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or width or self.default_sample_size * self.vae_scale_factor
        height, width = normalize_min_aligned_size(height, width, self.vae_scale_factor * 2)
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

        if prompt_ids is None and prompt_embeds is None:
            return DiffusionOutput(output=None, custom_output={})

        ctx = self._prepare_token_id_generation_context(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_mask=negative_prompt_mask,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            attention_kwargs=attention_kwargs,
            max_sequence_length=max_sequence_length,
        )

        latents = super().diffuse(
            ctx["prompt_embeds"],
            ctx["prompt_embeds_mask"],
            ctx["negative_prompt_embeds"],
            ctx["negative_prompt_embeds_mask"],
            ctx["latents"],
            ctx["img_shapes"],
            ctx["txt_seq_lens"],
            ctx["negative_txt_seq_lens"],
            ctx["timesteps"],
            ctx["do_true_cfg"],
            ctx["guidance"],
            true_cfg_scale,
            image_latents=None,
            cfg_normalize=True,
            additional_transformer_kwargs={
                "return_dict": False,
                "attention_kwargs": self.attention_kwargs,
            },
        )

        self._current_timestep = None
        latents_clean = latents.float()
        decoded = self._decode_latents(latents, height, width, output_type or "pil")

        return DiffusionOutput(
            output=decoded.output,
            custom_output={
                "latents_clean": latents_clean,
                "train_timesteps": ctx["timesteps"].unsqueeze(0).expand(latents_clean.shape[0], -1),
                "prompt_embeds": ctx["prompt_embeds"],
                "prompt_embeds_mask": ctx["prompt_embeds_mask"],
                "negative_prompt_embeds": ctx["negative_prompt_embeds"],
                "negative_prompt_embeds_mask": ctx["negative_prompt_embeds_mask"],
            },
            to_cpu=True,
        )
