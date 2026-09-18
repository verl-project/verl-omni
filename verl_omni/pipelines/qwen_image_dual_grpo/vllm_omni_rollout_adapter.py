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

"""Rollout adapter for Qwen-Image Dual-GRPO (text encoder + DiT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import rollout_output
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import build_img_shapes, coalesce_not_none
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
from verl_omni.pipelines.request_batch import (
    collate_prompt_mask as _collate_prompt_mask,
)
from verl_omni.pipelines.request_batch import (
    collate_prompt_rows as _collate_prompt_rows,
)
from verl_omni.pipelines.request_batch import (
    sample_per_sample_sde_windows as _sample_per_sample_sde_windows,
)
from verl_omni.pipelines.request_batch import (
    split_diffusion_output_by_request as _split_diffusion_output_by_request,
)

__all__ = ["QwenImagePipelineWithDualLogProb"]


@dataclass
class TextEncoderGenerationResult:
    """Autoregressive text-encoder generation results."""

    llm_response_ids: torch.Tensor
    llm_log_probs: torch.Tensor | None
    text_encoder_responses: list[str]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dual_grpo")
class QwenImagePipelineWithDualLogProb(QwenImagePipelineWithLogProb):
    """Rollout pipeline of Qwen-Image for Dual-GRPO.
    It involves two paths:
    (1) text encoder generate text tokens for each given prompt.
    For Simplicity, LLM only generates one response per prompt for each request.
    (2) image diffusion generates an image for each given prompt.

    Extends :class:`QwenImagePipelineWithDualLogProb` by autoregressively sampling
    tokens from the Qwen2.5-VL text encoder before image diffusion.
    The pipeline returns:

    * DiT trajectory log-probabilities (``all_log_probs``) from the SDE window.
    * Text-encoder token log-probabilities (``llm_all_log_probs``) for the generated text.
    * The decoded text (``text_encoder_responses``).
    """

    def _get_qwen_text_response(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        return_logprobs: bool = True,
        llm_kwargs: dict[str, Any] = None,
    ):
        outputs = self.text_encoder.generate(
            input_ids=prompt_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            return_dict_in_generate=True,
            output_scores=return_logprobs,
            **llm_kwargs,
        )
        if return_logprobs:
            scores = torch.stack(outputs.scores, dim=1)  # B x gen_seq_len x vocab_size
            logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
        else:
            logprobs = None
        output_ids = outputs.sequences
        output_ids = output_ids[:, prompt_ids.shape[1] :]  # remove prompt prefix
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return TextEncoderGenerationResult(
            llm_response_ids=output_ids,
            llm_log_probs=logprobs,
            text_encoder_responses=output_texts,
        )

    def generate_text_encoder_response(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_responses_per_prompt: int = 1,
        return_logprobs: bool = True,
        dtype: torch.dtype | None = None,
        llm_kwargs: dict[str, Any] = None,
    ):
        """Text encoder response generation.

        Args:
            prompt_ids (torch.Tensor): Token IDs of shape ``(B, L)`` or ``(L,)``.
            attention_mask (torch.Tensor, *optional*): Boolean mask of shape
                ``(B, L)`` for *prompt_ids*; inferred as all-ones when ``None``.
            num_responses_per_prompt (int): Number of responses to generate per prompt;
                tokens and logprobs are repeated accordingly.
            return_logprobs (bool): Whether to calculate log-probabilities for generated tokens.
            dtype (torch.dtype, *optiional*): Data type for text encoder.
            llm_kwargs (dict): Additional argmuents for text generation.

        Returns:
            tuple[torch.Tensor, torch.Tensor | None, list[str]]: A tuple of
            ``llm_response_ids``: tensor of shape ``(B * num_responses_per_prompt, input_len+gen_seq_len)``
            ``llm_all_log_probs``: tensor of shape ``(B * num_responses_per_prompt, gen_seq_len, vacab_size)``
            ``text_encoder_responses``: a list of text responses


        """
        # prepare input tensors
        dtype = dtype or self.text_encoder.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        # gerenete response for each prompt
        llm_response_ids = []
        llm_all_log_probs = []
        text_encoder_responses: list[str] = []
        for _ in range(num_responses_per_prompt):
            response = self._get_qwen_text_response(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                return_logprobs=return_logprobs,
                llm_kwargs=llm_kwargs,
            )
            llm_response_ids.append(response.llm_response_ids)
            if return_logprobs:
                llm_all_log_probs.append(response.llm_log_probs)
            text_encoder_responses.extend(response.text_encoder_responses)

        llm_response_ids = torch.cat(llm_response_ids, dim=0)  # B*num_responses_per_prompt x input_len+gen_seq_len
        if return_logprobs:
            llm_all_log_probs = torch.cat(
                llm_all_log_probs, dim=0
            )  # ，B*num_responses_per_prompt x gen_seq_len x vacab_size
        else:
            llm_all_log_probs = None

        return llm_response_ids, llm_all_log_probs, text_encoder_responses

    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        prompt_token_ids: torch.Tensor | list[int] | None = None,
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
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
        max_new_tokens: int = 256,  # llm max new tokens
        llm_logprobs: bool = True,  # calculate llm logprobs
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """End-to-end text generation and image generation with rollout data collection.

        Text generation:
        Autoregressively samples tokens from the text encoder.
        Returns the all generated token ids, per-token log-probabilities, and the decoded response text.

        Image generation:
        Encodes the prompt, prepares latents, runs the SDE diffusion loop via
        :meth:`diffuse`, and decodes the final latents through the VAE.  Sampling
        parameters in *req* take precedence over the keyword arguments.
        """

        # input and args preparation
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        prompts = request_batch.prompts
        prompt_token_ids, prompt_token_lengths = _collate_prompt_rows(
            prompts,
            ("prompt_token_ids", "prompt_ids"),
            prompt_token_ids,
            device=self.device,
            field_name="prompt_token_ids",
        )
        prompt_mask = _collate_prompt_mask(
            prompts,
            ("prompt_mask",),
            prompt_mask,
            device=self.device,
            field_name="prompt_mask",
            token_lengths=prompt_token_lengths,
            target_seq_len=None if prompt_token_ids is None else int(prompt_token_ids.shape[1]),
        )
        negative_prompt_ids, negative_prompt_lengths = _collate_prompt_rows(
            prompts,
            ("negative_prompt_ids",),
            negative_prompt_ids,
            device=self.device,
            field_name="negative_prompt_ids",
        )
        negative_prompt_mask = _collate_prompt_mask(
            prompts,
            ("negative_prompt_mask",),
            negative_prompt_mask,
            device=self.device,
            field_name="negative_prompt_mask",
            token_lengths=negative_prompt_lengths,
            target_seq_len=None if negative_prompt_ids is None else int(negative_prompt_ids.shape[1]),
        )

        sampling_params = request_batch.sampling_params_list[0]
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        output_type = sampling_params.output_type or output_type

        noise_level = coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = coalesce_not_none(sampling_params.extra_args.get("sde_window_range", None), sde_window_range)
        sde_type = coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        for request in request_batch.requests:
            request_sampling_params = request.sampling_params
            if request_sampling_params.generator is None and request_sampling_params.seed is not None:
                request_sampling_params.generator = torch.Generator(device=self.device).manual_seed(
                    request_sampling_params.seed
                )
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        if getattr(sampling_params, "guidance_scale_provided", False):
            guidance_scale = sampling_params.guidance_scale
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs
        generator = request_batch.collate_request_generators(num_images_per_prompt, generator)
        latents = request_batch.collate_request_tensors("latents", latents)

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt_token_ids is not None:
            if isinstance(prompt_token_ids, list):
                prompt_token_ids = torch.tensor(prompt_token_ids, device=self.device)
            batch_size = prompt_token_ids.shape[0] if prompt_token_ids.ndim == 2 else 1
        elif prompt_embeds is not None:
            batch_size = prompt_embeds.shape[0]
        else:
            # Both prompt_token_ids and prompt_embeds are None (e.g. during warmup/dummy run).
            # Return a minimal dummy output to avoid crashing.
            outputs = [DiffusionOutput(output=None) for _ in range(request_batch.num_reqs)]
            return outputs if return_batch else outputs[0]

        has_neg_prompt = negative_prompt_ids is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            prompt_ids=prompt_token_ids,
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

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        sde_window = _sample_per_sample_sde_windows(
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range if sde_window_range is not None else (0, 5),
            num_timesteps=len(timesteps),
            batch_size=latents.shape[0],
            generator=generator,
            device=self.device,
        )

        # diffusion rollout, image generation
        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
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
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        # LLM generation
        llm_logprobs = coalesce_not_none(sampling_params.extra_args.get("llm_logprobs", None), llm_logprobs)
        temperature = coalesce_not_none(sampling_params.extra_args.get("temperature", None), temperature)
        top_p = coalesce_not_none(sampling_params.extra_args.get("top_p", None), top_p)
        top_k = int(coalesce_not_none(sampling_params.extra_args.get("top_k", None), top_k))
        max_new_tokens = int(coalesce_not_none(sampling_params.extra_args.get("max_new_tokens", None), max_new_tokens))
        repetition_penalty = coalesce_not_none(sampling_params.extra_args.get("repetition_penalty", None), 1.0)

        llm_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        llm_response_ids, llm_all_log_probs, text_encoder_responses = self.generate_text_encoder_response(
            prompt_ids=prompt_token_ids,
            attention_mask=prompt_mask,
            num_responses_per_prompt=num_images_per_prompt,  # reused for num responses per prompt in LLM generation
            return_logprobs=llm_logprobs,
            llm_kwargs=llm_kwargs,
        )

        result = rollout_output(
            media=image,
            trajectory_latents=all_latents,
            trajectory_log_probs=all_log_probs,
            trajectory_timesteps=all_timesteps,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            },
            rl={
                "llm_response_ids": llm_response_ids,
                "llm_all_log_probs": llm_all_log_probs,
                "text_encoder_responses": text_encoder_responses,
            },
            to_cpu=True,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs if return_batch else outputs[0]
