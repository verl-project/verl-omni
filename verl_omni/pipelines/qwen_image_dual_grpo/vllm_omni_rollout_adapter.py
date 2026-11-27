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

"""Rollout adapter for Qwen-Image Dual-GRPO (text encoder writing + DiT image generation)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

import torch
from verl.utils.torch_functional import get_response_mask
from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.common import coalesce_not_none
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb
from verl_omni.pipelines.request_batch import (
    collate_prompt_mask as _collate_prompt_mask,
)
from verl_omni.pipelines.request_batch import (
    collate_prompt_rows as _collate_prompt_rows,
)
from verl_omni.pipelines.request_batch import (
    split_diffusion_output_by_request as _split_diffusion_output_by_request,
)

__all__ = ["QwenImagePipelineWithDualLogProb"]

logger = logging.getLogger(__name__)


def _seed_rollout_device(seed: int, device: torch.device) -> None:
    """Seed the RNG ``generate`` samples from, without touching other devices."""
    if device.type == "cpu":
        torch.default_generator.manual_seed(seed)
        return
    device_mod = torch.get_device_module(device.type)
    with device_mod.device(device):
        device_mod.manual_seed(seed)


# system prompt used for DiT
SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
)


@dataclass
class TextEncoderGenerationResult:
    """Autoregressive text-encoder generation results."""

    ar_response_ids: torch.Tensor  # padded response ids
    ar_response_mask: torch.Tensor  # padded response mask
    ar_log_probs: torch.Tensor | None  # padded response log-probs
    text_encoder_responses: list[str]  # CoT + refined prompt texts


def _token_rows_to_lists(value: torch.Tensor) -> list[list[int]]:
    """One ``list[int]`` per batch row.

    Request split indexes the leading axis, then ``_maybe_unbatch`` unwraps that
    row to the flat list ``_pad_token_ids`` expects.
    """
    return value.detach().cpu().tolist()


def extract_prompt(texts: list[str]) -> list[str]:
    """Extracts the refined prompt from the model's reasoning output."""
    refined_prompts = []
    for text in texts:
        m = re.search(r"Revised Prompt:\n(.*)", text, re.DOTALL)
        if not m:
            m = re.search(r"Revised Prompt:(.*)", text, re.DOTALL)
        if m:
            refined_prompt = m.group(1).strip()
        else:
            logger.warning("Revised Prompt: regex missed; feeding full CoT into DiT.")
            refined_prompt = text.strip()
        refined_prompts.append(refined_prompt)
    return refined_prompts


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="dual_grpo")
class QwenImagePipelineWithDualLogProb(QwenImagePipelineWithLogProb):
    """Rollout pipeline of Qwen-Image for Dual-GRPO.
    It involves two alternative paths:
    (1) text encoder generate text tokens for each given prompt.
    For Simplicity, LLM only generates one response per prompt for each request.
    (2) image diffusion generates an image for each refined prompt.

    Extends :class:`QwenImagePipelineWithDualLogProb` by autoregressively sampling
    tokens from the Qwen2.5-VL text encoder before image diffusion.
    The pipeline returns:

    * DiT trajectory fields plus named ``image_preview`` / ``image_latent`` artifacts.
    * AR token ids, masks, and log-probabilities in rollout ``rl`` metadata, with no media payload.
    * The decoded text (``text_encoder_responses``).
    """

    supports_request_batch = False

    @staticmethod
    def build_ar_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """Text-only Qwen2.5-VL M-RoPE layout for composite AR postprocess.

        Returns ``(1, 4, seq_len)`` so ``left_right_2_no_padding`` can unpad the
        multi-channel positions used by the text encoder. Vision channels reuse
        the 1D text index until multimodal AR inputs are wired.
        """
        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, attention_mask.shape[-1]), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        # M-RoPE layout: B x 4 x seq_len.
        return text_position_ids.unsqueeze(1).expand(-1, 4, -1)

    def _get_qwen_text_response(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        return_logprobs: bool = True,
        **ar_kwargs,
    ):
        # Sampling reads the process-global RNG. Fork that state so this seed
        # stays inside this generate and is restored before DiT runs.
        seed = ar_kwargs.pop("seed", None)
        device = torch.device(self.device)
        fork_devices = [device] if seed is not None and device.type != "cpu" else []
        with torch.random.fork_rng(devices=fork_devices, enabled=seed is not None):
            if seed is not None:
                _seed_rollout_device(int(seed), device)

            # ref： https://github.com/verl-project/verl/blob/main/verl/workers/rollout/hf_rollout.py#L54
            outputs = self.text_encoder.generate(
                input_ids=prompt_ids.to(device),
                attention_mask=attention_mask.to(device),
                return_dict_in_generate=True,
                output_scores=return_logprobs,  # apply temperature, top_p, top_k, repetition_penalty
                **ar_kwargs,
            )
        max_new_tokens = ar_kwargs["max_new_tokens"]

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        seq = outputs.sequences
        sequence_length = prompt_ids.shape[1] + max_new_tokens
        delta_length = sequence_length - seq.shape[1]
        if delta_length > 0:
            delta_tokens = torch.ones(size=(seq.shape[0], delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = self.tokenizer.pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        output_ids = seq[:, prompt_ids.shape[1] :]  # remove prompt prefix
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)  # CoT+refined prompt

        logprobs = None
        if return_logprobs:
            scores = torch.stack(outputs.scores, dim=1)  # B x gen_len x vocab_size
            full_logprobs = torch.nn.functional.log_softmax(scores, dim=-1)
            gen_len = full_logprobs.shape[1]
            # Logprob of the sampled token only: (B, gen_len), not the full vocab.
            token_logprobs = full_logprobs.gather(-1, output_ids[:, :gen_len].unsqueeze(-1)).squeeze(-1)
            if gen_len < max_new_tokens:
                token_logprobs = torch.nn.functional.pad(token_logprobs, (0, max_new_tokens - gen_len), value=0.0)
            logprobs = token_logprobs

        return output_ids, logprobs, output_texts

    def generate_text_encoder_response(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        num_responses_per_prompt: int = 1,
        return_logprobs: bool = True,
        dtype: torch.dtype | None = None,
        **ar_kwargs,
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
            ar_kwargs: Additional argmuents for text generation.
        """
        # prepare input tensors
        dtype = dtype or self.text_encoder.dtype
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        prompt_ids = prompt_ids.unsqueeze(0) if prompt_ids.ndim == 1 else prompt_ids
        attention_mask = attention_mask.unsqueeze(0) if attention_mask.ndim == 1 else attention_mask

        # generate response for each prompt
        ar_response_ids_list: list[torch.Tensor] = []
        ar_all_log_probs: list[torch.Tensor] = []
        all_text_encoder_responses: list[str] = []
        for _ in range(num_responses_per_prompt):
            response_ids, ar_log_probs, text_encoder_responses = self._get_qwen_text_response(
                prompt_ids=prompt_ids,
                attention_mask=attention_mask,
                return_logprobs=return_logprobs,
                **ar_kwargs,
            )
            ar_response_ids_list.append(response_ids)
            if return_logprobs:
                ar_all_log_probs.append(ar_log_probs)
            all_text_encoder_responses.extend(text_encoder_responses)

        ar_response_ids = torch.cat(ar_response_ids_list, dim=0)  # B*num_responses_per_prompt x response_length
        if return_logprobs:
            ar_all_log_probs = torch.cat(ar_all_log_probs, dim=0)  # B*num_responses_per_prompt x response_length
        else:
            ar_all_log_probs = None

        # compute attention masks and position ids
        eos_ids = getattr(self.text_encoder.generation_config, "eos_token_id", None)
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        response_mask = get_response_mask(
            response_id=ar_response_ids, eos_token=eos_ids, dtype=attention_mask.dtype
        )  # B*num_responses_per_prompt x seq_len
        if ar_all_log_probs is not None:
            ar_all_log_probs = torch.where(response_mask.bool(), ar_all_log_probs, torch.zeros_like(ar_all_log_probs))

        return TextEncoderGenerationResult(
            ar_response_ids=ar_response_ids,
            ar_response_mask=response_mask,
            ar_log_probs=ar_all_log_probs,
            text_encoder_responses=all_text_encoder_responses,
        )

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
        response_length: int = 1024,  # llm max new tokens
        ar_logprobs: bool = True,  # calculate llm logprobs
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stage: Literal["ar", "diffusion"] = "diffusion",
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """End-to-end either text generation or image generation with rollout data collection.

        AR text generation:
        Autoregressively samples tokens from the text encoder.
        Returns token ids, per-token log-probabilities, and the decoded response text
        in rollout metadata, with an empty media payload.

        Diffusion image generation:
        Calls :meth:`QwenImagePipelineWithLogProb.forward` with the diffusion
        arguments. Sampling parameters in *req* take precedence over the keyword
        arguments.
        """

        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        sampling_params = request_batch.sampling_params_list[0]
        stage = coalesce_not_none(sampling_params.extra_args.get("stage", None), stage)

        if stage == "ar":
            prompts = request_batch.prompts
            prompt_token_ids, prompt_token_lengths = _collate_prompt_rows(
                prompts,
                ("prompt_ids",),
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
            ar_logprobs = coalesce_not_none(sampling_params.extra_args.get("ar_logprobs", None), ar_logprobs)
            temperature = coalesce_not_none(sampling_params.extra_args.get("temperature", None), temperature)
            top_p = coalesce_not_none(sampling_params.extra_args.get("top_p", None), top_p)
            top_k = int(coalesce_not_none(sampling_params.extra_args.get("top_k", None), top_k))
            response_length = int(
                coalesce_not_none(sampling_params.extra_args.get("response_length", None), response_length)
            )
            repetition_penalty = coalesce_not_none(
                sampling_params.extra_args.get("repetition_penalty", None), repetition_penalty
            )
            ar_kwargs = dict(
                max_new_tokens=response_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
            )
            if sampling_params.seed is not None:
                ar_kwargs["seed"] = int(sampling_params.seed)
            # generation
            num_responses_per_prompt = 1
            response = self.generate_text_encoder_response(
                prompt_ids=prompt_token_ids,
                attention_mask=prompt_mask,
                num_responses_per_prompt=num_responses_per_prompt,
                return_logprobs=ar_logprobs,
                **ar_kwargs,
            )

            # extract the actual image generation prompt
            refined_prompts = extract_prompt(response.text_encoder_responses)
            # apply chat template
            messages = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": e},
                ]
                for e in refined_prompts
            ]

            # Token stage only. An empty payload keeps the engine from treating this
            # call as an image; ``rl`` carries the AR tensors for the strategy.
            result = DiffusionOutput(
                output={
                    "payload": {},
                    "metadata": {
                        "rl": {
                            "ar_response_ids": _token_rows_to_lists(response.ar_response_ids),
                            "ar_response_mask": _token_rows_to_lists(response.ar_response_mask),
                            "ar_all_log_probs": response.ar_log_probs,
                            "refined_prompt": messages,  # formatted prompt
                            "text_encoder_responses": response.text_encoder_responses,  # CoT + prompt
                        }
                    },
                },
                to_cpu=True,
            )
            outputs = _split_diffusion_output_by_request(
                result,
                request_batch,
                num_outputs_per_prompt=num_images_per_prompt,
            )
            return outputs if return_batch else outputs[0]

        return super().forward(
            req,
            prompt_token_ids=prompt_token_ids,
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
            output_type=output_type,
            attention_kwargs=attention_kwargs,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            noise_level=noise_level,
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            sde_type=sde_type,
            logprobs=logprobs,
        )
