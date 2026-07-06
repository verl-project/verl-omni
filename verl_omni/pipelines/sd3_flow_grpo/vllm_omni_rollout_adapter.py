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

"""SD3.5 vLLM-Omni rollout adapter for FlowGRPO."""

from __future__ import annotations

import ast
import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.utils import split_diffusion_output_by_request as _split_diffusion_output_by_request

__all__ = ["StableDiffusion3PipelineWithLogProb"]


def _coalesce_not_none(value, default):
    return default if value is None else value


def _normalize_sde_window_args(
    sde_window_size: int | str | None,
    sde_window_range: tuple[int, int] | list[int] | str,
) -> tuple[int | None, tuple[int, int]]:
    if sde_window_size is not None:
        sde_window_size = int(sde_window_size)
    if isinstance(sde_window_range, str):
        sde_window_range = ast.literal_eval(sde_window_range)
    if len(sde_window_range) != 2:
        raise ValueError("SD3 rollout sde_window_range must contain exactly two values.")
    return sde_window_size, (int(sde_window_range[0]), int(sde_window_range[1]))


def _extract_text_prompts(prompts: list) -> tuple[list[str] | None, list[str] | None]:
    if not prompts:
        return None, None

    prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts]
    if all(not item for item in prompt):
        prompt = None

    negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
    if all(not item for item in negative_prompt):
        negative_prompt = None

    return prompt, negative_prompt


def _to_token_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, list):
        return value
    return None


def _extract_prompt_ids(prompts: list) -> tuple[list[list[int]] | None, list[list[int]] | None]:
    if not prompts:
        return None, None

    prompt_ids_list: list[list[int]] = []
    negative_prompt_ids_list: list[list[int]] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            continue
        prompt_ids = _to_token_list(prompt.get("prompt_token_ids"))
        negative_prompt_ids = _to_token_list(prompt.get("negative_prompt_ids"))
        if prompt_ids is not None:
            prompt_ids_list.append(prompt_ids)
        if negative_prompt_ids is not None:
            negative_prompt_ids_list.append(negative_prompt_ids)

    if not prompt_ids_list:
        return None, None
    negative_prompt_ids = negative_prompt_ids_list if negative_prompt_ids_list else None
    return prompt_ids_list, negative_prompt_ids


def _decode_prompt_ids(tokenizer, prompt_ids_list: list[list[int]]) -> list[str]:
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in prompt_ids_list]


def _sd3_image_seq_len(height: int, width: int, vae_scale_factor: int = 8) -> int:
    patch_size = 2
    return (int(height) // vae_scale_factor // patch_size) * (int(width) // vae_scale_factor // patch_size)


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")
class StableDiffusion3PipelineWithLogProb(StableDiffusion3Pipeline):
    """SD3.5 rollout pipeline that returns FlowGRPO trajectory data.

    Differences from the upstream pipeline:

    - the Euler flow-match scheduler is replaced by
      :class:`FlowMatchSDEDiscreteScheduler` so SDE-window sampling produces
      per-step log-probabilities;
    - ``forward`` collects ``all_latents`` / ``all_log_probs`` /
      ``all_timesteps`` and ships prompt embeddings (sequence + pooled)
      through ``custom_output`` for training-side log-prob recomputation;
    - CFG is plain SD3 guidance (``guidance_scale``); the convergence-test
      default is non-CFG (``guidance_scale <= 1`` skips the negative branch
      entirely, halving the transformer NFE).
    """

    supports_request_batch = True

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    def _resolve_sde_args(self, sampling) -> dict[str, Any]:
        extra = sampling.extra_args or {}
        noise_level = _coalesce_not_none(extra.get("noise_level", None), 0.7)
        sde_window_size = _coalesce_not_none(extra.get("sde_window_size", None), None)
        sde_window_range = _coalesce_not_none(extra.get("sde_window_range", None), (0, 5))
        sde_window_size, sde_window_range = _normalize_sde_window_args(sde_window_size, sde_window_range)
        sde_type = _coalesce_not_none(extra.get("sde_type", None), "sde")
        logprobs = _coalesce_not_none(extra.get("logprobs", None), True)
        return {
            "noise_level": noise_level,
            "sde_window_size": sde_window_size,
            "sde_window_range": sde_window_range,
            "sde_type": sde_type,
            "logprobs": logprobs,
        }

    def _sample_sde_window(
        self,
        sde_window_size: int | None,
        sde_window_range: tuple[int, int],
        num_timesteps: int,
        generator: torch.Generator | None,
    ) -> tuple[int, int]:
        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            return (start, start + sde_window_size)
        return (0, num_timesteps - 1)

    def _model_dtype(self) -> torch.dtype:
        return self.od_config.dtype

    def _to_encode_prompt_text(self, prompts: list) -> tuple[list[str] | None, list[str] | None]:
        """Convert vLLM-Omni request prompts to plain text for ``encode_prompt()``."""
        prompt, negative_prompt = _extract_text_prompts(prompts)
        if prompt is not None:
            return prompt, negative_prompt

        prompt_ids, negative_prompt_ids = _extract_prompt_ids(prompts)
        if prompt_ids is None:
            return None, None

        prompt = _decode_prompt_ids(self.tokenizer, prompt_ids)
        if negative_prompt_ids is not None:
            negative_prompt = _decode_prompt_ids(self.tokenizer, negative_prompt_ids)
        return prompt, negative_prompt

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.output_type == "latent":
            return latents
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0]

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        do_cfg: bool,
        guidance_scale: float,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run the full SDE diffusion loop and collect per-step rollout data."""
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        model_dtype = self._model_dtype()
        self.scheduler.set_begin_index(0)

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(device=self.device, dtype=model_dtype)

            # Cast to model dtype for the transformer forward (the scheduler
            # returns fp32 latents).
            x = latents.to(model_dtype)

            positive_kwargs = {
                "hidden_states": x,
                "timestep": timestep,
                "encoder_hidden_states": prompt_embeds,
                "pooled_projections": pooled_prompt_embeds,
                "return_dict": False,
            }
            negative_kwargs = None
            if do_cfg:
                negative_kwargs = {
                    "hidden_states": x,
                    "timestep": timestep,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "pooled_projections": negative_pooled_prompt_embeds,
                    "return_dict": False,
                }

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_cfg,
                guidance_scale,
                positive_kwargs,
                negative_kwargs,
                False,
            )

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            # Save fp32 trajectory BEFORE casting back to model dtype, so the
            # trainer recomputes log-probs on full-precision latents.
            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents.float())
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        max_sequence_length: int = 256,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """End-to-end SD3.5 generation with rollout trajectory collection."""
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        req_prompt, req_negative_prompt = self._to_encode_prompt_text(request_batch.prompts or [])
        prompt = req_prompt if req_prompt is not None else prompt
        negative_prompt = req_negative_prompt if req_negative_prompt is not None else negative_prompt

        if prompt is None:
            # Engine warm-up / dummy run without a usable prompt.
            outputs = [DiffusionOutput(output=None, custom_output={}) for _ in range(request_batch.num_reqs)]
            return outputs if return_batch else outputs[0]
        if isinstance(prompt, str):
            prompt = [prompt]

        sampling_params = request_batch.sampling_params_list[0]
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length
        if sampling_params.guidance_scale_provided:
            guidance_scale = sampling_params.guidance_scale

        noise_level = _coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = _coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = _coalesce_not_none(
            sampling_params.extra_args.get("sde_window_range", None), sde_window_range
        )
        sde_window_size, sde_window_range = _normalize_sde_window_args(sde_window_size, sde_window_range)
        sde_type = _coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = _coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs
        for request in request_batch.requests:
            request_sampling_params = request.sampling_params
            if request_sampling_params.generator is None and request_sampling_params.seed is not None:
                request_sampling_params.generator = torch.Generator(device=self.device).manual_seed(
                    request_sampling_params.seed
                )
        generator = request_batch.collate_request_generators(num_images_per_prompt, generator)
        sde_generator = generator[0] if isinstance(generator, list) else generator
        latents = request_batch.collate_request_tensors("latents", latents)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        batch_size = len(prompt)
        do_cfg = guidance_scale > 1

        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt,
        )
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_cfg:
            negative_prompt = negative_prompt or [""] * batch_size
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=None,
                prompt_3=None,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )

        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.int64, device=prompt_embeds.device)
        negative_prompt_embeds_mask = (
            torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.int64, device=negative_prompt_embeds.device)
            if negative_prompt_embeds is not None
            else None
        )

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            generator,
            latents,
        )

        timesteps, num_inference_steps = self.prepare_timesteps(
            num_inference_steps, sigmas, _sd3_image_seq_len(height, width)
        )
        self._num_timesteps = len(timesteps)

        sde_window = self._sample_sde_window(
            sde_window_size,
            sde_window_range,
            len(timesteps),
            sde_generator,
        )

        if request_batch.requests[0].request_id == DUMMY_DIFFUSION_REQUEST_ID and sde_window[0] == sde_window[1]:
            image = self._decode_latents(latents)
            result = DiffusionOutput(output=image, custom_output={}, to_cpu=True)
            outputs = _split_diffusion_output_by_request(
                result,
                request_batch,
                num_outputs_per_prompt=num_images_per_prompt,
            )
            return outputs if return_batch else outputs[0]

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            latents,
            timesteps,
            do_cfg,
            guidance_scale,
            noise_level,
            sde_window,
            sde_type,
            sde_generator,
            logprobs,
        )

        self._current_timestep = None
        image = self._decode_latents(latents)

        result = DiffusionOutput(
            output=image,
            custom_output={
                "all_latents": all_latents,
                "all_log_probs": all_log_probs,
                "all_timesteps": all_timesteps,
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            },
            to_cpu=True,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs if return_batch else outputs[0]
