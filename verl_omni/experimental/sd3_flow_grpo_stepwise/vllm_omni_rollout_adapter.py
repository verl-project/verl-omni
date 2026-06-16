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

"""Experimental SD3.5 stepwise vLLM-Omni rollout adapter for FlowGRPO."""

from __future__ import annotations

import ast
import copy
import logging
import os
from typing import Any, ClassVar, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import DiffusionRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

__all__ = ["StableDiffusion3PipelineWithLogProb"]

logger = logging.getLogger(__name__)


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


def _prompt_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    return None


def _extract_model_prompts(prompts: list, extra_args: dict[str, Any] | None) -> tuple[list[str] | None, list[str] | None]:
    prompt, negative_prompt = _extract_text_prompts(prompts)
    extra_args = extra_args or {}
    prompt = prompt or _prompt_list(extra_args.get("_model_prompt"))
    negative_prompt = negative_prompt or _prompt_list(extra_args.get("_model_negative_prompt"))
    return prompt, negative_prompt


def _sd3_image_seq_len(height: int, width: int, vae_scale_factor: int = 8) -> int:
    patch_size = 2
    return (int(height) // vae_scale_factor // patch_size) * (int(width) // vae_scale_factor // patch_size)


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")
class StableDiffusion3PipelineWithLogProb(StableDiffusion3Pipeline):
    """Experimental stepwise SD3.5 rollout pipeline that returns FlowGRPO trajectory data.

    Differences from the upstream pipeline:

    - the Euler flow-match scheduler is replaced by
      :class:`FlowMatchSDEDiscreteScheduler` so SDE-window sampling produces
      per-step log-probabilities;
    - ``forward`` collects ``all_latents`` / ``all_log_probs`` /
      ``all_timesteps`` and ships prompt embeddings (sequence + pooled)
      through ``custom_output`` for training-side log-prob recomputation;
    - the step-execution contract (``prepare_encode`` / ``denoise_step`` /
      ``step_scheduler`` / ``post_decode``) is implemented so the rollout can
      run with ``step_execution=True`` (step-wise continuous batching);
    - CFG is plain SD3 guidance (``guidance_scale``); the convergence-test
      default is non-CFG (``guidance_scale <= 1`` skips the negative branch
      entirely, halving the transformer NFE).
    """

    supports_step_execution: ClassVar[bool] = True

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
        # request_id -> pooled embeds, consumed by ``denoise_step`` because
        # ``InputBatch`` only carries the shared step-execution fields.
        self._pooled_cache: dict[str, torch.Tensor] = {}
        self._negative_pooled_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Shared request preparation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Step-execution contract
    # ------------------------------------------------------------------

    def prepare_encode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionRequestState:
        """One-time request preparation for ``step_execution=True``.

        Mirrors the setup phase of :meth:`forward`: raw text prompts are
        encoded with SD3's three text encoders, latents and the SDE timestep
        schedule are initialised, and a per-request scheduler copy is stored
        on *state* so concurrent requests do not share scheduler state.
        """
        del kwargs
        sampling = state.sampling
        prompt, negative_prompt = _extract_model_prompts(state.prompts or [], sampling.extra_args)
        if prompt is None:
            raise ValueError(
                "StableDiffusion3PipelineWithLogProb.prepare_encode requires a text 'prompt' in state.prompts."
            )

        height = sampling.height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling.num_inference_steps or 28
        sigmas = sampling.sigmas
        guidance_scale = sampling.guidance_scale if sampling.guidance_scale_provided else 0.0
        num_images_per_prompt = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        max_sequence_length = sampling.max_sequence_length or 256

        generator = sampling.generator
        if generator is None and sampling.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling.seed)

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
        if do_cfg:
            negative_prompt = negative_prompt or [""] * batch_size
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=None,
                prompt_3=None,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None

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
            None,
        )

        timesteps, _ = self.prepare_timesteps(num_inference_steps, sigmas, _sd3_image_seq_len(height, width))
        self._num_timesteps = len(timesteps)

        req_scheduler = copy.deepcopy(self.scheduler)
        req_scheduler.set_begin_index(0)

        sde_args = self._resolve_sde_args(sampling)
        sde_window = self._sample_sde_window(
            sde_args["sde_window_size"], sde_args["sde_window_range"], len(timesteps), generator
        )

        state.prompt_embeds = prompt_embeds
        state.prompt_embeds_mask = prompt_embeds_mask
        state.negative_prompt_embeds = negative_prompt_embeds
        state.negative_prompt_embeds_mask = negative_prompt_embeds_mask
        state.latents = latents
        state.timesteps = timesteps
        state.step_index = 0
        state.scheduler = req_scheduler
        state.do_true_cfg = do_cfg
        state.guidance = None
        # ``InputBatch`` reads the CFG scale from ``sampling.true_cfg_scale``;
        # SD3 uses plain ``guidance_scale``, so mirror it there for batching.
        state.sampling.true_cfg_scale = guidance_scale if do_cfg else None
        state.sampling.cfg_normalize = False
        # Persist the resolved generator so ``step_scheduler`` keeps drawing
        # from the same RNG stream as ``forward()``.
        state.sampling.generator = generator
        # Rollout / SDE state consumed by ``step_scheduler`` and packaged
        # into ``custom_output`` by ``post_decode``.
        state.sde_window = sde_window
        state.noise_level = sde_args["noise_level"]
        state.sde_type = sde_args["sde_type"]
        state.logprobs = sde_args["logprobs"]
        state.all_latents = []
        state.all_log_probs = []
        state.all_timesteps = []

        self._pooled_cache[state.request_id] = pooled_prompt_embeds
        if negative_pooled_prompt_embeds is not None:
            self._negative_pooled_cache[state.request_id] = negative_pooled_prompt_embeds

        return state

    def _gather_pooled(self, cache: dict[str, torch.Tensor], request_ids: list[str]) -> torch.Tensor | None:
        rows = [cache.get(request_id) for request_id in request_ids]
        if any(row is None for row in rows):
            return None
        return torch.cat(rows, dim=0)

    def denoise_step(self, input_batch: InputBatch, **kwargs: Any) -> torch.Tensor | None:
        """One transformer forward (optionally with CFG) for the current step."""
        del kwargs
        if self.interrupt:
            return None

        t = input_batch.timesteps
        self._current_timestep = t

        model_dtype = self._model_dtype()
        x = input_batch.latents.to(model_dtype)
        timestep = t.to(device=x.device, dtype=model_dtype)

        pooled_prompt_embeds = self._gather_pooled(self._pooled_cache, input_batch.request_ids)
        if pooled_prompt_embeds is None:
            raise RuntimeError("SD3 step execution is missing pooled prompt embeds for the current batch.")

        positive_kwargs = {
            "hidden_states": x,
            "timestep": timestep,
            "encoder_hidden_states": input_batch.prompt_embeds.to(model_dtype),
            "pooled_projections": pooled_prompt_embeds.to(model_dtype),
            "return_dict": False,
        }
        negative_kwargs = None
        if input_batch.do_true_cfg:
            negative_pooled = self._gather_pooled(self._negative_pooled_cache, input_batch.request_ids)
            if input_batch.negative_prompt_embeds is None or negative_pooled is None:
                raise RuntimeError("SD3 CFG step execution is missing negative prompt embeds for the current batch.")
            negative_kwargs = {
                "hidden_states": x,
                "timestep": timestep,
                "encoder_hidden_states": input_batch.negative_prompt_embeds.to(model_dtype),
                "pooled_projections": negative_pooled.to(model_dtype),
                "return_dict": False,
            }

        noise_pred = self.predict_noise_maybe_with_cfg(
            input_batch.do_true_cfg,
            input_batch.true_cfg_scale,
            positive_kwargs,
            negative_kwargs,
            input_batch.cfg_normalize,
        )
        # ``step_scheduler`` expects an fp32 noise prediction.
        return noise_pred.float()

    def step_scheduler(
        self,
        state: DiffusionRequestState,
        noise_pred: torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """One SDE scheduler step with FlowGRPO trajectory bookkeeping."""
        del kwargs
        if self.interrupt:
            return

        i = state.step_index
        timestep_value = state.timesteps[i]
        sde_window = state.sde_window

        if i < sde_window[0]:
            cur_noise_level = 0.0
        elif i == sde_window[0]:
            cur_noise_level = state.noise_level
            state.all_latents.append(state.latents.float())
        elif i > sde_window[0] and i < sde_window[1]:
            cur_noise_level = state.noise_level
        else:
            cur_noise_level = 0.0

        new_latents, log_prob, _, _ = state.scheduler.step(
            noise_pred.float(),
            timestep_value,
            state.latents,
            generator=state.sampling.generator,
            noise_level=cur_noise_level,
            sde_type=state.sde_type,
            return_logprobs=state.logprobs,
            return_dict=False,
        )
        # Save fp32 trajectory BEFORE casting live state to model dtype,
        # so the trainer later recomputes log-probs on full-precision latents.
        if i >= sde_window[0] and i < sde_window[1]:
            state.all_latents.append(new_latents.float())
            state.all_log_probs.append(log_prob)
            state.all_timesteps.append(timestep_value)

        state.latents = new_latents.to(self._model_dtype())
        state.step_index += 1

    def post_decode(
        self,
        state: DiffusionRequestState,
        **kwargs: Any,
    ) -> DiffusionOutput:
        """Decode final latents, package rollout trajectory, and move to CPU.

        In ``step_execution`` mode the worker ships the returned
        :class:`DiffusionOutput` across an inter-process MessageQueue to the
        ``vLLMOmniHttpServer`` actor, so tensors are moved to CPU and
        ``custom_output`` is populated with the same trajectory fields that
        :meth:`forward` produces.
        """
        del kwargs
        self._current_timestep = None

        pooled_prompt_embeds = self._pooled_cache.get(state.request_id)
        negative_pooled_prompt_embeds = self._negative_pooled_cache.get(state.request_id)
        try:
            image = self._decode_latents(state.latents)

            all_latents = state.all_latents
            all_log_probs = state.all_log_probs
            all_timesteps = state.all_timesteps

            stacked_latents = torch.stack(all_latents, dim=1) if all_latents else None
            stacked_log_probs = (
                torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
            )
            stacked_timesteps = (
                torch.stack(all_timesteps).unsqueeze(0).expand(state.latents.shape[0], -1) if all_timesteps else None
            )

            return DiffusionOutput(
                output=image,
                custom_output={
                    "all_latents": stacked_latents,
                    "all_log_probs": stacked_log_probs,
                    "all_timesteps": stacked_timesteps,
                    "prompt_embeds": state.prompt_embeds,
                    "prompt_embeds_mask": state.prompt_embeds_mask,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "negative_prompt_embeds": state.negative_prompt_embeds,
                    "negative_prompt_embeds_mask": state.negative_prompt_embeds_mask,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
                },
                to_cpu=True,
            )
        finally:
            self._pooled_cache.pop(state.request_id, None)
            self._negative_pooled_cache.pop(state.request_id, None)

    def _decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        if self.output_type == "latent":
            return latents
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0]

    # ------------------------------------------------------------------
    # Request-level path (step_execution=False)
    # ------------------------------------------------------------------

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
        req: OmniDiffusionRequest,
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
    ) -> DiffusionOutput:
        """End-to-end SD3.5 generation with rollout trajectory collection."""
        req_prompt, req_negative_prompt = _extract_model_prompts(req.prompts or [], req.sampling_params.extra_args)
        prompt = req_prompt if req_prompt is not None else prompt
        negative_prompt = req_negative_prompt if req_negative_prompt is not None else negative_prompt

        if prompt is None:
            # Engine warm-up / dummy run without a usable prompt.
            return DiffusionOutput(output=None, custom_output={})
        if isinstance(prompt, str):
            prompt = [prompt]

        sampling_params = req.sampling_params
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

        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        req_num_outputs = getattr(sampling_params, "num_outputs_per_prompt", None)
        if req_num_outputs and req_num_outputs > 0:
            num_images_per_prompt = req_num_outputs

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
            generator if not isinstance(generator, list) else (generator[0] if generator else None),
        )

        if req.request_id == DUMMY_DIFFUSION_REQUEST_ID and sde_window[0] == sde_window[1]:
            image = self._decode_latents(latents)
            return DiffusionOutput(output=image, custom_output={}, to_cpu=True)

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
            generator if not isinstance(generator, list) else (generator[0] if generator else None),
            logprobs,
        )

        self._current_timestep = None
        image = self._decode_latents(latents)

        return DiffusionOutput(
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
