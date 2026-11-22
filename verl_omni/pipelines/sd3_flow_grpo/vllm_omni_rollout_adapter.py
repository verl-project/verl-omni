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
import copy
import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.sd3 import pipeline_sd3
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.models.sd3.step_state import export_step_state, restore_step_state, step_execution_profile
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from verl_omni.pipelines.diffusion_rollout_output import (
    install_rollout_postprocessor,
    rollout_output,
    with_visual_artifacts,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import requested_outputs_for_batch
from verl_omni.pipelines.request_batch import (
    sample_per_sample_sde_windows as _sample_per_sample_sde_windows,
)
from verl_omni.pipelines.request_batch import (
    split_diffusion_output_by_request as _split_diffusion_output_by_request,
)
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.rollout_request import prompt_ids_from_payload
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.sd3_flow_grpo.common import (
    SD3_CLIP_TOKENS_KEY,
    SD3_ENCODER_TOKEN_KEYS,
    SD3_T5_TOKENS_KEY,
    SD3TokenIdPromptMixin,
)

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


def _extract_extra_prompt_ids(prompts: list, key: str = "extra_prompt_ids") -> dict[str, list[list[int]]] | None:
    """Extract per-text-encoder token ids (agent-loop tokenized) from request prompts.

    Returns ``{encoder_name: [ids_per_prompt, ...]}`` or None when any request
    prompt lacks the per-encoder ids.
    """
    if not prompts:
        return None

    per_prompt: list[dict] = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            return None
        extra = (prompt.get("extra_args") or {}).get(key)
        if prompt.get(key) is not None:
            raise ValueError(f"SD3 {key} must be nested under extra_args, not at the prompt top level")
        if not extra:
            return None
        per_prompt.append(extra)

    for idx, extra in enumerate(per_prompt):
        missing = [name for name in SD3_ENCODER_TOKEN_KEYS if name not in extra]
        if missing:
            raise ValueError(
                f"SD3 rollout requires {list(SD3_ENCODER_TOKEN_KEYS)} entries in '{key}' but {missing} "
                f"are missing for prompt index {idx}. Configure `actor_rollout_ref.model.extra_tokenizers` "
                "with these names (see examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh)."
            )
        for name in SD3_ENCODER_TOKEN_KEYS:
            if _to_token_list(extra[name]) is None:
                raise ValueError(
                    f"SD3 rollout expected list or tensor token ids for '{key}[{name}]' at prompt index {idx}, "
                    f"got {type(extra[name]).__name__}."
                )
    return {name: [_to_token_list(extra[name]) for extra in per_prompt] for name in SD3_ENCODER_TOKEN_KEYS}


def _sd3_image_seq_len(height: int, width: int, vae_scale_factor: int = 8) -> int:
    patch_size = 2
    return (int(height) // vae_scale_factor // patch_size) * (int(width) // vae_scale_factor // patch_size)


def _validate_output_type(output_type: str) -> Literal["image", "latent", "both"]:
    if output_type not in ("image", "latent", "both"):
        raise ValueError(f"SD3 rollout output_type must be one of ['image', 'latent', 'both'], got {output_type!r}.")
    return output_type


def _resolve_output_type(sampling_params, default: str) -> Literal["image", "latent", "both"]:
    """Resolve output_type from vLLM-Omni's first-class sampling field."""
    output_type = getattr(sampling_params, "output_type", None)
    if output_type is None:
        output_type = sampling_params.extra_args.get("output_type", None)
    return _validate_output_type(_coalesce_not_none(output_type, default))


install_rollout_postprocessor(pipeline_sd3, "get_sd3_image_post_process_func")


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="flow_grpo")
class StableDiffusion3PipelineWithLogProb(SD3TokenIdPromptMixin, StableDiffusion3Pipeline):
    """SD3.5 rollout pipeline that returns FlowGRPO trajectory data.

    Differences from the upstream pipeline:

    - the Euler flow-match scheduler is replaced by
      :class:`FlowMatchSDEDiscreteScheduler` so SDE-window sampling produces
      per-step log-probabilities;
    - prompts are encoded from per-text-encoder token ids (CLIP + T5) produced
      once by the agent loop (``actor_rollout_ref.model.extra_tokenizers``),
      so no decode/re-encode happens here; plain-text prompts are only used
      for engine warm-up / serving-style requests;
    - ``forward`` collects ``all_latents`` / ``all_log_probs`` /
      ``all_timesteps`` and ships prompt embeddings (sequence + pooled)
      through ``trajectory_*`` / metadata for training-side log-prob recomputation;
    - CFG is plain SD3 guidance (``guidance_scale``); the convergence-test
      default is non-CFG (``guidance_scale <= 1`` skips the negative branch
      entirely, halving the transformer NFE).
    """

    # Forward-process anchors may require negative conditioning even with unguided sampling.
    always_encode_negative_prompt = False

    supports_request_batch = True
    supports_step_execution = True
    #: Also return the final latents as ``latents_clean`` when the output type is not ``both``.
    emit_clean_latents = False

    diffusion_io_spec = DiffusionIOSpec(
        artifacts={
            "image_preview": MediaSpec("image", "decoded", "CHW"),
            "image_latent": MediaSpec("image", "latent", "CHW"),
        }
    )

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

    def _model_dtype(self) -> torch.dtype:
        return self.od_config.dtype

    def export_step_state(self, state: StepRequestState, *, policy_ref: str, max_bytes: int) -> bytes:
        """Export SDE state; the caller supplies its actual generation-policy reference."""
        if state.sampling.generator is None:
            raise ValueError("SD3 SDE recovery requires a request-owned generator")
        return export_step_state(
            state,
            policy_ref=policy_ref,
            max_bytes=max_bytes,
            execution_profile=step_execution_profile(self.od_config, self.device),
            scheduler_cls=FlowMatchSDEDiscreteScheduler,
        )

    def restore_step_state(self, payload: bytes, *, policy_ref: str, max_bytes: int) -> StepRequestState:
        """Restore SDE state after the caller verifies the installed policy."""
        return restore_step_state(
            payload,
            policy_ref=policy_ref,
            device=self.device,
            max_bytes=max_bytes,
            execution_profile=step_execution_profile(self.od_config, self.device),
            scheduler_cls=FlowMatchSDEDiscreteScheduler,
        )

    def _decode_latents(self, latents: torch.Tensor, output_type: str | None = None) -> torch.Tensor:
        output_type = self.output_type if output_type is None else output_type
        if output_type == "latent":
            return latents
        latents = latents.to(self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        return self.vae.decode(latents, return_dict=False)[0]

    @staticmethod
    def _sde_history(batch_size: int) -> dict[str, list[list[Any]]]:
        return {name: [[] for _ in range(batch_size)] for name in ("all_latents", "all_log_probs", "all_timesteps")}

    @staticmethod
    def _trajectory_output(
        state: StepRequestState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        history = state.extra
        latents = torch.stack([torch.stack(row) for row in history["all_latents"]])
        log_probs = history["all_log_probs"]
        stacked_log_probs = (
            torch.stack([torch.stack(row) for row in log_probs])
            if log_probs and log_probs[0] and log_probs[0][0] is not None
            else None
        )
        timesteps = torch.stack([torch.stack(row) for row in history["all_timesteps"]])
        return state.latents, latents, stacked_log_probs, timesteps

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Prepare one request using the full-loop encoder, sampling and window rules."""
        request = OmniDiffusionRequest(
            request_id=state.request_id, prompt=state.prompt, sampling_params=copy.copy(state.sampling)
        )
        context = self._prepare_generation_context(DiffusionRequestBatch(requests=[request]))
        if context is None:
            raise ValueError("SD3 FlowGRPO step execution requires a usable text or tokenized prompt")
        state.latents = context.pop("latents")
        state.timesteps = context.pop("timesteps")
        state.prompt_embeds = context.pop("prompt_embeds")
        state.negative_prompt_embeds = context.pop("negative_prompt_embeds")
        state.prompt_embeds_mask = context.pop("prompt_embeds_mask")
        state.negative_prompt_embeds_mask = context.pop("negative_prompt_embeds_mask")
        state.do_true_cfg = context.pop("do_cfg")
        state.sampling.generator = context.pop("generator")
        state.scheduler = copy.deepcopy(self.scheduler)
        state.scheduler.set_begin_index(0)
        state.step_index = 0
        context["sde_windows"] = context.pop("sde_window")
        state.extra.update(context)
        state.extra.update(self._sde_history(state.latents.shape[0]))
        return state

    def _predict_noise(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor | None,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        do_true_cfg: bool,
        guidance_scale: float,
        cfg_normalize: bool = False,
    ) -> torch.Tensor:
        return super()._predict_noise(
            latents.to(self._model_dtype()),
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            do_true_cfg,
            guidance_scale,
            cfg_normalize,
        )

    def post_decode(self, state: StepRequestState, **kwargs: Any) -> DiffusionOutput:
        """Return complete training tensors and the same named media as full execution."""
        # InputBatch normalizes masks to bool; retain the training-output dtype.
        context = {
            **state.extra,
            "prompt_embeds": state.prompt_embeds,
            "negative_prompt_embeds": state.negative_prompt_embeds,
            "prompt_embeds_mask": state.prompt_embeds_mask.to(torch.int64),
            "negative_prompt_embeds_mask": (
                state.negative_prompt_embeds_mask.to(torch.int64)
                if state.negative_prompt_embeds_mask is not None
                else None
            ),
        }
        request = OmniDiffusionRequest(request_id=state.request_id, prompt=state.prompt, sampling_params=state.sampling)
        windows = state.extra["sde_windows"]
        if state.request_id == DUMMY_DIFFUSION_REQUEST_ID and windows[0][0] == windows[0][1]:
            trajectory = (state.latents, None, None, None)
        else:
            trajectory = self._trajectory_output(state)
        return self._finish_rollout(context, DiffusionRequestBatch(requests=[request]), *trajectory)[0]

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        """Commit one SDE step and its FP32 training trajectory to request-owned state."""
        i, timestep = state.step_index, state.current_timestep
        windows = state.extra["sde_windows"]
        latents = state.latents
        batch_size = latents.shape[0]
        for row, (start, end) in enumerate(windows):
            if i == start:
                state.extra["all_latents"][row].append(latents[row].detach().float().clone())
        levels = [float(state.extra["noise_level"]) if start <= i < end else 0.0 for start, end in windows]
        noise_level = (
            levels[0]
            if all(level == levels[0] for level in levels)
            else torch.tensor(levels, device=latents.device, dtype=torch.float32).view(
                batch_size, *([1] * (latents.ndim - 1))
            )
        )
        state.latents, log_prob, _, _ = state.scheduler.step(
            noise_pred.float(),
            timestep,
            latents,
            generator=state.sampling.generator,
            noise_level=noise_level,
            sde_type=state.extra["sde_type"],
            return_logprobs=state.extra["logprobs"],
            return_dict=False,
        )
        for row, (start, end) in enumerate(windows):
            if start <= i < end:
                state.extra["all_latents"][row].append(state.latents[row].detach().float().clone())
                state.extra["all_log_probs"][row].append(None if log_prob is None else log_prob[row])
                state.extra["all_timesteps"][row].append(timestep)
        state.step_index += 1

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
        sde_window: tuple[int, int] | list[tuple[int, int]],
        sde_type: str,
        generator: torch.Generator | list[torch.Generator] | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Run the full SDE diffusion loop and collect per-step rollout data."""
        batch_size = latents.shape[0]
        windows = [sde_window] * batch_size if isinstance(sde_window, tuple) else list(sde_window)
        if len(windows) != batch_size:
            raise ValueError(f"Expected {batch_size} SDE windows, got {len(windows)}.")
        if len({end - start for start, end in windows}) != 1:
            raise ValueError("Packed SDE windows must share the same size.")
        self.scheduler.set_begin_index(0)
        state = StepRequestState(
            request_id="",
            sampling=OmniDiffusionSamplingParams(generator=generator),
            latents=latents,
            timesteps=timesteps,
            scheduler=self.scheduler,
            extra={
                "sde_windows": windows,
                "noise_level": noise_level,
                "sde_type": sde_type,
                "logprobs": logprobs,
                **self._sde_history(batch_size),
            },
        )

        for timestep_value in timesteps:
            if self.interrupt:
                continue

            self._current_timestep = timestep_value
            noise_pred = self._predict_noise(
                latents,
                timestep_value,
                prompt_embeds,
                pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                do_cfg,
                guidance_scale,
            )

            self.step_scheduler(state, noise_pred)
            latents = state.latents

        return self._trajectory_output(state)

    def _prepare_generation_context(
        self,
        request_batch: DiffusionRequestBatch,
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
        output_type: Literal["image", "latent", "both"] = "image",
    ) -> dict[str, Any] | None:
        """Prepare shared full-request and incremental FlowGRPO inputs."""
        req_prompts = request_batch.prompts or []
        extra_prompt_ids = _extract_extra_prompt_ids(req_prompts)
        negative_extra_prompt_ids = (
            _extract_extra_prompt_ids(req_prompts, key="negative_extra_prompt_ids")
            if extra_prompt_ids is not None
            else None
        )
        req_prompt, req_negative_prompt = _extract_text_prompts(req_prompts)
        prompt = req_prompt if req_prompt is not None else prompt
        negative_prompt = req_negative_prompt if req_negative_prompt is not None else negative_prompt

        if extra_prompt_ids is None and prompt is None:
            if any(isinstance(p, dict) and prompt_ids_from_payload(p) is not None for p in req_prompts):
                raise ValueError(
                    "SD3 rollout received tokenized prompts without per-text-encoder token ids; decoding "
                    "token ids back to text inside the pipeline is not supported. Configure "
                    f"`actor_rollout_ref.model.extra_tokenizers` with {list(SD3_ENCODER_TOKEN_KEYS)} entries "
                    "(see examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh)."
                )
            # Engine warm-up / dummy run without a usable prompt.
            return None
        if isinstance(prompt, str):
            prompt = [prompt]

        sampling_params = request_batch.sampling_params_list[0]
        requested_outputs = requested_outputs_for_batch(request_batch)
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
        output_type = _resolve_output_type(sampling_params, output_type)

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
        latents = request_batch.collate_request_tensors("latents", latents)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        batch_size = len(extra_prompt_ids[SD3_CLIP_TOKENS_KEY]) if extra_prompt_ids is not None else len(prompt)
        do_cfg = guidance_scale > 1

        if extra_prompt_ids is not None:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt_from_token_ids(
                clip_prompt_ids=extra_prompt_ids[SD3_CLIP_TOKENS_KEY],
                t5_prompt_ids=extra_prompt_ids[SD3_T5_TOKENS_KEY],
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )
        else:
            prompt_embeds, pooled_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                prompt_3=None,
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_images_per_prompt,
            )
        negative_prompt_embeds = None
        negative_pooled_prompt_embeds = None
        if do_cfg or self.always_encode_negative_prompt:
            if negative_extra_prompt_ids is not None:
                negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt_from_token_ids(
                    clip_prompt_ids=negative_extra_prompt_ids[SD3_CLIP_TOKENS_KEY],
                    t5_prompt_ids=negative_extra_prompt_ids[SD3_T5_TOKENS_KEY],
                    max_sequence_length=max_sequence_length,
                    num_images_per_prompt=num_images_per_prompt,
                )
            else:
                # Upstream SD3 default: CFG negatives fall back to empty strings.
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

        sde_window = _sample_per_sample_sde_windows(
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            num_timesteps=len(timesteps),
            batch_size=latents.shape[0],
            generator=generator,
            device=self.device,
        )

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            "latents": latents,
            "timesteps": timesteps,
            "do_cfg": do_cfg,
            "guidance_scale": guidance_scale,
            "noise_level": noise_level,
            "sde_window": sde_window,
            "sde_type": sde_type,
            "generator": generator,
            "logprobs": logprobs,
            "prompt_embeds_mask": prompt_embeds_mask,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "output_type": output_type,
            "requested_outputs": requested_outputs,
            "num_images_per_prompt": num_images_per_prompt,
        }

    def _finish_rollout(
        self,
        context: dict[str, Any],
        request_batch: DiffusionRequestBatch,
        latents: torch.Tensor,
        all_latents: torch.Tensor | None,
        all_log_probs: torch.Tensor | None,
        all_timesteps: torch.Tensor | None,
    ) -> list[DiffusionOutput]:
        output_type = context["output_type"]
        requested_outputs = context["requested_outputs"]
        num_images_per_prompt = context["num_images_per_prompt"]
        prompt_embeds = context["prompt_embeds"]
        pooled_prompt_embeds = context["pooled_prompt_embeds"]
        negative_prompt_embeds = context["negative_prompt_embeds"]
        negative_pooled_prompt_embeds = context["negative_pooled_prompt_embeds"]
        prompt_embeds_mask = context["prompt_embeds_mask"]
        negative_prompt_embeds_mask = context["negative_prompt_embeds_mask"]
        decode = output_type != "latent" or "image_preview" in requested_outputs
        self._current_timestep = None
        output = self._decode_latents(latents, "image") if decode else None
        rl = {}
        if output_type == "both" or self.emit_clean_latents:
            rl["latents_clean"] = latents.float()

        result = rollout_output(
            media=output,
            trajectory_latents=all_latents,
            trajectory_log_probs=all_log_probs,
            trajectory_timesteps=all_timesteps,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            },
            rl=rl,
            to_cpu=True,
        )
        result = with_visual_artifacts(
            result,
            decoded=output,
            latents=latents,
            latent_layout="CHW",
            output_type=output_type,
            context=f"pipeline={type(self).__name__}, request_id={[r.request_id for r in request_batch.requests]}",
            requested=requested_outputs,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs

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
        output_type: Literal["image", "latent", "both"] = "image",
    ) -> DiffusionOutput | list[DiffusionOutput]:
        """End-to-end SD3.5 generation with rollout trajectory collection."""
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        context = self._prepare_generation_context(
            request_batch,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            max_sequence_length=max_sequence_length,
            noise_level=noise_level,
            sde_window_size=sde_window_size,
            sde_window_range=sde_window_range,
            sde_type=sde_type,
            logprobs=logprobs,
            output_type=output_type,
        )
        if context is None:
            outputs = [DiffusionOutput(output=None) for _ in range(request_batch.num_reqs)]
            return outputs if return_batch else outputs[0]
        latents = context["latents"]
        sde_window = context["sde_window"]
        output_type = context["output_type"]
        requested_outputs = context["requested_outputs"]
        num_images_per_prompt = context["num_images_per_prompt"]
        decode = output_type != "latent" or "image_preview" in requested_outputs
        if request_batch.requests[0].request_id == DUMMY_DIFFUSION_REQUEST_ID and sde_window[0][0] == sde_window[0][1]:
            output = self._decode_latents(latents, "image") if decode else None
            result = with_visual_artifacts(
                DiffusionOutput(output=output, to_cpu=True),
                decoded=output,
                latents=latents,
                latent_layout="CHW",
                output_type=output_type,
                context=f"pipeline={type(self).__name__}, request_id={request_batch.requests[0].request_id}",
                requested=requested_outputs,
            )
            outputs = _split_diffusion_output_by_request(
                result,
                request_batch,
                num_outputs_per_prompt=num_images_per_prompt,
            )
            return outputs if return_batch else outputs[0]

        trajectory = self.diffuse(
            context["prompt_embeds"],
            context["pooled_prompt_embeds"],
            context["negative_prompt_embeds"],
            context["negative_pooled_prompt_embeds"],
            context["latents"],
            context["timesteps"],
            context["do_cfg"],
            context["guidance_scale"],
            context["noise_level"],
            context["sde_window"],
            context["sde_type"],
            context["generator"],
            context["logprobs"],
        )
        outputs = self._finish_rollout(context, request_batch, *trajectory)
        return outputs if return_batch else outputs[0]
