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
from vllm_omni.diffusion.models.sd3 import pipeline_sd3
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import (
    sample_per_sample_sde_windows as _sample_per_sample_sde_windows,
)
from verl_omni.pipelines.request_batch import (
    split_diffusion_output_by_request as _split_diffusion_output_by_request,
)
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
        extra = prompt.get(key)
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


_SD3_IMAGE_POST_PROCESS_FUNC = pipeline_sd3.get_sd3_image_post_process_func


def get_latent_post_process_func(od_config):
    """Keep SD3 latents untouched while normally postprocessing decoded images."""
    image_postprocess = _SD3_IMAGE_POST_PROCESS_FUNC(od_config)

    def postprocess(output):
        if isinstance(output, torch.Tensor) and output.ndim >= 3 and output.shape[-3] == 16:
            return output
        return image_postprocess(output)

    return postprocess


# vLLM-Omni resolves this module-level factory before initializing the custom
# pipeline, so install the SD3-specific override while registering this adapter.
pipeline_sd3.get_sd3_image_post_process_func = get_latent_post_process_func


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

    def _model_dtype(self) -> torch.dtype:
        return self.od_config.dtype

    def _decode_latents(self, latents: torch.Tensor, output_type: str | None = None) -> torch.Tensor:
        output_type = self.output_type if output_type is None else output_type
        if output_type == "latent":
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
        all_latents: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        all_log_probs: list[list[Any]] = [[] for _ in range(batch_size)]
        all_timesteps: list[list[Any]] = [[] for _ in range(batch_size)]

        model_dtype = self._model_dtype()
        self.scheduler.set_begin_index(0)

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            for batch_idx, (start, end) in enumerate(windows):
                if i == start:
                    all_latents[batch_idx].append(latents[batch_idx].detach().float().clone())
            levels = [float(noise_level) if start <= i < end else 0.0 for start, end in windows]
            cur_noise_level: float | torch.Tensor = (
                levels[0]
                if all(level == levels[0] for level in levels)
                else torch.tensor(levels, device=latents.device, dtype=torch.float32).view(
                    batch_size, *([1] * (latents.ndim - 1))
                )
            )

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
            for batch_idx, (start, end) in enumerate(windows):
                if start <= i < end:
                    all_latents[batch_idx].append(latents[batch_idx].detach().float().clone())
                    all_log_probs[batch_idx].append(None if log_prob is None else log_prob[batch_idx])
                    all_timesteps[batch_idx].append(timestep_value)

        all_latents_t = torch.stack([torch.stack(traj, dim=0) for traj in all_latents], dim=0)
        if all_log_probs and all_log_probs[0] and all_log_probs[0][0] is not None:
            all_log_probs_t = torch.stack([torch.stack(traj, dim=0) for traj in all_log_probs], dim=0)
        else:
            all_log_probs_t = None
        all_timesteps_t = torch.stack([torch.stack(traj, dim=0) for traj in all_timesteps], dim=0)
        return latents, all_latents_t, all_log_probs_t, all_timesteps_t

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
            if any(isinstance(p, dict) and p.get("prompt_token_ids") is not None for p in req_prompts):
                raise ValueError(
                    "SD3 rollout received tokenized prompts without per-text-encoder token ids; decoding "
                    "token ids back to text inside the pipeline is not supported. Configure "
                    f"`actor_rollout_ref.model.extra_tokenizers` with {list(SD3_ENCODER_TOKEN_KEYS)} entries "
                    "(see examples/flowgrpo_trainer/sd35/run_sd35_medium_ocr_lora.sh)."
                )
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
        if do_cfg:
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

        if request_batch.requests[0].request_id == DUMMY_DIFFUSION_REQUEST_ID and sde_window[0][0] == sde_window[0][1]:
            output = self._decode_latents(latents, output_type)
            result = DiffusionOutput(output=output, custom_output={}, to_cpu=True)
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
            generator,
            logprobs,
        )

        self._current_timestep = None
        output = self._decode_latents(latents, output_type)
        custom_output = {
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
            "all_timesteps": all_timesteps,
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_embeds_mask": negative_prompt_embeds_mask,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }
        if output_type == "both":
            custom_output["latents_clean"] = latents.float()

        result = DiffusionOutput(
            output=output,
            custom_output=custom_output,
            to_cpu=True,
        )
        outputs = _split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_images_per_prompt,
        )
        return outputs if return_batch else outputs[0]
