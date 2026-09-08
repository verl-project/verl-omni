# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""FLUX.1-dev rollout adapter for DanceGRPO.

The adapter follows the request-batch and per-text-encoder token-id contracts
of the current verl-omni SD3 adapter. CLIP/T5 run only in rollout; the actor
receives fixed-size embeddings and trains the complete FLUX transformer.
"""

from __future__ import annotations

import ast
import os
from typing import Any

import numpy as np
import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.models.flux import pipeline_flux
from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import rollout_output, wrap_rollout_postprocessor
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.request_batch import split_diffusion_output_by_request
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.pipelines.wan22_dance_grpo.common import seed_from_prompt_ids

from .common import predict_original_sample, sd3_time_shift, select_dance_grpo_transitions

__all__ = ["FluxDanceGRPOPipelineWithLogProb"]

FLUX_CLIP_TOKENS_KEY = "clip"
FLUX_T5_TOKENS_KEY = "t5"
FLUX_ENCODER_TOKEN_KEYS = (FLUX_CLIP_TOKENS_KEY, FLUX_T5_TOKENS_KEY)

_FLUX_POST_PROCESS_FACTORY = pipeline_flux.get_flux_post_process_func


def get_rollout_post_process_func(od_config):
    """Postprocess generated images while preserving rollout metadata."""
    return wrap_rollout_postprocessor(_FLUX_POST_PROCESS_FACTORY(od_config))


pipeline_flux.get_flux_post_process_func = get_rollout_post_process_func


def _coalesce_not_none(value, default):
    return default if value is None else value


def _to_token_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    if isinstance(value, list):
        return [int(item) for item in value]
    return None


def _extract_extra_prompt_ids(prompts: list[Any]) -> dict[str, list[list[int]]] | None:
    """Return CLIP/T5 token ids produced by the latest agent loop."""
    if not prompts:
        return None
    per_prompt: list[dict[str, Any]] = []
    for prompt in prompts:
        if not isinstance(prompt, dict) or not prompt.get("extra_prompt_ids"):
            return None
        per_prompt.append(prompt["extra_prompt_ids"])

    for index, extra in enumerate(per_prompt):
        missing = [key for key in FLUX_ENCODER_TOKEN_KEYS if key not in extra]
        if missing:
            raise ValueError(
                f"FLUX request {index} is missing extra_prompt_ids entries {missing}. "
                "Configure actor_rollout_ref.model.extra_tokenizers.clip and .t5."
            )
        for key in FLUX_ENCODER_TOKEN_KEYS:
            if _to_token_list(extra[key]) is None:
                raise TypeError(f"extra_prompt_ids[{key!r}] must be a list or tensor of token ids")
    return {key: [_to_token_list(extra[key]) for extra in per_prompt] for key in FLUX_ENCODER_TOKEN_KEYS}


def _extract_text_prompts(prompts: list[Any]) -> list[str] | None:
    values = [prompt if isinstance(prompt, str) else (prompt.get("prompt") or "") for prompt in prompts]
    return values if any(values) else None


def _pad_token_ids(
    ids_list: list[list[int]],
    *,
    max_length: int,
    pad_token_id: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for ids in ids_list:
        ids = ids[:max_length]
        rows.append(ids + [pad_token_id] * (max_length - len(ids)))
    return torch.tensor(rows, dtype=torch.long, device=device)


def _normalize_window_range(value: list[int] | tuple[int, int] | str | None, num_steps: int) -> tuple[int, int]:
    if value is None:
        return 0, num_steps
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if len(value) != 2:
        raise ValueError("sde_window_range must contain exactly [start, end]")
    return int(value[0]), int(value[1])


def _sample_sde_windows(
    *,
    window_size: int | None,
    window_range: list[int] | tuple[int, int] | str | None,
    num_steps: int,
    batch_size: int,
    generators: torch.Generator | list[torch.Generator] | None,
    device: torch.device,
) -> list[tuple[int, int]]:
    if window_size is None:
        return [(0, num_steps)] * batch_size
    window_size = int(window_size)
    start, end = _normalize_window_range(window_range, num_steps)
    if window_size <= 0 or start < 0 or end > num_steps or start + window_size > end:
        raise ValueError(
            f"invalid sde_window_size={window_size}, sde_window_range={(start, end)}, num_steps={num_steps}"
        )
    high = end - window_size + 1

    def sample_one(generator: torch.Generator | None) -> tuple[int, int]:
        selected = int(torch.randint(start, high, (1,), device=device, generator=generator).item())
        return selected, selected + window_size

    if isinstance(generators, list):
        if len(generators) != batch_size:
            raise ValueError(f"expected {batch_size} generators, got {len(generators)}")
        return [sample_one(generator) for generator in generators]
    sampled = sample_one(generators)
    return [sampled] * batch_size


@VllmOmniPipelineBase.register("FluxPipeline", algorithm="dance_grpo")
class FluxDanceGRPOPipelineWithLogProb(FluxPipeline):
    """FLUX.1-dev generation plus aligned DanceGRPO transition capture."""

    supports_request_batch = True
    diffusion_io_spec = DiffusionIOSpec(primary=MediaSpec("image"))

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        # ``interrupt`` is a read-only property on the vllm_omni FluxPipeline
        # backed by ``_interrupt``, which FluxPipeline only initialises inside
        # ``__call__``. This adapter drives generation via ``diffuse()`` instead,
        # so initialise the flag here (mirrors the wan22_dance_grpo adapter).
        self._interrupt = False
        self.device = get_local_device()
        model = od_config.model
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=os.path.exists(model),
        )
        # We pass an already shifted sigma schedule. Keep the scheduler itself
        # at identity shift so rollout and actor cannot double-shift.
        if getattr(scheduler.config, "use_dynamic_shifting", False):
            scheduler.register_to_config(use_dynamic_shifting=False)
        if getattr(scheduler, "shift", 1.0) != 1.0:
            scheduler.register_to_config(shift=1.0)
            scheduler.set_shift(1.0)
        if scheduler.shift != 1.0:
            raise RuntimeError(f"FLUX rollout scheduler shift must be 1.0, got {scheduler.shift}")
        self.scheduler = scheduler

    def encode_prompt_from_token_ids(
        self,
        *,
        clip_prompt_ids: list[list[int]],
        t5_prompt_ids: list[list[int]],
        max_sequence_length: int,
        num_images_per_prompt: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Canonical FLUX CLIP pooled + T5 sequence encoding from fixed ids."""
        if len(clip_prompt_ids) != len(t5_prompt_ids):
            raise ValueError("CLIP and T5 prompt batches must have the same size")
        batch_size = len(clip_prompt_ids)
        clip_ids = _pad_token_ids(
            clip_prompt_ids,
            max_length=self.tokenizer_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        pooled_prompt_embeds = self.text_encoder(clip_ids, output_hidden_states=False).pooler_output
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=self.device, dtype=self.text_encoder.dtype)

        t5_ids = _pad_token_ids(
            t5_prompt_ids,
            max_length=max_sequence_length,
            pad_token_id=self.tokenizer_2.pad_token_id,
            device=self.device,
        )
        prompt_embeds = self.text_encoder_2(t5_ids)[0]
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.text_encoder_2.dtype)

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1).view(
            batch_size * num_images_per_prompt, max_sequence_length, -1
        )
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            batch_size * num_images_per_prompt, -1
        )
        text_ids = torch.zeros(
            max_sequence_length,
            3,
            device=self.device,
            dtype=prompt_embeds.dtype,
        )
        return prompt_embeds, pooled_prompt_embeds, text_ids

    def _set_timesteps(self, *, num_steps: int, shift: float) -> torch.Tensor:
        sigmas_np = np.linspace(1.0, 0.0, num_steps + 1)
        shifted = sd3_time_shift(shift, torch.from_numpy(sigmas_np).float())
        self.scheduler.set_timesteps(
            num_steps,
            device=self.device,
            sigmas=shifted[:num_steps].cpu().numpy(),
        )
        return self.scheduler.timesteps

    def diffuse(
        self,
        *,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        image_ids: torch.Tensor,
        guidance_scale: float,
        noise_level: float,
        sde_type: str,
        generators: torch.Generator | list[torch.Generator] | None,
        windows: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run chronological diffusion and collect selected explicit pairs."""
        batch_size = latents.shape[0]
        if len(windows) != batch_size:
            raise ValueError(f"expected {batch_size} SDE windows, got {len(windows)}")
        if len({end - start for start, end in windows}) != 1:
            raise ValueError("all packed SDE windows must have the same length")

        current_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        next_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        log_prob_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        timestep_rows: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        pred_original_sample: torch.Tensor | None = None
        model_dtype = self.od_config.dtype
        self.scheduler.set_begin_index(0)

        for step_index, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue
            self._current_timestep = timestep_value
            set_forward_context_denoise_step_idx(step_index)
            current_sample = latents.float()
            timestep = timestep_value.expand(batch_size).to(device=self.device, dtype=model_dtype)

            guidance = None
            if getattr(self.transformer, "guidance_embeds", False):
                guidance = torch.full((batch_size,), guidance_scale, device=self.device, dtype=torch.float32)

            noise_pred = self.transformer(
                hidden_states=latents.to(model_dtype),
                timestep=timestep / 1000.0,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=image_ids,
                guidance=guidance,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            sigma_index = self.scheduler.index_for_timestep(timestep_value)
            sigma = self.scheduler.sigmas[sigma_index].to(device=current_sample.device, dtype=current_sample.dtype)
            pred_original_sample = predict_original_sample(current_sample, noise_pred.float(), sigma)

            levels = [float(noise_level) if start <= step_index < end else 0.0 for start, end in windows]
            step_noise: float | torch.Tensor
            if all(level == levels[0] for level in levels):
                step_noise = levels[0]
            else:
                step_noise = torch.tensor(levels, device=self.device, dtype=torch.float32).view(batch_size, 1, 1)

            has_active_row = any(start <= step_index < end for start, end in windows)
            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                current_sample,
                generator=generators,
                noise_level=step_noise,
                sde_type=sde_type,
                return_logprobs=has_active_row,
                return_dict=False,
            )
            if has_active_row and log_prob is None:
                raise RuntimeError("DanceGRPO scheduler did not return rollout log-probabilities")

            for row, (start, end) in enumerate(windows):
                if start <= step_index < end:
                    if log_prob is None:
                        raise RuntimeError("active DanceGRPO transition is missing a rollout log-probability")
                    current_rows[row].append(current_sample[row].detach().clone())
                    next_rows[row].append(latents[row].detach().float().clone())
                    log_prob_rows[row].append(log_prob[row].detach().float().clone())
                    timestep_rows[row].append(timestep_value.detach().float().clone())

        if pred_original_sample is None or any(not row for row in current_rows):
            raise RuntimeError("FLUX rollout collected no DanceGRPO transitions")
        return (
            torch.stack([torch.stack(row) for row in current_rows]),
            torch.stack([torch.stack(row) for row in next_rows]),
            torch.stack([torch.stack(row) for row in log_prob_rows]),
            torch.stack([torch.stack(row) for row in timestep_rows]),
            pred_original_sample,
        )

    @torch.no_grad()
    def forward(
        self,
        req: OmniDiffusionRequest | DiffusionRequestBatch,
        *,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 16,
        guidance_scale: float = 3.5,
        output_type: str = "image",
        init_same_noise: bool = True,
        **kwargs: Any,
    ) -> DiffusionOutput | list[DiffusionOutput]:
        request_batch = req if isinstance(req, DiffusionRequestBatch) else DiffusionRequestBatch(requests=[req])
        return_batch = isinstance(req, DiffusionRequestBatch)
        if all(request.request_id == DUMMY_DIFFUSION_REQUEST_ID for request in request_batch.requests):
            outputs = [DiffusionOutput(output=None) for _ in range(request_batch.num_reqs)]
            return outputs if return_batch else outputs[0]
        prompts = request_batch.prompts or []
        extra_prompt_ids = _extract_extra_prompt_ids(prompts)
        text_prompts = _extract_text_prompts(prompts)

        if extra_prompt_ids is None and text_prompts is None:
            if any(
                request.request_id != DUMMY_DIFFUSION_REQUEST_ID
                and isinstance(prompt, dict)
                and prompt.get("prompt_token_ids") is not None
                for request, prompt in zip(request_batch.requests, prompts, strict=False)
            ):
                raise ValueError(
                    "FLUX received only the generic prompt_token_ids. Configure "
                    "actor_rollout_ref.model.extra_tokenizers.clip={path: tokenizer, max_length: 77} "
                    "and .t5={path: tokenizer_2, max_length: 512}."
                )
            outputs = [DiffusionOutput(output=None) for _ in range(request_batch.num_reqs)]
            return outputs if return_batch else outputs[0]

        sampling = request_batch.sampling_params_list[0]
        extra_args = sampling.extra_args or {}
        height = sampling.height or height or self.default_sample_size * self.vae_scale_factor
        width = sampling.width or width or self.default_sample_size * self.vae_scale_factor
        num_steps = sampling.num_inference_steps or num_inference_steps
        max_sequence_length = sampling.max_sequence_length or 512
        if sampling.guidance_scale_provided:
            guidance_scale = sampling.guidance_scale
        output_type = sampling.output_type or extra_args.get("output_type") or output_type
        if output_type not in ("image", "latent"):
            raise ValueError(f"FLUX DanceGRPO output_type must be 'image' or 'latent', got {output_type!r}")
        num_outputs = sampling.num_outputs_per_prompt if sampling.num_outputs_per_prompt > 0 else 1
        self._guidance_scale = guidance_scale
        self._interrupt = False

        shift = float(_coalesce_not_none(extra_args.get("shift"), kwargs.get("shift", 3.0)))
        noise_level = float(_coalesce_not_none(extra_args.get("noise_level"), kwargs.get("noise_level", 1.0)))
        sde_type = _coalesce_not_none(extra_args.get("sde_type"), kwargs.get("sde_type", "dance_sde"))
        window_size = _coalesce_not_none(extra_args.get("sde_window_size"), kwargs.get("sde_window_size"))
        window_range = _coalesce_not_none(extra_args.get("sde_window_range"), kwargs.get("sde_window_range"))
        strategy = _coalesce_not_none(
            extra_args.get("timestep_sample_strategy"), kwargs.get("timestep_sample_strategy", "random_subset")
        )
        fraction = float(_coalesce_not_none(extra_args.get("timestep_fraction"), kwargs.get("timestep_fraction", 0.6)))
        drop_last = bool(
            _coalesce_not_none(extra_args.get("drop_last_transition"), kwargs.get("drop_last_transition", True))
        )
        if strategy == "random_subset" and window_size is not None:
            raise ValueError("random_subset requires the full trajectory; unset sde_window_size")

        if extra_prompt_ids is not None:
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt_from_token_ids(
                clip_prompt_ids=extra_prompt_ids[FLUX_CLIP_TOKENS_KEY],
                t5_prompt_ids=extra_prompt_ids[FLUX_T5_TOKENS_KEY],
                max_sequence_length=max_sequence_length,
                num_images_per_prompt=num_outputs,
            )
        else:
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                prompt=text_prompts,
                prompt_2=None,
                num_images_per_prompt=num_outputs,
                max_sequence_length=max_sequence_length,
            )
        prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], device=self.device, dtype=torch.long)

        for request in request_batch.requests:
            params = request.sampling_params
            if params.generator is None and params.seed is not None:
                params.generator = torch.Generator(device=self.device).manual_seed(params.seed)
        generators = request_batch.collate_request_generators(num_outputs, None)
        initial_generators = generators
        if init_same_noise and extra_prompt_ids is not None:
            global_steps = extra_args.get("global_steps")
            # Repeated generations of one prompt start from the same noise,
            # matching MindSpeed-MM. Keep the request-specific generators for
            # SDE updates so the trajectories do not become identical.
            initial_generators = [
                torch.Generator(device=self.device).manual_seed(seed_from_prompt_ids(ids, global_steps))
                for ids in extra_prompt_ids[FLUX_T5_TOKENS_KEY]
                for _ in range(num_outputs)
            ]
        latents = request_batch.collate_request_tensors("latents", None)
        num_channels_latents = self.transformer.in_channels // 4
        latents, image_ids = self.prepare_latents(
            prompt_embeds.shape[0],
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            initial_generators,
            latents,
        )

        timesteps = self._set_timesteps(num_steps=num_steps, shift=shift)
        self._num_timesteps = len(timesteps)
        windows = _sample_sde_windows(
            window_size=window_size,
            window_range=window_range,
            num_steps=len(timesteps),
            batch_size=latents.shape[0],
            generators=generators,
            device=self.device,
        )
        current, next_latents, log_probs, selected_timesteps, pred_original = self.diffuse(
            latents=latents,
            timesteps=timesteps,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            image_ids=image_ids,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            sde_type=sde_type,
            generators=generators,
            windows=windows,
        )
        current, next_latents, selected_timesteps, log_probs = select_dance_grpo_transitions(
            current,
            next_latents,
            selected_timesteps,
            log_probs,
            strategy=strategy,
            fraction=fraction,
            drop_last=drop_last,
            generator=generators,
        )

        self._current_timestep = None
        if output_type == "latent":
            output = pred_original
        else:
            clean = self._unpack_latents(pred_original, height, width, self.vae_scale_factor)
            clean = (clean / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            output = self.vae.decode(clean.to(self.vae.dtype), return_dict=False)[0]

        batch_size = prompt_embeds.shape[0]
        result = rollout_output(
            media=output,
            trajectory_latents=current,
            trajectory_log_probs=log_probs,
            trajectory_timesteps=selected_timesteps,
            prompt_embeddings={
                "prompt_embeds": prompt_embeds,
                "prompt_embeds_mask": prompt_embeds_mask,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "text_ids": text_ids.unsqueeze(0).expand(batch_size, -1, -1),
                "image_ids": image_ids.unsqueeze(0).expand(batch_size, -1, -1),
            },
            rl={"all_next_latents": next_latents},
            to_cpu=True,
        )
        outputs = split_diffusion_output_by_request(
            result,
            request_batch,
            num_outputs_per_prompt=num_outputs,
        )
        return outputs if return_batch else outputs[0]
