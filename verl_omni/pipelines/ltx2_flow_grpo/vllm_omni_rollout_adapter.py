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

"""vLLM-Omni rollout adapter for joint LTX-2.3 audio-video FlowGRPO."""

from __future__ import annotations

import copy
import os
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXPromptContext
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import (
    LTXDenoiseContext,
    LTXDenoiseExecutor,
    LTXForwardContext,
    LTXPhaseResult,
    LTXVideoAudioStepAdapter,
    prepare_rope_coords_stage,
)
from vllm_omni.diffusion.models.ltx2.ltx2_latents import (
    LTXAVState,
    unpack_audio_latents,
    unpad_audio_latents,
)
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTXPhaseRecipe
from vllm_omni.diffusion.models.ltx2.ltx2_request import LTXRequestInputs
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.diffusion_rollout_output import (
    rollout_output,
    wrap_rollout_postprocessor,
)
from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import calculate_shift, normalize_ltx_output_type

__all__ = ["LTX23PipelineWithLogProb"]

_LTX2_POST_PROCESS_FACTORY = pipeline_ltx2.get_ltx2_post_process_func


def get_rollout_post_process_func(od_config: Any):
    """Postprocess LTX media while preserving rollout metadata."""
    return wrap_rollout_postprocessor(_LTX2_POST_PROCESS_FACTORY(od_config))


# vllm-omni resolves the built-in architecture's factory in the engine process.
pipeline_ltx2.get_ltx2_post_process_func = get_rollout_post_process_func


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="flow_grpo")
class LTX23PipelineWithLogProb(LTX2Pipeline):
    """Sample LTX-2.3 with CPS/SDE transitions and return joint log-probs."""

    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.set_progress_bar_config(disable=True)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            od_config.model,
            subfolder="scheduler",
            local_files_only=os.path.exists(od_config.model),
        )
        self._flow_grpo_noise_level = 0.8
        self._flow_grpo_sde_type = "cps"
        self._flow_grpo_window_size: int | None = None
        self._flow_grpo_window_range: list[int] | None = None
        self._flow_grpo_sde_contiguous = True
        self._flow_grpo_logprobs = True
        self._flow_grpo_seed = 42
        self._flow_grpo_prompt_context: LTXPromptContext | None = None
        self._flow_grpo_trajectory: dict[str, torch.Tensor | None] = {}
        self._selected_sde_steps: set[int] = set()
        self._current_latents: list[torch.Tensor] = []
        self._next_latents: list[torch.Tensor] = []
        self._selected_timesteps: list[torch.Tensor] = []
        self._log_probs: list[torch.Tensor] = []

    def _encode_token_ids(
        self,
        token_ids: torch.Tensor | list[int],
        attention_mask: torch.Tensor | None,
        max_sequence_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode pre-tokenized prompts through Gemma-3's full hidden-state stack."""
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long)
        else:
            token_ids = token_ids.to(device=self.device, dtype=torch.long)
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)

        if attention_mask is None:
            attention_mask = torch.ones_like(token_ids)
        else:
            attention_mask = attention_mask.to(device=self.device)
            if attention_mask.ndim == 1:
                attention_mask = attention_mask.unsqueeze(0)

        token_ids = token_ids[:, :max_sequence_length]
        attention_mask = attention_mask[:, :max_sequence_length]
        pad_length = max_sequence_length - token_ids.shape[1]
        if pad_length > 0:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            token_ids = torch.nn.functional.pad(token_ids, (pad_length, 0), value=pad_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (pad_length, 0), value=0)

        encoded = self.text_encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        prompt_embeds = torch.stack(encoded.hidden_states, dim=-1).flatten(2, 3)
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype)
        return prompt_embeds, attention_mask

    def _inject_precomputed_prompt_embeds(self, req: OmniDiffusionRequest) -> None:
        """Convert verl token-ID request fields into LTX raw text-encoder embeddings."""
        if not isinstance(req.prompt, dict):
            raise TypeError("LTX-2.3 FlowGRPO expects a dict prompt containing `prompt_token_ids`.")
        payload = dict(req.prompt)
        prompt_ids = payload.get("prompt_token_ids")
        if prompt_ids is None:
            return

        max_sequence_length = req.sampling_params.max_sequence_length or self.tokenizer_max_length
        prompt_embeds, prompt_mask = self._encode_token_ids(
            prompt_ids,
            payload.get("prompt_mask"),
            max_sequence_length,
        )
        payload["prompt_embeds"] = prompt_embeds[0]
        payload["prompt_attention_mask"] = prompt_mask[0]

        negative_ids = payload.get("negative_prompt_ids")
        if negative_ids is not None:
            negative_embeds, negative_mask = self._encode_token_ids(
                negative_ids,
                payload.get("negative_prompt_mask"),
                max_sequence_length,
            )
            payload["negative_prompt_embeds"] = negative_embeds[0]
            payload["negative_prompt_attention_mask"] = negative_mask[0]
        req.prompt = payload

    def _configure_flow_grpo(self, req: OmniDiffusionRequest) -> None:
        req.sampling_params.output_type = normalize_ltx_output_type(req.sampling_params.output_type)
        extra_args = req.sampling_params.extra_args or {}
        self._flow_grpo_noise_level = float(extra_args.get("noise_level", 0.8))
        self._flow_grpo_sde_type = extra_args.get("sde_type", "cps")
        self._flow_grpo_window_size = extra_args.get("sde_window_size")
        self._flow_grpo_window_range = extra_args.get("sde_window_range")
        self._flow_grpo_sde_contiguous = bool(extra_args.get("sde_contiguous", True))
        self._flow_grpo_logprobs = bool(extra_args.get("logprobs", True))
        scheduler_seed = int(extra_args.get("sde_window_seed", 42))
        global_step = int(extra_args.get("global_steps", 1))
        self._flow_grpo_seed = scheduler_seed + max(global_step - 1, 0)

    def _select_sde_steps(self, num_steps: int, device: torch.device) -> list[int]:
        del device
        if self._flow_grpo_window_size is not None:
            window_size = int(self._flow_grpo_window_size)
            window_range = self._flow_grpo_window_range or [0, num_steps]
            low = int(window_range[0])
            high = min(int(window_range[1]), num_steps)
            if low < 0 or window_size <= 0 or high - low < window_size:
                raise ValueError(
                    f"Invalid LTX SDE window: size={window_size}, range={window_range}, num_steps={num_steps}."
                )
            generator = torch.Generator().manual_seed(self._flow_grpo_seed)
            if self._flow_grpo_sde_contiguous:
                start = int(torch.randint(low, high - window_size + 1, (1,), generator=generator).item())
                return list(range(start, start + window_size))

            order = torch.randperm(high - low, generator=generator)[:window_size].tolist()
            return sorted(low + index for index in order)

        return list(range(max(num_steps - 1, 0)))

    def _prepare_prompt_context(self, **kwargs: Any) -> LTXPromptContext:
        prompt_context = super()._prepare_prompt_context(**kwargs)
        self._flow_grpo_prompt_context = prompt_context
        return prompt_context

    def run_phase(
        self,
        req: DiffusionRequestBatch,
        request_inputs: LTXRequestInputs,
        *,
        noise_scale: float,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        attention_kwargs: dict[str, Any] | None,
        phase_recipe: LTXPhaseRecipe,
        image: Any | None = None,
        prompt_context: LTXPromptContext | None = None,
    ) -> LTXPhaseResult:
        """Prepare and execute one phase with FlowGRPO SDE transitions."""
        del phase_recipe
        self._check_forward_inputs(request_inputs, image=image)
        guidance_parallel_ready = self._setup_forward_runtime(req, request_inputs, attention_kwargs)
        device = self.device
        if prompt_context is None:
            prompt_context = self._prepare_prompt_context(
                prompt=request_inputs.prompt,
                negative_prompt=request_inputs.negative_prompt,
                prompt_embeds=request_inputs.prompt_embeds,
                negative_prompt_embeds=request_inputs.negative_prompt_embeds,
                prompt_attention_mask=request_inputs.prompt_attention_mask,
                negative_prompt_attention_mask=request_inputs.negative_prompt_attention_mask,
                num_videos_per_prompt=request_inputs.num_videos_per_prompt,
                max_sequence_length=request_inputs.max_sequence_length,
            )

        latent_num_frames, latent_height, latent_width = self._resolve_video_latent_dimensions(request_inputs)
        latents, conditioning_mask = self._prepare_video_latents_stage(
            request_inputs,
            prompt_context,
            device=device,
            noise_scale=noise_scale,
            image=image,
        )
        audio_latents, original_audio_num_frames, padded_audio_num_frames, latent_mel_bins = (
            self._prepare_audio_latents_stage(
                request_inputs,
                prompt_context,
                device=device,
                noise_scale=noise_scale,
            )
        )

        sigmas = (
            np.linspace(1.0, 1.0 / request_inputs.num_inference_steps, request_inputs.num_inference_steps)
            if sigmas is None
            else sigmas
        )
        video_seq_len = latent_num_frames * latent_height * latent_width
        mu = calculate_shift(
            video_seq_len,
            self.scheduler.config.get("base_image_seq_len", 1024),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.95),
            self.scheduler.config.get("max_shift", 2.05),
        )
        audio_scheduler = copy.deepcopy(self.scheduler)
        video_audio_step_adapter = LTXVideoAudioStepAdapter(
            self,
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
            image_conditioned=conditioning_mask is not None,
        )
        _ = retrieve_timesteps(
            audio_scheduler,
            request_inputs.num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        timesteps_tensor, _ = retrieve_timesteps(
            self.scheduler,
            request_inputs.num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        forward_ctx = LTXForwardContext(
            req=req,
            request_inputs=request_inputs,
            prompt_context=prompt_context,
            device=device,
            guidance_parallel_ready=guidance_parallel_ready,
            attention_kwargs=attention_kwargs,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            latent_mel_bins=latent_mel_bins,
            original_audio_num_frames=original_audio_num_frames,
            padded_audio_num_frames=padded_audio_num_frames,
            timesteps=timesteps_tensor,
            audio_scheduler=audio_scheduler,
            video_audio_step_adapter=video_audio_step_adapter,
        )
        video_coords, audio_coords = prepare_rope_coords_stage(self, forward_ctx, latents, audio_latents)
        denoise_ctx = LTXDenoiseContext(
            latents=latents,
            audio_latents=audio_latents,
            video_coords=video_coords,
            audio_coords=audio_coords,
            conditioning_mask=conditioning_mask,
        )
        denoise_ctx = self._prepare_denoise_context_for_guidance(forward_ctx, denoise_ctx)

        self.scheduler.set_begin_index(0)
        selected_steps = set(self._select_sde_steps(len(forward_ctx.timesteps), device))
        self._selected_sde_steps = selected_steps
        self._current_latents = []
        self._next_latents = []
        self._selected_timesteps = []
        self._log_probs = []

        state = LTXDenoiseExecutor.run(
            self,
            LTXAVState(video=denoise_ctx.latents, audio=denoise_ctx.audio_latents),
            forward_ctx.timesteps,
            lambda index, timestep, current_state: self._denoise_step(
                index,
                timestep,
                current_state,
                forward_ctx,
                denoise_ctx,
            ),
        )
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio

        if not self._current_latents:
            raise RuntimeError("LTX-2.3 rollout selected no SDE transitions.")
        batch_size = denoise_ctx.latents.shape[0]
        self._flow_grpo_trajectory = {
            "all_latents": torch.stack(self._current_latents, dim=1),
            "all_next_latents": torch.stack(self._next_latents, dim=1),
            "all_timesteps": torch.stack(self._selected_timesteps).unsqueeze(0).expand(batch_size, -1),
            "all_log_probs": torch.stack(self._log_probs, dim=1) if self._log_probs else None,
            "video_seq_len": torch.full(
                (batch_size,),
                video_seq_len,
                device=denoise_ctx.latents.device,
                dtype=torch.long,
            ),
        }
        unpacked_latents, unpacked_audio = self._unpack_and_denormalize_stage(
            forward_ctx,
            state.video,
            state.audio,
        )
        normalized_audio = unpack_audio_latents(
            unpad_audio_latents(state.audio, forward_ctx.original_audio_num_frames),
            num_mel_bins=forward_ctx.latent_mel_bins,
        )
        return LTXPhaseResult(
            forward_context=forward_ctx,
            video=unpacked_latents,
            audio=unpacked_audio,
            audio_for_next_phase=normalized_audio,
        )

    def _denoise_step(
        self,
        index: int,
        timestep: torch.Tensor,
        state: LTXAVState,
        forward_ctx: LTXForwardContext,
        denoise_ctx: LTXDenoiseContext,
    ) -> LTXAVState:
        denoise_ctx.latents = state.video
        denoise_ctx.audio_latents = state.audio
        noise_pred_video, noise_pred_audio = self._predict_noise_for_step(
            index,
            timestep,
            state,
            forward_ctx,
            denoise_ctx,
        )
        video_seq_len = state.video.shape[1]
        unified_sample = torch.cat([state.video, state.audio], dim=1).float()
        unified_pred = torch.cat([noise_pred_video, noise_pred_audio], dim=1).float()
        is_selected = index in self._selected_sde_steps

        stepped, log_prob, _, _ = self.scheduler.step(
            unified_pred,
            timestep,
            unified_sample,
            generator=forward_ctx.request_inputs.generator,
            noise_level=self._flow_grpo_noise_level if is_selected else 0.0,
            sde_type=self._flow_grpo_sde_type,
            return_logprobs=self._flow_grpo_logprobs and is_selected,
            return_dict=False,
        )
        next_video = stepped[:, :video_seq_len]
        next_audio = stepped[:, video_seq_len:]

        if is_selected:
            self._current_latents.append(unified_sample)
            self._next_latents.append(stepped.float())
            self._selected_timesteps.append(timestep)
            if log_prob is not None:
                self._log_probs.append(log_prob)

        video, audio = self._synchronize_guidance_parallel_step_output(
            (next_video, next_audio),
            guidance_parallel_ready=forward_ctx.guidance_parallel_ready,
        )
        return LTXAVState(video=video, audio=audio)

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> DiffusionOutput | list[DiffusionOutput]:
        """Generate one request and attach the FlowGRPO trajectory contract."""
        if req.num_reqs != 1:
            raise ValueError(f"LTX-2.3 FlowGRPO expects one request, got {req.num_reqs}.")
        request = req.requests[0]
        self._configure_flow_grpo(request)
        self._inject_precomputed_prompt_embeds(request)
        output = super().forward(req, **kwargs)
        if isinstance(output, list):
            if len(output) != 1:
                raise RuntimeError(f"Single-request LTX rollout returned {len(output)} outputs.")
            output = output[0]
        video, audio = output.output
        if isinstance(video, torch.Tensor) and video.ndim == 5:
            if video.shape[0] != 1:
                raise ValueError(f"Expected one video per diffusion request, got shape {tuple(video.shape)}.")
            video = video[0]
        prompt_context = self._flow_grpo_prompt_context
        if prompt_context is None:
            raise RuntimeError("LTX-2.3 rollout did not prepare prompt connector outputs.")

        audio_sample_rate = (
            self.vocoder.config.output_sampling_rate
            if hasattr(self, "vocoder") and self.vocoder is not None and hasattr(self.vocoder, "config")
            else 24000
        )
        result = rollout_output(
            media=(video, audio),
            media_key="video",
            trajectory_latents=self._flow_grpo_trajectory.get("all_latents"),
            trajectory_log_probs=self._flow_grpo_trajectory.get("all_log_probs"),
            trajectory_timesteps=self._flow_grpo_trajectory.get("all_timesteps"),
            prompt_embeddings={
                "prompt_embeds": prompt_context.positive_connector_prompt_embeds,
                "audio_prompt_embeds": prompt_context.positive_connector_audio_prompt_embeds,
                "prompt_embeds_mask": prompt_context.positive_connector_attention_mask,
                "negative_prompt_embeds": prompt_context.negative_connector_prompt_embeds,
                "negative_audio_prompt_embeds": prompt_context.negative_connector_audio_prompt_embeds,
                "negative_prompt_embeds_mask": prompt_context.negative_connector_attention_mask,
            },
            rl={
                "all_next_latents": self._flow_grpo_trajectory.get("all_next_latents"),
                "video_seq_len": self._flow_grpo_trajectory.get("video_seq_len"),
                "audio": audio,
                "audio_sample_rate": audio_sample_rate,
            },
            to_cpu=True,
        )
        self._current_timestep = None
        return result
