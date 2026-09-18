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

import math
from typing import Any

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2
from vllm_omni.diffusion.models.ltx2.ltx2_conditioning import LTXPromptContext
from vllm_omni.diffusion.models.ltx2.ltx2_denoise import LTXDenoiseContext, LTXForwardContext, LTXPhaseResult
from vllm_omni.diffusion.models.ltx2.ltx2_latents import LTXAVState, clear_audio_padding
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
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import normalize_ltx_output_type

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

    #: Declares the joint video/audio rollout streams. The runtime audio sample
    #: rate (from the vocoder) is attached via rollout metadata and takes
    #: precedence over this declared default.
    diffusion_io_spec = DiffusionIOSpec(
        primary=MediaSpec("video"),
        auxiliary=(MediaSpec("audio", sample_rate=24000),),
    )

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__(od_config=od_config, prefix=prefix)
        self.set_progress_bar_config(disable=True)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_config(self.scheduler.config)
        self._flow_grpo_noise_level = 0.8
        self._flow_grpo_sde_type = "cps"
        self._flow_grpo_window_size: int | None = None
        self._flow_grpo_window_range: list[int] | None = None
        self._flow_grpo_sde_contiguous = True
        self._flow_grpo_logprobs = True
        self._flow_grpo_seed = 42
        self._flow_grpo_task: str | None = None
        self._flow_grpo_prompt_context: LTXPromptContext | None = None
        self._flow_grpo_trajectory: dict[str, torch.Tensor | None] = {}
        self._selected_sde_steps: set[int] = set()
        self._current_latents: list[torch.Tensor] = []
        self._next_latents: list[torch.Tensor] = []
        self._selected_timesteps: list[torch.Tensor] = []
        self._log_probs: list[torch.Tensor] = []
        self._flow_grpo_video_seq_len = 0
        self._flow_grpo_condition_image_latents: torch.Tensor | None = None

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
        extra_args = dict(req.sampling_params.extra_args or {})
        guidance_scale = float(getattr(req.sampling_params, "guidance_scale", None) or 1.0)
        for key in ("video_cfg_scale", "audio_cfg_scale"):
            configured = float(extra_args.get(key, guidance_scale))
            if not math.isclose(configured, guidance_scale):
                raise NotImplementedError(
                    f"LTX-2.3 FlowGRPO requires {key} to match guidance_scale={guidance_scale}, got {configured}."
                )
            extra_args[key] = guidance_scale
        for key, expected in {
            "video_stg_scale": 0.0,
            "audio_stg_scale": 0.0,
            "video_modality_scale": 1.0,
            "audio_modality_scale": 1.0,
            "video_rescale_scale": 0.0,
            "audio_rescale_scale": 0.0,
        }.items():
            configured = float(extra_args.get(key, expected))
            if not math.isclose(configured, expected):
                raise NotImplementedError(f"LTX-2.3 FlowGRPO does not support {key}={configured}; expected {expected}.")
            extra_args[key] = expected
        req.sampling_params.extra_args = extra_args
        self._flow_grpo_noise_level = float(extra_args.get("noise_level", 0.8))
        self._flow_grpo_sde_type = extra_args.get("sde_type", "cps")
        self._flow_grpo_window_size = extra_args.get("sde_window_size")
        self._flow_grpo_window_range = extra_args.get("sde_window_range")
        self._flow_grpo_sde_contiguous = bool(extra_args.get("sde_contiguous", True))
        self._flow_grpo_logprobs = bool(extra_args.get("logprobs", True))
        self._flow_grpo_task = extra_args.get("task")
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
        """Execute one full-model phase while recording policy transitions."""
        if phase_recipe.sampler != "euler" or phase_recipe.adapter_slot is not None:
            raise NotImplementedError("LTX-2.3 FlowGRPO supports only the regular one-stage checkpoint.")
        if getattr(self, "_flow_grpo_task", None) == "ti2va" and image is None:
            raise ValueError("LTX-2.3 TI2VA requires exactly one first-frame image.")

        if sigmas is not None:
            num_steps = len(sigmas) - int(bool(sigmas) and float(sigmas[-1]) == 0.0)
        elif timesteps is not None:
            num_steps = len(timesteps)
        else:
            num_steps = request_inputs.num_inference_steps
        self._selected_sde_steps = set(self._select_sde_steps(num_steps, self.device))
        self._current_latents = []
        self._next_latents = []
        self._selected_timesteps = []
        self._log_probs = []
        self._flow_grpo_video_seq_len = 0
        self._flow_grpo_condition_image_latents = None

        result = super().run_phase(
            req,
            request_inputs,
            noise_scale=noise_scale,
            sigmas=sigmas,
            timesteps=timesteps,
            attention_kwargs=attention_kwargs,
            phase_recipe=phase_recipe,
            image=image,
            prompt_context=prompt_context,
        )
        if not self._current_latents:
            raise RuntimeError("LTX-2.3 rollout selected no SDE transitions.")
        if self._flow_grpo_condition_image_latents is None:
            raise RuntimeError("LTX-2.3 rollout did not capture the video condition state.")

        batch_size = self._current_latents[0].shape[0]
        self._flow_grpo_trajectory = {
            "all_latents": torch.stack(self._current_latents, dim=1),
            "all_next_latents": torch.stack(self._next_latents, dim=1),
            "all_timesteps": torch.stack(self._selected_timesteps).unsqueeze(0).expand(batch_size, -1),
            "all_log_probs": torch.stack(self._log_probs, dim=1) if self._log_probs else None,
            "video_seq_len": torch.full(
                (batch_size,),
                self._flow_grpo_video_seq_len,
                device=self._current_latents[0].device,
                dtype=torch.long,
            ),
            "condition_image_latents": self._flow_grpo_condition_image_latents,
        }
        return result

    @staticmethod
    def _video_policy_mask(state: LTXAVState, denoise_ctx: LTXDenoiseContext) -> torch.Tensor:
        condition_mask = denoise_ctx.conditioning_mask
        if condition_mask is None:
            return torch.ones(state.video.shape[1], device=state.video.device, dtype=torch.bool)
        if condition_mask.shape != state.video.shape[:2]:
            raise ValueError("LTX-2.3 conditioning mask must match the packed video batch and sequence dimensions.")
        condition_mask = condition_mask.to(device=state.video.device, dtype=torch.bool)
        if not torch.equal(condition_mask, condition_mask[:1].expand_as(condition_mask)):
            raise ValueError("LTX-2.3 FlowGRPO requires one shared conditioning layout per request batch.")
        return ~condition_mask[0]

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
        video_policy_mask = self._video_policy_mask(state, denoise_ctx)
        condition_count = int((~video_policy_mask).sum().item())
        if state.video.shape[1] % int(forward_ctx.latent_num_frames):
            raise ValueError("LTX-2.3 packed video rows do not divide into latent frames.")
        rows_per_frame = state.video.shape[1] // int(forward_ctx.latent_num_frames)
        if condition_count not in (0, rows_per_frame) or (
            condition_count and video_policy_mask[:condition_count].any()
        ):
            raise ValueError("LTX-2.3 TI2VA supports exactly one fixed first latent frame.")
        audio_length = int(forward_ctx.original_audio_num_frames)
        if not 0 < audio_length <= state.audio.shape[1]:
            raise ValueError(
                f"LTX-2.3 logical audio length must be in [1, {state.audio.shape[1]}], got {audio_length}."
            )

        current_video = state.video[:, video_policy_mask]
        current_audio = state.audio[:, :audio_length]
        unified_sample = torch.cat([current_video, current_audio], dim=1).float()
        unified_pred = torch.cat(
            [noise_pred_video[:, video_policy_mask], noise_pred_audio[:, :audio_length]],
            dim=1,
        ).float()
        self._flow_grpo_video_seq_len = current_video.shape[1]
        if self._flow_grpo_condition_image_latents is None:
            self._flow_grpo_condition_image_latents = state.video[:, ~video_policy_mask].float()
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
        next_video = state.video.clone()
        next_video[:, video_policy_mask] = stepped[:, : current_video.shape[1]].to(next_video.dtype)
        next_audio = state.audio.clone()
        next_audio[:, :audio_length] = stepped[:, current_video.shape[1] :].to(next_audio.dtype)
        next_audio = clear_audio_padding(next_audio, audio_length)
        next_sample = torch.cat(
            [next_video[:, video_policy_mask], next_audio[:, :audio_length]],
            dim=1,
        ).float()

        if is_selected:
            self._current_latents.append(unified_sample)
            self._next_latents.append(next_sample)
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
                "condition_image_latents": self._flow_grpo_trajectory.get("condition_image_latents"),
                "audio": audio,
                "audio_sample_rate": audio_sample_rate,
            },
            to_cpu=True,
        )
        self._current_timestep = None
        return result
