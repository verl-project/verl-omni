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

import copy
import os
from typing import Any

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import (
    LTX23Pipeline,
    _LTX23DenoiseContext,
    _LTX23ForwardContext,
    _LTX23PromptContext,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import calculate_shift

__all__ = ["LTX23PipelineWithLogProb"]


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="flow_grpo")
class LTX23PipelineWithLogProb(LTX23Pipeline):
    """Sample LTX-2.3 with CPS/SDE transitions and return joint log-probs."""

    supports_request_batch = False

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)
        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            od_config.model,
            subfolder="scheduler",
            local_files_only=os.path.exists(od_config.model),
        )
        self._flow_grpo_noise_level = 0.8
        self._flow_grpo_sde_type = "cps"
        self._flow_grpo_sde_steps: list[int] | None = None
        self._flow_grpo_num_sde_steps: int | None = None
        self._flow_grpo_window_size: int | None = None
        self._flow_grpo_window_range: list[int] | None = None
        self._flow_grpo_logprobs = True
        self._flow_grpo_seed = 42
        self._flow_grpo_prompt_context: _LTX23PromptContext | None = None
        self._flow_grpo_trajectory: dict[str, torch.Tensor | None] = {}

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
        extra_args = req.sampling_params.extra_args or {}
        self._flow_grpo_noise_level = float(extra_args.get("noise_level", 0.8))
        self._flow_grpo_sde_type = extra_args.get("sde_type", "cps")
        self._flow_grpo_sde_steps = extra_args.get("sde_steps")
        self._flow_grpo_num_sde_steps = extra_args.get("num_sde_steps")
        self._flow_grpo_window_size = extra_args.get("sde_window_size")
        self._flow_grpo_window_range = extra_args.get("sde_window_range")
        self._flow_grpo_logprobs = bool(extra_args.get("logprobs", True))
        scheduler_seed = int(extra_args.get("sde_window_seed", 42))
        global_step = int(extra_args.get("global_steps", 1))
        self._flow_grpo_seed = scheduler_seed + max(global_step - 1, 0)

    def _select_sde_steps(self, num_steps: int, device: torch.device) -> list[int]:
        del device
        if self._flow_grpo_sde_steps is not None:
            eligible = sorted({int(step) for step in self._flow_grpo_sde_steps if 0 <= int(step) < num_steps})
            if not eligible:
                raise ValueError(
                    f"No valid LTX SDE steps remain after filtering {self._flow_grpo_sde_steps} "
                    f"against num_inference_steps={num_steps}."
                )
            count = self._flow_grpo_num_sde_steps
            if count is None or count >= len(eligible):
                return eligible
            generator = torch.Generator().manual_seed(self._flow_grpo_seed)
            order = torch.randperm(len(eligible), generator=generator)[: int(count)].tolist()
            return sorted(eligible[index] for index in order)

        if self._flow_grpo_window_size is not None:
            window_size = int(self._flow_grpo_window_size)
            window_range = self._flow_grpo_window_range or [0, num_steps]
            low = int(window_range[0])
            high = min(int(window_range[1]), num_steps)
            if window_size <= 0 or high - low < window_size:
                raise ValueError(
                    f"Invalid LTX SDE window: size={window_size}, range={window_range}, num_steps={num_steps}."
                )
            generator = torch.Generator().manual_seed(self._flow_grpo_seed)
            start = int(torch.randint(low, high - window_size + 1, (1,), generator=generator).item())
            return list(range(start, start + window_size))

        return list(range(max(num_steps - 1, 0)))

    def _prepare_prompt_context(self, **kwargs) -> _LTX23PromptContext:
        prompt_context = super()._prepare_prompt_context(**kwargs)
        self._flow_grpo_prompt_context = prompt_context
        return prompt_context

    def _prepare_scheduler_stage(
        self,
        request_inputs,
        *,
        device: torch.device,
        sigmas: list[float] | None,
        timesteps: list[int] | None,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> tuple[Any, Any, torch.Tensor]:
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
        video_audio_scheduler = self._make_video_audio_scheduler(
            audio_scheduler,
            latent_num_frames,
            latent_height,
            latent_width,
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
        self._num_timesteps = len(timesteps_tensor)
        return audio_scheduler, video_audio_scheduler, timesteps_tensor

    def _denoise_loop(
        self,
        forward_ctx: _LTX23ForwardContext,
        denoise_ctx: _LTX23DenoiseContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run joint audio-video denoising and retain selected stochastic transitions."""
        request_inputs = forward_ctx.request_inputs
        prompt_context = forward_ctx.prompt_context
        guidance_scale = request_inputs.guidance_scale
        audio_scheduler = forward_ctx.audio_scheduler
        selected_steps = set(self._select_sde_steps(len(forward_ctx.timesteps), denoise_ctx.latents.device))

        current_latents = []
        next_latents = []
        log_probs = []
        selected_timesteps = []
        self.scheduler.set_begin_index(0)

        with self.progress_bar(total=len(forward_ctx.timesteps)) as pbar:
            for index, timestep_value in enumerate(forward_ctx.timesteps):
                if self.interrupt:
                    continue
                self._current_timestep = timestep_value

                if forward_ctx.cfg_parallel_ready:
                    video_input = denoise_ctx.latents.to(prompt_context.positive_connector_prompt_embeds.dtype)
                    audio_input = denoise_ctx.audio_latents.to(prompt_context.positive_connector_prompt_embeds.dtype)
                    timestep = timestep_value.expand(video_input.shape[0])
                    positive_kwargs = self._build_transformer_kwargs(
                        forward_ctx,
                        denoise_ctx,
                        hidden_states=video_input,
                        audio_hidden_states=audio_input,
                        encoder_hidden_states=prompt_context.positive_connector_prompt_embeds,
                        audio_encoder_hidden_states=prompt_context.positive_connector_audio_prompt_embeds,
                        encoder_attention_mask=prompt_context.positive_connector_attention_mask,
                        audio_encoder_attention_mask=prompt_context.positive_connector_attention_mask,
                        ts=timestep,
                    )
                    negative_kwargs = {
                        **positive_kwargs,
                        "encoder_hidden_states": prompt_context.negative_connector_prompt_embeds,
                        "audio_encoder_hidden_states": prompt_context.negative_connector_audio_prompt_embeds,
                        "encoder_attention_mask": prompt_context.negative_connector_attention_mask,
                        "audio_encoder_attention_mask": prompt_context.negative_connector_attention_mask,
                    }
                    video_pred, audio_pred = self.predict_noise_with_parallel_cfg(
                        true_cfg_scale=guidance_scale,
                        positive_kwargs=positive_kwargs,
                        negative_kwargs=negative_kwargs,
                        cfg_normalize=False,
                        video_latents=denoise_ctx.latents,
                        audio_latents=denoise_ctx.audio_latents,
                        video_sigma=self.scheduler.sigmas[index],
                        audio_sigma=audio_scheduler.sigmas[index],
                    )
                else:
                    video_input = (
                        torch.cat([denoise_ctx.latents] * 2)
                        if self.do_classifier_free_guidance
                        else denoise_ctx.latents
                    ).to(prompt_context.connector_prompt_embeds.dtype)
                    audio_input = (
                        torch.cat([denoise_ctx.audio_latents] * 2)
                        if self.do_classifier_free_guidance
                        else denoise_ctx.audio_latents
                    ).to(prompt_context.connector_prompt_embeds.dtype)
                    timestep = timestep_value.expand(video_input.shape[0])
                    transformer_kwargs = self._build_transformer_kwargs(
                        forward_ctx,
                        denoise_ctx,
                        hidden_states=video_input,
                        audio_hidden_states=audio_input,
                        encoder_hidden_states=prompt_context.connector_prompt_embeds,
                        audio_encoder_hidden_states=prompt_context.connector_audio_prompt_embeds,
                        encoder_attention_mask=prompt_context.connector_attention_mask,
                        audio_encoder_attention_mask=prompt_context.connector_attention_mask,
                        ts=timestep,
                    )
                    with self._transformer_cache_context("cond_uncond"):
                        video_pred, audio_pred = self.transformer(**transformer_kwargs)
                    video_pred = video_pred.float()
                    audio_pred = audio_pred.float()

                    if self.do_classifier_free_guidance:
                        negative_video, positive_video = video_pred.chunk(2)
                        negative_audio, positive_audio = audio_pred.chunk(2)
                        video_pred = self._combine_x0_space_cfg(
                            denoise_ctx.latents,
                            positive_video,
                            negative_video,
                            self.scheduler.sigmas[index],
                            guidance_scale,
                        )
                        audio_pred = self._combine_x0_space_cfg(
                            denoise_ctx.audio_latents,
                            positive_audio,
                            negative_audio,
                            audio_scheduler.sigmas[index],
                            guidance_scale,
                        )

                video_seq_len = denoise_ctx.latents.shape[1]
                unified_sample = torch.cat([denoise_ctx.latents, denoise_ctx.audio_latents], dim=1).float()
                unified_pred = torch.cat([video_pred, audio_pred], dim=1).float()
                is_selected = index in selected_steps
                stepped, log_prob, _, _ = self.scheduler.step(
                    unified_pred,
                    timestep_value,
                    unified_sample,
                    generator=request_inputs.generator,
                    noise_level=self._flow_grpo_noise_level if is_selected else 0.0,
                    sde_type=self._flow_grpo_sde_type,
                    return_logprobs=self._flow_grpo_logprobs and is_selected,
                    return_dict=False,
                )
                denoise_ctx.latents = stepped[:, :video_seq_len]
                denoise_ctx.audio_latents = stepped[:, video_seq_len:]

                if is_selected:
                    current_latents.append(unified_sample)
                    next_latents.append(stepped.float())
                    selected_timesteps.append(timestep_value)
                    if log_prob is not None:
                        log_probs.append(log_prob)
                pbar.update()

        if not current_latents:
            raise RuntimeError("LTX-2.3 rollout selected no SDE transitions.")
        batch_size = denoise_ctx.latents.shape[0]
        self._flow_grpo_trajectory = {
            "all_latents": torch.stack(current_latents, dim=1),
            "all_next_latents": torch.stack(next_latents, dim=1),
            "all_timesteps": torch.stack(selected_timesteps).unsqueeze(0).expand(batch_size, -1),
            "all_log_probs": torch.stack(log_probs, dim=1) if log_probs else None,
            "video_seq_len": torch.full(
                (batch_size,),
                denoise_ctx.latents.shape[1],
                device=denoise_ctx.latents.device,
                dtype=torch.long,
            ),
        }
        return denoise_ctx.latents, denoise_ctx.audio_latents

    @torch.no_grad()
    def forward(self, req: DiffusionRequestBatch, **kwargs: Any) -> list[DiffusionOutput]:
        """Generate one request and attach the FlowGRPO trajectory contract."""
        if req.num_reqs != 1:
            raise ValueError(f"LTX-2.3 FlowGRPO expects one request, got {req.num_reqs}.")
        request = req.requests[0]
        self._configure_flow_grpo(request)
        self._inject_precomputed_prompt_embeds(request)
        outputs = super().forward(req, **kwargs)
        if len(outputs) != 1:
            raise RuntimeError(f"Single-request LTX rollout returned {len(outputs)} outputs.")
        output = outputs[0]
        prompt_context = self._flow_grpo_prompt_context
        if prompt_context is None:
            raise RuntimeError("LTX-2.3 rollout did not prepare prompt connector outputs.")

        custom_output = {
            **self._flow_grpo_trajectory,
            "prompt_embeds": prompt_context.positive_connector_prompt_embeds,
            "audio_prompt_embeds": prompt_context.positive_connector_audio_prompt_embeds,
            "prompt_embeds_mask": prompt_context.positive_connector_attention_mask,
            "negative_prompt_embeds": prompt_context.negative_connector_prompt_embeds,
            "negative_audio_prompt_embeds": prompt_context.negative_connector_audio_prompt_embeds,
            "negative_prompt_embeds_mask": prompt_context.negative_connector_attention_mask,
            "audio_sample_rate": self.vocoder.config.output_sampling_rate,
        }
        output.custom_output = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in custom_output.items()
        }
        self._current_timestep = None
        return [output]
