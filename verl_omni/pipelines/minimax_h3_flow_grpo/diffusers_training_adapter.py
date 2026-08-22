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

"""Diffusers Actor adapter for MiniMax H3 T2VA FlowGRPO."""

from __future__ import annotations

from typing import Optional

import torch
from diffusers import ModelMixin
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    combine_log_probs,
    configure_flow_scheduler,
    flatten_joint_latents,
    h3_sigma_schedules,
    sample_h3_transition,
    split_joint_latents,
)
from .weight_sync import H3_LORA_TARGETS

__all__ = ["MiniMaxH3FlowGRPO"]

H3SchedulerPair = tuple[FlowMatchSDEDiscreteScheduler, FlowMatchSDEDiscreteScheduler]


def _shared_int(value: torch.Tensor, name: str) -> int:
    values = value.reshape(-1)
    if values.numel() == 0 or not torch.all(values == values[0]):
        raise ValueError(f"MiniMax H3 requires one shared {name} per Actor micro-batch.")
    return int(values[0].item())


def _shared_layout(value: torch.Tensor, length: int, name: str) -> torch.Tensor:
    value = value[:, :length]
    if value.shape[0] > 1 and not torch.all(value == value[0]):
        raise ValueError(
            f"MiniMax H3 Actor micro-batch contains different {name} layouts. "
            "Use ppo_micro_batch_size_per_gpu=1 in the first implementation."
        )
    return value[0]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="flow_grpo")
class MiniMaxH3FlowGRPO(DiffusionModelBase):
    """Replay flattened joint video/audio transitions with the H3 DiT."""

    @classmethod
    def validate_lora_config(cls, model_config: DiffusionModelConfig) -> None:
        """Reject LoRA targets that the rollout weight sync cannot transport."""
        if model_config.lora_rank <= 0:
            return

        target_modules = model_config.target_modules
        requested = {target_modules} if isinstance(target_modules, str) else set(target_modules or [])
        unsupported = {
            target
            for target in requested
            if not any(target == supported or target.endswith("." + supported) for supported in H3_LORA_TARGETS)
        }
        if not requested or unsupported:
            raise ValueError(
                "MiniMax H3 LoRA supports only transformer/refiner attention and MLP projections "
                f"{sorted(H3_LORA_TARGETS)}, got unsupported targets {sorted(unsupported or requested)}. "
                "Other targets cannot be synchronized to the rollout model."
            )

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig) -> H3SchedulerPair:
        # H3 uses different sigma schedules for video and audio latents.
        schedulers = (FlowMatchSDEDiscreteScheduler(), FlowMatchSDEDiscreteScheduler())
        cls.set_timesteps(schedulers, model_config, get_device_name())
        return schedulers

    @classmethod
    def set_timesteps(
        cls,
        scheduler: H3SchedulerPair,
        model_config: DiffusionModelConfig,
        device: str,
    ) -> None:
        num_steps = model_config.pipeline.num_inference_steps
        video_sigmas, audio_sigmas = h3_sigma_schedules(num_steps)
        video_scheduler, audio_scheduler = scheduler
        configure_flow_scheduler(video_scheduler, video_sigmas, device)
        configure_flow_scheduler(audio_scheduler, audio_sigmas, device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module: ModelMixin,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, None]:
        del module, model_config, negative_prompt_embeds, negative_prompt_embeds_mask
        required = {
            "all_next_latents",
            "h3_step_indices",
            "h3_audio_timesteps",
            "h3_video_rows",
            "h3_audio_rows",
            "h3_seq_len",
            "h3_position_ids",
            "h3_token_tags",
            "h3_video_indices",
            "h3_audio_indices",
            "h3_text_indices",
        }
        missing = sorted(required - set(micro_batch.keys()))
        if missing:
            raise KeyError(f"MiniMax H3 rollout is missing fields: {missing}.")

        # Diffusers treats H3's batch dimension as a replication axis: every
        # sample in one forward must share the same packed-sequence structure.
        video_rows = _shared_int(micro_batch["h3_video_rows"], "video row count")
        audio_rows = _shared_int(micro_batch["h3_audio_rows"], "audio row count")
        seq_len = _shared_int(micro_batch["h3_seq_len"], "packed sequence length")
        current_video, current_audio = split_joint_latents(latents[:, step], video_rows, audio_rows)

        if prompt_embeds_mask is not None:
            text_len = _shared_int(prompt_embeds_mask.sum(dim=-1), "text length")
            prompt_embeds = prompt_embeds[:, :text_len]

        position_ids = _shared_layout(micro_batch["h3_position_ids"], seq_len, "position_ids")
        token_tags = _shared_layout(micro_batch["h3_token_tags"], seq_len, "token_tags")
        video_indices = _shared_layout(micro_batch["h3_video_indices"], video_rows, "video indices").long()
        audio_indices = _shared_layout(micro_batch["h3_audio_indices"], audio_rows, "audio indices").long()
        text_indices = _shared_layout(micro_batch["h3_text_indices"], prompt_embeds.shape[1], "text indices").long()

        original_step = _shared_int(micro_batch["h3_step_indices"][:, step], "scheduler step")
        step_timesteps = torch.stack((timesteps[:, step], micro_batch["h3_audio_timesteps"][:, step]), dim=-1)
        if step_timesteps.shape[0] > 1 and not torch.all(step_timesteps == step_timesteps[0]):
            raise ValueError("MiniMax H3 requires shared video/audio timesteps per Actor micro-batch.")
        video_t = float(step_timesteps[0, 0].item())
        audio_t = float(step_timesteps[0, 1].item())
        row_timesteps = torch.full((seq_len,), video_t, device=current_video.device, dtype=torch.float32)
        row_timesteps[audio_indices.to(current_video.device)] = audio_t
        unique_t, inverse = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        return (
            {
                "hidden_states": current_video,
                "audio_hidden_states": current_audio,
                "encoder_hidden_states": prompt_embeds,
                "timestep": unique_t,
                "timestep_indices": inverse,
                "token_tags": token_tags.to(current_video.device),
                "position_ids": position_ids.to(current_video.device),
                "video_indices": video_indices.to(current_video.device),
                "audio_indices": audio_indices.to(current_video.device),
                "text_indices": text_indices.to(current_video.device),
                "return_dict": False,
                "_h3_scheduler_step": original_step,
            },
            None,
        )

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module: ModelMixin,
        scheduler: H3SchedulerPair,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        del negative_model_inputs
        if scheduler_inputs is None:
            raise ValueError("MiniMax H3 replay requires rollout scheduler inputs.")
        model_inputs = dict(model_inputs)
        original_step = int(model_inputs.pop("_h3_scheduler_step"))
        video_velocity, audio_velocity = module(**model_inputs)
        video = model_inputs["hidden_states"].float()
        audio = model_inputs["audio_hidden_states"].float()
        next_video, next_audio = split_joint_latents(
            scheduler_inputs["all_next_latents"][:, step],
            video.shape[1],
            audio.shape[1],
        )
        video_scheduler, audio_scheduler = scheduler
        video_out = sample_h3_transition(
            video_scheduler,
            video,
            video_velocity.float(),
            original_step,
            noise_level=model_config.algo.noise_level,
            sde_type=model_config.algo.sde_type,
            prev_sample=next_video,
        )
        audio_out = sample_h3_transition(
            audio_scheduler,
            audio,
            audio_velocity.float(),
            original_step,
            noise_level=model_config.algo.noise_level,
            sde_type=model_config.algo.sde_type,
            prev_sample=next_audio,
        )
        video_log_prob = video_out[1]
        audio_log_prob = audio_out[1]
        if video_log_prob is None or audio_log_prob is None:
            raise RuntimeError("MiniMax H3 replay did not compute log probabilities.")

        video_weight = video[0].numel()
        audio_weight = audio[0].numel()
        total_weight = video_weight + audio_weight
        log_prob = combine_log_probs(video_log_prob, audio_log_prob)
        mean = flatten_joint_latents(video_out[2], audio_out[2])
        video_std = video_out[3].reshape(-1).mean()
        audio_std = audio_out[3].reshape(-1).mean()
        effective_std = (
            torch.sqrt((video_weight * video_std.square() + audio_weight * audio_std.square()) / total_weight)
            .reshape(1, 1, 1)
            .expand(video.shape[0], -1, -1)
        )
        video_dt = video_out[4].reshape(-1).mean().square()
        audio_dt = audio_out[4].reshape(-1).mean().square()
        effective_sqrt_dt = torch.sqrt((video_weight * video_dt + audio_weight * audio_dt) / total_weight)
        effective_sqrt_dt = effective_sqrt_dt.expand(video.shape[0])
        return log_prob, mean, effective_std, effective_sqrt_dt
