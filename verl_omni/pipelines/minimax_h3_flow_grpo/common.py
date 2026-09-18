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

"""Shared MiniMax H3 trajectory and packed-layout helpers."""

from __future__ import annotations

from typing import Literal

import torch
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0
H3_VIDEO_LOG_PROB_WEIGHT = 0.5
H3_AUDIO_LOG_PROB_WEIGHT = 0.5

H3_VIDEO_WIDTH = 96
H3_AUDIO_WIDTH = 32


def h3_sigma_schedules(
    num_steps: int,
    video_shift: float = H3_VIDEO_SHIFT,
    audio_shift: float = H3_AUDIO_SHIFT,
) -> tuple[list[float], list[float]]:
    """Call vLLM-Omni's H3 time-shift function for the video and audio schedules."""
    return (
        minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=video_shift),
        minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=audio_shift),
    )


def configure_flow_scheduler(
    scheduler: FlowMatchSDEDiscreteScheduler,
    sigmas: torch.Tensor | list[float],
    device: torch.device | str,
) -> None:
    """Configure the shared FlowGRPO scheduler with an exact H3 sigma grid."""
    sigmas = torch.as_tensor(sigmas, dtype=torch.float32)
    if sigmas.ndim != 1 or sigmas.numel() < 2 or not torch.isclose(sigmas[-1], sigmas.new_zeros(())):
        raise ValueError("MiniMax H3 sigma grid must be one-dimensional and end at zero.")
    scheduler.set_timesteps(sigmas=sigmas[:-1].cpu().tolist(), device=device)
    expected = sigmas.to(scheduler.sigmas.device)
    if scheduler.sigmas.shape != expected.shape or not torch.allclose(scheduler.sigmas, expected):
        raise RuntimeError("Shared FlowGRPO scheduler changed the MiniMax H3 sigma grid.")


def sample_h3_transition(
    scheduler: FlowMatchSDEDiscreteScheduler,
    sample: torch.Tensor,
    h3_velocity: torch.Tensor,
    step: int,
    *,
    noise_level: float,
    sde_type: Literal["sde", "cps"],
    generator: torch.Generator | None = None,
    prev_sample: torch.Tensor | None = None,
    return_log_prob: bool = True,
):
    """Run one H3 transition through the shared scheduler."""
    if not 0 <= step < len(scheduler.timesteps):
        raise IndexError(f"MiniMax H3 scheduler step {step} is out of range.")
    timestep = scheduler.timesteps[step].expand(sample.shape[0])
    return scheduler.sample_previous_step(
        sample=sample.float(),
        model_output=-h3_velocity.float(),
        timestep=timestep,
        generator=generator,
        noise_level=noise_level,
        prev_sample=None if prev_sample is None else prev_sample.float(),
        sde_type=sde_type,
        return_logprobs=return_log_prob,
        return_sqrt_dt=True,
    )


def combine_log_probs(
    video_log_prob: torch.Tensor,
    audio_log_prob: torch.Tensor,
    *,
    video_weight: float = H3_VIDEO_LOG_PROB_WEIGHT,
    audio_weight: float = H3_AUDIO_LOG_PROB_WEIGHT,
) -> torch.Tensor:
    """Combine per-modality mean log densities using explicit H3 weights."""
    return video_weight * video_log_prob + audio_weight * audio_log_prob


def flatten_joint_latents(video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
    """Encode unequal H3 row widths as one Engine-compatible row."""
    if video.shape[0] != audio.shape[0]:
        raise ValueError("MiniMax H3 video and audio batch sizes must match.")
    return torch.cat([video.flatten(1), audio.flatten(1)], dim=1).unsqueeze(1)


def split_joint_latents(
    joint: torch.Tensor,
    video_rows: int,
    audio_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse :func:`flatten_joint_latents`."""
    if joint.ndim == 3:
        if joint.shape[1] != 1:
            raise ValueError(f"MiniMax H3 joint trajectory row must be singleton, got {joint.shape}.")
        joint = joint[:, 0]
    video_numel = video_rows * H3_VIDEO_WIDTH
    audio_numel = audio_rows * H3_AUDIO_WIDTH
    if joint.shape[-1] != video_numel + audio_numel:
        raise ValueError(
            f"MiniMax H3 joint width {joint.shape[-1]} does not match video/audio metadata "
            f"({video_numel} + {audio_numel})."
        )
    return (
        joint[:, :video_numel].reshape(joint.shape[0], video_rows, H3_VIDEO_WIDTH),
        joint[:, video_numel:].reshape(joint.shape[0], audio_rows, H3_AUDIO_WIDTH),
    )
