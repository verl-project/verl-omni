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

from collections.abc import Sequence
from typing import Literal

import torch
from vllm_omni.diffusion.models.minimax_h3.time_request import minimax_h3_time_shift_sigmas

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    AUDIO_ROW_WIDTH,
    VIDEO_ROW_WIDTH,
    pack_video_audio_rows,
    unpack_video_audio_rows,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

H3_VIDEO_SHIFT = 12.0
H3_AUDIO_SHIFT = 3.0
H3_VIDEO_LOG_PROB_WEIGHT = 0.5
H3_AUDIO_LOG_PROB_WEIGHT = 0.5

H3_VIDEO_WIDTH = VIDEO_ROW_WIDTH
H3_AUDIO_WIDTH = AUDIO_ROW_WIDTH


def h3_sigma_schedules(
    num_steps: int,
    video_shift: float = H3_VIDEO_SHIFT,
    audio_shift: float = H3_AUDIO_SHIFT,
) -> tuple[list[float], list[float]]:
    """Call vLLM-Omni's H3 time-shift function for the video and audio schedules."""
    video_sigmas = minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=video_shift)
    audio_sigmas = minimax_h3_time_shift_sigmas(num_steps=num_steps, shift_scale=audio_shift)
    h3_transition_count(video_sigmas, audio_sigmas)
    return video_sigmas, audio_sigmas


def h3_transition_count(video_sigmas: Sequence[float], audio_sigmas: Sequence[float]) -> int:
    """Return the number of denoiser transitions in a sigma schedule, validating its shape.

    The count is always taken from the schedule itself (``len - 1``) rather than from
    ``num_inference_steps``: whether a schedule holds ``num_steps`` or ``num_steps + 1``
    boundaries is an upstream convention that has changed between vLLM-Omni releases.
    Rollout and Actor replay must walk the same grid down to sigma 0, otherwise the sample
    is left partly noisy.
    """
    if len(video_sigmas) != len(audio_sigmas):
        raise RuntimeError(
            f"MiniMax H3 video/audio sigma schedules differ in length: {len(video_sigmas)} vs {len(audio_sigmas)}."
        )
    if len(video_sigmas) < 2:
        raise RuntimeError("MiniMax H3 sigma schedule needs at least two boundaries.")
    for name, sigmas in (("video", video_sigmas), ("audio", audio_sigmas)):
        values = [float(value) for value in sigmas]
        if abs(values[0] - 1.0) > 1e-6 or abs(values[-1]) > 1e-6:
            raise RuntimeError(
                f"MiniMax H3 {name} sigma schedule must run from 1.0 to 0.0, got {values[0]} -> {values[-1]}."
            )
        if any(nxt >= cur for cur, nxt in zip(values, values[1:], strict=False)):
            raise RuntimeError(f"MiniMax H3 {name} sigma schedule must be strictly decreasing.")
    return len(video_sigmas) - 1


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
    return pack_video_audio_rows(video.flatten(1).unsqueeze(1), audio.flatten(1).unsqueeze(1)).unsqueeze(1)


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
    return unpack_video_audio_rows(joint, video_rows, audio_rows)
