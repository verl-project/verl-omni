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

"""Shared FLUX DanceGRPO rollout utilities."""

from typing import Optional

import torch

from verl_omni.pipelines.wan22_dance_grpo.common import sd3_time_shift  # noqa: F401


def predict_original_sample(
    sample: torch.Tensor, model_output: torch.Tensor, sigma: torch.Tensor | float
) -> torch.Tensor:
    """Return FLUX's clean latent prediction ``x0 = x_t - sigma * v``."""
    if sample.shape != model_output.shape:
        raise ValueError(f"sample/model_output shapes differ: {tuple(sample.shape)} vs {tuple(model_output.shape)}")
    sigma_tensor = torch.as_tensor(sigma, device=sample.device, dtype=sample.dtype)
    return sample - sigma_tensor * model_output.to(dtype=sample.dtype)


def select_dance_grpo_transitions(
    current_latents: torch.Tensor,
    next_latents: torch.Tensor,
    timesteps: torch.Tensor,
    log_probs: torch.Tensor,
    *,
    strategy: str = "continuous",
    fraction: float = 1.0,
    drop_last: bool = False,
    generator: Optional[torch.Generator | list[torch.Generator]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select aligned transitions after the chronological rollout completes."""
    if strategy not in {"continuous", "random_subset"}:
        raise ValueError(f"unknown timestep selection strategy: {strategy!r}")
    if not 0 < fraction <= 1:
        raise ValueError(f"timestep fraction must be in (0, 1], got {fraction}")
    if current_latents.ndim < 3 or next_latents.ndim != current_latents.ndim:
        raise ValueError("current_latents and next_latents must have matching [B,T,...] shapes")
    if current_latents.shape != next_latents.shape:
        raise ValueError(
            f"current/next latent shapes differ: {tuple(current_latents.shape)} vs {tuple(next_latents.shape)}"
        )
    if timesteps.ndim != 2 or timesteps.shape != current_latents.shape[:2]:
        raise ValueError(
            f"timesteps must match latent [B,T]={tuple(current_latents.shape[:2])}, got {tuple(timesteps.shape)}"
        )
    if log_probs.ndim < 2 or log_probs.shape[:2] != timesteps.shape:
        raise ValueError(f"log_probs first dimensions must be {tuple(timesteps.shape)}, got {tuple(log_probs.shape)}")

    if drop_last:
        if timesteps.shape[1] <= 1:
            raise ValueError("cannot drop the last transition when T <= 1")
        current_latents = current_latents[:, :-1]
        next_latents = next_latents[:, :-1]
        timesteps = timesteps[:, :-1]
        log_probs = log_probs[:, :-1]

    batch_size, num_transitions = timesteps.shape
    if strategy == "continuous":
        return current_latents, next_latents, timesteps, log_probs

    num_selected = max(1, int(num_transitions * fraction))
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(f"expected {batch_size} transition generators, got {len(generator)}")
    indices = torch.stack(
        [
            torch.randperm(
                num_transitions,
                device=timesteps.device,
                generator=generator[row] if isinstance(generator, list) else generator,
            )[:num_selected]
            for row in range(batch_size)
        ],
        dim=0,
    )

    def _gather(value: torch.Tensor) -> torch.Tensor:
        gather_index = indices.view(batch_size, num_selected, *([1] * (value.ndim - 2)))
        return torch.gather(value, 1, gather_index.expand(-1, -1, *value.shape[2:]))

    return (
        _gather(current_latents),
        _gather(next_latents),
        _gather(timesteps),
        _gather(log_probs),
    )
