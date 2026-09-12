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
"""Pure fp32 tensor utilities for Qwen-Image DMD2 distribution matching.

The score-difference normalizer spans every non-batch dimension per sample.
Only the surrogate reduction is masked. These functions own no model or runtime.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "velocity_to_x0",
    "dmd_gradient",
    "dmd_surrogate_loss",
    "fake_score_target",
    "fake_score_loss",
    "ode_euler_step",
    "standard_cfg",
    "timestep_shift",
]


def velocity_to_x0(noisy: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """Convert flow velocity ``epsilon - x0`` to ``x0 = x_sigma - sigma * v``."""
    return noisy.float() - sigma.float() * velocity.float()


@torch.no_grad()
def dmd_gradient(
    x0_fake: Tensor,
    x0_real: Tensor,
    x_g: Tensor,
    normalization_epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor, int]:
    """Return detached normalized fake-minus-real scores for tensors ``(B, ...)``.

    Return the gradient, per-sample clamped normalizer and replacement count.
    The normalizer is unmasked, including when the caller masks the final loss.
    """
    if not math.isfinite(normalization_epsilon) or normalization_epsilon <= 0:
        raise ValueError(f"normalization_epsilon must be finite and greater than zero, got {normalization_epsilon}.")
    if x_g.shape != x0_fake.shape or x_g.shape != x0_real.shape:
        raise ValueError("x_g, x0_fake and x0_real must have identical shapes.")
    if x_g.ndim < 2 or x_g.numel() == 0:
        raise ValueError("DMD tensors require a nonempty batch and at least one non-batch dimension.")
    x_g, x0_fake, x0_real = x_g.float(), x0_fake.float(), x0_real.float()
    normalizer = (x_g - x0_real).abs().mean(dim=tuple(range(1, x_g.ndim)), keepdim=True)
    normalizer = normalizer.clamp_min(normalization_epsilon)
    gradient = (x0_fake - x0_real) / normalizer
    nonfinite = int((~torch.isfinite(gradient)).sum().item())
    return torch.nan_to_num(gradient), normalizer, nonfinite


def expand_loss_mask(mask: Tensor, prediction: Tensor) -> Tensor:
    """Expand a prefix or broadcastable mask to individual loss elements."""
    shape = tuple(mask.shape)
    mask = mask.bool()
    if mask.ndim < prediction.ndim and prediction.shape[: mask.ndim] == mask.shape:
        mask = mask.reshape(*mask.shape, *((1,) * (prediction.ndim - mask.ndim)))
    try:
        return torch.broadcast_to(mask, prediction.shape)
    except RuntimeError as exc:
        raise ValueError(f"Loss mask shape {shape} is not broadcastable to {tuple(prediction.shape)}.") from exc


def dmd_surrogate_loss(
    x_g: Tensor,
    g_normalized: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Compute ``0.5 * mean((x_g - stop_gradient(x_g - g))**2)`` in fp32."""
    if x_g.shape != g_normalized.shape or x_g.numel() == 0:
        raise ValueError("Student and normalized gradient must have identical nonempty shapes.")
    x_g = x_g.float()
    target = (x_g - g_normalized.float()).detach()
    if gradient_mask is None:
        return 0.5 * (x_g - target).square().mean(), x_g.numel()
    mask = expand_loss_mask(gradient_mask, x_g)
    active = int(mask.sum().item())
    if active == 0:
        raise ValueError("all-masked DMD loss is an error, not zero.")
    return 0.5 * (x_g - target).square()[mask].sum() / active, active


def fake_score_target(noise: Tensor, x_g: Tensor) -> Tensor:
    """Construct the detached rectified-flow target ``epsilon - x_g``."""
    if noise.shape != x_g.shape:
        raise ValueError("Fake-score noise and generated latents must have identical shapes.")
    return noise.detach().float() - x_g.detach().float()


def fake_score_loss(
    model_output: Tensor,
    noise: Tensor,
    x_g: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Compute fp32 denoising MSE; the engine must also detach its noisy inputs."""
    target = fake_score_target(noise, x_g)
    if model_output.shape != target.shape or model_output.numel() == 0:
        raise ValueError("Fake-score prediction and target must have identical nonempty shapes.")
    difference = (model_output.float() - target).square()
    if gradient_mask is None:
        return difference.mean(), difference.numel()
    mask = expand_loss_mask(gradient_mask, model_output)
    active = int(mask.sum().item())
    if active == 0:
        raise ValueError("all-masked fake-score loss is an error, not zero.")
    return difference[mask].sum() / active, active


def ode_euler_step(latents: Tensor, velocity: Tensor, sigma_from: Tensor, sigma_to: Tensor) -> Tensor:
    """Apply one deterministic Euler transition during backward simulation."""
    return latents.float() + (sigma_to.float() - sigma_from.float()) * velocity.float()


def standard_cfg(
    cond: Tensor,
    uncond: Tensor,
    guidance_scale: float,
    cfg_norm: Optional[str] = "none",
) -> Tensor:
    """Apply standard CFG, with LightX2V last-vector or batch-global rescaling."""
    cond, uncond = cond.float(), uncond.float()
    guided = uncond + guidance_scale * (cond - uncond)
    if cfg_norm in (None, "none"):
        return guided
    if cfg_norm == "layer_norm":
        return guided * (cond.norm(dim=-1, keepdim=True) / guided.norm(dim=-1, keepdim=True).clamp_min(1e-12))
    if cfg_norm == "scalar":
        return guided * (cond.norm() / guided.norm().clamp_min(1e-12)).clamp_max(1.0)
    raise ValueError(f"Unknown cfg_norm {cfg_norm!r}; expected one of {{'none', 'layer_norm', 'scalar'}}.")


def timestep_shift(timestep: Tensor, num_train_timesteps: int, shift: float = 1.0) -> Tensor:
    """Apply one rational shift, normalized by the model's actual training grid."""
    if not math.isfinite(shift) or shift < 1 or num_train_timesteps <= 0:
        raise ValueError("The timestep shift must be finite and at least 1, with a positive training grid size.")
    t = timestep.float()
    frac = t / num_train_timesteps
    return shift * frac / (1.0 + (shift - 1.0) * frac) * num_train_timesteps
