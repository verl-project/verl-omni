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
"""Pure DMD-family math used by the distillation trainer.

This module is independent of Ray and model libraries: it implements only the
detached-normalized distribution-matching gradient, the surrogate student loss,
the fake-score target, canonical x0 conversion, and the CFG forms that the
reference implementations use. The equations follow RFC §7 and §8.

Why this is a separate module (and not part of ``recipes.py`` or
``contracts.py``):

- ``contracts.py`` holds *data types* (immutable plan pieces, role layout, the
  execution state machine). It carries no equations.
- ``recipes.py`` holds *declarations* (which objective, which rollout strategy,
  which initialization, how roles map onto groups). It never computes a quantity.
- ``equations.py`` holds the only *executable equations* in the package. Every value
  is a pure function of its tensors; nothing here reads config, weights, or the
  prompt. Keeping these functions together means they can be unit-tested as
  algebraic identities and finite-difference checks without building a plan or an
  executor (see ``test_distillation_dmd_math_on_cpu.py``).

Boundary conditions (all must match the reviewed reference implementations):

- ``normalizer`` is formed over the **entire** ``x_g - x0_real`` tensor across all
  non-batch (block, frame, channel, spatial) dimensions of one sample, ``keepdim``
  per sample. It is **not** restricted by ``gradient_mask``.
- Only the surrogate loss is masked by ``gradient_mask``.
- ``normalization_epsilon`` is applied as ``max(normalizer, normalization_epsilon)``
  before division, and a non-finite ``g / normalizer`` is replaced by
  ``nan_to_num`` and counted.
- All score/objective arithmetic is fp32.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

__all__ = [
    "epsilon_to_x0",
    "velocity_to_x0",
    "dmd_gradient",
    "dmd_surrogate_loss",
    "fake_score_target",
    "fake_score_loss",
    "ode_regression_loss",
    "ode_euler_step",
    "consistency_renoise_step",
    "standard_cfg",
    "legacy_cfg",
    "timestep_shift",
]


def epsilon_to_x0(noisy: Tensor, epsilon: Tensor, sigma: Tensor, a_fn, b_fn) -> Tensor:
    """Convert an epsilon prediction to canonical ``x0`` for ``a(sigma)``/``b(sigma)``.

    For rectified flow, ``a(sigma) = 1 - sigma`` and ``b(sigma) = sigma``. For an
    epsilon-prediction model, ``x_sigma = a*x0 + b*epsilon`` so
    ``x0 = (x_sigma - b*epsilon) / a``.
    """
    noisy = noisy.float()
    epsilon = epsilon.float()
    sigma = sigma.float()
    a = a_fn(sigma).float()
    b = b_fn(sigma).float()
    if torch.any(a == 0):
        raise ValueError("epsilon_to_x0 is undefined where a(sigma) is zero.")
    return (noisy - b * epsilon) / a


def velocity_to_x0(noisy: Tensor, velocity: Tensor, sigma: Tensor) -> Tensor:
    """Convert a velocity prediction to canonical ``x0`` (rectified flow).

    ``x_sigma = (1 - sigma) * x0 + sigma * epsilon`` and ``v = epsilon - x0``, so
    ``x0 = x_sigma - sigma * v``.
    """
    return noisy.float() - sigma.float() * velocity.float()


def dmd_gradient(
    x0_fake: Tensor,
    x0_real: Tensor,
    x_g: Tensor,
    normalization_epsilon: float = 1e-5,
) -> tuple[Tensor, Tensor, int]:
    """Compute the detached normalized fake-minus-real score gradient.

    Follows Self-Forcing ``_compute_kl_grad`` and LightX2V ``dmd_loss`` exactly:

    ``g = x0_fake - x0_real``
    ``normalizer = mean(abs(x_g - x0_real), non-batch dimensions)``
    ``g_normalized = nan_to_num(g / max(normalizer, normalization_epsilon))``

    Returns ``(g_normalized, normalizer, nonfinite_count)``. ``normalizer`` is
    formed over the entire ``x_g - x0_real`` tensor across all non-batch
    dimensions of one sample, ``keepdim`` per sample, and is **not** restricted by
    ``gradient_mask``.
    """
    if normalization_epsilon <= 0:
        raise ValueError(f"normalization_epsilon must be greater than zero, got {normalization_epsilon}.")
    if x_g.shape != x0_fake.shape or x_g.shape != x0_real.shape:
        raise ValueError(
            f"x_g, x0_fake, and x0_real must have identical shapes, got "
            f"{tuple(x_g.shape)}, {tuple(x0_fake.shape)}, and {tuple(x0_real.shape)}."
        )
    if x_g.ndim < 2:
        raise ValueError("DMD tensors must include a batch dimension and at least one non-batch dimension.")

    x_g = x_g.float()
    x0_fake = x0_fake.float()
    x0_real = x0_real.float()

    g = x0_fake - x0_real
    # All non-batch dims (keep batch dim), per-sample keepdim.
    reduction_dims = tuple(range(1, x_g.dim()))
    normalizer = torch.abs(x_g - x0_real).mean(dim=reduction_dims, keepdim=True)
    normalizer = torch.maximum(normalizer, torch.as_tensor(normalization_epsilon, device=normalizer.device))
    g_normalized = g / normalizer
    nonfinite = (~torch.isfinite(g_normalized)).sum().item()
    g_normalized = torch.nan_to_num(g_normalized)
    return g_normalized, normalizer, nonfinite


def dmd_surrogate_loss(
    x_g: Tensor,
    g_normalized: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Surrogate objective ``L_DMD = 0.5 * mean((x_g - stop_gradient(x_g - g_norm))^2)``.

    Only the surrogate loss is masked by ``gradient_mask``. An all-masked loss is
    an error, not zero. Returns ``(loss, active_elements)``.
    """
    x_g = x_g.float()
    g_normalized = g_normalized.float()
    target = (x_g - g_normalized).detach()
    if gradient_mask is None:
        loss = 0.5 * torch.mean((x_g - target) ** 2)
        active = x_g.numel()
    else:
        mask = gradient_mask.bool()
        active = int(mask.sum().item())
        if active == 0:
            raise ValueError("all-masked DMD loss is an error, not zero.")
        diff = (x_g - target) ** 2
        loss = 0.5 * torch.sum(diff[mask]) / active
    return loss, active


def fake_score_target(noise: Tensor, x_g: Tensor) -> Tensor:
    """Rectified-flow fake-score target ``v_target = epsilon - x_g``.

    The fake score is trained to denoise the generated clean latent with
    ``x_sigma = (1 - sigma) * x_g + sigma * epsilon``.
    """
    return noise.detach().float() - x_g.detach().float()


def fake_score_loss(
    model_output: Tensor,
    noise: Tensor,
    x_g: Tensor,
    gradient_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Fake-score denoising MSE against the rectified-flow velocity target.

    ``L_fake = mean((model_output - (epsilon - x_g))^2)``. With a mask, the loss
    divides by the number of active elements after applying the mask.
    """
    target = fake_score_target(noise, x_g)
    model_output = model_output.float()
    if gradient_mask is None:
        loss = torch.mean((model_output - target) ** 2)
        active = model_output.numel()
    else:
        mask = gradient_mask.bool()
        active = int(mask.sum().item())
        if active == 0:
            raise ValueError("all-masked fake-score loss is an error, not zero.")
        diff = (model_output - target) ** 2
        loss = torch.sum(diff[mask]) / active
    return loss, active


def ode_regression_loss(
    prediction: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> tuple[Tensor, int]:
    """Compute fp32 ODE-target MSE over nonzero-timestep positions."""
    prediction = prediction.float()
    target = target.detach().float()
    if prediction.shape != target.shape:
        raise ValueError(
            f"ODE prediction and target must have identical shapes, got {tuple(prediction.shape)} and "
            f"{tuple(target.shape)}."
        )
    if valid_mask is None:
        return torch.mean((prediction - target) ** 2), prediction.numel()
    mask = valid_mask.bool()
    if mask.shape != prediction.shape:
        if prediction.shape[: mask.ndim] == mask.shape:
            mask = mask.reshape(*mask.shape, *((1,) * (prediction.ndim - mask.ndim)))
        try:
            mask = torch.broadcast_to(mask, prediction.shape)
        except RuntimeError as exc:
            raise ValueError(
                f"ODE valid_mask shape {tuple(valid_mask.shape)} is not broadcastable to {tuple(prediction.shape)}."
            ) from exc
    active = int(mask.sum().item())
    if active == 0:
        raise ValueError("all-masked ODE regression loss is an error, not zero.")
    return torch.sum((prediction - target)[mask] ** 2) / active, active


def ode_euler_step(latents: Tensor, velocity: Tensor, sigma_from: Tensor, sigma_to: Tensor) -> Tensor:
    """Apply one deterministic Euler transition used by backward simulation."""
    return latents.float() + (sigma_to.float() - sigma_from.float()) * velocity.float()


def consistency_renoise_step(x0: Tensor, noise: Tensor, sigma_to: Tensor) -> Tensor:
    """Re-noise a clean prediction with fresh noise for a consistency transition."""
    return (1.0 - sigma_to.float()) * x0.float() + sigma_to.float() * noise.float()


def standard_cfg(
    cond: Tensor,
    uncond: Tensor,
    guidance_scale: float,
    cfg_norm: Optional[str] = "none",
) -> Tensor:
    """Apply ``uncond + scale * (cond - uncond)`` classifier-free guidance.

    ``cfg_norm`` follows the LightX2V reference: ``layer_norm`` rescales each
    last-dimension vector, while ``scalar`` uses one norm ratio for the entire
    tensor.
    """
    cond = cond.float()
    uncond = uncond.float()
    guided = uncond + guidance_scale * (cond - uncond)
    if cfg_norm in (None, "none"):
        return guided
    if cfg_norm == "layer_norm":
        cond_norm = torch.norm(cond, dim=-1, keepdim=True)
        guided_norm = torch.norm(guided, dim=-1, keepdim=True)
        return guided * (cond_norm / guided_norm.clamp_min(1e-12))
    if cfg_norm == "scalar":
        ratio = torch.norm(cond) / torch.norm(guided).clamp_min(1e-12)
        return guided * min(1.0, ratio.item())
    raise ValueError(f"Unknown cfg_norm {cfg_norm!r}; expected one of {{'none', 'layer_norm', 'scalar'}}.")


def legacy_cfg(cond: Tensor, uncond: Tensor, guidance_scale: float) -> Tensor:
    """Self-Forcing CFG form ``cond + legacy_scale*(cond - uncond)``.

    This is *not* numerically identical to :func:`standard_cfg` with the same
    scale. A parity recipe must convert explicitly rather than silently reusing the
    number.
    """
    return cond.float() + guidance_scale * (cond.float() - uncond.float())


def timestep_shift(timestep: Tensor, num_train_timesteps: int, shift: float = 1.0) -> Tensor:
    """Apply the time-shift remapping exactly once.

    ``shifted = shift * (t / T) / (1 + (shift - 1) * (t / T)) * T``. The
    normalization is by the model's ``num_train_timesteps``, so it diverges from a
    hardcoded 1000 for any model whose ``num_train_timesteps`` is not 1000.
    """
    if shift <= 1.0:
        return timestep.float()
    t = timestep.float()
    frac = t / num_train_timesteps
    shifted = shift * frac / (1.0 + (shift - 1.0) * frac) * num_train_timesteps
    return shifted
