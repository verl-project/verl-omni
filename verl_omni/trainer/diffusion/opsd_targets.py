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

"""Torch-only math primitives for DiffusionOPSD (online preference self-distillation).

DiffusionOPSD turns a differentiable reward on the predicted clean sample into a
positive (reward-ascent) and a negative (reward-descent) target, then trains the
policy with a two-branch DPO-style objective. This module contains the whole
numerical core so it can be unit tested on CPU without importing ``verl``,
``tensordict``, ``omegaconf`` or ``diffusers``.

Notation follows the paper: ``xt`` is the noised latent, ``t_expanded`` is the
timestep broadcast to the latent shape, ``velocity`` / ``old_prediction`` are
flow-matching velocity predictions, and ``y0 = xt - t * v`` is the implied clean
sample. The first tensor dimension is always the batch dimension; every other
dimension (spatial, temporal, channel, ...) is reduced over per sample.

The two target construction routines implement Eq. 6-7 of the paper:

1. Per-sample trust-region budget ``b_i = rho * ||anchor_i||_2``.
2. ``n_ascent`` normalized-gradient steps of length ``eta * b_i / n_ascent``.
3. A final projection of the total displacement back into the
   ``b_i``-ball, so ``||target_i - anchor_i|| <= rho * ||anchor_i||`` holds by
   construction.

``build_positive_negative_targets`` shares the first reward gradient between the
ascent and the descent branch, which saves one reward backward per construction
when ``n_ascent == 2`` (the reference setting).

``opsd_two_branch_loss`` implements the two-branch policy objective (Eq. 11-13),
using a detached per-sample normalizer ``gamma`` computed in float64 for
stability. Only the current policy prediction carries gradient: anchors, old
predictions, targets and reward probabilities are detached inside this module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

__all__ = [
    "build_positive_negative_targets",
    "build_trust_region_targets",
    "clean_output",
    "opsd_anchor",
    "opsd_two_branch_loss",
    "reward_gradient",
]


def clean_output(xt: Tensor, t_expanded: Tensor, velocity: Tensor) -> Tensor:
    """Return the clean-sample estimate implied by a flow velocity.

    Args:
        xt: Noised latents of shape ``(B, ...)``.
        t_expanded: Timesteps broadcastable to ``xt``, e.g. ``(B, 1, 1, 1)``.
        velocity: Predicted velocity, same shape as ``xt``.

    Returns:
        ``y0 = xt - t_expanded * velocity``, differentiable w.r.t. its inputs.
    """
    return xt - t_expanded * velocity


def opsd_anchor(xt: Tensor, t_expanded: Tensor, old_prediction: Tensor) -> Tensor:
    """Return the detached OPSD anchor ``y0 = xt - t_expanded * old_prediction``.

    The anchor is the starting point of both trust-region branches and is
    treated as a fixed reference, so it is always detached.
    """
    return (xt - t_expanded * old_prediction).detach()


def reward_gradient(anchor: Tensor, reward_fn: Callable[[Tensor], Tensor]) -> Tensor:
    """Return ``d(sum(reward_fn(anchor))) / d(anchor)``.

    Args:
        anchor: Leaf tensor with ``requires_grad=True``.
        reward_fn: Differentiable callable mapping ``(B, ...)`` to ``(B,)``.

    Returns:
        Gradient tensor with the same shape as ``anchor``.

    Raises:
        ValueError: If ``anchor`` does not require gradients.
    """
    if not isinstance(anchor, Tensor) or not anchor.requires_grad:
        raise ValueError("anchor must be a tensor with requires_grad=True to compute a reward gradient.")
    reward = reward_fn(anchor)
    (grad,) = torch.autograd.grad(reward.sum(), anchor, create_graph=False)
    return grad


def _per_sample_norm(x: Tensor) -> Tensor:
    """Return the L2 norm over all non-batch dimensions, shape ``(B,)``."""
    if x.dim() <= 1:
        return x.abs()
    return x.flatten(1).norm(dim=1)


def _mean_non_batch(x: Tensor, *, keepdim: bool = False) -> Tensor:
    """Mean over all non-batch dimensions, shape ``(B,)`` or ``(B, 1, ..., 1)``."""
    if x.dim() <= 1:
        return x
    return x.mean(dim=tuple(range(1, x.dim())), keepdim=keepdim)


def _broadcast_per_sample(value: Tensor, like: Tensor) -> Tensor:
    """Reshape a per-sample ``(B,)`` tensor to broadcast against ``like``."""
    return value.reshape((value.shape[0],) + (1,) * (like.dim() - 1))


def _ascent_step(
    x: Tensor,
    reward_fn: Callable[[Tensor], Tensor],
    step_len: Tensor,
    *,
    direction: float,
    eps_g: float,
) -> Tensor:
    """One detached, normalized-gradient trust-region step from ``x``."""
    x = x.detach().clone().requires_grad_(True)
    grad = reward_gradient(x, reward_fn)
    grad_norm = _per_sample_norm(grad) + eps_g
    step = _broadcast_per_sample(step_len, grad) * grad / _broadcast_per_sample(grad_norm, grad)
    return (x.detach() + direction * step).detach()


def _project_to_budget(x: Tensor, anchor: Tensor, budget: Tensor, *, eps_g: float) -> Tensor:
    """Project ``x`` back into the per-sample ``budget``-ball around ``anchor``.

    The displacement is only ever shrunk (never grown): the scale factor is
    clamped to ``max=1``, which guarantees ``||target - anchor|| <= budget``.
    """
    displacement = x - anchor
    displacement_norm = _per_sample_norm(displacement)
    scale = (budget / (displacement_norm + eps_g)).clamp(max=1.0)
    return (anchor + displacement * _broadcast_per_sample(scale, displacement)).detach()


def build_trust_region_targets(
    anchor: Tensor,
    reward_fn: Callable[[Tensor], Tensor],
    *,
    rho: float = 0.10,
    n_ascent: int = 2,
    eta: float = 1.0,
    direction: float = 1,
    eps_g: float = 1e-12,
) -> Tensor:
    """Build one detached trust-region target (paper Eq. 6-7).

    Per sample ``i``:

    ``b_i = rho * ||anchor_i||_2`` (norm over all non-batch dims)
    ``step_len_i = eta * b_i / n_ascent``
    repeat ``n_ascent`` times: ``x <- x + direction * step_len_i * g / (||g||_2 + eps_g)``
    finally project: ``x <- anchor + d * clamp(b_i / (||d||_2 + eps_g), max=1)``
    with ``d = x - anchor``.

    Args:
        anchor: Detached clean-sample anchor of shape ``(B, ...)``.
        reward_fn: Differentiable reward mapping ``(B, ...)`` to ``(B,)``.
        rho: Relative trust-region radius.
        n_ascent: Number of normalized-gradient steps (must be >= 1).
        eta: Fraction of the budget actually traversed before projection.
        direction: ``+1`` for reward ascent, ``-1`` for reward descent.
        eps_g: Numerical epsilon for the gradient-norm and displacement-norm denominators.

    Returns:
        Detached target tensor with the same shape as ``anchor``.

    Raises:
        ValueError: If ``n_ascent < 1``.
    """
    if n_ascent < 1:
        raise ValueError(f"n_ascent must be >= 1, got {n_ascent}.")

    anchor = anchor.detach()
    budget = rho * _per_sample_norm(anchor)
    step_len = eta * budget / n_ascent

    x = anchor.clone()
    for _ in range(n_ascent):
        x = _ascent_step(x, reward_fn, step_len, direction=direction, eps_g=eps_g)
    return _project_to_budget(x, anchor, budget, eps_g=eps_g)


def build_positive_negative_targets(
    anchor: Tensor,
    reward_fn: Callable[[Tensor], Tensor],
    *,
    rho: float = 0.10,
    n_ascent: int = 2,
    eta: float = 1.0,
    eps_g: float = 1e-12,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    """Build the positive (ascent) and negative (descent) targets, sharing the first gradient.

    Both branches need the gradient of ``reward_fn`` at the anchor as their first
    step. That gradient is computed once here and used with opposite signs, which
    saves one reward backward pass compared to calling
    :func:`build_trust_region_targets` twice (``2 * n_ascent - 1`` reward calls
    instead of ``2 * n_ascent``).

    Args:
        anchor: Detached clean-sample anchor of shape ``(B, ...)``.
        reward_fn: Differentiable reward mapping ``(B, ...)`` to ``(B,)``.
        rho: Relative trust-region radius.
        n_ascent: Number of normalized-gradient steps per branch (must be >= 1).
        eta: Fraction of the budget actually traversed before projection.
        eps_g: Numerical epsilon for the gradient-norm and displacement-norm denominators.

    Returns:
        ``(target_pos, target_neg, diagnostics)``. Both targets are detached and
        have the shape of ``anchor``. ``diagnostics`` holds small Python floats
        for logging (relative displacements, budgets, norms).

    Raises:
        ValueError: If ``n_ascent < 1``.
    """
    if n_ascent < 1:
        raise ValueError(f"n_ascent must be >= 1, got {n_ascent}.")

    anchor = anchor.detach()
    budget = rho * _per_sample_norm(anchor)
    step_len = eta * budget / n_ascent

    # Shared first step: one reward backward gives the ascent and the descent direction.
    anchor_leaf = anchor.clone().requires_grad_(True)
    first_grad = reward_gradient(anchor_leaf, reward_fn)
    first_grad_norm = _per_sample_norm(first_grad) + eps_g
    first_step = (
        _broadcast_per_sample(step_len, first_grad) * first_grad / _broadcast_per_sample(first_grad_norm, first_grad)
    )
    target_pos = anchor + first_step
    target_neg = anchor - first_step

    for _ in range(n_ascent - 1):
        target_pos = _ascent_step(target_pos, reward_fn, step_len, direction=1, eps_g=eps_g)
        target_neg = _ascent_step(target_neg, reward_fn, step_len, direction=-1, eps_g=eps_g)

    target_pos = _project_to_budget(target_pos, anchor, budget, eps_g=eps_g)
    target_neg = _project_to_budget(target_neg, anchor, budget, eps_g=eps_g)

    with torch.no_grad():
        anchor_norm = _per_sample_norm(anchor)
        safe_anchor_norm = anchor_norm + eps_g
        diagnostics = {
            "displacement_pos_rel_mean": (_per_sample_norm(target_pos - anchor) / safe_anchor_norm).mean().item(),
            "displacement_neg_rel_mean": (_per_sample_norm(target_neg - anchor) / safe_anchor_norm).mean().item(),
            "budget_rel": (budget / safe_anchor_norm).mean().item(),
            "anchor_norm_mean": anchor_norm.mean().item(),
            "target_pos_norm_mean": _per_sample_norm(target_pos).mean().item(),
            "target_neg_norm_mean": _per_sample_norm(target_neg).mean().item(),
            "first_grad_norm_mean": first_grad_norm.detach().mean().item(),
        }

    return target_pos, target_neg, diagnostics


def opsd_two_branch_loss(
    *,
    forward_prediction: Tensor,
    old_prediction: Tensor,
    xt: Tensor,
    t_expanded: Tensor,
    target_pos: Tensor,
    target_neg: Tensor,
    reward_prob: Tensor,
    beta: float = 1.0,
    adv_clip_max: float = 5.0,
    eps_gamma: float = 1e-5,
) -> tuple[Tensor, dict[str, float]]:
    """Two-branch OPSD policy loss (paper Eq. 11-13).

    With ``old = sg(old_prediction)`` and ``theta = forward_prediction``:

    ``v_pos = beta * theta + (1 - beta) * old``
    ``v_neg = (1 + beta) * old - beta * theta``
    ``y_pos = xt - t_expanded * v_pos``, ``y_neg = xt - t_expanded * v_neg``
    ``gamma_pos = max(sg(mean_abs(y_pos - target_pos)), eps_gamma)`` (float64)
    ``gamma_neg = max(sg(mean_abs(y_neg - target_neg)), eps_gamma)`` (float64)
    ``per_sample = reward_prob * mean((y_pos - target_pos)^2 / gamma_pos)
                 + (1 - reward_prob) * mean((y_neg - target_neg)^2 / gamma_neg)``
    ``loss = mean(per_sample) * adv_clip_max``

    Reductions run over all non-batch dimensions. Only ``forward_prediction``
    (and any other input the caller explicitly left attached to the graph, such
    as ``xt``) can carry gradient: ``old_prediction``, both targets and
    ``reward_prob`` are detached here.

    Args:
        forward_prediction: Current policy velocity prediction, shape ``(B, ...)``.
        old_prediction: Velocity prediction of the sampling policy.
        xt: Noised latents, shape ``(B, ...)``.
        t_expanded: Timesteps broadcastable to ``xt``, e.g. ``(B, 1, 1, 1)``.
        target_pos: Detached reward-ascent target, shape ``(B, ...)``.
        target_neg: Detached reward-descent target, shape ``(B, ...)``.
        reward_prob: Per-sample probability of the positive branch, ``(B,)`` or
            ``(B, 1, ...)``.
        beta: Interpolation strength of the current policy against the old one.
        adv_clip_max: Scale applied to the averaged policy loss (advantage clip).
        eps_gamma: Lower clamp for the per-sample normalizer ``gamma``.

    Returns:
        ``(loss, metrics)`` where ``loss`` is a scalar tensor and ``metrics``
        holds ``opsd_pos_loss``, ``opsd_neg_loss``, ``opsd_policy_loss``,
        ``opsd_reward_prob_mean`` and ``opsd_total_loss`` as Python floats.
    """
    old = old_prediction.detach()
    target_pos = target_pos.detach()
    target_neg = target_neg.detach()
    reward_prob = reward_prob.detach()

    v_pos = beta * forward_prediction + (1.0 - beta) * old
    v_neg = (1.0 + beta) * old - beta * forward_prediction
    y_pos = xt - t_expanded * v_pos
    y_neg = xt - t_expanded * v_neg

    with torch.no_grad():
        # Normalizers are detached and computed in float64: gamma scales the
        # squared error, so losing precision here directly biases the loss.
        gamma_pos = _mean_non_batch((y_pos.detach().double() - target_pos.double()).abs(), keepdim=True)
        gamma_neg = _mean_non_batch((y_neg.detach().double() - target_neg.double()).abs(), keepdim=True)
        gamma_pos = gamma_pos.clamp_min(eps_gamma).to(y_pos.dtype)
        gamma_neg = gamma_neg.clamp_min(eps_gamma).to(y_neg.dtype)

    pos = _mean_non_batch((y_pos - target_pos) ** 2 / gamma_pos)
    neg = _mean_non_batch((y_neg - target_neg) ** 2 / gamma_neg)

    pos_weight = _broadcast_per_sample(reward_prob.to(pos.dtype), pos)
    per_sample = pos_weight * pos + (1.0 - pos_weight) * neg
    policy_loss = per_sample.mean()
    loss = policy_loss * adv_clip_max

    metrics = {
        "opsd_pos_loss": pos.detach().mean().item(),
        "opsd_neg_loss": neg.detach().mean().item(),
        "opsd_policy_loss": policy_loss.detach().item(),
        "opsd_reward_prob_mean": reward_prob.to(policy_loss.dtype).mean().item(),
        "opsd_total_loss": loss.detach().item(),
    }
    return loss, metrics
