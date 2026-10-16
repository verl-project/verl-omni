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
"""DiffusionART tensor math: aggregation-first guidance, trust-region mask and frozen-target MSE.

DiffusionART (Wan et al., "DiffusionART: Efficient, Stable and Scalable Reinforcement of
Diffusion Models", arXiv:2610.12572; authors' code https://github.com/vanzll/DiffusionART)
is implemented here from the paper's Section 4 and Appendix C, in verl-omni's conventions.
The pure functions below have no trainer dependencies; ``DiffusionARTLoss`` in
``diffusion_algos.py`` wires them into the worker loss and the actor batch preparation.

Velocity convention: verl-omni's flow-matching adapters predict ``v = noise - x0``
(``x0 = x_t - t * v``), whereas the paper writes ``u_t = x0 - noise``. The paper's target
``v_old + G`` therefore becomes ``v_old - G`` here; every inner product used by the mask is
convention-free because it only involves differences of velocities.
"""

import numpy as np
import torch

__all__ = [
    "ART_LOSS_SPACES",
    "ART_MASK_METRICS",
    "art_guidance",
    "art_loss",
    "art_pairwise_labels",
    "art_reference_drift",
    "group_index_from_uid",
]

ART_LOSS_SPACES = ("velocity_linear_sigma", "velocity_uniform")
ART_MASK_METRICS = ("sde_kl", "fixed_mse")


def group_index_from_uid(uid: np.ndarray) -> tuple[torch.Tensor, int]:
    """Number prompts ``0, 1, ...`` in order of first appearance of their ``uid``.

    Returns:
        ``(group_index, num_groups)`` with ``group_index`` an int64 tensor of shape ``(B,)``.
    """
    _, first, inverse = np.unique(np.asarray(uid), return_index=True, return_inverse=True)
    rank = np.empty_like(first)
    rank[np.argsort(first, kind="stable")] = np.arange(first.size)
    group_index = torch.from_numpy(rank[inverse.reshape(-1)].astype(np.int64))
    return group_index, int(first.size)


@torch.no_grad()
def art_guidance(
    endpoints: torch.Tensor,
    rewards: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int,
    return_covariance_rms: bool = False,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, ...]:
    """Per-group reward-endpoint covariance normalised to unit RMS (paper Eq. 5).

    ``Cov(R, x0 | c) = 1/K sum_i (R_i - mean R)(x0_i - mean x0)`` over the ``K`` endpoints of
    prompt ``c``; the ``mean x0`` term vanishes because the centred rewards sum to zero. The
    covariance, scaled by the group's mean absolute centred reward, is divided by its RMS (floored at
    ``eps``) so every active prompt contributes a direction of unit RMS independent of the reward
    scale, except that near-zero reward-endpoint correlations are damped rather than amplified (the
    reference's ``historical`` direction mode). A group yields zero guidance (and is marked inactive)
    when it has a single sample, constant rewards, or an exactly zero covariance.

    Args:
        endpoints: Clean final latents, ``(B, *latent_shape)``.
        rewards: One scalar reward (or pairwise label) per endpoint, ``(B,)``.
        group_index: Prompt index of every endpoint in ``[0, num_groups)``, ``(B,)`` int64.
        num_groups: Number of prompts in the batch.
        return_covariance_rms: Also return the RMS of the raw covariance per group, ``(num_groups,)``
            float32 (the strength of the reward-endpoint correlation before normalisation; a diagnostic).
        eps: Floor of the RMS normaliser applied to the reward-MAD-scaled covariance (reference default
            ``1e-5``); groups below it receive a damped direction instead of a unit-RMS one.

    Returns:
        ``(guidance, active)``: ``guidance`` has shape ``(num_groups, *latent_shape)`` in
        float32 and is zero for inactive groups; ``active`` is a ``(num_groups,)`` bool mask.
        With ``return_covariance_rms`` a third element ``covariance_rms`` is appended.
    """
    if endpoints.ndim < 2:
        raise ValueError(f"endpoints must have shape (B, ...), got {tuple(endpoints.shape)}.")
    flat = endpoints.detach().float().flatten(1)
    rewards = rewards.detach().to(device=flat.device, dtype=torch.float32).reshape(-1)
    groups = group_index.to(device=flat.device, dtype=torch.long).reshape(-1)
    if rewards.numel() != flat.shape[0] or groups.numel() != flat.shape[0]:
        raise ValueError("rewards and group_index must have one entry per endpoint.")
    if num_groups < 1 or (groups < 0).any() or (groups >= num_groups).any():
        raise ValueError(f"group_index must lie in [0, {num_groups}).")
    if not torch.isfinite(flat).all() or not torch.isfinite(rewards).all():
        raise ValueError("endpoints and rewards must be finite.")

    counts = torch.bincount(groups, minlength=num_groups).float()
    safe_counts = counts.clamp_min(1.0)
    reward_mean = torch.zeros(num_groups, device=flat.device).index_add_(0, groups, rewards) / safe_counts
    reward_max = torch.full((num_groups,), -torch.inf, device=flat.device).scatter_reduce(
        0, groups, rewards, reduce="amax"
    )
    reward_min = torch.full((num_groups,), torch.inf, device=flat.device).scatter_reduce(
        0, groups, rewards, reduce="amin"
    )
    constant = reward_max == reward_min  # exact check: summing equal fp32 rewards can round the mean
    centered = torch.where(constant[groups], torch.zeros_like(rewards), rewards - reward_mean[groups])

    covariance = torch.zeros(num_groups, flat.shape[1], device=flat.device, dtype=torch.float64)
    covariance.index_add_(0, groups, centered.double()[:, None] * flat.double())
    covariance = covariance / safe_counts.double()[:, None]
    rms = covariance.square().mean(dim=1).sqrt()
    # Reference normalisation (vanzll/DiffusionART ``art_scalar_covariance``, ``direction_mode="historical"``): the
    # covariance is first scaled by the group's mean absolute centred reward (MAD) and then divided by its RMS
    # floored at ``eps``, so a group whose reward-scaled covariance is below ``eps`` RMS gets a *damped* push
    # instead of a unit-RMS push along what is mostly noise. For every other group this equals ``cov / rms(cov)``.
    mad = (
        torch.zeros(num_groups, device=flat.device, dtype=torch.float64).index_add_(0, groups, centered.abs().double())
        / safe_counts.double()
    )
    signal = covariance / torch.where(mad > 0, mad, torch.ones_like(mad))[:, None]
    signal_rms = signal.square().mean(dim=1).sqrt()
    active = (counts > 1) & ~constant & (mad > 0) & (signal_rms > 0)
    direction = signal / signal_rms.clamp_min(eps)[:, None]
    direction = torch.where(active[:, None], direction, torch.zeros_like(direction)).float()
    guidance = direction.reshape(num_groups, *endpoints.shape[1:])
    if return_covariance_rms:
        return guidance, active, rms.float()
    return guidance, active


@torch.no_grad()
def art_pairwise_labels(
    scores: torch.Tensor,
    group_index: torch.Tensor,
    num_groups: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Winner/loser labels for DiffusionART-Pairwise (paper Sec. 4.4, App. C.3).

    Every group's ``K`` endpoints are split into ``K / 2`` random disjoint pairs; the winner
    gets ``1``, the loser ``0`` and a tie gives both ``0.5``. The labels replace the scalar
    rewards in :func:`art_guidance`, which is unchanged otherwise.

    Args:
        scores: Scalar scores used only to decide each pair's winner, ``(B,)``.
        group_index: Prompt index of every endpoint, ``(B,)`` int64.
        num_groups: Number of prompts in the batch.
        generator: CPU generator for the pairing permutation.

    Returns:
        ``(B,)`` float32 labels in ``{0, 0.5, 1}``.

    Raises:
        ValueError: If a group has an odd number of endpoints.
    """
    scores = scores.detach().float().reshape(-1)
    groups = group_index.to(device=scores.device, dtype=torch.long).reshape(-1)
    labels = torch.full_like(scores, 0.5)
    for group in range(num_groups):
        rows = torch.nonzero(groups == group, as_tuple=True)[0]
        if rows.numel() == 0:
            continue
        if rows.numel() % 2:
            raise ValueError(f"Pairwise guidance needs an even group size; group {group} has {rows.numel()} rows.")
        perm = torch.randperm(rows.numel(), generator=generator).to(rows.device)
        left, right = rows[perm].reshape(-1, 2).unbind(1)
        win = torch.where(scores[left] > scores[right], 1.0, torch.where(scores[left] < scores[right], 0.0, 0.5))
        labels[left] = win
        labels[right] = 1.0 - win
    return labels


def art_reference_drift(
    movement_energy: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    reference_noise_level: float,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Per-coordinate KL proxy between the current and old reference SDE transitions (Eq. 7, App. C.2).

    ODE rollouts have no finite KL, so the paper measures policy change through Flow-GRPO's
    ODE-to-SDE reference transition ``q(. | s_t) = N(mu(s_t, v), sigma_ref^2 I)`` with
    ``mu(s_t, v) = x_t + h [(1 + nu^2 / 2) v - nu^2 / (2 (1 - t)) x_t]`` and
    ``sigma_ref^2 = nu^2 t / (1 - t) h`` for ``0 < t < 1``, where ``h = t - t_next`` and ``nu`` is
    the reference noise level. At ``t = 1`` the paper uses ``mu = x_t + h v`` and
    ``sigma_ref^2 = nu^2``. The proxy is ``||mu_cur - mu_old||^2 / (2 D sigma_ref^2)``.

    Args:
        movement_energy: ``mean((v_theta - v_old)^2)`` over latent coordinates, ``(B,)``.
        sigma: Flow time ``t`` in ``(0, 1]`` of every state, ``(B,)``.
        sigma_next: Flow time of the next ODE state (``0`` for the last step), ``(B,)``.
        reference_noise_level: ``nu`` of the reference SDE (paper default ``0.8``).
        eps: Tolerance of the ``t = 1`` band, floor for ``1 - t``, and square-root floor of the variance
            (reference default ``1e-5``).

    Raises:
        ValueError: if any ``sigma - sigma_next`` is not a positive finite step.

    Returns:
        ``(B,)`` drift scores ``m(s_t)``.
    """
    nu2 = float(reference_noise_level) ** 2
    sigma = sigma.float()
    step = sigma - sigma_next.float()
    if not torch.isfinite(step).all() or bool((step <= 0).any()):
        raise ValueError("art_reference_drift needs sigma > sigma_next for every item (a forward ODE step).")
    # Reference edge handling (vanzll ``art_loss``): the ``t = 1`` form applies within ``atol=eps`` of 1, the
    # variance is floored at ``eps**2`` (a std floor of ``eps``), and ``1 - t`` is floored at ``eps``.
    initial = torch.isclose(sigma, torch.ones_like(sigma), atol=eps, rtol=0.0)
    coefficient = torch.where(initial, step, step * (1.0 + 0.5 * nu2))
    variance = torch.where(initial, torch.full_like(sigma, nu2), nu2 * sigma / (1.0 - sigma).clamp_min(eps) * step)
    return coefficient.square() * movement_energy.float() / (2.0 * variance.clamp_min(eps**2))


def art_loss(
    current_velocity: torch.Tensor,
    old_velocity: torch.Tensor,
    guidance: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    *,
    guidance_gain: float = 1.0,
    loss_space: str = "velocity_linear_sigma",
    mask_metric: str = "sde_kl",
    mask_threshold: float = 1e-5,
    reference_noise_level: float = 0.8,
    corrective_mask_enabled: bool = True,
    restore_anchor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Trust-region-masked frozen-target velocity MSE (paper Eq. 6-8, Algorithm 1).

    Every state regresses toward the frozen target ``v_old - gain * G(c)`` (see the module
    docstring for the sign). With ``d = v_theta - v_old`` the descent direction is
    ``target - v_theta``; the second update of a rollout therefore carries both the guidance
    and a restoration ``-d`` proportional to the drift. The mask zeroes items whose drift
    score ``m(s_t)`` reaches ``mask_threshold`` *and* whose displacement points along the
    descent direction (``<d, target - v_theta> > 0``, an outward move). The weight is ``t``
    for ``velocity_linear_sigma`` and ``1`` for ``velocity_uniform``. Items with zero
    guidance (inactive prompts) or ``t = 0`` contribute zero loss but stay in the batch
    denominator, unless they are *restore anchors*: pairwise tie rows of a batch that holds
    at least one decided pair (paper App. C.3) keep a pure restoration loss toward ``v_old``.

    Args:
        current_velocity: ``v_theta(s_t)``, ``(B, *latent_shape)``; the only input with gradient.
        old_velocity: Rollout-policy prediction ``v_old(s_t)``, same shape.
        guidance: Unit-RMS guidance ``G(c)`` of each item's prompt, same shape; zero when inactive.
        sigma: Flow time ``t`` of every state, ``(B,)``.
        sigma_next: Flow time of the next ODE state, ``(B,)``.
        guidance_gain: Scalar multiplier on ``G(c)`` (paper: ``1``).
        loss_space: One of :data:`ART_LOSS_SPACES`.
        mask_metric: ``"sde_kl"`` (Eq. 7) or ``"fixed_mse"`` (raw ``mean(d^2)``).
        mask_threshold: Drift threshold ``delta`` (paper: ``1e-5``).
        reference_noise_level: ``nu`` of the reference SDE used by ``"sde_kl"``.
        corrective_mask_enabled: Disable to ablate the trust-region mask.
        restore_anchor: Optional ``(B,)`` bool; inactive rows flagged here train toward ``v_old``.

    Returns:
        Dict with ``per_item_loss`` ``(B,)`` (differentiable), ``per_item_mse``, ``target``,
        ``active``, ``anchor``, ``keep_mask``, ``masked``, ``movement_energy``,
        ``movement_score`` and ``outward_alignment`` (all ``(B,)`` except ``target``).
    """
    if loss_space not in ART_LOSS_SPACES:
        raise ValueError(f"loss_space must be one of {ART_LOSS_SPACES}, got {loss_space!r}.")
    if mask_metric not in ART_MASK_METRICS:
        raise ValueError(f"mask_metric must be one of {ART_MASK_METRICS}, got {mask_metric!r}.")
    if current_velocity.shape != old_velocity.shape or current_velocity.shape != guidance.shape:
        raise ValueError("current_velocity, old_velocity and guidance must share one shape.")
    velocity = current_velocity.float()
    old = old_velocity.detach().to(velocity)
    direction = guidance.detach().to(velocity)
    sigma = sigma.detach().to(velocity).reshape(-1)
    sigma_next = sigma_next.detach().to(velocity).reshape(-1)
    if sigma.numel() != velocity.shape[0] or sigma_next.numel() != velocity.shape[0]:
        raise ValueError("sigma and sigma_next need one value per item.")

    active = (direction.flatten(1).square().sum(dim=1) > 0) & (sigma > 0)
    if restore_anchor is None:
        anchor = torch.zeros_like(active)
    else:
        anchor = restore_anchor.detach().to(device=active.device).reshape(-1).bool() & ~active & (sigma > 0)
    target = (old - float(guidance_gain) * direction).detach()
    residual = target - velocity
    per_item_mse = residual.square().flatten(1).mean(dim=1)
    weight = sigma if loss_space == "velocity_linear_sigma" else torch.ones_like(sigma)

    with torch.no_grad():
        movement = velocity - old
        movement_energy = movement.square().flatten(1).mean(dim=1)
        outward_alignment = (movement * residual).flatten(1).mean(dim=1)
        if mask_metric == "sde_kl":
            movement_score = art_reference_drift(movement_energy, sigma, sigma_next, reference_noise_level)
        else:
            movement_score = movement_energy
        masked = active & (movement_score >= float(mask_threshold)) & (outward_alignment > 0)
        if not corrective_mask_enabled:
            masked = torch.zeros_like(masked)
        keep = anchor | (active & ~masked)

    per_item_loss = torch.where(keep, weight * per_item_mse, torch.zeros_like(per_item_mse))
    return {
        "per_item_loss": per_item_loss,
        "per_item_mse": per_item_mse,
        "target": target,
        "active": active,
        "anchor": anchor,
        "keep_mask": keep,
        "masked": masked,
        "movement_energy": movement_energy,
        "movement_score": movement_score,
        "outward_alignment": outward_alignment,
    }
