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

"""Rollout Correction for diffusion training (experimental).

Mirrors ``verl.trainer.ppo.rollout_corr_helper``.  All IS/RS math is delegated
to ``verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_and_rejection_mask``
so the algorithm semantics are byte-for-byte identical to the verl LLM stack.

Key differences from the verl LLM helper:

* **No ``response_mask``** — diffusion log-probs are dense across the SDE window
  (no padding).  RS rejection is folded into ``rollout_is_weights`` as a 0/1
  multiplier instead of modifying a separate mask.
* **Per-step engine correction** — in bypass mode, IS/RS is computed inside the
  FSDP engine's ``forward_step`` for each micro-batch / SDE step rather than in
  a dedicated loss function.  This is necessary because the diffusion actor
  already dispatches losses by registered name (``flow_grpo``, ``grpo_guard``)
  and the per-step granularity is finer than a global-batch correction.
* **``loss_type`` respected** — ``ppo_clip`` (default) omits IS weights from the
  loss multiplier (PPO ratio handles IS); ``reinforce`` (reserved) would apply
  IS weights explicitly.  This matches verl's ``compute_policy_loss_bypass_mode``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from verl import DataProto
from verl.trainer.ppo.rollout_corr_helper import (
    compute_offpolicy_metrics,
    compute_rollout_correction_and_rejection_mask,
)

__all__ = [
    "apply_bypass_mode_to_diffusion_batch",
    "apply_rollout_correction_to_diffusion_batch",
    "compute_per_step_rollout_correction",
    "compute_rollout_corr_metrics_from_logprobs",
    "rollout_correction_enabled",
]

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def rollout_correction_enabled(rollout_corr_config) -> bool:
    """Return True if the config requests any IS or RS computation."""
    if rollout_corr_config is None:
        return False
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    return bool(rollout_is) or bool(rollout_rs)


def _is_ppo_clip(rollout_corr_config) -> bool:
    """Return True if the loss_type is ppo_clip (or unset, for backward compat)."""
    loss_type = rollout_corr_config.get("loss_type", "ppo_clip") if rollout_corr_config else "ppo_clip"
    return loss_type == "ppo_clip"


# ---------------------------------------------------------------------------
# Bypass-mode entrypoint (mirrors verl.trainer.ppo.rollout_corr_helper.apply_bypass_mode)
# ---------------------------------------------------------------------------


def apply_bypass_mode_to_diffusion_batch(batch: DataProto, rollout_corr_config) -> None:
    """Set up bypass mode for diffusion: ``old_log_probs := rollout_log_probs``.

    Mirrors :func:`verl.trainer.ppo.rollout_corr_helper.apply_bypass_mode`.

    Verl additionally mutates ``policy_loss_config.loss_mode`` to ``"bypass_mode"``
    to dispatch a dedicated loss function.  verl-omni achieves the same semantics
    without a dedicated loss function because:

    * The PPO ratio ``exp(current − rollout)`` already provides IS correction
      (matching verl's ``rollout_is_weights=None`` for ``ppo_clip``).
    * RS rejection is applied per-step via
      :func:`compute_per_step_rollout_correction` inside the engine.
    * The config is forwarded to the engine via non-tensor batch metadata
      (a common verl pattern).
    """
    if "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "rollout_correction.bypass_mode=True requires `rollout_log_probs` in the batch. "
            "Ensure the rollout backend records log probs (calculate_log_probs=true)."
        )
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]


# ---------------------------------------------------------------------------
# Per-step engine correction (bypass mode)
# ---------------------------------------------------------------------------


def compute_per_step_rollout_correction(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
    rollout_corr_config,
) -> tuple[Optional[torch.Tensor], dict[str, float]]:
    """Per-step (micro-batch slice) IS / RS computation for bypass mode.

    This is the diffusion analogue of verl's ``compute_policy_loss_bypass_mode``
    (the registered ``"bypass_mode"`` loss).  It is called inside the FSDP engine's
    ``forward_step`` for each micro-batch / SDE step using the *current* policy
    log-prob vs the recorded rollout log-prob — the only non-trivial ratio under
    bypass.

    Args:
        log_prob: Current policy log-prob, shape ``(B,)`` (one SDE step).
        rollout_log_prob: Rollout log-prob, shape ``(B,)``.
        rollout_corr_config: Dict-like config with keys matching
            ``RolloutCorrectionConfig``.

    Returns:
        ``(weights, metrics)`` tuple:

        * **weights** — ``None`` when neither IS nor RS is configured, or a
          ``(B,)`` tensor:

          - ``loss_type=ppo_clip``: RS keep-mask only (1=keep, 0=reject).
            IS weights are in *metrics* but **not** applied to the loss.
          - ``loss_type=reinforce`` (reserved): IS weights × RS mask, matching
            verl's ``compute_policy_loss_reinforce``.
        * **metrics** — dict of ``rollout_corr/*`` scalars.
    """
    if not rollout_correction_enabled(rollout_corr_config):
        return None, {}

    if log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            f"log_prob and rollout_log_prob must have identical shapes; "
            f"got {tuple(log_prob.shape)} vs {tuple(rollout_log_prob.shape)}."
        )

    # Reshape (B,) → (B, 1) so the verl helper's masked-mean / per-token
    # semantics work on our dense 1-token-per-step representation.
    log_prob_2d = log_prob.unsqueeze(-1)
    rollout_log_prob_2d = rollout_log_prob.unsqueeze(-1)
    response_mask = torch.ones_like(log_prob_2d)

    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    is_weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=log_prob_2d,
        rollout_log_prob=rollout_log_prob_2d,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )

    # Build the per-element multiplier that will be passed to the loss function.
    # Behaviour mirrors verl's compute_policy_loss_bypass_mode:
    #   ppo_clip   → IS weights are NOT applied (rollout_is_weights=None in verl)
    #   reinforce  → IS weights ARE applied
    weights: Optional[torch.Tensor] = None
    ppo_clip = _is_ppo_clip(rollout_corr_config)

    # IS weights (only forwarded for reinforce; always present in metrics)
    is_w: Optional[torch.Tensor] = None
    if is_weights_proto is not None:
        is_w = is_weights_proto.batch["rollout_is_weights"]  # (B, 1)
        if not ppo_clip:
            weights = is_w

    # RS rejection mask (always forwarded when configured)
    if rollout_rs:
        rs_mask = modified_mask  # (B, 1), 1=keep, 0=reject
        weights = rs_mask if weights is None else weights * rs_mask

    if weights is not None:
        weights = weights.squeeze(-1).to(dtype=log_prob.dtype)

    return weights, metrics


# ---------------------------------------------------------------------------
# Decoupled-mode batch correction (mirrors verl.trainer.ppo.rollout_corr_helper
#   .compute_rollout_correction_and_add_to_batch)
# ---------------------------------------------------------------------------


def apply_rollout_correction_to_diffusion_batch(
    batch: DataProto,
    rollout_corr_config,
) -> tuple[DataProto, dict[str, float]]:
    """Compute IS weights / rejection mask for a *decoupled* diffusion batch.

    Called once per global batch when ``bypass_mode=False``.  Uses the recomputed
    ``old_log_probs`` vs the recorded ``rollout_log_probs`` to compute:

    * IS weights (``old / rollout`` ratio, TIS/IcePop truncated)
    * RS keep-mask (``1`` = keep, ``0`` = reject)

    Both are folded into a single ``rollout_is_weights`` tensor on the batch so
    the loss path only needs to handle one optional tensor.

    Mirrors :func:`verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_and_add_to_batch`
    with two intentional differences:

    1. No ``response_mask`` modification — diffusion has no padding, so RS
       rejection is expressed as a 0-weight instead of a mask mutation.
    2. Combined IS+RS tensor — the verl LLM helper adds separate
       ``rollout_is_weights`` and ``modified_response_mask`` to the batch;
       we merge them because the diffusion loss functions already accept a
       single ``rollout_is_weights`` multiplier.

    Returns the (mutated) batch plus a metrics dict.
    """
    _logger.warning(
        "[verl-omni] Rollout Correction for diffusion is an EXPERIMENTAL feature. "
        "See docs/algo/rollout_correction.md for usage, supported presets, and caveats. "
        "Treat reported metrics as diagnostics and tune thresholds for your model / "
        "rollout precision before relying on it in production."
    )

    if "old_log_probs" not in batch.batch or "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "Rollout Correction requires both 'old_log_probs' and 'rollout_log_probs' in the "
            "batch. Ensure the rollout backend records log probs (calculate_log_probs=true) "
            "and that the trainer runs the old_log_prob recompute step."
        )

    old_log_prob: torch.Tensor = batch.batch["old_log_probs"]
    rollout_log_prob: torch.Tensor = batch.batch["rollout_log_probs"]

    if old_log_prob.shape != rollout_log_prob.shape:
        raise ValueError(
            "old_log_probs and rollout_log_probs must have identical shapes; "
            f"got {tuple(old_log_prob.shape)} vs {tuple(rollout_log_prob.shape)}."
        )
    if old_log_prob.dim() != 2:
        raise ValueError(
            "Rollout Correction expects 2D log-prob tensors of shape (batch, sde_window_size); "
            f"got shape {tuple(old_log_prob.shape)}."
        )

    # Diffusion log-probs are dense (no padding) — response_mask is all-ones.
    response_mask = torch.ones_like(old_log_prob)

    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)

    is_weights_proto: Optional[DataProto]
    modified_mask: torch.Tensor
    metrics: dict[str, float]
    is_weights_proto, modified_mask, metrics = compute_rollout_correction_and_rejection_mask(
        old_log_prob=old_log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
    )

    # Fold IS weights and RS mask into a single per-element multiplier.
    effective_weights: Optional[torch.Tensor] = None
    if is_weights_proto is not None:
        effective_weights = is_weights_proto.batch["rollout_is_weights"]
    if rollout_rs:
        rs_keep_mask = modified_mask  # 1 = keep, 0 = reject
        effective_weights = rs_keep_mask if effective_weights is None else effective_weights * rs_keep_mask
    if effective_weights is not None:
        batch.batch["rollout_is_weights"] = effective_weights.to(dtype=old_log_prob.dtype)

    return batch, metrics


# ---------------------------------------------------------------------------
# Off-policy diagnostics (mirrors verl.trainer.ppo.rollout_corr_helper
#   .compute_rollout_corr_metrics_from_logprobs)
# ---------------------------------------------------------------------------


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
) -> dict[str, float]:
    """Compute rollout correction diagnostics from (current, rollout) log probs.

    Mirrors :func:`verl.trainer.ppo.rollout_corr_helper.compute_rollout_corr_metrics_from_logprobs`.
    Called during training to track the off-policy gap as the current policy
    evolves away from the rollout policy.

    Unlike the verl LLM version, diffusion has no ``response_mask`` — all SDE
    steps are valid (no padding).

    Args:
        log_prob: Current policy log-prob, shape ``(B,)`` or ``(B, T)``.
        rollout_log_prob: Rollout policy log-prob, same shape.

    Returns:
        Dict of metrics with ``rollout_corr/`` prefix (KL, PPL, χ², etc.).
    """
    if log_prob.dim() == 1:
        log_prob = log_prob.unsqueeze(-1)
        rollout_log_prob = rollout_log_prob.unsqueeze(-1)

    response_mask = torch.ones_like(log_prob)
    offpolicy_metrics = compute_offpolicy_metrics(
        old_log_prob=log_prob,
        rollout_log_prob=rollout_log_prob,
        response_mask=response_mask,
    )

    metrics_with_prefix: dict[str, float] = {}
    for key, value in offpolicy_metrics.items():
        if isinstance(value, torch.Tensor):
            metrics_with_prefix[f"rollout_corr/{key}"] = value.item()
        else:
            metrics_with_prefix[f"rollout_corr/{key}"] = value

    return metrics_with_prefix
