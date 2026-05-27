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

Diffusion-specific notes:
- No ``response_mask`` — log-probs are dense (no padding).  RS rejection is
  expressed as a 0-weight in ``rollout_is_weights`` instead of a mask.
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
    "compute_rollout_corr_metrics_from_logprobs",
    "rollout_correction_enabled",
]

_logger = logging.getLogger(__name__)
_warned_experimental = False

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


# ---------------------------------------------------------------------------
# Bypass mode
# ---------------------------------------------------------------------------


def apply_bypass_mode_to_diffusion_batch(batch: DataProto) -> None:
    """Set ``old_log_probs := rollout_log_probs`` (zero-cost substitution).

    Bypass-mode IS/RS is computed per SDE step inside ``diffusion_loss``,
    which reads ``config.rollout_correction`` from ``DiffusionActorConfig``.
    """
    global _warned_experimental
    if not _warned_experimental:
        _warned_experimental = True
        _logger.warning(
            "[verl-omni] Rollout Correction for diffusion is an EXPERIMENTAL feature. "
            "See docs/algo/rollout_correction.md for usage and caveats."
        )
    if batch.batch is None or "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "rollout_correction.bypass_mode=True requires `rollout_log_probs` in the batch. "
            "Ensure the rollout backend records log probs (calculate_log_probs=true)."
        )
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]


# ---------------------------------------------------------------------------
# Decoupled mode
# ---------------------------------------------------------------------------


def apply_rollout_correction_to_diffusion_batch(
    batch: DataProto,
    rollout_corr_config,
) -> tuple[DataProto, dict[str, float]]:
    """Compute IS weights / RS mask for decoupled mode (``bypass_mode=False``).

    Uses ``old_log_probs`` vs ``rollout_log_probs`` to compute IS weights and
    RS keep-mask, then folds both into a single ``rollout_is_weights`` tensor.
    Called once per global batch; in bypass mode this is skipped (old == rollout).
    """
    global _warned_experimental
    if not _warned_experimental:
        _warned_experimental = True
        _logger.warning(
            "[verl-omni] Rollout Correction for diffusion is an EXPERIMENTAL feature. "
            "See docs/algo/rollout_correction.md for usage and caveats."
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
# Diagnostics
# ---------------------------------------------------------------------------


def compute_rollout_corr_metrics_from_logprobs(
    log_prob: torch.Tensor,
    rollout_log_prob: torch.Tensor,
) -> dict[str, float]:
    """Off-policy diagnostics from (current, rollout) log probs.

    Diffusion has no ``response_mask`` — all SDE steps are valid (no padding).

    Args:
        log_prob: Current policy log-prob, shape ``(B,)`` or ``(B, T)``.
        rollout_log_prob: Rollout policy log-prob, same shape.

    Returns:
        Dict of ``rollout_corr/`` metrics (KL, PPL, χ², etc.).
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
