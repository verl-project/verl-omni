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
"""Step-metric plumbing for ``bagel_corl_sync``: allowlist, EMA trend, lane renames, divergence tripwire."""

from __future__ import annotations

import math

#: ``extra`` -> step-log allowlist for the dual-lane bookkeeping.
STEP_METRIC_KEYS: tuple[str, ...] = (
    "und/no_image_credit",
    "gen/dropped_incomplete_groups",
    "episode/J",
    "episode/K",
    "episode/K_zero",
)

#: Prefix form of the allowlist for data-dependent key names (one key per reward dim present).
STEP_METRIC_PREFIXES: tuple[str, ...] = (
    "bagel_corl/reward_dim/",
    "bagel_corl/reward_valid/",
    "bagel_corl/reward_no_image/",
)

#: Keys whose bias-corrected EMA is published as ``bagel_corl/ema/<key>``.
METRIC_EMA_KEYS: tuple[str, ...] = (
    "critic/score/mean",
    "bagel_corl/reward_dim/image/mean",
    "episode/J",
    "episode/K_zero",
)
METRIC_EMA_BETA = 0.95

#: Runaway-policy thresholds: an order of magnitude above a healthy run, below the readings
#: that made the ``_084724`` collapse unrecoverable.
DIVERGENCE_KL_LIMIT = 20.0
DIVERGENCE_GRAD_NORM_LIMIT = 100.0
DIVERGENCE_MESSAGE = "actor/rollout mismatch or gradient health threshold exceeded"

#: Validation keys that make up the RFC §8.2 dual-lane evidence record.
RFC82_EVIDENCE_MARKERS = ("episode/J", "episode/K", "pattern", "no_image_credit", "skipped_no_groups")


def metric_scalar(metrics: dict, key: str) -> float | None:
    """Read ``metrics[key]`` as a float (plain, scalar tensor or ``Metric``), or ``None``."""
    value = metrics.get(key)
    if value is None:
        return None
    aggregate = getattr(value, "aggregate", None)
    if callable(aggregate):
        try:
            value = aggregate()
        except (TypeError, ValueError):
            return None
    if hasattr(value, "numel") and hasattr(value, "item"):
        try:
            if value.numel() != 1:
                return None
            value = value.item()
        except (RuntimeError, TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def step_metric_keys(extra: dict) -> list[str]:
    """Step-log keys ``extra`` is currently carrying, in a stable order (absent stays absent)."""
    keys = [key for key in STEP_METRIC_KEYS if extra.get(key) is not None]
    keys.extend(
        sorted(key for key, value in extra.items() if value is not None and key.startswith(STEP_METRIC_PREFIXES))
    )
    return keys


def update_metric_ema(state: dict, metrics: dict, *, keys=METRIC_EMA_KEYS, beta: float = METRIC_EMA_BETA) -> None:
    """Publish a bias-corrected EMA of each key in ``keys``; ``state`` persists ``(ema, n)`` per key."""
    for key in keys:
        value = metric_scalar(metrics, key)
        if value is None or not math.isfinite(value):
            continue
        ema, n = state.get(key, (0.0, 0))
        ema = beta * ema + (1.0 - beta) * value
        n += 1
        state[key] = (ema, n)
        metrics[f"bagel_corl/ema/{key}"] = ema / (1.0 - beta**n)


def publish_lane_metrics(metrics: dict) -> None:
    """Mirror ``actor/und/*`` and ``actor/gen/*`` under their lane names and derive the GEN ratio std."""
    for key, value in list(metrics.items()):
        if key.startswith(("actor/und/", "actor/gen/")):
            metrics[key.removeprefix("actor/")] = value
    if "rollout_corr/kl" in metrics:
        metrics["und/rollout_actor_logp_gap"] = metrics["rollout_corr/kl"]
    if "actor/entropy" in metrics:
        metrics["und/entropy"] = metrics["actor/entropy"]
    if "gen/ratio_square_mean" in metrics and "gen/ratio_mean" in metrics:
        metrics["gen/ratio_std"] = math.sqrt(
            max(0.0, metrics["gen/ratio_square_mean"] - metrics["gen/ratio_mean"] ** 2)
        )


def _runaway(value: float | None, limit: float) -> bool:
    """A non-finite reading is the loudest divergence signal; ``nan > limit`` alone would miss it."""
    return value is not None and (not math.isfinite(value) or value > limit)


def divergence_triggers(metrics: dict, *, kl_limit: float, grad_norm_limit: float) -> list[str]:
    """Human-readable triggers for the provisional rollout/actor mismatch diagnostics."""
    kl = metric_scalar(metrics, "rollout_corr/kl")
    grad_norm = metric_scalar(metrics, "actor/grad_norm")
    triggers = []
    if _runaway(kl, kl_limit):
        triggers.append(f"rollout_corr/kl={kl:.4g} > {kl_limit:g}")
    if _runaway(grad_norm, grad_norm_limit):
        triggers.append(f"actor/grad_norm={grad_norm:.4g} > {grad_norm_limit:g}")
    return triggers


def divergence_message(step, metrics: dict, triggers: list[str]) -> str:
    """The warning/abort text for one divergence excursion."""
    entropy = metric_scalar(metrics, "actor/entropy")
    return (
        f"bagel_corl_sync step={step}: {DIVERGENCE_MESSAGE} ({'; '.join(triggers)}"
        + (f"; actor/entropy={entropy:.4g}" if entropy is not None else "")
        + "). Check serving synchronization, probability conventions, and update scale; "
        "check NORM_ADV_BY_STD / loss_weight_* / optim.lr, and inspect checkpoint health before resuming."
    )


def rfc82_evidence(val_metrics: dict | None) -> dict[str, float]:
    """The RFC §8.2 dual-lane evidence subset of a validation metrics dict."""
    return {
        key: float(value)
        for key, value in (val_metrics or {}).items()
        if isinstance(value, int | float)
        and any(marker in (str(key).lower() if marker == "pattern" else str(key)) for marker in RFC82_EVIDENCE_MARKERS)
    }
