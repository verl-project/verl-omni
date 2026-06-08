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
"""Metrics for train–rollout log-probability consistency checks."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from verl_omni.trainer.diffusion import diffusion_algos
from verl_omni.workers.config.diffusion.actor import FSDPDiffusionActorConfig

logger = logging.getLogger(__name__)

_DEFAULT_LOGPROB_TOL = 0.01


@dataclass
class TrainRolloutConsistencyMetrics:
    """Aggregated consistency metrics between rollout and train log-probs."""

    logprob_abs_max: float
    logprob_abs_mean: float
    logprob_rmse: float
    logprob_rel_err_p99: float
    frac_steps_over_tol: float
    ratio_mean: float
    ratio_std: float
    pg_clipfrac: float
    pg_clipfrac_higher: float
    pg_clipfrac_lower: float
    ppo_kl: float
    prev_sample_mean_abs_max: Optional[float] = None
    std_dev_t_abs_max: Optional[float] = None
    num_steps: int = 0
    logprob_tol: float = _DEFAULT_LOGPROB_TOL
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _flatten_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    tensor = log_probs.detach().float().reshape(-1)
    return tensor


def compute_train_rollout_consistency_metrics(
    rollout_log_probs: torch.Tensor,
    train_log_probs: torch.Tensor,
    *,
    actor_config: Optional[FSDPDiffusionActorConfig] = None,
    clip_ratio: float = 0.0001,
    logprob_tol: float = _DEFAULT_LOGPROB_TOL,
    prev_sample_mean_rollout: Optional[torch.Tensor] = None,
    prev_sample_mean_train: Optional[torch.Tensor] = None,
    std_dev_t_rollout: Optional[torch.Tensor] = None,
    std_dev_t_train: Optional[torch.Tensor] = None,
    extra: Optional[dict[str, Any]] = None,
) -> TrainRolloutConsistencyMetrics:
    """Compare rollout vs train log-probs and derive FlowGRPO ratio metrics.

    When ``actor_config`` is provided, clip-fraction metrics are computed with
    the registered ``flow_grpo`` loss helper and unit advantages.
    Otherwise ``clip_ratio`` is used with a lightweight inline computation.
    """
    rollout_flat = _flatten_log_probs(rollout_log_probs)
    train_flat = _flatten_log_probs(train_log_probs)
    if rollout_flat.shape != train_flat.shape:
        raise ValueError(
            f"rollout/train log-prob shapes must match, got {tuple(rollout_flat.shape)} vs {tuple(train_flat.shape)}"
        )

    diff = train_flat - rollout_flat
    abs_diff = diff.abs()
    rel_err = abs_diff / (rollout_flat.abs() + 1e-8)

    advantages = torch.ones_like(rollout_flat)
    if actor_config is not None:
        flow_grpo_loss = diffusion_algos.get_diffusion_loss_fn("flow_grpo")
        _, pg_metrics = flow_grpo_loss.compute_loss(
            old_log_prob=rollout_flat,
            log_prob=train_flat,
            advantages=advantages,
            config=actor_config,
        )
    else:
        log_ratio = train_flat - rollout_flat
        ratio = torch.exp(log_ratio)
        pg_metrics = {
            "actor/ppo_kl": torch.mean(-log_ratio).item(),
            "actor/pg_clipfrac": torch.mean((torch.abs(ratio - 1.0) > clip_ratio).float()).item(),
            "actor/pg_clipfrac_higher": torch.mean((ratio - 1.0 > clip_ratio).float()).item(),
            "actor/pg_clipfrac_lower": torch.mean((1.0 - ratio > clip_ratio).float()).item(),
            "actor/ratio_mean": ratio.mean().item(),
            "actor/ratio_std": ratio.std(unbiased=False).item(),
        }

    prev_sample_mean_abs_max = None
    if prev_sample_mean_rollout is not None and prev_sample_mean_train is not None:
        prev_sample_mean_abs_max = (
            (prev_sample_mean_train.detach().float() - prev_sample_mean_rollout.detach().float()).abs().max().item()
        )

    std_dev_t_abs_max = None
    if std_dev_t_rollout is not None and std_dev_t_train is not None:
        std_dev_t_abs_max = (std_dev_t_train.detach().float() - std_dev_t_rollout.detach().float()).abs().max().item()

    return TrainRolloutConsistencyMetrics(
        logprob_abs_max=abs_diff.max().item(),
        logprob_abs_mean=abs_diff.mean().item(),
        logprob_rmse=torch.sqrt(torch.mean(diff**2)).item(),
        logprob_rel_err_p99=torch.quantile(rel_err, 0.99).item(),
        frac_steps_over_tol=(abs_diff > logprob_tol).float().mean().item(),
        ratio_mean=pg_metrics["actor/ratio_mean"],
        ratio_std=pg_metrics["actor/ratio_std"],
        pg_clipfrac=pg_metrics["actor/pg_clipfrac"],
        pg_clipfrac_higher=pg_metrics["actor/pg_clipfrac_higher"],
        pg_clipfrac_lower=pg_metrics["actor/pg_clipfrac_lower"],
        ppo_kl=pg_metrics["actor/ppo_kl"],
        prev_sample_mean_abs_max=prev_sample_mean_abs_max,
        std_dev_t_abs_max=std_dev_t_abs_max,
        num_steps=int(rollout_flat.numel()),
        logprob_tol=logprob_tol,
        extra=extra or {},
    )


def format_metrics_report(
    metrics: TrainRolloutConsistencyMetrics,
    *,
    label: str = "train_rollout_consistency",
) -> str:
    """Return a single-line human-readable metrics summary."""
    parts = [
        f"logprob_abs_max={metrics.logprob_abs_max:.6f}",
        f"logprob_abs_mean={metrics.logprob_abs_mean:.6f}",
        f"logprob_rmse={metrics.logprob_rmse:.6f}",
        f"logprob_rel_err_p99={metrics.logprob_rel_err_p99:.6f}",
        f"frac_steps_over_tol={metrics.frac_steps_over_tol:.6f}",
        f"ratio_mean={metrics.ratio_mean:.6f}",
        f"ratio_std={metrics.ratio_std:.6f}",
        f"pg_clipfrac={metrics.pg_clipfrac:.6f}",
        f"pg_clipfrac_higher={metrics.pg_clipfrac_higher:.6f}",
        f"pg_clipfrac_lower={metrics.pg_clipfrac_lower:.6f}",
        f"ppo_kl={metrics.ppo_kl:.6f}",
        f"num_steps={metrics.num_steps}",
        f"logprob_tol={metrics.logprob_tol:.6f}",
    ]
    if metrics.prev_sample_mean_abs_max is not None:
        parts.append(f"prev_sample_mean_abs_max={metrics.prev_sample_mean_abs_max:.6f}")
    if metrics.std_dev_t_abs_max is not None:
        parts.append(f"std_dev_t_abs_max={metrics.std_dev_t_abs_max:.6f}")
    return f"[{label}] " + "  ".join(parts)


def write_metrics_json(
    path: str | Path,
    metrics: TrainRolloutConsistencyMetrics,
    *,
    metadata: Optional[dict[str, Any]] = None,
) -> Path:
    """Write metrics and optional metadata to JSON for CI artifacts."""
    output_path = Path(path)
    payload = {
        "metadata": metadata or {},
        "metrics": metrics.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("Wrote train–rollout consistency metrics to %s", output_path)
    return output_path


def log_metrics(
    metrics: TrainRolloutConsistencyMetrics,
    *,
    label: str = "train_rollout_consistency",
    json_path: str | Path | None = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log metrics to the module logger and optionally persist JSON."""
    report = format_metrics_report(metrics, label=label)
    print(report)
    logger.info(report)
    if json_path is not None:
        write_metrics_json(json_path, metrics, metadata=metadata)
