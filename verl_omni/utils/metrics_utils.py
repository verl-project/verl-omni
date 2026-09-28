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
"""Metric aggregation helpers shared across verl-omni."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

__all__ = [
    "AgenticRewardMetrics",
    "GroupedMetricMean",
    "judge_parse_ok_rate",
]


def judge_parse_ok_rate(n_ok: int, n_fail: int) -> float | None:
    """Parse-health rate for one rollout, or ``None`` when no judge was attempted.

    Single source of truth for the derived metric. It is computed here rather
    than in ``compute_score`` because the trainer reduces every reward-extra key
    with ``np.mean`` (``verl.trainer.ppo.metric_utils.process_validation_metrics``),
    which raises ``TypeError`` on ``None`` — so the honest "absent measurement"
    value can only live on the per-row dump path, never in the reward dict.

    Args:
        n_ok: Judge observations that parsed to a real ``judge_image`` call.
        n_fail: Judge observations that failed to parse.

    Returns:
        ``n_ok / (n_ok + n_fail)``, or ``None`` when ``n_ok + n_fail == 0``.
    """
    attempts = int(n_ok) + int(n_fail)
    if attempts <= 0:
        return None
    return float(n_ok) / float(attempts)


class _MetricMeanStats:
    """Accumulate batch-mean metrics weighted by sample count."""

    def __init__(self) -> None:
        self.total = 0
        self.sums: dict[str, float] = defaultdict(float)

    def update(self, metrics: dict[str, Any], *, weight: int) -> None:
        if weight <= 0:
            return
        self.total += weight
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().cpu().item()
            elif hasattr(value, "item"):
                value = value.item()
            self.sums[key] += float(value) * weight

    def to_prefixed_dict(self, prefix: str, metric_keys: tuple[str, ...]) -> dict[str, float | int]:
        result: dict[str, float | int] = {f"{prefix}/num_samples": self.total}
        for key in metric_keys:
            if key in self.sums:
                result[f"{prefix}/{key}"] = self.sums[key] / self.total if self.total else 0.0
        return result


class GroupedMetricMean:
    """Accumulate weighted metric means overall and optionally by group.

    What is class:
        Aggregates metrics that are already averaged over each batch, weighting
        them by the number of logical samples represented by that batch. When
        ``group_attribute`` is set, the class also tracks per-group means using
        values supplied through ``attributes`` in ``update``.

    Args:
        metric_keys: Metric names to include in emitted summaries.
        group_attribute: Optional attribute name used to split metrics into
            per-group summaries. If ``None``, only overall metrics are emitted.

    Returns:
        A ``GroupedMetricMean`` instance that can be updated with batch metrics
        and converted to a prefixed metrics dictionary.
    """

    def __init__(self, *, metric_keys: tuple[str, ...], group_attribute: str | None = None) -> None:
        self.metric_keys = metric_keys
        self.group_attribute = group_attribute
        self.overall = _MetricMeanStats()
        self.by_group: dict[str, _MetricMeanStats] = defaultdict(_MetricMeanStats)

    def update(
        self,
        metrics: dict[str, Any],
        *,
        weight: int,
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self.overall.update(metrics, weight=weight)
        if self.group_attribute is None:
            return
        attributes = attributes or {}
        if self.group_attribute not in attributes:
            raise KeyError(f"Missing grouping attribute {self.group_attribute!r}.")
        group_value = str(attributes[self.group_attribute])
        self.by_group[group_value].update(metrics, weight=weight)

    def to_prefixed_dict(self, prefix: str) -> dict[str, float | int]:
        metrics = self.overall.to_prefixed_dict(prefix, self.metric_keys)
        if self.group_attribute is None:
            return metrics
        for group_value, stats in sorted(self.by_group.items()):
            metrics.update(stats.to_prefixed_dict(f"{prefix}/{group_value}", self.metric_keys))
        return metrics


class AgenticRewardMetrics:
    """Read-only views of agentic reward extras on a rollout batch.

    ``MIX_KEYS`` feed ``agentic_reward/<name>/{mean,min,max}``. ``ROLLOUT_KEYS``
    are available at ``generate_sequences``. ``ARTIFACT_KEYS`` copy into dump rows.
    """

    MIX_KEYS: tuple[str, ...] = (
        "reward_tool_call",
        "reward_correctness",
        "reward_aesthetics",
        "reward_done",
    )
    # Available at generate_sequences time (before the reward manager).
    ROLLOUT_KEYS: tuple[str, ...] = (
        "num_generate_image_prompts",
        "rollout_has_generate",
        "rollout_valid",
        "forced_reflection",
        "forced_first_generate",
        "forced_first_judge",
    )
    ARTIFACT_KEYS: tuple[str, ...] = (
        "reward_tool_call",
        "reward_correctness",
        "reward_aesthetics",
        "reward_done",
        "num_hermes_tool_calls",
        "num_generate_image_prompts",
        "num_judge_image_calls",
        "judge_parse_ok",
        "judge_parse_fail",
        "protocol_ok",
        "rewrite_after_yes",
        "reward_delta_c",
        "reward_rewrite_yes",
        "first_correctness",
        "first_judge_no",
        "rollout_valid",
    )
    INTEGER_KEYS: frozenset[str] = frozenset(
        {
            "num_hermes_tool_calls",
            "num_generate_image_prompts",
            "num_judge_image_calls",
            "protocol_ok",
            "rollout_valid",
        }
    )

    @classmethod
    def aggregate(cls, non_tensor_batch: dict[str, Any]) -> dict[str, float]:
        """Batch mean/min/max for mix terms and rollout-time counters.

        Args:
            non_tensor_batch: ``DataProto.non_tensor_batch`` mapping.

        Returns:
            Flat dict of ``agentic_reward/*`` and ``agentic_rollout/*/mean`` keys.
        """
        metrics: dict[str, float] = {}
        for key in cls.MIX_KEYS:
            if key not in non_tensor_batch:
                continue
            values = np.asarray(non_tensor_batch[key], dtype=np.float64)
            if values.size == 0:
                continue
            prefix = f"agentic_reward/{key.removeprefix('reward_')}"
            metrics[f"{prefix}/mean"] = float(np.mean(values))
            metrics[f"{prefix}/min"] = float(np.min(values))
            metrics[f"{prefix}/max"] = float(np.max(values))
        for key in cls.ROLLOUT_KEYS:
            if key not in non_tensor_batch:
                continue
            try:
                values = np.asarray(non_tensor_batch[key], dtype=np.float64)
            except (TypeError, ValueError):
                continue
            if values.size == 0:
                continue
            metrics[f"agentic_rollout/{key}/mean"] = float(np.mean(values))
        # Derived parse health, pooled over rows that actually attempted a judge.
        # Zero-attempt rows contribute nothing (they are not "0% healthy" either),
        # and no None ever enters the pinned np.mean reduction.
        if "judge_parse_ok" in non_tensor_batch and "judge_parse_fail" in non_tensor_batch:
            try:
                ok = np.asarray(non_tensor_batch["judge_parse_ok"], dtype=np.float64)
                fail = np.asarray(non_tensor_batch["judge_parse_fail"], dtype=np.float64)
            except (TypeError, ValueError):
                ok = fail = np.zeros(0, dtype=np.float64)
            attempts = float(ok.sum() + fail.sum())
            if attempts > 0:
                metrics["agentic_rollout/judge_parse_ok_rate/mean"] = float(ok.sum()) / attempts
        return metrics

    @classmethod
    def for_rollout(cls, output: Any, index: int) -> dict[str, float | int | None]:
        """Per-row scorer outputs for one ``hermes_actions`` JSONL record.

        Args:
            output: Rollout ``DataProto``.
            index: Row index in the batch.

        Returns:
            Compact dict of score and ``ARTIFACT_KEYS`` present on that row, plus
            the derived ``judge_parse_ok_rate``. A derived key is ``None`` when
            the row carries no measurement for it (zero judge attempts), which
            JSONL encodes as ``null``.
        """
        if not isinstance(index, int) or index < 0:
            raise IndexError(f"rollout index must be a non-negative int, got {index!r}")
        metrics: dict[str, float | int | None] = {}
        batch = getattr(output, "batch", None)
        rm_scores = batch.get("rm_scores") if batch is not None else None
        if rm_scores is not None:
            # AgentLoopManager writes the scalar reward on the final valid response
            # token; the first token is normally zero. Sum the token-level tensor.
            row = cls._row(rm_scores, index)
            if row is not None:
                metrics["score"] = float(np.asarray(row.detach().cpu()).sum())

        non_tensor_batch = getattr(output, "non_tensor_batch", None) or {}
        for key in cls.ARTIFACT_KEYS:
            values = non_tensor_batch.get(key)
            if values is None:
                continue
            row = cls._row(values, index)
            if row is None:
                continue
            value = np.asarray(row).reshape(-1)[0]
            if value is None:
                # Defensive: a None row value stays None so JSONL dumps null
                # rather than raising in float().
                metrics[key] = None
            elif key in cls.INTEGER_KEYS:
                metrics[key] = int(value)
            else:
                metrics[key] = float(value)
        # Derived, not stored on the row: counts are the SoT (see
        # ``judge_parse_ok_rate`` for why this must not live in the reward dict).
        if "judge_parse_ok" in metrics and "judge_parse_fail" in metrics:
            metrics["judge_parse_ok_rate"] = judge_parse_ok_rate(metrics["judge_parse_ok"], metrics["judge_parse_fail"])
        return metrics

    @staticmethod
    def _row(values: Any, index: int) -> Any | None:
        try:
            length = len(values)
        except TypeError:
            return None
        if index >= length:
            return None
        return values[index]
