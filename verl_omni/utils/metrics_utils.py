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
]


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

    ``MIX_KEYS`` are the scalar terms mixed into ``compute_score`` (logged as
    ``agentic_reward/<name>/{mean,min,max}``). ``ARTIFACT_KEYS`` are copied
    into compact ``hermes_actions`` JSONL rows; ``INTEGER_KEYS`` stay ints.
    """

    MIX_KEYS: tuple[str, ...] = (
        "reward_tool_call",
        "reward_correctness",
        "reward_aesthetics",
        "reward_done",
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
        "judge_parse_ok_rate",
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
        """Batch mean/min/max for mix terms already returned by the reward manager."""
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
        return metrics

    @classmethod
    def for_rollout(cls, output: Any, index: int) -> dict[str, float | int]:
        """Per-row scorer outputs for one ``hermes_actions`` JSONL record."""
        if not isinstance(index, int) or index < 0:
            raise IndexError(f"rollout index must be a non-negative int, got {index!r}")
        metrics: dict[str, float | int] = {}
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
            metrics[key] = int(value) if key in cls.INTEGER_KEYS else float(value)
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
