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

import torch


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
    """Aggregate metric means overall and optionally by one attribute.

    Each update consumes metrics that are already averaged over a batch and a
    weight equal to the number of logical samples represented by that batch.
    When ``group_attribute`` is set, callers pass the group value through
    ``attributes`` and the aggregator emits both overall and per-group means.
    When ``group_attribute`` is ``None``, only the overall means are emitted.
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
