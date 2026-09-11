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
"""CPU tests for ``verl_omni.utils.metrics_utils``."""

from __future__ import annotations

import pytest
import torch

from verl_omni.utils.metrics_utils import AgenticRewardMetrics, GroupedMetricMean


def test_grouped_metric_mean_without_attribute_returns_overall_weighted_mean():
    aggregator = GroupedMetricMean(
        metric_keys=("reward_accuracy", "reward_margin"),
        group_attribute=None,
    )

    aggregator.update({"reward_accuracy": torch.tensor(1.0), "reward_margin": 0.5}, weight=1)
    aggregator.update({"reward_accuracy": torch.tensor(0.0), "reward_margin": 1.0}, weight=3)

    assert aggregator.to_prefixed_dict("val") == {
        "val/num_samples": 4,
        "val/reward_accuracy": pytest.approx(0.25),
        "val/reward_margin": pytest.approx(0.875),
    }


def test_grouped_metric_mean_groups_by_attribute_and_keeps_overall():
    aggregator = GroupedMetricMean(
        metric_keys=("reward_accuracy", "reward_margin"),
        group_attribute="modality",
    )

    aggregator.update(
        {"reward_accuracy": torch.tensor(1.0), "reward_margin": 0.5},
        weight=2,
        attributes={"modality": "image"},
    )
    aggregator.update(
        {"reward_accuracy": torch.tensor(0.0), "reward_margin": 1.5},
        weight=1,
        attributes={"modality": "audio"},
    )

    assert aggregator.to_prefixed_dict("val") == {
        "val/num_samples": 3,
        "val/reward_accuracy": pytest.approx(2 / 3),
        "val/reward_margin": pytest.approx(5 / 6),
        "val/audio/num_samples": 1,
        "val/audio/reward_accuracy": pytest.approx(0.0),
        "val/audio/reward_margin": pytest.approx(1.5),
        "val/image/num_samples": 2,
        "val/image/reward_accuracy": pytest.approx(1.0),
        "val/image/reward_margin": pytest.approx(0.5),
    }


def test_grouped_metric_mean_requires_grouping_attribute_when_configured():
    aggregator = GroupedMetricMean(metric_keys=("loss",), group_attribute="modality")

    with pytest.raises(KeyError, match="Missing grouping attribute"):
        aggregator.update({"loss": 1.0}, weight=1)


def test_agentic_reward_metrics_aggregate_mix_keys_only():
    metrics = AgenticRewardMetrics.aggregate(
        {
            "reward_tool_call": torch.tensor([1.0, 1.0]),
            "reward_correctness": torch.tensor([0.8, 0.6]),
            "reward_done": torch.tensor([]),
            "reward_plan": torch.tensor([0.4]),
        }
    )
    assert metrics["agentic_reward/tool_call/mean"] == pytest.approx(1.0)
    assert metrics["agentic_reward/correctness/min"] == pytest.approx(0.6)
    assert "agentic_reward/done/mean" not in metrics
    assert "agentic_reward/plan/mean" not in metrics
