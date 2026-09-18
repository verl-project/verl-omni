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
"""CPU tests for trainer metric aggregation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "verl_omni" / "utils" / "metrics_utils.py"


def _load_metrics_utils():
    spec = importlib.util.spec_from_file_location("verl_omni.utils.metrics_utils", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics utils from {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


metrics_utils = _load_metrics_utils()


def test_grouped_metric_mean_without_attribute_returns_overall_weighted_mean():
    aggregator = metrics_utils.GroupedMetricMean(
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
    aggregator = metrics_utils.GroupedMetricMean(
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
    aggregator = metrics_utils.GroupedMetricMean(metric_keys=("loss",), group_attribute="modality")

    with pytest.raises(KeyError, match="Missing grouping attribute"):
        aggregator.update({"loss": 1.0}, weight=1)
