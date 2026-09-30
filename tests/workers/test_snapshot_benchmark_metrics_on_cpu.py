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

"""CPU tests for paired snapshot benchmark metric aggregation."""

import importlib.util
import math
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "snapshot_benchmark_metrics_under_test", Path(__file__).parents[2] / "scripts/snapshot_benchmark_metrics.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_METRICS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_METRICS)
arm_order = _METRICS.arm_order
summarize_pairs = _METRICS.summarize_pairs


def _rows(pairs: list[tuple[list[float], list[float]]]) -> list[dict]:
    return [
        {"pair": pair, "arm": arm, "rank": rank, "cycle_seconds": seconds}
        for pair, (full, trainable) in enumerate(pairs)
        for arm, values in (("full", full), ("trainable", trainable))
        for rank, seconds in enumerate(values)
    ]


def test_abba_pairing_and_paired_summary() -> None:
    assert arm_order(0) == ("full", "trainable")
    assert arm_order(1) == ("trainable", "full")
    with pytest.raises(ValueError):
        arm_order(-1)
    summary = summarize_pairs(_rows([([10], [8]), ([12], [9])]), world_size=1, repeats=2)
    assert summary["full_median_seconds"] == 11
    assert summary["trainable_median_seconds"] == 8.5
    assert summary["median_paired_saved_seconds"] == 2.5
    assert summary["median_paired_reduction_percent"] == 22.5


def test_uses_slowest_rank_not_rank_average() -> None:
    summary = summarize_pairs(_rows([([4, 10], [6, 7])]), world_size=2, repeats=1)
    assert summary["full_median_seconds"] == 10
    assert summary["trainable_median_seconds"] == 7
    assert summary["median_paired_reduction_percent"] == 30
    assert summary["paired_slowest_rank_observations"] == [
        {
            "pair": 0,
            "full_slowest_rank_seconds": 10.0,
            "trainable_slowest_rank_seconds": 7.0,
            "saved_seconds": 3.0,
            "reduction_percent": 30.0,
        }
    ]


def test_reports_ten_pairs_without_rank_pseudoreplication() -> None:
    pairs = [([float(pair + 1), float(pair + 11)], [float(pair + 2), float(pair + 8)]) for pair in range(10)]
    summary = summarize_pairs(_rows(pairs), world_size=2, repeats=10, bootstrap_samples=20)
    observations = summary["paired_slowest_rank_observations"]
    assert len(observations) == summary["n_pairs"] == 10
    assert [item["pair"] for item in observations] == list(range(10))
    assert [item["full_slowest_rank_seconds"] for item in observations] == [float(pair + 11) for pair in range(10)]
    assert [item["trainable_slowest_rank_seconds"] for item in observations] == [float(pair + 8) for pair in range(10)]


@pytest.mark.parametrize(
    ("rows", "world_size", "repeats"),
    [
        (_rows([([1], [1])])[:-1], 1, 1),
        (_rows([([1], [1])]) + [_rows([([1], [1])])[0]], 1, 1),
        ([{"arm": "full", "rank": 0, "cycle_seconds": 1}], 1, 1),
        ([{"pair": 0, "rank": 0, "cycle_seconds": 1}], 1, 1),
        ([{"pair": 0, "arm": "full", "cycle_seconds": 1}], 1, 1),
        ([{"pair": 1, "arm": "full", "rank": 0, "cycle_seconds": 1}], 1, 1),
        ([{"pair": 0, "arm": "full", "rank": 0, "cycle_seconds": math.nan}], 1, 1),
        ([{"pair": 0, "arm": "full", "rank": 0, "cycle_seconds": 0}], 1, 1),
        (_rows([([1, 2], [1, 2])]), 1, 1),
        (_rows([([1], [1])]), 2, 1),
    ],
)
def test_rejects_incomplete_duplicate_or_invalid_samples(rows: list[dict], world_size: int, repeats: int) -> None:
    with pytest.raises(ValueError):
        summarize_pairs(rows, world_size=world_size, repeats=repeats)


def test_retains_negative_improvement() -> None:
    summary = summarize_pairs(_rows([([5], [6])]), world_size=1, repeats=1)
    assert summary["median_paired_saved_seconds"] == -1
    assert summary["median_paired_reduction_percent"] == -20


def test_bootstrap_interval_is_deterministic() -> None:
    rows = _rows([([10], [8]), ([10], [9]), ([10], [12])])
    first = summarize_pairs(rows, world_size=1, repeats=3, bootstrap_samples=100, seed=7)
    second = summarize_pairs(rows, world_size=1, repeats=3, bootstrap_samples=100, seed=7)
    assert first["reduction_percent_bootstrap_ci95"] == second["reduction_percent_bootstrap_ci95"]
