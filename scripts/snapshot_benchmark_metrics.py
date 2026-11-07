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

"""Paired metric aggregation for the LoRA proximal benchmark."""

import math
import random
import statistics

_ARMS = ("full", "trainable")


def arm_order(pair_index: int) -> tuple[str, str]:
    """Return the ABBA arm order assigned to a non-negative pair index."""
    if not isinstance(pair_index, int) or isinstance(pair_index, bool) or pair_index < 0:
        raise ValueError("pair_index must be a non-negative integer")
    return _ARMS if pair_index % 2 == 0 else _ARMS[::-1]


def _percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _positive_count(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def summarize_pairs(
    rows: list[dict],
    world_size: int,
    repeats: int,
    bootstrap_samples: int = 2000,
    seed: int = 20260911,
) -> dict:
    """Summarize complete paired repeats using each arm's slowest rank time."""
    _positive_count(world_size, "world_size")
    _positive_count(repeats, "repeats")
    _positive_count(bootstrap_samples, "bootstrap_samples")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    if not isinstance(rows, list):
        raise ValueError("rows must be a list")

    samples: dict[tuple[int, str, int], float] = {}
    required = {"pair", "arm", "rank", "cycle_seconds"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("each row must contain pair, arm, rank, and cycle_seconds")
        pair, arm, rank, seconds = (
            row["pair"],
            row["arm"],
            row["rank"],
            row["cycle_seconds"],
        )
        if not isinstance(pair, int) or isinstance(pair, bool) or not 0 <= pair < repeats:
            raise ValueError("pair is out of range")
        if arm not in _ARMS:
            raise ValueError("arm must be full or trainable")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("rank is out of range")
        if (
            not isinstance(seconds, int | float)
            or isinstance(seconds, bool)
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            raise ValueError("cycle_seconds must be finite and positive")
        identity = (pair, arm, rank)
        if identity in samples:
            raise ValueError("duplicate pair, arm, and rank identity")
        samples[identity] = float(seconds)

    expected = {(pair, arm, rank) for pair in range(repeats) for arm in _ARMS for rank in range(world_size)}
    if set(samples) != expected:
        raise ValueError("rows must cover every pair, arm, and rank exactly once")

    full = [max(samples[pair, "full", rank] for rank in range(world_size)) for pair in range(repeats)]
    trainable = [max(samples[pair, "trainable", rank] for rank in range(world_size)) for pair in range(repeats)]
    saved = [full_time - trainable_time for full_time, trainable_time in zip(full, trainable, strict=True)]
    reductions = [100 * saved_time / full_time for saved_time, full_time in zip(saved, full, strict=True)]
    paired_observations = [
        {
            "pair": pair,
            "full_slowest_rank_seconds": full[pair],
            "trainable_slowest_rank_seconds": trainable[pair],
            "saved_seconds": saved[pair],
            "reduction_percent": reductions[pair],
        }
        for pair in range(repeats)
    ]
    generator = random.Random(seed)
    bootstrap_medians = [
        statistics.median(reductions[generator.randrange(repeats)] for _ in range(repeats))
        for _ in range(bootstrap_samples)
    ]
    return {
        "n_pairs": repeats,
        "paired_slowest_rank_observations": paired_observations,
        "full_median_seconds": statistics.median(full),
        "trainable_median_seconds": statistics.median(trainable),
        "median_paired_saved_seconds": statistics.median(saved),
        "median_paired_reduction_percent": statistics.median(reductions),
        "reduction_percent_bootstrap_ci95": [
            _percentile(bootstrap_medians, 2.5),
            _percentile(bootstrap_medians, 97.5),
        ],
        "uncertainty_scope": "within_allocation_paired_repeats_not_independent_runs",
    }
