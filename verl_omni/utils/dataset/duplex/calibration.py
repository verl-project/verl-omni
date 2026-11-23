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
"""Reward-calibration audit for the duplex GRPO entry gate (verl-omni#905 §3.6).

Before any gradient step, the unmodified starting checkpoint is rolled over the
mined event set. Per-axis reward distributions must land in a sane mid band
(target: mean normalized reward ≥ ~0.4 of axis maximum). If an axis floors the
base model, the event construction or reward band is misaligned with the
checkpoint's actual behavior and must be fixed before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from verl_omni.utils.reward_score.duplex.rule_rewards import (
    DuplexTrajectory,
    compute_all_rewards,
    compute_pause_violation,
    compute_interruption_yield,
    compute_turn_latency,
)

__all__ = ["CalibrationReport", "run_calibration_audit"]


@dataclass(frozen=True)
class CalibrationReport:
    """Per-axis calibration summary for one rollout batch."""

    axis: str
    mean_reward: float
    min_reward: float
    max_reward: float
    sample_count: int
    passes: bool


def run_calibration_audit(
    trajectories: Sequence[DuplexTrajectory],
    min_mean_reward: float = -0.6,
) -> list[CalibrationReport]:
    """Check that the base policy's per-axis rewards land in a sane mid band.

    A mean of −1.0 means the base model completely fails the axis (floored);
    0.0 means perfect behavior. We require the mean to be ≥ ``min_mean_reward``
    (e.g. −0.6 means the model gets at least 40% of the way to optimal on
    average, which is the "mid band" from the RFC).

    Parameters:
        trajectories: One DuplexTrajectory per episode from the base rollout.
        min_mean_reward: Threshold below which an axis is considered floored.
    """
    reports: list[CalibrationReport] = []

    axis_fns = {
        "pause_violation": compute_pause_violation,
        "turn_latency": compute_turn_latency,
        "interruption_yield": compute_interruption_yield,
    }

    for axis, fn in axis_fns.items():
        relevant = [t for t in trajectories if t.event_type in _axis_event_types(axis)]
        if not relevant:
            reports.append(CalibrationReport(axis=axis, mean_reward=0.0, min_reward=0.0, max_reward=0.0, sample_count=0, passes=True))
            continue

        rewards = [fn(t).get(axis, 0.0) for t in relevant]
        mean_r = sum(rewards) / len(rewards)
        passes = mean_r >= min_mean_reward
        reports.append(
            CalibrationReport(
                axis=axis,
                mean_reward=mean_r,
                min_reward=min(rewards),
                max_reward=max(rewards),
                sample_count=len(rewards),
                passes=passes,
            )
        )

    return reports


def _axis_event_types(axis: str) -> set[str]:
    if axis == "pause_violation":
        return {"pause"}
    if axis == "turn_latency":
        return {"turn"}
    if axis == "interruption_yield":
        return {"interruption"}
    return set()
