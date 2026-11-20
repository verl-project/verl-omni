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
"""Reward-calibration entry gate for duplex GRPO (RFC verl-omni #905 §3.6).

Before any gradient step, the unmodified starting checkpoint is rolled over
the mined event set and each axis's normalized reward must land in a sane
mid band: an axis that floors the base model means the event construction or
the reward band is misaligned with the checkpoint's actual behavior, and must
be fixed before training. This gate keeps initial rewards reasonable rather
than near-zero, so training refines existing behavior instead of rebuilding
it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

__all__ = [
    "CALIBRATION_MIN_MEAN_RATIO",
    "DEFAULT_AXIS_MAXIMUM",
    "DEFAULT_AXIS_MINIMUM",
    "CalibrationReport",
    "audit_calibration",
]

#: Target: per-axis mean normalized reward at least this fraction of the axis
#: maximum (the RFC's "mean normalized reward >= ~0.4 of axis maximum").
CALIBRATION_MIN_MEAN_RATIO = 0.4


class CalibrationReport(dict):
    """The audit's verdict, keyed per axis plus an ``overall_pass`` flag."""

    @property
    def passed(self) -> bool:
        return bool(self.get("overall_pass", False))


#: The rule rewards live in [-1, 0] per axis (0 is the best possible).
DEFAULT_AXIS_MINIMUM = -1.0
DEFAULT_AXIS_MAXIMUM = 0.0


def audit_calibration(
    axis_rewards: Mapping[str, Sequence[float]],
    *,
    axis_minima: Mapping[str, float] | None = None,
    axis_maxima: Mapping[str, float] | None = None,
    min_mean_ratio: float = CALIBRATION_MIN_MEAN_RATIO,
    min_samples: int = 8,
) -> CalibrationReport:
    """Audit the base policy's per-axis rewards against the mid-band gate.

    Args:
        axis_rewards: Raw reward values per axis, from the calibration rollout.
        axis_minima: Each axis's normalization minimum (default ``-1``).
        axis_maxima: Each axis's normalization maximum (default ``0``, the
            rule rewards' best possible value).
        min_mean_ratio: Required ``(mean - min) / (max - min)`` ratio.
        min_samples: Minimum episodes per axis for the audit to be meaningful.

    Returns:
        A report with per-axis ``mean``, ``normalized_mean``, ``passed`` and a
        global ``overall_pass``; axes with too few samples fail closed.
    """
    axis_minima = axis_minima or {}
    axis_maxima = axis_maxima or {}
    report = CalibrationReport()
    overall = True
    for axis, rewards in sorted(axis_rewards.items()):
        minimum = float(axis_minima.get(axis, DEFAULT_AXIS_MINIMUM))
        maximum = float(axis_maxima.get(axis, DEFAULT_AXIS_MAXIMUM))
        entry: dict[str, float | bool | int] = {"num_samples": len(rewards)}
        if len(rewards) < min_samples:
            entry.update(
                {
                    "mean": float("nan"),
                    "normalized_mean": float("nan"),
                    "passed": False,
                    "reason": f"only {len(rewards)} samples (< {min_samples})",
                }
            )
            overall = False
        else:
            mean = sum(float(value) for value in rewards) / len(rewards)
            span = maximum - minimum
            normalized = (mean - minimum) / span if span > 0 else float("nan")
            passed = span > 0 and normalized >= min_mean_ratio
            entry.update({"mean": mean, "normalized_mean": normalized, "passed": passed})
            if not passed:
                entry["reason"] = (
                    "degenerate axis band; fix the reward band"
                    if span <= 0
                    else f"normalized mean {normalized:.3f} < {min_mean_ratio:.2f}; "
                    "the event construction or reward band is misaligned with the checkpoint"
                )
                overall = False
        report[axis] = entry  # type: ignore[assignment]
    report["overall_pass"] = overall
    return report
