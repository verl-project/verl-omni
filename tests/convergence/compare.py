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
"""Compare an L4 convergence run against a reviewed release baseline.

Two things are separated on purpose:

1. **Comparability.**  A baseline produced from different weights, a different
   dataset, a different algorithm, a different scoring window, or a different
   runner shape cannot support a convergence verdict.  That is reported as
   ``incomparable`` — never as "no regression".
2. **Convergence.**  For comparable runs, each tracked metric is scored with a
   direction-aware, one-sided tolerance ``atol + rtol * |reference|`` so that an
   improvement never fails, while a regression beyond tolerance does.

The code SHA of the two runs is recorded for provenance but is deliberately *not*
part of comparability: the commit under test is normally exactly what changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .curves import CurveSummary

#: Contract keys that must match for a baseline to be usable.  ``commit_sha`` is
#: intentionally absent: it is provenance, not a comparability constraint.
COMPARABLE_FIELDS = (
    "algorithm",
    "precision",
    "models",
    "dataset",
    "metrics",
    "steps",
    "hardware",
)

MISSING = "<missing>"


@dataclass
class MetricVerdict:
    """Scoring verdict for one tracked metric."""

    metric: str
    direction: str
    baseline_final_window_mean: float | None
    current_final_window_mean: float | None
    delta: float | None
    tolerance: float | None
    relative_delta: float | None
    improvement: float | None
    min_improvement: float | None
    passed: bool
    scored: bool = True
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "baseline_final_window_mean": self.baseline_final_window_mean,
            "current_final_window_mean": self.current_final_window_mean,
            "delta": self.delta,
            "tolerance": self.tolerance,
            "relative_delta": self.relative_delta,
            "improvement": self.improvement,
            "min_improvement": self.min_improvement,
            "passed": self.passed,
            "scored": self.scored,
            "reasons": list(self.reasons),
        }


@dataclass
class ComparisonReport:
    """Full comparison outcome for one case."""

    comparable: bool
    mismatches: list[dict[str, Any]]
    verdicts: dict[str, MetricVerdict]
    baseline_present: bool

    @property
    def failed_metrics(self) -> list[str]:
        return [name for name, verdict in self.verdicts.items() if not verdict.passed]

    @property
    def passed(self) -> bool:
        if not self.baseline_present or not self.comparable:
            return False
        return not self.failed_metrics

    @property
    def status(self) -> str:
        """Return the status this comparison supports.

        ``passed``/``failed`` are only reachable for comparable runs against a
        present baseline; everything else is explicitly non-conclusive.
        """
        if not self.baseline_present:
            return "invalid"
        if not self.comparable:
            return "incomparable"
        return "passed" if not self.failed_metrics else "failed"

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_present": self.baseline_present,
            "comparable": self.comparable,
            "mismatches": list(self.mismatches),
            "failed_metrics": self.failed_metrics,
            "passed": self.passed,
            "metrics": {name: verdict.as_dict() for name, verdict in sorted(self.verdicts.items())},
        }


def build_contract(
    *,
    algorithm: str,
    precision: str,
    recipe_data: dict[str, Any],
    observed_hardware: dict[str, Any] | None,
    commit_sha: str | None,
) -> dict[str, Any]:
    """Assemble the comparability contract recorded next to every run.

    The contract is stored with the baseline artifact so a later run can prove it
    was scored against a like-for-like reference.
    """
    models = {
        name: {"source": entry.get("source"), "revision": entry.get("revision")}
        for name, entry in sorted(recipe_data["models"].items())
    }
    dataset = {
        "train": recipe_data["dataset"]["train"],
        "val": recipe_data["dataset"]["val"],
    }
    if "train_max_samples" in recipe_data["dataset"]:
        dataset["train_max_samples"] = recipe_data["dataset"]["train_max_samples"]
    metrics = {
        "tracked": [
            {
                "name": str(item["name"]),
                "direction": str(item["direction"]),
                "optional": bool(item.get("optional", False)),
            }
            for item in recipe_data["metrics"]["tracked"]
        ],
        "warmup_steps": int(recipe_data["metrics"]["warmup_steps"]),
        "final_window": int(recipe_data["metrics"]["final_window"]),
    }
    hardware: dict[str, Any] = {}
    if observed_hardware:
        hardware = {
            "gpu_architecture": observed_hardware.get("gpu_architecture"),
            "gpu_count": observed_hardware.get("gpu_count"),
        }
    return {
        "algorithm": algorithm,
        "precision": precision,
        "models": models,
        "dataset": dataset,
        "metrics": metrics,
        "steps": int(recipe_data["budget"]["total_training_steps"]),
        "hardware": hardware,
        "commit_sha": commit_sha,
    }


def compare_contracts(
    baseline_contract: dict[str, Any] | None,
    current_contract: dict[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Return ``(comparable, mismatches)`` for two run contracts."""
    if not baseline_contract:
        return False, [{"field": "<contract>", "baseline": MISSING, "current": current_contract or MISSING}]
    if not current_contract:
        return False, [{"field": "<contract>", "baseline": baseline_contract, "current": MISSING}]

    mismatches: list[dict[str, Any]] = []
    for key in COMPARABLE_FIELDS:
        baseline_value = baseline_contract.get(key, MISSING)
        current_value = current_contract.get(key, MISSING)
        if baseline_value != current_value:
            mismatches.append({"field": key, "baseline": baseline_value, "current": current_value})
    return not mismatches, mismatches


def _tolerance(reference: float, atol: float, rtol: float) -> float:
    return atol + rtol * abs(reference)


def score_metric(
    metric: str,
    *,
    direction: str,
    baseline: CurveSummary | None,
    current: CurveSummary | None,
    min_points: int,
    require_finite: bool,
    atol: float,
    rtol: float,
    min_improvement: float,
    optional: bool = False,
) -> MetricVerdict:
    """Score one metric's curve against its baseline curve.

    An ``optional`` metric that the baseline simply does not contain is reported
    as not scored rather than as a failure, because the baseline was reviewed
    without that signal.  Once the baseline *does* contain it, it is scored like
    any other metric.
    """
    reasons: list[str] = []
    baseline_final = baseline.final_window_mean if baseline else None
    current_final = current.final_window_mean if current else None

    if baseline is None and optional:
        return MetricVerdict(
            metric=metric,
            direction=direction,
            baseline_final_window_mean=None,
            current_final_window_mean=current_final,
            delta=None,
            tolerance=None,
            relative_delta=None,
            improvement=current.improvement if current else None,
            min_improvement=min_improvement,
            passed=True,
            scored=False,
            reasons=["optional metric is absent from the baseline; not scored"],
        )

    if baseline is None:
        return MetricVerdict(
            metric=metric,
            direction=direction,
            baseline_final_window_mean=None,
            current_final_window_mean=current_final,
            delta=None,
            tolerance=None,
            relative_delta=None,
            improvement=current.improvement if current else None,
            min_improvement=min_improvement,
            passed=False,
            reasons=["metric absent from the baseline; the baseline cannot score it"],
        )

    if current is None or current.num_points == 0:
        return MetricVerdict(
            metric=metric,
            direction=direction,
            baseline_final_window_mean=baseline_final,
            current_final_window_mean=None,
            delta=None,
            tolerance=None,
            relative_delta=None,
            improvement=None,
            min_improvement=min_improvement,
            passed=False,
            reasons=["metric not observed in the current run"],
        )

    if require_finite and current.non_finite_steps:
        reasons.append(f"non-finite values at steps {current.non_finite_steps[:10]}")
    if current.num_finite_points < min_points:
        reasons.append(f"only {current.num_finite_points} finite point(s); the recipe requires at least {min_points}")
    if current_final is None or baseline_final is None:
        reasons.append("final window could not be computed for both runs")

    delta: float | None = None
    tolerance: float | None = None
    relative_delta: float | None = None
    if current_final is not None and baseline_final is not None:
        delta = current_final - baseline_final
        tolerance = _tolerance(baseline_final, atol, rtol)
        if abs(baseline_final) < 1e-12:
            relative_delta = 0.0 if abs(current_final) < 1e-12 else math.inf
        else:
            relative_delta = delta / abs(baseline_final)
        if not math.isfinite(delta) or not math.isfinite(tolerance):
            reasons.append("delta or tolerance is not finite")
        elif direction == "higher":
            if (baseline_final - current_final) > tolerance:
                reasons.append(
                    f"reward/loss dropped from {baseline_final:.6g} to {current_final:.6g}, "
                    f"beyond tolerance {tolerance:.6g}"
                )
        elif (current_final - baseline_final) > tolerance:
            reasons.append(
                f"metric rose from {baseline_final:.6g} to {current_final:.6g}, beyond tolerance {tolerance:.6g}"
            )

    improvement = current.improvement
    if improvement is None:
        reasons.append("no in-run improvement could be measured")
    elif improvement < min_improvement:
        reasons.append(f"in-run improvement {improvement:.6g} is below the required minimum {min_improvement:.6g}")

    return MetricVerdict(
        metric=metric,
        direction=direction,
        baseline_final_window_mean=baseline_final,
        current_final_window_mean=current_final,
        delta=delta,
        tolerance=tolerance,
        relative_delta=relative_delta,
        improvement=improvement,
        min_improvement=min_improvement,
        passed=not reasons,
        reasons=reasons,
    )


def compare_curves(
    *,
    current_summaries: dict[str, CurveSummary],
    baseline_summaries: dict[str, CurveSummary],
    metric_specs: list[dict[str, Any]],
    convergence: dict[str, Any],
    min_points: int,
    baseline_contract: dict[str, Any] | None,
    current_contract: dict[str, Any] | None,
    baseline_present: bool,
) -> ComparisonReport:
    """Compare every declared tracked metric and report comparability separately."""
    comparable, mismatches = compare_contracts(baseline_contract, current_contract)
    atol = float(convergence["atol"])
    rtol = float(convergence["rtol"])
    min_improvement = float(convergence["min_improvement"])
    require_finite = bool(convergence.get("require_finite", True))

    verdicts: dict[str, MetricVerdict] = {}
    for spec in metric_specs:
        name = str(spec["name"])
        verdicts[name] = score_metric(
            name,
            direction=str(spec["direction"]),
            baseline=baseline_summaries.get(name),
            current=current_summaries.get(name),
            min_points=min_points,
            require_finite=require_finite,
            atol=atol,
            rtol=rtol,
            min_improvement=min_improvement,
            optional=bool(spec.get("optional", False)),
        )
    return ComparisonReport(
        comparable=comparable,
        mismatches=mismatches,
        verdicts=verdicts,
        baseline_present=baseline_present,
    )
