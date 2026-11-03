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
"""Bounded step-time regression detection for synchronous diffusion training."""

import hashlib
import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from statistics import median
from types import MappingProxyType
from typing import Any

_TIME_PER_STEP = "perf/time_per_step"
_TOTAL_IMAGES = "perf/total_num_images"
_PHASE_PREFIX = "timing_s/"
_WORKLOAD_FIELDS = frozenset(
    {
        "train_batch_size",
        "rollout_n",
        "height",
        "width",
        "num_inference_steps",
        "parameter_sync_step",
        "nnodes",
        "n_gpus_per_node",
    }
)
_SUPPORTED_TOOLS = frozenset({"torch", "npu", "torch_memory"})


@dataclass(frozen=True)
class PhaseDelta:
    """Median phase-time delta between the baseline and regression windows."""

    name: str
    baseline_s: float
    current_s: float
    delta_s: float
    delta_ratio: float | None


@dataclass(frozen=True)
class StepTimeRegressionAlert:
    """Evidence for one sustained step-time regression episode."""

    first_step: int
    last_step: int
    baseline_first_step: int
    baseline_last_step: int
    baseline_observations: int
    current_observations: int
    baseline_median_s: float
    current_median_s: float
    slowdown_ratio: float
    workload_fingerprint: str
    phase_deltas: tuple[PhaseDelta, ...]


@dataclass(frozen=True)
class PerformanceWatchdogResult:
    """Outcome for one completed step observation."""

    state: str
    reason: str | None = None
    alert: StepTimeRegressionAlert | None = None

    @property
    def insufficient_evidence(self) -> bool:
        return self.state == "insufficient_evidence"


@dataclass(frozen=True)
class _StepSample:
    step: int
    step_time_s: float
    phase_times_s: Mapping[str, float]


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _nested_get(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        current = _config_get(current, key, default)
        if current is default:
            return default
    return current


def validate_performance_watchdog_config(
    config: Any,
    *,
    trainer_mode: str,
    use_reference_policy: bool,
) -> tuple[str, tuple[str, ...]]:
    """Validate runtime-capture prerequisites and return tool plus enabled roles.

    The automatic trigger intentionally supports only backends that can be
    started and stopped dynamically by the existing worker RPC. Nsight's Ray
    runtime environment and capture count are fixed when workers are created.
    """
    if trainer_mode != "sync":
        raise ValueError("trainer.v1.performance_watchdog.enabled is supported only with trainer_mode='sync'")

    profiler = _nested_get(config, "global_profiler", default={})
    tool = _config_get(profiler, "tool")
    if tool not in _SUPPORTED_TOOLS:
        valid = ", ".join(sorted(_SUPPORTED_TOOLS))
        if tool == "nsys":
            raise ValueError("trainer.v1.performance_watchdog does not support global_profiler.tool='nsys'")
        raise ValueError(f"trainer.v1.performance_watchdog requires global_profiler.tool in {{{valid}}}")

    if _config_get(profiler, "profile_continuous_steps", False) is True:
        raise ValueError(
            "trainer.v1.performance_watchdog requires global_profiler.profile_continuous_steps=false "
            "to keep the triggered capture bounded to one step"
        )

    save_path = _config_get(profiler, "save_path")
    if not isinstance(save_path, str) or not save_path.strip():
        raise ValueError("trainer.v1.performance_watchdog requires global_profiler.save_path")

    role_config = _nested_get(config, "actor_rollout_ref", default={})
    enabled_roles: list[str] = []
    for role in ("actor", "ref", "rollout"):
        role_profiler = _nested_get(role_config, role, "profiler", default={})
        enabled = _config_get(role_profiler, "enable", False)
        if not isinstance(enabled, bool):
            raise ValueError(f"actor_rollout_ref.{role}.profiler.enable must be a boolean")
        if not enabled:
            continue
        if role == "ref" and not use_reference_policy:
            continue
        role_tool = _config_get(role_profiler, "tool")
        if role_tool != tool:
            raise ValueError(
                f"actor_rollout_ref.{role}.profiler.tool must match global_profiler.tool ({tool!r}) "
                "when the performance watchdog is enabled"
            )
        enabled_roles.append(role)

    if not enabled_roles:
        raise ValueError(
            "trainer.v1.performance_watchdog requires at least one active role profiler: "
            "actor, rollout, or ref when reference policy is enabled"
        )

    return str(tool), tuple(enabled_roles)


def _positive_integer_metric(metrics: Mapping[str, object], key: str) -> int | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0 or not numeric_value.is_integer():
        return None
    return int(numeric_value)


def _positive_time_metric(metrics: Mapping[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        return None
    return numeric_value


def _finite_nonnegative_metric(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value < 0:
        return None
    return numeric_value


class StepTimePerformanceWatchdog:
    """Compare bounded, comparable timing windows without touching training state.

    The current window and baseline are disjoint. Every step in the current
    window must exceed the baseline median by the configured ratio to create
    one alert for that regression episode. Cooldown is measured in training
    step numbers and survives workload-segment resets.
    """

    def __init__(
        self,
        warmup_steps: int = 5,
        baseline_window_size: int = 20,
        minimum_observations: int = 8,
        regression_window_size: int = 3,
        regression_ratio: float = 1.25,
        cooldown_steps: int = 20,
        max_phase_deltas: int = 3,
    ) -> None:
        self.warmup_steps = self._nonnegative_int("warmup_steps", warmup_steps)
        self.baseline_window_size = self._positive_int("baseline_window_size", baseline_window_size)
        self.minimum_observations = self._positive_int("minimum_observations", minimum_observations)
        if self.minimum_observations > self.baseline_window_size:
            raise ValueError("minimum_observations must be <= baseline_window_size")
        self.regression_window_size = self._positive_int("regression_window_size", regression_window_size)
        if isinstance(regression_ratio, bool) or not isinstance(regression_ratio, Real):
            raise ValueError("regression_ratio must be a finite real number greater than 1")
        self.regression_ratio = float(regression_ratio)
        if not math.isfinite(self.regression_ratio) or self.regression_ratio <= 1.0:
            raise ValueError("regression_ratio must be a finite real number greater than 1")
        self.cooldown_steps = self._nonnegative_int("cooldown_steps", cooldown_steps)
        self.max_phase_deltas = self._positive_int("max_phase_deltas", max_phase_deltas)
        self.reset()

    @staticmethod
    def _nonnegative_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return int(value)

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    def reset(self) -> None:
        """Clear all run-local state, including cooldown after a fresh fit."""
        self._last_step: int | None = None
        self._workload_signature: tuple[tuple[str, float | int | None], ...] | None = None
        self._warmup_observations = 0
        self._history: deque[_StepSample] = deque(maxlen=self.baseline_window_size + self.regression_window_size - 1)
        self._regression_active = False
        self._last_alert_step: int | None = None

    @staticmethod
    def _make_workload_signature(
        workload: Mapping[str, object], total_images: int
    ) -> tuple[tuple[str, float | int | None], ...] | None:
        if not isinstance(workload, Mapping) or set(workload) - _WORKLOAD_FIELDS:
            return None
        dimensions: list[tuple[str, float | int | None]] = [("total_images", total_images)]
        for key in sorted(workload):
            value = workload[key]
            if value is None:
                normalized = None
            elif isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                return None
            elif float(value).is_integer():
                normalized = int(value)
            else:
                normalized = float(value)
            dimensions.append((key, normalized))
        return tuple(dimensions)

    def _reset_segment(self, signature: tuple[tuple[str, float | int | None], ...] | None = None) -> None:
        self._workload_signature = signature
        self._warmup_observations = 0
        self._history.clear()
        self._regression_active = False

    def _sample(self, step: int, metrics: Mapping[str, object]) -> _StepSample | None:
        step_time = _positive_time_metric(metrics, _TIME_PER_STEP)
        total_images = _positive_integer_metric(metrics, _TOTAL_IMAGES)
        if step_time is None:
            return None
        if total_images is None:
            return None
        phases = {
            key.removeprefix(_PHASE_PREFIX): value
            for key, raw_value in metrics.items()
            if isinstance(key, str)
            and key.startswith(_PHASE_PREFIX)
            and key.removeprefix(_PHASE_PREFIX) != "step"
            and (value := _finite_nonnegative_metric(raw_value)) is not None
        }
        return _StepSample(step, step_time, MappingProxyType(phases))

    def _baseline_samples(self, previous: list[_StepSample]) -> list[_StepSample]:
        non_baseline_count = self.regression_window_size - 1
        baseline_end = len(previous) - non_baseline_count
        baseline_start = max(0, baseline_end - self.baseline_window_size)
        return previous[baseline_start:baseline_end]

    def _phase_deltas(self, baseline: list[_StepSample], current: list[_StepSample]) -> tuple[PhaseDelta, ...]:
        baseline_values: dict[str, list[float]] = {}
        current_values: dict[str, list[float]] = {}
        for sample in baseline:
            for name, value in sample.phase_times_s.items():
                baseline_values.setdefault(name, []).append(value)
        for sample in current:
            for name, value in sample.phase_times_s.items():
                current_values.setdefault(name, []).append(value)

        deltas = []
        for name in baseline_values.keys() & current_values.keys():
            baseline_s = float(median(baseline_values[name]))
            current_s = float(median(current_values[name]))
            delta_s = current_s - baseline_s
            deltas.append(
                PhaseDelta(
                    name=name,
                    baseline_s=baseline_s,
                    current_s=current_s,
                    delta_s=delta_s,
                    delta_ratio=delta_s / baseline_s if baseline_s > 0 else None,
                )
            )
        deltas.sort(key=lambda item: (-abs(item.delta_s), item.name))
        return tuple(deltas[: self.max_phase_deltas])

    def observe(
        self,
        step: int,
        metrics: Mapping[str, object],
        workload: Mapping[str, object],
    ) -> PerformanceWatchdogResult:
        """Observe one completed step and return bounded evidence or an alert."""
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise ValueError("step must be an integer")
        step = int(step)
        if self._last_step is not None and step <= self._last_step:
            return PerformanceWatchdogResult("insufficient_evidence", "non_increasing_step")

        had_step_gap = self._last_step is not None and step != self._last_step + 1
        self._last_step = step
        if had_step_gap:
            self._reset_segment()

        if not isinstance(metrics, Mapping):
            self._reset_segment()
            return PerformanceWatchdogResult("insufficient_evidence", "missing_metrics")
        sample = self._sample(step, metrics)
        if sample is None:
            self._reset_segment()
            return PerformanceWatchdogResult("insufficient_evidence", "missing_or_invalid_step_metrics")

        total_images = _positive_integer_metric(metrics, _TOTAL_IMAGES)
        assert total_images is not None
        signature = self._make_workload_signature(workload, total_images)
        if signature is None:
            self._reset_segment()
            return PerformanceWatchdogResult("insufficient_evidence", "invalid_workload_signature")

        workload_changed = self._workload_signature is not None and signature != self._workload_signature
        if workload_changed:
            self._reset_segment(signature)
        elif self._workload_signature is None:
            self._workload_signature = signature

        if self._warmup_observations < self.warmup_steps:
            self._warmup_observations += 1
            reason = "workload_changed" if workload_changed else ("step_gap" if had_step_gap else "warmup")
            return PerformanceWatchdogResult("insufficient_evidence", reason)

        if workload_changed or had_step_gap:
            self._history.append(sample)
            return PerformanceWatchdogResult(
                "insufficient_evidence", "workload_changed" if workload_changed else "step_gap"
            )

        previous = list(self._history)
        previous_window_size = self.regression_window_size - 1
        current_prefix = previous[-previous_window_size:] if previous_window_size else []
        current_window = current_prefix + [sample]
        if len(current_window) < self.regression_window_size:
            self._history.append(sample)
            return PerformanceWatchdogResult("insufficient_evidence", "collecting_regression_window")

        baseline = self._baseline_samples(previous)
        if len(baseline) < self.minimum_observations:
            self._history.append(sample)
            return PerformanceWatchdogResult("insufficient_evidence", "collecting_baseline")

        baseline_median = float(median(item.step_time_s for item in baseline))
        current_median = float(median(item.step_time_s for item in current_window))
        threshold_s = baseline_median * self.regression_ratio
        is_regression = all(item.step_time_s >= threshold_s for item in current_window)

        alert = None
        if is_regression:
            if not self._regression_active:
                self._regression_active = True
                cooldown_elapsed = self._last_alert_step is None or step - self._last_alert_step >= self.cooldown_steps
                if cooldown_elapsed:
                    fingerprint = hashlib.sha256(repr(signature).encode("utf-8")).hexdigest()[:12]
                    alert = StepTimeRegressionAlert(
                        first_step=current_window[0].step,
                        last_step=current_window[-1].step,
                        baseline_first_step=baseline[0].step,
                        baseline_last_step=baseline[-1].step,
                        baseline_observations=len(baseline),
                        current_observations=len(current_window),
                        baseline_median_s=baseline_median,
                        current_median_s=current_median,
                        slowdown_ratio=current_median / baseline_median,
                        workload_fingerprint=fingerprint,
                        phase_deltas=self._phase_deltas(baseline, current_window),
                    )
                    self._last_alert_step = step

        else:
            self._regression_active = False

        self._history.append(sample)
        if alert is not None:
            return PerformanceWatchdogResult("regression", alert=alert)
        if is_regression and self._last_alert_step is not None and step - self._last_alert_step < self.cooldown_steps:
            return PerformanceWatchdogResult("cooldown", "profile_cooldown")
        return PerformanceWatchdogResult("observing")


__all__ = [
    "PerformanceWatchdogResult",
    "PhaseDelta",
    "StepTimePerformanceWatchdog",
    "StepTimeRegressionAlert",
    "validate_performance_watchdog_config",
]
