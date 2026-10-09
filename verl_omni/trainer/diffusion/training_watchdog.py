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
"""Bounded, reporting-only checks for reduced diffusion training metrics.

One trainer owns one instance and resets it for each fit. Only accepted,
increasing steps advance observation time; no state is persisted in checkpoints.
"""

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType

_RATIO_LOSS_MODES = frozenset({"flow_grpo", "dance_grpo", "flow_dppo", "grpo_guard"})
_RULES = ("nonfinite_grad", "reward_zero_grad", "ratio_out_of_bounds", "trajectory_staleness")
_GRAD = "actor/grad_norm"
_RATIO = "actor/ratio_mean"
_ZERO_STD = "critic/rewards/zero_std_ratio"
_STD_MEAN = "critic/rewards/std_mean"
_GROUP_SIZE = "critic/rewards/group_size"
_STALENESS = "training/off_policy/trajectory_staleness_worst/max"


@dataclass(frozen=True)
class WatchdogAlert:
    """A single warning episode with a snapshot of its triggering evidence."""

    rule: str
    step: int
    first_step: int
    last_step: int
    observations: int
    evidence: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


def _nonnegative_finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _nonnegative_int(name: str, value: int, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < int(positive):
        kind = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {kind} integer")
    return int(value)


def _scalar(metrics: Mapping[str, object], name: str, *, allow_nonfinite: bool = False) -> float | None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    result = float(value)
    if math.isfinite(result) or allow_nonfinite:
        return result
    return None


class TrainingWatchdog:
    """Observe reduced scalar metrics without affecting the training path.

    Windows hold only consecutive faulty observations. A known healthy value
    ends an episode; missing or invalid prerequisites break the window but do
    not prove recovery. Cooldown is measured in accepted observations.
    """

    def __init__(
        self,
        loss_mode: str,
        window_size: int = 3,
        warmup_steps: int = 5,
        cooldown_steps: int = 20,
        reward_std_max: float = 1e-8,
        grad_norm_max: float = 1e-12,
        ratio_lower: float = 0.5,
        ratio_upper: float = 2.0,
        max_staleness: float | None = None,
    ) -> None:
        if not isinstance(loss_mode, str):
            raise ValueError("loss_mode must be a string")
        self.loss_mode = loss_mode
        self.window_size = _nonnegative_int("window_size", window_size, positive=True)
        self.warmup_steps = _nonnegative_int("warmup_steps", warmup_steps)
        self.cooldown_steps = _nonnegative_int("cooldown_steps", cooldown_steps)
        self.reward_std_max = _nonnegative_finite("reward_std_max", reward_std_max)
        self.grad_norm_max = _nonnegative_finite("grad_norm_max", grad_norm_max)
        self.ratio_lower = _nonnegative_finite("ratio_lower", ratio_lower)
        self.ratio_upper = _nonnegative_finite("ratio_upper", ratio_upper)
        if not 0 < self.ratio_lower < 1 < self.ratio_upper:
            raise ValueError("ratio bounds must satisfy 0 < ratio_lower < 1 < ratio_upper")
        self.max_staleness = None if max_staleness is None else _nonnegative_finite("max_staleness", max_staleness)
        self.reset()

    def reset(self) -> None:
        """Discard episode and warmup state, including on a resumed fit."""
        self._last_step: int | None = None
        self._observations = 0
        self._windows: dict[str, deque[tuple[int, dict[str, float]]]] = {
            rule: deque(maxlen=self.window_size) for rule in _RULES if rule != "nonfinite_grad"
        }
        self._active = dict.fromkeys(_RULES, False)
        self._last_alert_observation: dict[str, int | None] = dict.fromkeys(_RULES)

    def _emit(self, rule: str, step: int, window: list[tuple[int, dict[str, float]]]) -> WatchdogAlert:
        self._active[rule] = True
        self._last_alert_observation[rule] = self._observations
        evidence: dict[str, float] = {}
        for _, scalars in window:
            for key, value in scalars.items():
                evidence[key] = value
        if len(window) > 1:
            for key in window[0][1]:
                values = [scalars[key] for _, scalars in window]
                evidence[f"{key}/window_min"] = min(values)
                evidence[f"{key}/window_max"] = max(values)
        return WatchdogAlert(rule, step, window[0][0], window[-1][0], len(window), evidence)

    def _ready(self, rule: str) -> bool:
        last_alert = self._last_alert_observation[rule]
        return last_alert is None or self._observations - last_alert >= self.cooldown_steps

    def _window_rule(
        self, rule: str, step: int, evidence: dict[str, float] | None, faulty: bool | None
    ) -> WatchdogAlert | None:
        window = self._windows[rule]
        if evidence is None:
            window.clear()
            return None
        if not faulty:
            window.clear()
            self._active[rule] = False
            return None
        window.append((step, evidence))
        if len(window) == self.window_size and not self._active[rule] and self._ready(rule):
            return self._emit(rule, step, list(window))
        return None

    def observe(self, step: int, metrics: Mapping[str, object]) -> tuple[list[WatchdogAlert], dict[str, float]]:
        """Return new alerts and numeric logger metrics for one accepted step."""
        if isinstance(step, bool) or not isinstance(step, Integral):
            raise ValueError("step must be an integer")
        if self._last_step is not None and step <= self._last_step:
            return [], {
                "watchdog/alerts": 0.0,
                "watchdog/ignored_step": 1.0,
                "watchdog/warmup_remaining": float(max(0, self.warmup_steps - self._observations)),
                **{f"watchdog/coverage/{rule}": 0.0 for rule in _RULES},
            }
        if self._last_step is not None and step > self._last_step + 1:
            for window in self._windows.values():
                window.clear()
        self._last_step = int(step)
        self._observations += 1
        warmup = self._observations <= self.warmup_steps
        coverage = dict.fromkeys(_RULES, 0.0)
        alerts: list[WatchdogAlert] = []

        grad = _scalar(metrics, _GRAD, allow_nonfinite=True)
        if grad is not None:
            coverage["nonfinite_grad"] = 1.0
            if math.isfinite(grad):
                self._active["nonfinite_grad"] = False
            elif not self._active["nonfinite_grad"] and self._ready("nonfinite_grad"):
                alerts.append(self._emit("nonfinite_grad", int(step), [(int(step), {_GRAD: grad})]))

        if self.loss_mode in _RATIO_LOSS_MODES:
            zero_std = _scalar(metrics, _ZERO_STD)
            std_mean = _scalar(metrics, _STD_MEAN)
            group_size = _scalar(metrics, _GROUP_SIZE)
            if (
                zero_std is not None
                and 0 <= zero_std <= 1
                and std_mean is not None
                and std_mean >= 0
                and group_size is not None
                and group_size >= 1
                and grad is not None
                and math.isfinite(grad)
                and grad >= 0
            ):
                coverage["reward_zero_grad"] = 1.0
                evidence = {_ZERO_STD: zero_std, _STD_MEAN: std_mean, _GROUP_SIZE: group_size, _GRAD: grad}
                faulty = (
                    zero_std == 1.0
                    and group_size > 1
                    and std_mean <= self.reward_std_max
                    and grad <= self.grad_norm_max
                )
            else:
                evidence, faulty = None, None
            alert = self._window_rule("reward_zero_grad", int(step), None if warmup else evidence, faulty)
            if alert is not None:
                alerts.append(alert)

            ratio = _scalar(metrics, _RATIO)
            if ratio is not None and ratio >= 0:
                coverage["ratio_out_of_bounds"] = 1.0
                ratio_evidence = {_RATIO: ratio}
                ratio_faulty = ratio < self.ratio_lower or ratio > self.ratio_upper
            else:
                ratio_evidence, ratio_faulty = None, None
            alert = self._window_rule(
                "ratio_out_of_bounds", int(step), None if warmup else ratio_evidence, ratio_faulty
            )
            if alert is not None:
                alerts.append(alert)

        staleness = _scalar(metrics, _STALENESS)
        if self.max_staleness is not None and staleness is not None and staleness >= 0:
            coverage["trajectory_staleness"] = 1.0
            stale_evidence = {_STALENESS: staleness}
            stale_faulty = staleness > self.max_staleness
        else:
            stale_evidence, stale_faulty = None, None
        alert = self._window_rule("trajectory_staleness", int(step), None if warmup else stale_evidence, stale_faulty)
        if alert is not None:
            alerts.append(alert)

        result = {
            "watchdog/alerts": float(len(alerts)),
            "watchdog/ignored_step": 0.0,
            "watchdog/warmup_remaining": float(max(0, self.warmup_steps - self._observations)),
        }
        result.update({f"watchdog/coverage/{rule}": value for rule, value in coverage.items()})
        return alerts, result
