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
"""Extract per-step convergence curves from a real training run.

L4 needs the *trajectory* of a run, not a single number: convergence is a claim
about how reward/loss evolves over steps.  This module reads the two raw sources
the trainer already produces

* the console log, whose step records contain ``training/global_step`` and the
  flattened metrics dict; and
* an optional ``metrics.jsonl`` side channel written by the same trainer,

and turns them into ``{metric: [{step, value}, ...]}`` curves.

Non-finite values (``NaN``/``inf``) are preserved as curve points and reported
separately instead of being dropped, so that a run which diverges can never be
scored as converged.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Console step records from ``verl`` look like ``... training/global_step:3 - ...``
#: or embed a python-dict payload.  Both shapes are handled below.
_STEP_RE = re.compile(r"(?:step|global_step)['\"]?\s*[:=]\s*(\d+)")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NUMERIC_RE = re.compile(
    r"^\s*(?:np\.\w+\()?("
    r"nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r")\)?"
)

#: Metric keys that carry the training signal L4 scores.  A recipe names the
#: subset it tracks; this set only documents the well-known keys.
KNOWN_SIGNAL_KEYS = (
    "critic/rewards/mean",
    "critic/score/mean",
    "critic/returns/mean",
    "critic/advantages/mean",
    "actor/loss",
    "actor/pg_loss",
    "actor/grad_norm",
    "val/rewards/mean",
    "val/score/mean",
)


class CurveError(Exception):
    """Raised when no usable curve can be derived from a run's outputs."""


@dataclass(frozen=True)
class CurvePoint:
    """One observed value of one metric at one training step."""

    step: int
    value: float

    def as_dict(self) -> dict[str, Any]:
        return {"step": self.step, "value": self.value}


@dataclass
class CurveSummary:
    """Aggregate view of a single metric's trajectory."""

    metric: str
    direction: str
    num_points: int
    num_finite_points: int
    warmup_steps: int
    final_window: int
    first_window_mean: float | None
    final_window_mean: float | None
    improvement: float | None
    min_value: float | None
    max_value: float | None
    non_finite_steps: list[int]
    steps: list[int]

    @property
    def finite_fraction(self) -> float:
        if self.num_points == 0:
            return 0.0
        return self.num_finite_points / self.num_points

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "direction": self.direction,
            "num_points": self.num_points,
            "num_finite_points": self.num_finite_points,
            "finite_fraction": self.finite_fraction,
            "warmup_steps": self.warmup_steps,
            "final_window": self.final_window,
            "first_window_mean": self.first_window_mean,
            "final_window_mean": self.final_window_mean,
            "improvement": self.improvement,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "non_finite_steps": self.non_finite_steps,
            "steps": self.steps,
        }


def _to_float(value: Any) -> float | None:
    """Coerce a scalar (including a numpy scalar) to a finite-or-nonfinite float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if hasattr(value, "item"):
        try:
            return _to_float(value.item())
        except Exception:
            return None
    if isinstance(value, str):
        parsed = _parse_console_value(value)
        return parsed
    return None


def _parse_console_value(value: str) -> float | None:
    """Parse a console-rendered numeric value, preserving ``nan``/``inf``.

    Returns ``None`` only when the text is not a number at all.  ``nan`` and
    ``inf`` are returned as floats so the caller can fail closed on them.
    """
    match = _NUMERIC_RE.match(value.strip())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def read_jsonl_records(path: Path | None) -> list[dict[str, Any]]:
    """Read step records from a ``metrics.jsonl`` side channel.

    Raises:
        CurveError: a line is present but is not valid JSON.
    """
    records: list[dict[str, Any]] = []
    if not path:
        return records
    path = Path(path)
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise CurveError(f"malformed metrics JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(payload, dict):
                raise CurveError(f"metrics JSONL record at {path}:{line_number} is not an object")
            records.append(payload)
    return records


def read_console_records(path: Path | None) -> list[dict[str, Any]]:
    """Best-effort extraction of step records from the trainer console log."""
    records: list[dict[str, Any]] = []
    if not path:
        return records
    path = Path(path)
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "training/global_step" not in line:
                continue
            payload: dict[str, Any] = {}
            clean = _ANSI_RE.sub("", line)
            parsed = _try_literal_dict(clean)
            if parsed is not None:
                payload = parsed
            else:
                # The payload is not a valid Python literal.  That is exactly what
                # happens when a run diverges, because ``repr`` renders NaN/Inf as
                # bare ``nan``/``inf`` tokens, and numpy scalars as ``np.float64(..)``.
                # Fall back to pair extraction so the divergence is still visible.
                payload = _extract_pairs(clean)
            match = _STEP_RE.search(clean)
            step = int(match.group(1)) if match else _coerce_step(payload.get("training/global_step"))
            if step is None:
                continue
            records.append({"step": step, "data": payload})
    return records


def _try_literal_dict(line: str) -> dict[str, Any] | None:
    try:
        start = line.index("{")
        end = line.rindex("}")
    except ValueError:
        return None
    try:
        payload = ast.literal_eval(line[start : end + 1])
    except (ValueError, SyntaxError):
        return None
    return payload if isinstance(payload, dict) else None


#: Fallback key/value extractor for console records whose payload is not a valid
#: Python literal (``nan``, ``inf``, ``np.float64(...)``, ...).  Only numeric
#: values are matched, so the pattern cannot run past the end of a value and
#: swallow the keys that follow it.  Timestamps and log levels are ignored
#: because their key must start with a letter and be followed by a number.
_KEY_VALUE_RE = re.compile(
    r"['\"]?([A-Za-z_][\w./\-]*)['\"]?\s*:\s*"
    r"((?:np\.\w+\()?(?:nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\)?)"
)


def _extract_pairs(line: str) -> dict[str, Any]:
    """Extract ``key: value`` pairs from a line whose payload is not a literal."""
    payload: dict[str, Any] = {}
    for key, raw_value in _KEY_VALUE_RE.findall(line):
        number = _parse_console_value(raw_value)
        if number is not None:
            payload[key] = number
    return payload


def _coerce_step(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def numeric_metrics(record: dict[str, Any]) -> dict[str, float]:
    """Return the numeric subset of one step record's payload."""
    data = record.get("data", record)
    if not isinstance(data, dict):
        return {}
    numeric: dict[str, float] = {}
    for key, value in data.items():
        number = _to_float(value)
        if number is not None:
            numeric[str(key)] = number
    return numeric


def load_step_records(
    *,
    metrics_jsonl: Path | None = None,
    log_file: Path | None = None,
) -> list[dict[str, Any]]:
    """Load step records, preferring the structured JSONL side channel.

    Raises:
        CurveError: neither source produced a single step record.
    """
    records = read_jsonl_records(metrics_jsonl)
    if not records:
        records = read_console_records(log_file)
    if not records:
        raise CurveError(
            f"no step records found (metrics_jsonl={metrics_jsonl}, log_file={log_file}); "
            "the run produced no parseable training/global_step output"
        )
    return sorted(records, key=lambda item: int(item["step"]))


def build_curves(
    records: list[dict[str, Any]],
    metric_names: list[str],
) -> dict[str, list[CurvePoint]]:
    """Build one curve per requested metric.

    Metrics that never appear are returned as empty curves; the caller decides
    whether that is an error, because recipes may declare optional metrics.
    """
    curves: dict[str, list[CurvePoint]] = {name: [] for name in metric_names}
    for record in records:
        step = _coerce_step(record.get("step"))
        if step is None:
            continue
        metrics = numeric_metrics(record)
        for name in metric_names:
            if name in metrics:
                curves[name].append(CurvePoint(step=step, value=metrics[name]))
    for name, points in curves.items():
        points.sort(key=lambda point: point.step)
    return curves


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_curve(
    points: list[CurvePoint],
    *,
    metric: str,
    direction: str,
    warmup_steps: int,
    final_window: int,
) -> CurveSummary:
    """Summarize a curve after discarding the warmup steps.

    ``improvement`` is signed so that a positive value always means "better":
    for ``higher``-is-better metrics it is ``final - first``, for
    ``lower``-is-better metrics it is ``first - final``.

    Non-finite points are counted and listed, and they are excluded from the
    window means so that a single ``NaN`` cannot silently become a passing mean.
    """
    steps = [point.step for point in points]
    finite = [point for point in points if math.isfinite(point.value)]
    non_finite_steps = [point.step for point in points if not math.isfinite(point.value)]

    scored = [point for point in finite if point.step > warmup_steps]
    first_window = scored[:final_window]
    last_window = scored[-final_window:] if scored else []
    first_mean = _mean([point.value for point in first_window])
    final_mean = _mean([point.value for point in last_window])

    improvement: float | None = None
    if first_mean is not None and final_mean is not None:
        improvement = (final_mean - first_mean) if direction == "higher" else (first_mean - final_mean)

    finite_values = [point.value for point in finite]
    return CurveSummary(
        metric=metric,
        direction=direction,
        num_points=len(points),
        num_finite_points=len(finite),
        warmup_steps=warmup_steps,
        final_window=final_window,
        first_window_mean=first_mean,
        final_window_mean=final_mean,
        improvement=improvement,
        min_value=min(finite_values) if finite_values else None,
        max_value=max(finite_values) if finite_values else None,
        non_finite_steps=non_finite_steps,
        steps=steps,
    )


def summarize_curves(curve_specs: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, CurveSummary]:
    """Build and summarize every curve declared by a recipe's ``metrics`` block."""
    names = [str(spec["name"]) for spec in curve_specs]
    curves = build_curves(records, names)
    summaries: dict[str, CurveSummary] = {}
    for spec in curve_specs:
        name = str(spec["name"])
        summaries[name] = summarize_curve(
            curves[name],
            metric=name,
            direction=str(spec["direction"]),
            warmup_steps=int(spec["warmup_steps"]),
            final_window=int(spec["final_window"]),
        )
    return summaries


def completed_steps(records: list[dict[str, Any]]) -> int:
    """Return the highest observed training step (0 when there is none)."""
    steps = [step for step in (_coerce_step(record.get("step")) for record in records) if step is not None]
    return max(steps) if steps else 0
