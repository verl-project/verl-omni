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
"""Collect and compare performance metrics for the nightly FlowGRPO run."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NUMERIC_VALUE_RE = re.compile(
    r"^\s*(?:np\.\w+\()?("
    r"nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r")\)?"
)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return _as_float(value.item())
        except Exception:
            return None
    return None


def _parse_console_value(value: str) -> float | None:
    match = _NUMERIC_VALUE_RE.match(value.strip())
    if not match:
        return None
    raw = match.group(1).lower()
    if raw == "nan":
        return None
    if raw in {"inf", "+inf", "-inf"}:
        return None
    return _as_float(float(raw))


def _read_jsonl(path: Path) -> list[dict]:
    records = []
    if not path or not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _read_console_log(path: Path) -> list[dict]:
    """Best-effort fallback for console logs when DEBUG_METRICS_JSONL is unavailable."""
    records = []
    if not path or not path.exists():
        return records
    step_pattern = re.compile(r"(?:step|global_step)['\"]?\s*[:=]\s*(\d+)")
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if "training/global_step" not in line:
                continue
            try:
                payload = ast.literal_eval(line[line.index("{") : line.rindex("}") + 1])
            except Exception:
                clean_line = _ANSI_RE.sub("", line)
                match = step_pattern.search(clean_line)
                if not match:
                    continue
                payload = {}
                for item in clean_line.split(" - "):
                    if ":" not in item:
                        continue
                    key, value = item.split(":", 1)
                    key = key.strip().split()[-1]
                    number = _parse_console_value(value)
                    if number is not None:
                        payload[key] = number
                if not payload:
                    continue
                step = int(match.group(1))
                records.append({"step": step, "data": payload})
                continue
            match = step_pattern.search(line)
            step = int(match.group(1)) if match else int(payload.get("training/global_step", -1))
            records.append({"step": step, "data": payload})
    return records


def _numeric_metrics(record: dict) -> dict[str, float]:
    data = record.get("data", {})
    numeric = {}
    for key, value in data.items():
        number = _as_float(value)
        if number is not None:
            numeric[key] = number
    return numeric


def _summarize(records: list[dict]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {}
    for record in records:
        for key, value in _numeric_metrics(record).items():
            values.setdefault(key, []).append(value)
    summary = {}
    for key, items in sorted(values.items()):
        if not items:
            continue
        summary[key] = {
            "count": float(len(items)),
            "min": min(items),
            "max": max(items),
            "mean": sum(items) / len(items),
        }
    return summary


def _load_records(metrics_jsonl: Path | None, log_file: Path | None) -> list[dict]:
    records = _read_jsonl(metrics_jsonl) if metrics_jsonl else []
    if not records and log_file:
        records = _read_console_log(log_file)
    return sorted(records, key=lambda item: int(item["step"]))


def _compare_summary(current: dict, baseline: dict, threshold: float) -> tuple[bool, dict]:
    passed = True
    report = {}
    for key, baseline_stats in baseline.items():
        if key not in current:
            report[key] = {"missing_in_current": True}
            passed = False
            continue
        current_stats = current[key]
        baseline_mean = baseline_stats.get("mean")
        current_mean = current_stats.get("mean")
        if baseline_mean is None or current_mean is None:
            continue
        if abs(baseline_mean) < 1e-12:
            rel_delta = 0.0 if abs(current_mean) < 1e-12 else float("inf")
        else:
            rel_delta = (current_mean - baseline_mean) / abs(baseline_mean)
        failed = abs(rel_delta) > threshold
        if failed:
            passed = False
        report[key] = {
            "baseline_mean": baseline_mean,
            "current_mean": current_mean,
            "relative_delta": rel_delta,
            "failed": failed,
        }
    return passed, report


def collect(args: argparse.Namespace) -> tuple[bool, dict]:
    records = _load_records(args.metrics_jsonl, args.log_file)
    if not records:
        raise RuntimeError("No training metrics found in metrics JSONL or console log")

    metrics_all_steps = [{"step": int(record["step"]), "data": _numeric_metrics(record)} for record in records]
    metrics_for_compare = [record for record in metrics_all_steps if int(record["step"]) > args.perf_skip_steps]
    summary = _summarize(metrics_for_compare)

    output = {
        "perf_skip_steps": args.perf_skip_steps,
        "metrics_all_steps": metrics_all_steps,
        "metrics_for_compare": metrics_for_compare,
        "summary_for_compare": summary,
        "threshold": args.threshold,
    }

    passed = True
    baseline = args.baseline.expanduser().resolve() if args.baseline else None
    if baseline and baseline.exists():
        with baseline.open("r", encoding="utf-8") as file:
            baseline_payload = json.load(file)
        passed, compare_report = _compare_summary(
            summary,
            baseline_payload.get("summary_for_compare", {}),
            args.threshold,
        )
        output["baseline_compare"] = compare_report
    elif baseline and args.bootstrap_missing:
        baseline.parent.mkdir(parents=True, exist_ok=True)
        with baseline.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, sort_keys=True)
        output["bootstrapped_baseline"] = str(baseline)
    elif baseline:
        raise FileNotFoundError(f"Performance baseline does not exist: {baseline}")

    return passed, output


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Qwen-Image FlowGRPO nightly metrics")
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--perf-skip-steps", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--bootstrap-missing", action="store_true")
    args = parser.parse_args()

    passed, output = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, sort_keys=True)

    if args.baseline and not args.baseline.exists() and args.bootstrap_missing:
        shutil.copy2(args.output, args.baseline)
    if not passed:
        raise SystemExit(f"Performance comparison failed. See {args.output}")
    print(f"Performance metrics collected. Report: {args.output}")


if __name__ == "__main__":
    main()
