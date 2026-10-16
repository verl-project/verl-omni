#!/usr/bin/env python3
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
"""Per-step time attribution for the async diffusion V1 trainers (issue #712).

Reads one metrics dict per training step and answers "where did the seconds
go" from the always-on metrics alone -- no trace file needed:

- rollout-bound: the trainer waited for finished groups (sample_wait/*)
- transfer-bound: TransferQueue payload movement dominated (timing_s/tq_get,
  tq/put/*, weight-sync transfer window)
- sync-bound: weight synchronization dominated (weight_sync/phase/*)
- compute-bound: actor update / log-prob / reward compute dominated
- idle: separate_async/trainer_idle_ratio and its decomposition

Input formats (auto-detected per line):

1. JSONL -- one JSON object per line, holding the step's metrics:
   ``{"training/global_step": 12, "timing_s/step": 41.2, ...}``
   Export from wandb/TensorBoard, or emit directly by adding a small
   logger callback that dumps the metrics dict per step.

2. verl console log (``console_tracking``, ``LocalLogger``) -- lines like
   ``step:12 - training/global_step:12 - timing_s/step:41.2 - ...``; the
   numeric ``key:value`` fields are parsed and console ANSI escapes are
   stripped. The step number comes from the ``step:`` prefix.

3. ``step:12 metrics: {'timing_s/step': 41.2, ...}`` -- a metrics dict
   literal after the step prefix; parsed with ``ast.literal_eval``.

Usage:
    python scripts/analyze_step_metrics.py metrics.jsonl
    python scripts/analyze_step_metrics.py trainer_console.log --top 6
    python scripts/analyze_step_metrics.py metrics.jsonl --json   # machine-readable
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from decimal import Decimal, InvalidOperation

# Serialized trainer-step segments, in pipeline order. The wait segments are
# reported separately (they are idle, not work) and the remainder bucket
# catches anything unaccounted.
SERIAL_WORK_SEGMENTS = (
    ("timing_s/reward", "reward"),
    ("timing_s/old_log_prob", "old_log_prob"),
    ("timing_s/ref", "ref"),
    ("timing_s/adv", "adv"),
    ("timing_s/update_actor", "update_actor"),
    ("timing_s/update_critic", "update_critic"),
    ("timing_s/update_weights", "update_weights(serialized part)"),
    ("timing_s/tq_get", "tq_get(trainer reads)"),
)

WAIT_SEGMENTS = (
    ("timing_s/sample_wait/generation_bound", "wait:generation_bound(rollout)"),
    ("timing_s/sample_wait/detection", "wait:detection(poll quantization)"),
    ("timing_s/sample_wait/metadata_sync", "wait:metadata_sync"),
    ("timing_s/sample_wait/eviction", "wait:eviction"),
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_DICT_METRICS_LINE = re.compile(r"^step:(\d+)\s+metrics:\s+(\{.*\})\s*$")
_STEP_PREFIX = re.compile(r"^step:(\d+)\b")


def _parse_console_value(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None


def _to_step(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not number.is_finite() or number != number.to_integral_value():
        return None
    return int(number)


def _parse_line(line: str) -> tuple[int, dict] | None:
    stripped = _ANSI_ESCAPE.sub("", line).strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        step = _to_step(payload.get("training/global_step"))
        return (step, payload) if step is not None else None
    dict_match = _DICT_METRICS_LINE.match(stripped)
    if dict_match:
        try:
            payload = ast.literal_eval(dict_match.group(2))
        except (ValueError, SyntaxError):
            return None
        if not isinstance(payload, dict):
            return None
        step = _to_step(payload.get("training/global_step", dict_match.group(1)))
        return (step, payload) if step is not None else None
    # console_tracking lines (LocalLogger.concat_dict_to_str):
    # `step:12 - key:value - key:value - ...`
    step_match = _STEP_PREFIX.match(stripped)
    if not step_match:
        return None
    payload = {}
    for item in stripped.split(" - ")[1:]:
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        number = _parse_console_value(value.strip())
        if number is not None:
            payload[key.strip()] = number
    return (int(step_match.group(1)), payload) if payload else None


def load_steps(path: str) -> list[tuple[int, dict]]:
    steps = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_line(line)
            if parsed is not None:
                steps.append(parsed)
    return steps


def _number(metrics: dict, key: str) -> float:
    value = metrics.get(key)
    try:
        number = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def attribute_step(metrics: dict, step: int | None = None) -> dict:
    """Build the per-step attribution record from one metrics dict."""
    step_seconds = _number(metrics, "timing_s/step")
    parsed_step = step if step is not None else _number(metrics, "training/global_step")
    attribution = {"step": parsed_step, "step_s": step_seconds}
    waits = {label: _number(metrics, key) for key, label in WAIT_SEGMENTS}
    works = {label: _number(metrics, key) for key, label in SERIAL_WORK_SEGMENTS}
    wait_total = sum(waits.values())
    work_total = sum(works.values())
    other = max(0.0, step_seconds - wait_total - work_total)
    attribution["segments"] = {**waits, **works, "other(residual)": other}
    attribution["wait_s"] = wait_total
    attribution["work_s"] = work_total
    attribution["idle_ratio"] = _number(metrics, "separate_async/trainer_idle_ratio")
    if step_seconds <= 0:
        attribution["bound"] = "unknown(step_s<=0)"
        return attribution
    ranked = sorted(attribution["segments"].items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_seconds = ranked[0]
    attribution["bound"] = f"{top_name} {100.0 * top_seconds / step_seconds:.0f}%"
    attribution["ranked"] = [(name, seconds, seconds / step_seconds) for name, seconds in ranked if seconds > 0]
    # Context that is not step wall time but qualifies the classification.
    context = {}
    put_bytes = _number(metrics, "tq/put/bytes")
    put_seconds = _number(metrics, "tq/put/seconds")
    get_bytes = _number(metrics, "tq/get/bytes")
    get_seconds = _number(metrics, "timing_s/tq_get")
    if put_bytes and put_seconds:
        context["tq_put_GBps"] = put_bytes / put_seconds / 1e9
    if get_bytes and get_seconds:
        context["tq_get_GBps"] = get_bytes / get_seconds / 1e9
    for phase in ("abort", "kv_release", "topology", "transfer_and_finalize", "kv_resume", "generation_resume"):
        seconds = _number(metrics, f"weight_sync/phase/{phase}_s")
        if seconds:
            context[f"sync_{phase}_s"] = seconds
    attribution["context"] = context
    return attribution


def print_report(steps: list[tuple[int, dict]], top: int, as_json: bool) -> None:
    records = [attribute_step(metrics, step) for step, metrics in steps]
    if as_json:
        json.dump(records, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return
    if not records:
        print("no parsable step metrics found", file=sys.stderr)
        raise SystemExit(1)
    print(f"{len(records)} steps parsed\n")
    for record in records:
        if record["step_s"] <= 0:
            print(f"step {int(record['step'])}: no timing_s/step; skipped")
            continue
        print(f"step {int(record['step'])}: {record['step_s']:.1f}s -- {record['bound']}")
        for name, seconds, share in record["ranked"][:top]:
            print(f"    {name:<40s} {seconds:7.2f}s  {100 * share:5.1f}%")
        for key, value in sorted(record["context"].items()):
            print(f"    {key:<40s} {value:7.2f}")
        print()
    # Cross-step summary: mean step time and mean attribution shares.
    usable = [r for r in records if r["step_s"] > 0]
    if not usable:
        return
    mean_step = sum(r["step_s"] for r in usable) / len(usable)
    totals: dict[str, float] = {}
    for record in usable:
        for name, seconds, _share in record.get("ranked", []):
            totals[name] = totals.get(name, 0.0) + seconds
    print(f"mean step time over {len(usable)} steps: {mean_step:.1f}s")
    grand = sum(totals.values())
    for name, seconds in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        print(f"    {name:<40s} {seconds / len(usable):7.2f}s/step  {100 * seconds / grand:5.1f}% of attributed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="metrics JSONL file or verl console log")
    parser.add_argument("--top", type=int, default=6, help="segments to show per step (default 6)")
    parser.add_argument("--json", action="store_true", help="emit per-step attribution records as JSON")
    args = parser.parse_args()
    print_report(load_steps(args.input), args.top, args.json)


if __name__ == "__main__":
    main()
