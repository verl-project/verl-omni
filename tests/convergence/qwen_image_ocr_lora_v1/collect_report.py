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
"""L4 gates: train-infer gap, finite grad_norm, and a 100-step val-reward floor.

No baseline compare. Timing metrics are recorded only.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_NUMERIC_VALUE_RE = re.compile(
    r"^\s*(?:np\.\w+\()?("
    r"nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r")\)?"
)

# VeRL AR name, then the diffusion analogue logged by calculate_log_probs.
ROLLOUT_PROB_DIFF_MEAN_KEYS = (
    "training/rollout_probs_diff_mean",
    "rollout_corr/logprob_abs_diff_mean",
)
GRAD_NORM_KEY = "actor/grad_norm"
# v1 Metric aggregation logs ``actor/loss/mean``; older paths may log ``actor/loss``.
TRAIN_LOSS_KEYS = (
    "actor/loss/mean",
    "actor/loss",
    "actor/pg_loss",
    "actor/total_loss",
)
PERF_RECORD_KEYS = (
    "perf/time_per_step",
    "timing_s/step",
    "timing_s/gen",
    "timing_s/old_log_prob",
    "timing_s/reward",
    "timing_s/update_actor",
    "perf/throughput",
)

DEFAULT_SKIP_STEPS = 2
DEFAULT_MIN_TRAIN_STEPS = 100
DEFAULT_TEST_FREQ = 20
DEFAULT_ROLLOUT_PROB_DIFF_MEAN_MAX = 0.01
DEFAULT_VAL_REWARD_MIN = 0.9


@dataclass(frozen=True)
class PrecisionThresholds:
    skip_steps: int = DEFAULT_SKIP_STEPS
    min_train_steps: int = DEFAULT_MIN_TRAIN_STEPS
    rollout_prob_diff_mean_max: float = DEFAULT_ROLLOUT_PROB_DIFF_MEAN_MAX
    val_reward_min: float = DEFAULT_VAL_REWARD_MIN


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int | float)):
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
        return float("nan")
    if raw in {"inf", "+inf"}:
        return float("inf")
    if raw == "-inf":
        return float("-inf")
    return _as_float(float(raw))


def read_jsonl(path: Path) -> list[dict]:
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


def read_console_log(path: Path) -> list[dict]:
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
                records.append({"step": int(match.group(1)), "data": payload})
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


def load_records(metrics_jsonl: Path | None, log_file: Path | None) -> list[dict]:
    records = read_jsonl(metrics_jsonl) if metrics_jsonl else []
    if not records and log_file:
        records = read_console_log(log_file)
    normalized = []
    for record in records:
        step = int(record.get("step", record.get("data", {}).get("training/global_step", -1)))
        normalized.append({"step": step, "data": _numeric_metrics(record)})
    return sorted(normalized, key=lambda item: item["step"])


def train_records(records: list[dict], skip_steps: int) -> list[dict]:
    selected = []
    for record in records:
        if int(record["step"]) <= skip_steps:
            continue
        data = record["data"]
        if GRAD_NORM_KEY in data or any(key in data for key in ROLLOUT_PROB_DIFF_MEAN_KEYS):
            selected.append(record)
    return selected


def resolve_train_loss_key(records: list[dict]) -> str:
    for key in TRAIN_LOSS_KEYS:
        if any(key in record["data"] for record in records):
            return key
    return TRAIN_LOSS_KEYS[0]


def expected_val_steps(test_freq: int, total_steps: int) -> list[int]:
    if test_freq <= 0 or total_steps <= 0:
        return []
    steps = list(range(test_freq, total_steps + 1, test_freq))
    if not steps or steps[-1] != total_steps:
        steps.append(total_steps)
    return steps


def collect_curve_metrics(
    records: list[dict],
    *,
    test_freq: int,
    total_steps: int,
) -> dict[str, Any]:
    """Per-step train loss and val reward at each ``test_freq`` (and last step)."""
    loss_key = resolve_train_loss_key(records)
    train_loss = []
    val_reward = []
    for record in records:
        step = int(record["step"])
        data = record["data"]
        if loss_key in data:
            train_loss.append({"step": step, "value": data[loss_key]})
        items = val_reward_items(data)
        if items:
            val_reward.append({"step": step, "values": items})
    return {
        "test_freq": test_freq,
        "total_steps": total_steps,
        "train_loss_key": loss_key,
        "train_loss": train_loss,
        "val_reward": val_reward,
        "expected_val_steps": expected_val_steps(test_freq, total_steps),
    }


def resolve_rollout_prob_diff_key(records: list[dict]) -> str:
    for key in ROLLOUT_PROB_DIFF_MEAN_KEYS:
        if any(key in record["data"] for record in records):
            return key
    return ROLLOUT_PROB_DIFF_MEAN_KEYS[0]


def val_reward_items(data: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in data.items() if key.startswith("val-core/") and "/reward/mean@" in key}


def _series(records: list[dict], key: str) -> list[float]:
    return [record["data"][key] for record in records if key in record["data"]]


def _summarize(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    summary = {
        "count": float(len(values)),
        "finite_count": float(len(finite)),
        "non_finite_count": float(len(values) - len(finite)),
    }
    if finite:
        summary["min"] = min(finite)
        summary["max"] = max(finite)
        summary["mean"] = sum(finite) / len(finite)
    return summary


def _fail(report: dict, key: str, **payload: Any) -> None:
    report[key] = {"failed": True, **payload}


def _pass(report: dict, key: str, **payload: Any) -> None:
    report[key] = {"failed": False, **payload}


def _gate(report: dict, key: str, ok: bool, **payload: Any) -> bool:
    if ok:
        _pass(report, key, **payload)
    else:
        _fail(report, key, **payload)
    return ok


def collect_perf_report(records: list[dict]) -> dict[str, Any]:
    """Summarize timing/throughput. Never used as a pass/fail signal."""
    report = {}
    for key in PERF_RECORD_KEYS:
        values = _series(records, key)
        if values:
            report[key] = {"informational": True, "summary": _summarize(values)}
    return report


def evaluate_gates(
    actor_records: list[dict],
    all_records: list[dict],
    thresholds: PrecisionThresholds,
    *,
    logged_actor_steps: int,
) -> tuple[bool, dict]:
    report: dict[str, Any] = {}
    passed = True

    if logged_actor_steps < thresholds.min_train_steps:
        _fail(
            report,
            "min_train_steps",
            current=logged_actor_steps,
            required=thresholds.min_train_steps,
        )
        passed = False
    else:
        _pass(report, "min_train_steps", current=logged_actor_steps, required=thresholds.min_train_steps)

    if not actor_records:
        _fail(report, "compared_steps", current=0, reason="no_post_warmup_actor_steps")
        return False, report
    _pass(report, "compared_steps", current=len(actor_records))

    diff_key = resolve_rollout_prob_diff_key(actor_records)
    present = sum(1 for record in actor_records if diff_key in record["data"])
    if present < len(actor_records):
        _fail(
            report,
            "missing/rollout_prob_diff_mean",
            metric_key=diff_key,
            present=present,
            required=len(actor_records),
            reason="enable actor_rollout_ref.rollout.calculate_log_probs=true and keep bypass_mode=false",
        )
        passed = False
    else:
        _pass(report, "missing/rollout_prob_diff_mean", metric_key=diff_key, present=present)

    diff_values = _series(actor_records, diff_key)
    diff_summary = _summarize(diff_values)
    # "cannot exceed 0.01": every post-warmup step, not only the mean.
    diff_ok = (
        diff_summary.get("finite_count", 0) > 0
        and diff_summary.get("non_finite_count", 1) == 0
        and diff_summary.get("max", float("inf")) <= thresholds.rollout_prob_diff_mean_max
    )
    passed = (
        _gate(
            report,
            "rollout_prob_diff_mean",
            diff_ok,
            metric_key=diff_key,
            summary=diff_summary,
            cap=thresholds.rollout_prob_diff_mean_max,
        )
        and passed
    )

    grad_present = sum(1 for record in actor_records if GRAD_NORM_KEY in record["data"])
    if grad_present == 0:
        _fail(report, f"missing/{GRAD_NORM_KEY}", present=0)
        passed = False
    grad_values = _series(actor_records, GRAD_NORM_KEY)
    grad_summary = _summarize(grad_values)
    grad_ok = grad_summary.get("finite_count", 0) > 0 and grad_summary.get("non_finite_count", 1) == 0
    passed = (
        _gate(
            report,
            GRAD_NORM_KEY,
            grad_ok,
            summary=grad_summary,
            reason=None if grad_ok else "non_finite_grad_norm",
        )
        and passed
    )

    val_step = None
    val_items: dict[str, float] = {}
    for record in reversed(all_records):
        items = val_reward_items(record["data"])
        if items:
            val_step = int(record["step"])
            val_items = items
            break
    expected_val_step = thresholds.min_train_steps
    if not val_items:
        _fail(
            report,
            "val_reward",
            reason="missing_val-core_reward_mean",
            expected_step=expected_val_step,
        )
        passed = False
    else:
        finite_vals = {key: value for key, value in val_items.items() if math.isfinite(value)}
        worst = min(finite_vals.values()) if finite_vals else float("-inf")
        val_ok = (
            val_step == expected_val_step and len(finite_vals) == len(val_items) and worst >= thresholds.val_reward_min
        )
        passed = (
            _gate(
                report,
                "val_reward",
                val_ok,
                step=val_step,
                expected_step=expected_val_step,
                values=val_items,
                floor=thresholds.val_reward_min,
            )
            and passed
        )

    return passed, report


def collect(args: argparse.Namespace) -> tuple[bool, dict, dict]:
    thresholds = PrecisionThresholds(
        skip_steps=args.skip_steps,
        min_train_steps=args.min_train_steps,
        rollout_prob_diff_mean_max=args.rollout_prob_diff_mean_max,
        val_reward_min=args.val_reward_min,
    )
    records = load_records(args.metrics_jsonl, args.log_file)
    if not records:
        raise RuntimeError("No training metrics found in metrics JSONL or console log")

    all_actor = train_records(records, skip_steps=0)
    selected = train_records(records, thresholds.skip_steps)
    passed, gates = evaluate_gates(
        selected,
        records,
        thresholds,
        logged_actor_steps=len(all_actor),
    )
    output: dict[str, Any] = {
        "thresholds": asdict(thresholds),
        "num_logged_steps": len(records),
        "num_actor_steps": len(all_actor),
        "num_train_steps_compared": len(selected),
        "steps": [record["step"] for record in selected],
        "gates": gates,
        "perf": collect_perf_report(selected),
        "passed": passed,
    }
    curves = collect_curve_metrics(
        records,
        test_freq=int(getattr(args, "test_freq", DEFAULT_TEST_FREQ)),
        total_steps=thresholds.min_train_steps,
    )
    return passed, output, curves


def _print_conclusion(passed: bool, output: dict, report_path: Path) -> None:
    print("=" * 80)
    print(f"[L4] PRECISION GATES: {'PASS' if passed else 'FAIL'}")
    print(f"[L4] Compared actor steps: {output.get('num_train_steps_compared')}")
    print(f"[L4] Report: {report_path}")
    for key, stats in (output.get("gates") or {}).items():
        if stats.get("failed"):
            print(f"[L4] FAIL {key}: {json.dumps(stats, sort_keys=True, default=str)}")
    perf = output.get("perf") or {}
    time_stats = (perf.get("perf/time_per_step") or {}).get("summary") or {}
    if time_stats:
        print(
            "[L4] PERF (record only) perf/time_per_step: "
            f"mean={time_stats.get('mean')}, min={time_stats.get('min')}, max={time_stats.get('max')}"
        )
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect L4 report.json gates and metrics.json curves")
    parser.add_argument("--metrics-jsonl", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        help="Write per-step train loss and per-test_freq val reward here",
    )
    parser.add_argument("--skip-steps", type=int, default=DEFAULT_SKIP_STEPS)
    parser.add_argument("--min-train-steps", type=int, default=DEFAULT_MIN_TRAIN_STEPS)
    parser.add_argument("--test-freq", type=int, default=DEFAULT_TEST_FREQ)
    parser.add_argument("--rollout-prob-diff-mean-max", type=float, default=DEFAULT_ROLLOUT_PROB_DIFF_MEAN_MAX)
    parser.add_argument("--val-reward-min", type=float, default=DEFAULT_VAL_REWARD_MIN)
    args = parser.parse_args()

    passed, output, curves = collect(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, sort_keys=True)
    if args.metrics_json:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_json.open("w", encoding="utf-8") as file:
            json.dump(curves, file, indent=2, sort_keys=True)

    _print_conclusion(passed, output, args.output)
    if not passed:
        raise SystemExit(f"L4 precision check failed. See {args.output}")


if __name__ == "__main__":
    main()
