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
"""Verify diffusion smoke numerical health and step-1 rollout-actor parity."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\r")
_STEP_RE = re.compile(r"\b(?:global_step|step)\s*[:=]\s*(\d+)\b")
_NUMERIC_RE = re.compile(
    r"^\s*(?:(?:np\.\w+|(?:torch\.)?tensor)\()?("
    r"nan|[-+]?inf|[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    r")\)?\s*$",
    re.IGNORECASE,
)
_NP_WRAPPER_RE = re.compile(r"(?:np\.\w+|(?:torch\.)?tensor)\(([^()]+)\)")
_BARE_NONFINITE_RE = re.compile(r"(?<=[:,\s\[])(nan|[-+]?inf)(?=[,\]}\s]|$)", re.IGNORECASE)
_FP32_DTYPE_TOKENS = frozenset({"float32", "torch.float32", "fp32"})
_LATENT_DTYPE_KEYS = ("rollout/latents_dtype", "trajectory_latents_dtype", "rollout/trajectory_latents_dtype")


def normalize_metric_key(key: str) -> str:
    """Collapse repeated leading actor/ prefixes from nested engine metric renames."""
    while key.startswith("actor/actor/"):
        key = key.removeprefix("actor/")
    return key


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        match = _NUMERIC_RE.match(value)
        if match:
            return float(match.group(1))
    return None


def _is_nonfinite_or_null(value: Any) -> bool:
    return value is None or (isinstance(value, float) and not math.isfinite(value))


def _store_metric(target: dict[str, Any], key: str, value: Any) -> None:
    if key in target and _is_nonfinite_or_null(target[key]):
        return
    target[key] = value


def _merge_step_metrics(target: dict[str, Any], payload: Mapping[str, Any]) -> None:
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str):
            continue
        key = normalize_metric_key(raw_key.strip())
        if raw_value is None:
            _store_metric(target, key, None)
            continue
        number = _coerce_number(raw_value)
        if number is not None:
            _store_metric(target, key, number)
        elif isinstance(raw_value, str):
            _store_metric(target, key, raw_value.strip())
        else:
            _store_metric(target, key, raw_value)


def load_jsonl_metrics(path: Path) -> dict[int, dict[str, Any]]:
    """Load per-step metrics from a VERL_FILE_LOGGER_PATH JSONL file."""
    steps: dict[int, dict[str, Any]] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, Mapping):
            raise ValueError(f"Expected JSON object on line {line_no} of {path}.")
        step = record.get("step")
        data = record.get("data")
        if isinstance(step, bool) or not isinstance(step, int) or not isinstance(data, Mapping):
            raise ValueError(f"Expected integer 'step' and mapping 'data' on line {line_no} of {path}.")
        _merge_step_metrics(steps.setdefault(step, {}), data)
    return steps


def _parse_console_dict(raw_dict: str) -> Mapping[str, Any] | None:
    try:
        payload = ast.literal_eval(raw_dict)
    except (SyntaxError, ValueError):
        sanitized = _NP_WRAPPER_RE.sub(r"\1", raw_dict)
        sanitized = _BARE_NONFINITE_RE.sub(lambda m: repr(m.group(1)), sanitized)
        try:
            payload = ast.literal_eval(sanitized)
        except (SyntaxError, ValueError):
            return None
    return payload if isinstance(payload, Mapping) else None


def _parse_console_line(line: str) -> tuple[int, dict[str, Any]] | None:
    step_match = _STEP_RE.search(line)
    if step_match is None:
        return None
    step = int(step_match.group(1))
    if "{" in line and "}" in line:
        start, end = line.index("{"), line.rindex("}") + 1
        payload = _parse_console_dict(line[start:end])
        if payload is not None:
            metrics: dict[str, Any] = {}
            _merge_step_metrics(metrics, payload)
            return (step, metrics) if metrics else None
        return None
    metrics = {}
    suffix = line[step_match.end() :]
    for part in suffix.split(" - "):
        if ":" not in part:
            continue
        raw_key, raw_value = part.split(":", 1)
        key_parts = raw_key.strip().split()
        if len(key_parts) != 1:
            continue
        key = key_parts[0]
        if not key or key in {"step", "global_step"} or key.endswith(".py"):
            continue
        value = raw_value.strip().rstrip(",")
        number = _coerce_number(value)
        _store_metric(metrics, normalize_metric_key(key), number if number is not None else value)
    return (step, metrics) if metrics else None


def load_console_metrics(path: Path) -> tuple[dict[int, dict[str, Any]], bool]:
    """Parse per-step metrics and skipped-optimizer-step markers from a console log."""
    text = _ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    skipped_step = "skipping this optimizer step" in text.lower()
    steps: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        parsed = _parse_console_line(line)
        if parsed is None:
            continue
        step, metrics = parsed
        target = steps.setdefault(step, {})
        for key, value in metrics.items():
            _store_metric(target, key, value)
    return steps, skipped_step


def _require_finite(metrics: Mapping[str, Any], key: str, step: int) -> float:
    normalized = normalize_metric_key(key)
    if normalized not in metrics:
        raise ValueError(f"Missing required metric {normalized!r} at step {step}.")
    value = _coerce_number(metrics[normalized])
    if value is None or not math.isfinite(value):
        raise ValueError(f"Non-finite or non-numeric {normalized!r}={metrics[normalized]!r} at step {step}.")
    return value


def _check_step_metrics(
    step: int,
    metrics: Mapping[str, Any],
    loss_keys: Sequence[str],
    require_positive_grad_norm: bool,
    check_all_losses: bool,
) -> dict[str, float]:
    checked: dict[str, float] = {}
    grad_norm = _require_finite(metrics, "actor/grad_norm", step)
    if require_positive_grad_norm and grad_norm <= 0.0:
        raise ValueError(f"Expected positive actor/grad_norm at step {step}, got {grad_norm}.")
    checked["actor/grad_norm"] = grad_norm

    for key in loss_keys:
        checked[normalize_metric_key(key)] = _require_finite(metrics, key, step)

    if check_all_losses:
        for key in sorted(metrics):
            if key.startswith("actor/") and key.endswith("loss") and key not in checked:
                checked[key] = _require_finite(metrics, key, step)
    return checked


def _check_step1_invariants(
    step1_metrics: Mapping[str, Any],
    step1_ratio_tol: float | None,
    step1_kl_tol: float | None,
    step1_rollout_logprob_diff_tol: float | None,
    step1_rollout_kl_tol: float | None,
    require_fp32_latents: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if step1_ratio_tol is not None:
        ratio_mean = _require_finite(step1_metrics, "actor/ratio_mean", 1)
        summary["actor/ratio_mean"] = ratio_mean
        if abs(ratio_mean - 1.0) > step1_ratio_tol:
            raise ValueError(
                f"Step-1 actor/ratio_mean={ratio_mean:.6g} exceeds tolerance {step1_ratio_tol} around 1.0."
            )
    if step1_kl_tol is not None:
        ppo_kl = _require_finite(step1_metrics, "actor/ppo_kl", 1)
        summary["actor/ppo_kl"] = ppo_kl
        if abs(ppo_kl) > step1_kl_tol:
            raise ValueError(f"Step-1 actor/ppo_kl={ppo_kl:.6g} exceeds tolerance {step1_kl_tol}.")
    if step1_rollout_logprob_diff_tol is not None:
        logprob_diff = _require_finite(step1_metrics, "rollout_corr/logprob_abs_diff_mean", 1)
        summary["rollout_corr/logprob_abs_diff_mean"] = logprob_diff
        if abs(logprob_diff) > step1_rollout_logprob_diff_tol:
            raise ValueError(
                "Step-1 rollout_corr/logprob_abs_diff_mean="
                f"{logprob_diff:.6g} exceeds tolerance {step1_rollout_logprob_diff_tol}."
            )
    if step1_rollout_kl_tol is not None:
        rollout_kl = _require_finite(step1_metrics, "rollout_corr/kl", 1)
        summary["rollout_corr/kl"] = rollout_kl
        if abs(rollout_kl) > step1_rollout_kl_tol:
            raise ValueError(f"Step-1 rollout_corr/kl={rollout_kl:.6g} exceeds tolerance {step1_rollout_kl_tol}.")
    if require_fp32_latents:
        dtype_value = next((step1_metrics[key] for key in _LATENT_DTYPE_KEYS if key in step1_metrics), None)
        if not isinstance(dtype_value, str) or dtype_value.lower() not in _FP32_DTYPE_TOKENS:
            raise ValueError(f"Expected float32 stored rollout latents at step 1, got {dtype_value!r}.")
        summary["latents_dtype"] = dtype_value
    return summary


def check_diffusion_numerical_health(
    *,
    jsonl_path: Path | None = None,
    log_path: Path | None = None,
    steps: Sequence[int] = (1,),
    loss_keys: Sequence[str] = ("actor/loss",),
    step1_ratio_tol: float | None = None,
    step1_kl_tol: float | None = None,
    step1_rollout_logprob_diff_tol: float | None = None,
    step1_rollout_kl_tol: float | None = None,
    require_positive_grad_norm: bool = False,
    check_all_losses: bool = True,
    require_fp32_latents: bool = False,
) -> dict[str, Any]:
    """Validate numerical health across JSONL and/or console training logs."""
    if jsonl_path is None and log_path is None:
        raise ValueError("Provide at least one of jsonl_path or log_path.")
    for name, value in (
        ("step1_ratio_tol", step1_ratio_tol),
        ("step1_kl_tol", step1_kl_tol),
        ("step1_rollout_logprob_diff_tol", step1_rollout_logprob_diff_tol),
        ("step1_rollout_kl_tol", step1_rollout_kl_tol),
    ):
        if value is not None and (not math.isfinite(value) or value < 0.0):
            raise ValueError(f"{name} must be a finite non-negative number, got {value!r}.")

    recorded: dict[int, dict[str, Any]] = {}
    if log_path is not None:
        console_metrics, skipped_step = load_console_metrics(log_path)
        if skipped_step:
            raise ValueError("An optimizer update was skipped (non-finite gradient detected in log).")
        recorded.update(console_metrics)
    if jsonl_path is not None:
        for step, metrics in load_jsonl_metrics(jsonl_path).items():
            target = recorded.setdefault(step, {})
            for key, value in metrics.items():
                _store_metric(target, key, value)

    missing_steps = [step for step in steps if step not in recorded]
    if missing_steps:
        raise ValueError(f"Expected training steps {list(steps)}, missing {missing_steps} (found {sorted(recorded)}).")

    checked_steps: dict[str, dict[str, float]] = {}
    for step in steps:
        checked_steps[str(step)] = _check_step_metrics(
            step,
            recorded[step],
            loss_keys=loss_keys,
            require_positive_grad_norm=require_positive_grad_norm,
            check_all_losses=check_all_losses,
        )

    step1_summary: dict[str, Any] = {}
    if (
        step1_ratio_tol is not None
        or step1_kl_tol is not None
        or step1_rollout_logprob_diff_tol is not None
        or step1_rollout_kl_tol is not None
        or require_fp32_latents
    ):
        if 1 not in recorded:
            raise ValueError("Step 1 metrics are required for step-1 numerical invariant checks.")
        step1_summary = _check_step1_invariants(
            recorded[1],
            step1_ratio_tol=step1_ratio_tol,
            step1_kl_tol=step1_kl_tol,
            step1_rollout_logprob_diff_tol=step1_rollout_logprob_diff_tol,
            step1_rollout_kl_tol=step1_rollout_kl_tol,
            require_fp32_latents=require_fp32_latents,
        )

    return {
        "passed": True,
        "steps": list(steps),
        "checked_steps": checked_steps,
        "step1_invariants": step1_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for L2 diffusion smoke numerical health verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=None, help="VERL_FILE_LOGGER_PATH JSONL metrics file.")
    parser.add_argument("--log", type=Path, default=None, help="Console training log file.")
    parser.add_argument("--steps", type=int, nargs="+", default=[1], help="Training steps that must be present.")
    parser.add_argument("--loss-keys", nargs="*", default=["actor/loss"], help="Required finite loss metric keys.")
    parser.add_argument("--step1-ratio-tol", type=float, default=None, help="Max |actor/ratio_mean - 1.0| at step 1.")
    parser.add_argument("--step1-kl-tol", type=float, default=None, help="Max |actor/ppo_kl| at step 1.")
    parser.add_argument(
        "--step1-rollout-logprob-diff-tol",
        type=float,
        default=None,
        help="Max |rollout_corr/logprob_abs_diff_mean| at step 1.",
    )
    parser.add_argument(
        "--step1-rollout-kl-tol",
        type=float,
        default=None,
        help="Max |rollout_corr/kl| at step 1.",
    )
    parser.add_argument("--require-positive-grad-norm", action="store_true", help="Fail if actor/grad_norm <= 0.")
    parser.add_argument("--require-fp32-latents", action="store_true", help="Fail if stored latents dtype != float32.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional path to write structured summary.")
    args = parser.parse_args(argv)

    summary = check_diffusion_numerical_health(
        jsonl_path=args.jsonl,
        log_path=args.log,
        steps=args.steps,
        loss_keys=args.loss_keys,
        step1_ratio_tol=args.step1_ratio_tol,
        step1_kl_tol=args.step1_kl_tol,
        step1_rollout_logprob_diff_tol=args.step1_rollout_logprob_diff_tol,
        step1_rollout_kl_tol=args.step1_rollout_kl_tol,
        require_positive_grad_norm=args.require_positive_grad_norm,
        require_fp32_latents=args.require_fp32_latents,
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
