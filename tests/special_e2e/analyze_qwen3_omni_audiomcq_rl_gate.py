#!/usr/bin/env python3
"""Validate the scalar contracts of an AudioMCQ RL gate log."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

_METRIC_PATTERN = re.compile(r"(?:^| - )([^:\r\n]+):([^\s]+)")
_VERSION_PATTERN = re.compile(r"_fit_update_weights.*current_param_version:\s*(\d+)")


def _latest_metrics(text: str) -> dict[str, float]:
    metric_lines = [line for line in text.splitlines() if "training/global_step:" in line]
    if not metric_lines:
        raise ValueError("No aggregated training metrics line found")
    metrics: dict[str, float] = {}
    for key, raw_value in _METRIC_PATTERN.findall(metric_lines[-1]):
        try:
            metrics[key.strip()] = float(raw_value)
        except ValueError:
            continue
    return metrics


def _finite_metric(metrics: dict[str, float], key: str) -> float:
    if key not in metrics:
        raise ValueError(f"Required metric is absent: {key}")
    value = metrics[key]
    if not math.isfinite(value):
        raise ValueError(f"Metric is non-finite: {key}={value}")
    return value


def analyze(log_path: Path, *, require_nontrivial_reward: bool = True) -> dict[str, object]:
    text = log_path.read_text(errors="replace")
    metrics = _latest_metrics(text)
    versions = [int(value) for value in _VERSION_PATTERN.findall(text)]

    reward_min = _finite_metric(metrics, "critic/rewards/min")
    reward_max = _finite_metric(metrics, "critic/rewards/max")
    checked = {
        key: _finite_metric(metrics, key)
        for key in (
            "critic/rewards/mean",
            "actor/kl_loss",
            "actor/loss",
            "actor/grad_norm",
            "training/rollout_actor_logprob_pearson_corr",
            "training/rollout_log_probs/finite_fraction",
        )
    }
    reward_is_nontrivial = reward_max > reward_min and reward_max > 0
    if require_nontrivial_reward and not reward_is_nontrivial:
        raise ValueError(f"MCQ reward variance is trivial: min={reward_min}, max={reward_max}")
    if checked["training/rollout_log_probs/finite_fraction"] != 1.0:
        raise ValueError("Not all rollout logprobs are finite")
    if 0 not in versions or 1 not in versions or versions.index(0) > versions.index(1):
        raise ValueError(f"Expected ordered weight versions 0 -> 1, observed {versions}")

    return {
        "status": "PASS",
        "log": str(log_path),
        "weight_versions": versions,
        "reward": {
            "min": reward_min,
            "max": reward_max,
            "variance_nontrivial": reward_is_nontrivial,
            "nontrivial_required": require_nontrivial_reward,
        },
        "metrics": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--allow-zero-reward",
        action="store_true",
        help="Require finite reward metrics but not nonzero variance (pruned smoke only).",
    )
    args = parser.parse_args()

    result = analyze(args.log, require_nontrivial_reward=not args.allow_zero_reward)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
