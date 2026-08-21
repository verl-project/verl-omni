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
"""Compare debug dumps from the Qwen-Image FlowGRPO nightly run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch


def _payload_files(root: Path) -> dict[str, Path]:
    return {str(path.relative_to(root)): path for path in sorted(root.rglob("payload.pt"))}


def _flatten_tensors(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    flat = {}
    if isinstance(value, torch.Tensor):
        flat[prefix or "tensor"] = value.detach().cpu()
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_tensors(item, child))
    elif isinstance(value, (list | tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            flat.update(_flatten_tensors(item, child))
    return flat


def _tensor_metrics(
    reference: torch.Tensor, actual: torch.Tensor, atol: float
) -> dict[str, float | int | list[int] | str]:
    if reference.shape != actual.shape:
        return {
            "shape_mismatch": True,
            "baseline_shape": list(reference.shape),
            "current_shape": list(actual.shape),
        }
    ref = reference.float().reshape(-1)
    cur = actual.float().reshape(-1)
    if ref.numel() == 0:
        return {
            "numel": 0,
            "mean_abs_err": 0.0,
            "rmse": 0.0,
            "p99_abs_err": 0.0,
            "frac_abs_over_atol": 0.0,
            "cos_sim": 1.0,
        }
    diff = (cur - ref).abs()
    mean_abs = diff.mean().item()
    rmse = torch.sqrt(torch.mean((cur - ref).square())).item()
    p99_abs = torch.quantile(diff, 0.99).item()
    frac_abs_over_atol = diff.gt(atol).float().mean().item()
    denom = ref.norm() * cur.norm()
    cos_sim = 1.0 if denom.item() == 0 else torch.dot(ref, cur).div(denom).item()
    return {
        "numel": ref.numel(),
        "mean_abs_err": mean_abs,
        "rmse": rmse,
        "p99_abs_err": p99_abs,
        "frac_abs_over_atol": frac_abs_over_atol,
        "cos_sim": cos_sim,
    }


def _thresholds_for_key(key: str, thresholds: dict[str, Any]) -> dict[str, float]:
    if "default" not in thresholds:
        # Legacy flat report format: {"atol", "min_cos_sim"}.
        atol = float(thresholds.get("atol", 0.0))
        flat = {
            "atol": atol,
            "mean_atol": atol,
            "rmse_atol": atol,
            "p99_atol": atol,
            "max_frac_abs_over_atol": 0.0,
            "min_cos_sim": float(thresholds.get("min_cos_sim", 1.0)),
        }
        return flat
    return thresholds["default"]


def _exceeds_thresholds(metrics: dict[str, Any], thresholds: dict[str, float]) -> bool:
    return (
        metrics["mean_abs_err"] > thresholds["mean_atol"]
        or metrics["rmse"] > thresholds["rmse_atol"]
        or metrics["p99_abs_err"] > thresholds["p99_atol"]
        or metrics["frac_abs_over_atol"] > thresholds["max_frac_abs_over_atol"]
        or metrics["cos_sim"] < thresholds["min_cos_sim"]
    )


def _bootstrap_baseline(current: Path, baseline: Path) -> None:
    baseline.parent.mkdir(parents=True, exist_ok=True)
    if baseline.exists():
        shutil.rmtree(baseline)
    shutil.copytree(current, baseline)


def compare(args: argparse.Namespace) -> tuple[bool, dict]:
    current = args.current.expanduser().resolve()
    baseline = args.baseline.expanduser().resolve()
    if not current.exists():
        raise FileNotFoundError(f"Current dump directory does not exist: {current}")
    if not baseline.exists() or not any(baseline.rglob("payload.pt")):
        if args.bootstrap_missing:
            _bootstrap_baseline(current, baseline)
            return True, {"bootstrapped": True, "baseline": str(baseline)}
        raise FileNotFoundError(f"Baseline dump directory does not exist or has no payloads: {baseline}")

    current_files = _payload_files(current)
    baseline_files = _payload_files(baseline)
    thresholds = {
        "default": {
            "atol": args.atol,
            "mean_atol": args.mean_atol,
            "rmse_atol": args.rmse_atol,
            "p99_atol": args.p99_atol,
            "max_frac_abs_over_atol": args.max_frac_abs_over_atol,
            "min_cos_sim": args.min_cos_sim,
        },
    }
    results = {"files": {}, "missing_in_current": [], "missing_in_baseline": [], "thresholds": thresholds}
    passed = True

    for rel_path in sorted(set(baseline_files) - set(current_files)):
        results["missing_in_current"].append(rel_path)
        passed = False
    for rel_path in sorted(set(current_files) - set(baseline_files)):
        results["missing_in_baseline"].append(rel_path)
        passed = False

    for rel_path in sorted(set(current_files) & set(baseline_files)):
        baseline_payload = torch.load(baseline_files[rel_path], map_location="cpu", weights_only=False)
        current_payload = torch.load(current_files[rel_path], map_location="cpu", weights_only=False)
        baseline_tensors = _flatten_tensors(baseline_payload)
        current_tensors = _flatten_tensors(current_payload)
        file_result = {"tensors": {}, "missing_in_current": [], "missing_in_baseline": []}

        for key in sorted(set(baseline_tensors) - set(current_tensors)):
            file_result["missing_in_current"].append(key)
            passed = False
        for key in sorted(set(current_tensors) - set(baseline_tensors)):
            file_result["missing_in_baseline"].append(key)
            passed = False

        for key in sorted(set(baseline_tensors) & set(current_tensors)):
            key_thresholds = _thresholds_for_key(key, thresholds)
            metrics = _tensor_metrics(
                baseline_tensors[key], current_tensors[key], key_thresholds.get("atol", args.atol)
            )
            metrics["thresholds"] = key_thresholds
            file_result["tensors"][key] = metrics
            if metrics.get("shape_mismatch"):
                passed = False
                continue
            if _exceeds_thresholds(metrics, key_thresholds):
                passed = False

        results["files"][rel_path] = file_result

    results["passed"] = passed
    return passed, results


def _dump_failures(results: dict) -> list[str]:
    thresholds = results.get("thresholds", {})
    failures = []
    for rel_path in results.get("missing_in_current", []):
        failures.append(f"missing current file: {rel_path}")
    for rel_path in results.get("missing_in_baseline", []):
        failures.append(f"missing baseline file: {rel_path}")

    for rel_path, file_result in results.get("files", {}).items():
        for key in file_result.get("missing_in_current", []):
            failures.append(f"missing current tensor: {rel_path}::{key}")
        for key in file_result.get("missing_in_baseline", []):
            failures.append(f"missing baseline tensor: {rel_path}::{key}")
        for key, metrics in file_result.get("tensors", {}).items():
            if metrics.get("shape_mismatch"):
                failures.append(f"shape mismatch: {rel_path}::{key}")
                continue
            key_thresholds = metrics.get("thresholds") or _thresholds_for_key(key, thresholds)
            if _exceeds_thresholds(metrics, key_thresholds):
                failures.append(
                    f"tensor mismatch: {rel_path}::{key} "
                    f"numel={metrics['numel']} "
                    f"mean={metrics['mean_abs_err']:.6g} "
                    f"rmse={metrics['rmse']:.6g} "
                    f"p99={metrics['p99_abs_err']:.6g} "
                    f"frac_abs_over_atol={metrics['frac_abs_over_atol']:.6g} "
                    f"cos={metrics['cos_sim']:.6g}"
                )
    return failures


def _print_conclusion(passed: bool, results: dict, report_path: Path) -> None:
    print("=" * 80)
    if results.get("bootstrapped"):
        print("[DUMP] BASELINE BOOTSTRAPPED")
        print(f"[DUMP] Baseline: {results['baseline']}")
        print(f"[DUMP] Report:   {report_path}")
        print("=" * 80)
        return

    files = results.get("files", {})
    tensor_count = sum(len(file_result.get("tensors", {})) for file_result in files.values())
    failures = _dump_failures(results)
    print(f"[DUMP] DEBUG DUMP COMPARISON: {'PASS' if passed else 'FAIL'}")
    print(f"[DUMP] Compared files: {len(files)}")
    print(f"[DUMP] Compared tensors: {tensor_count}")
    print(f"[DUMP] Failed items: {len(failures)}")
    print(f"[DUMP] Thresholds: {results.get('thresholds', {})}")
    print(f"[DUMP] Report: {report_path}")
    if failures:
        print("[DUMP] First failures:")
        for item in failures[:10]:
            print(f"[DUMP] {item}")
        if len(failures) > 10:
            print(f"[DUMP] ... {len(failures) - 10} more failed items in report")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen-Image FlowGRPO nightly debug dumps")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--mean-atol", type=float, default=1e-4)
    parser.add_argument("--rmse-atol", type=float, default=1e-3)
    parser.add_argument("--p99-atol", type=float, default=2e-3)
    parser.add_argument("--max-frac-abs-over-atol", type=float, default=2e-2)
    parser.add_argument("--min-cos-sim", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rtol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-atol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-mean-atol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-rmse-atol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-p99-atol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-max-frac-abs-over-atol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-min-cos-sim", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--image-rtol", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-missing", action="store_true")
    args = parser.parse_args()

    passed, results = compare(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)

    _print_conclusion(passed, results, args.output)
    if not passed:
        raise SystemExit(f"Debug dump comparison failed. See {args.output}")


if __name__ == "__main__":
    main()
