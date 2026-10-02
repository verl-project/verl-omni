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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

# Tensors that are measured and reported but never allowed to fail the run.
# `batch.responses` is uint8 rollout pixel output and rollout sampling is not
# bit-exact, so pixel-level differences are expected. The reward-path tensors
# derive from that image through the OCR reward, so vLLM-Omni request packing
# (`max_num_seqs` > 1) flips individual scores by ~1/255 pixel drift. A shape
# mismatch on any of them still fails.
INFORMATIONAL_TENSORS = frozenset(
    {
        "batch.responses",
        "batch.advantages",
        "batch.sample_level_rewards",
        "batch.sample_level_scores",
    }
)


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


def _row_group_keys(payload: Any) -> list[int] | None:
    """Return the run-stable row group key for a payload, or None if unavailable.

    The training batch is ordered by ``uid``, which is a fresh ``uuid4()`` per
    run, so the same prompt lands at a different offset every run. ``extra_info``
    carries the dataset position (``repeat_index``), which is stable across runs
    and therefore lets two independent runs be compared row-for-row.
    """
    non_tensor = payload.get("non_tensor") or {}
    extra = non_tensor.get("extra_info")
    if extra is None:
        return None
    try:
        rows = list(extra)
    except TypeError:
        return None
    keys: list[int] = []
    for row in rows:
        if isinstance(row, Mapping):
            value = row.get("repeat_index")
        else:
            value = getattr(row, "repeat_index", None)
        if value is None:
            return None
        try:
            keys.append(int(value))
        except (TypeError, ValueError):
            return None
    return keys or None


def _sort_rows(keys: list[int]) -> list[int]:
    # Stable sort keeps the per-prompt session order (already ascending) intact.
    return sorted(range(len(keys)), key=lambda i: keys[i])


def _apply_row_order(payload: Any, order: list[int]) -> Any:
    aligned = dict(payload)
    batch = payload.get("batch") or {}
    aligned["batch"] = {
        key: (
            value[order]
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == len(order)
            else value
        )
        for key, value in batch.items()
    }
    return aligned


def _align_rows(baseline_payload: Any, current_payload: Any) -> tuple[Any, Any, str | None]:
    """Reorder both payloads onto their shared run-stable row order."""
    baseline_keys = _row_group_keys(baseline_payload)
    current_keys = _row_group_keys(current_payload)
    if baseline_keys is None or current_keys is None:
        return baseline_payload, current_payload, None
    if sorted(baseline_keys) != sorted(current_keys):
        return baseline_payload, current_payload, None
    return (
        _apply_row_order(baseline_payload, _sort_rows(baseline_keys)),
        _apply_row_order(current_payload, _sort_rows(current_keys)),
        "extra_info.repeat_index",
    )


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

    unaligned: list[str] = []
    for rel_path in sorted(set(current_files) & set(baseline_files)):
        baseline_payload = torch.load(baseline_files[rel_path], map_location="cpu", weights_only=False)
        current_payload = torch.load(current_files[rel_path], map_location="cpu", weights_only=False)
        baseline_payload, current_payload, row_alignment = _align_rows(baseline_payload, current_payload)
        if row_alignment is None:
            unaligned.append(rel_path)
        baseline_tensors = _flatten_tensors(baseline_payload)
        current_tensors = _flatten_tensors(current_payload)
        file_result = {
            "tensors": {},
            "missing_in_current": [],
            "missing_in_baseline": [],
            "row_alignment": row_alignment,
        }

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
            informational = key in INFORMATIONAL_TENSORS
            if informational:
                metrics["informational"] = True
            file_result["tensors"][key] = metrics
            if metrics.get("shape_mismatch"):
                passed = False
                continue
            if informational:
                continue
            if _exceeds_thresholds(metrics, key_thresholds):
                passed = False

        results["files"][rel_path] = file_result

    results["unaligned_files"] = unaligned
    results["informational_tensors"] = sorted(
        {
            key
            for file_result in results["files"].values()
            for key, metrics in file_result.get("tensors", {}).items()
            if metrics.get("informational")
        }
    )
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
            if metrics.get("informational"):
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
    informational = results.get("informational_tensors") or []
    if informational:
        print(f"[DUMP] Informational tensors (measured, never fail): {len(informational)} {informational}")
    print(f"[DUMP] Thresholds: {results.get('thresholds', {})}")
    unaligned = results.get("unaligned_files") or []
    if unaligned:
        print(
            "[DUMP] WARNING: row order not realigned for "
            f"{len(unaligned)} file(s) (missing/differing extra_info.repeat_index): "
            f"{unaligned[:3]}"
        )
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
    parser.add_argument("--min-cos-sim", type=float, default=0.99)
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
