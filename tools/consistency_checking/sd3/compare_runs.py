#!/usr/bin/env python
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

"""Compare two SD3.5 first-forward capture batches for replay equality.

Tensors compare bitwise by default; pass ``--atol KEY=VALUE`` for tensors that
are allowed to drift within a declared tolerance (for example rewards).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

BATCH_NAME = "actor_batch.pt"


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / BATCH_NAME
    if not candidate.is_file():
        raise FileNotFoundError(f"capture batch not found: {candidate}")
    return candidate


def _load(path: Path) -> dict[str, torch.Tensor]:
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(batch, dict):
        raise TypeError(f"{path} must hold a tensor dict, got {type(batch).__name__}")
    return batch


def _parse_atol(entries: list[str]) -> dict[str, float]:
    tolerances: dict[str, float] = {}
    for entry in entries:
        key, sep, value = entry.partition("=")
        if not sep or not key:
            raise ValueError(f"--atol expects KEY=VALUE, got {entry!r}")
        tolerances[key] = float(value)
    return tolerances


def compare(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor], atol: dict[str, float]) -> dict:
    """Compare two capture batches and return a JSON-ready report."""
    report: dict = {"equal": True, "mismatched": [], "missing": [], "tensors": {}}
    only_first = sorted(set(first) - set(second))
    only_second = sorted(set(second) - set(first))
    if only_first or only_second:
        report["missing"] = [*only_first, *only_second]
    for key in sorted(set(first) & set(second)):
        value_a, value_b = first[key], second[key]
        if not isinstance(value_a, torch.Tensor) or not isinstance(value_b, torch.Tensor):
            raise TypeError(f"{key} must be a tensor in both captures")
        tolerance = atol.get(key, 0.0)
        if value_a.shape != value_b.shape:
            entry = {"max_abs_diff": None, "atol": tolerance, "equal": False}
        else:
            max_abs = float((value_a - value_b).abs().max()) if value_a.numel() else 0.0
            if tolerance == 0:
                equal = torch.equal(value_a, value_b)
            else:
                equal = torch.allclose(value_a, value_b, rtol=0.0, atol=tolerance)
            entry = {"max_abs_diff": max_abs, "atol": tolerance, "equal": bool(equal)}
        report["tensors"][key] = entry
        if not entry["equal"]:
            report["mismatched"].append(key)
    report["equal"] = not report["mismatched"] and not report["missing"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", help="run directory or actor_batch.pt of the first run")
    parser.add_argument("second", help="run directory or actor_batch.pt of the second run")
    parser.add_argument(
        "--atol",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="absolute tolerance for one tensor key; repeatable",
    )
    args = parser.parse_args()
    first = _load(_resolve(args.first))
    second = _load(_resolve(args.second))
    tolerances = _parse_atol(args.atol)
    unknown = sorted(set(tolerances) - (set(first) & set(second)))
    if unknown:
        raise ValueError(f"--atol keys not present in both captures: {unknown}")
    report = compare(first, second, tolerances)
    print(json.dumps(report, indent=2))
    return 0 if report["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
