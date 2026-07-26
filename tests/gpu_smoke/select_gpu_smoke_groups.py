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
"""Select GPU smoke groups from changed files and assign CUDA devices."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

MAX_GPUS = 8
GROUP_ORDER = ("ci-core", "ci-e2e-omni", "ci-e2e-diffusion")


@dataclass(frozen=True)
class SmokeGroup:
    name: str
    num_gpus: int
    script: str


GROUPS = {
    "ci-core": SmokeGroup(
        name="ci-core",
        num_gpus=2,
        script="tests/gpu_smoke/run_gpu_smoke_core.sh",
    ),
    "ci-e2e-omni": SmokeGroup(
        name="ci-e2e-omni",
        num_gpus=2,
        script="tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh",
    ),
    "ci-e2e-diffusion": SmokeGroup(
        name="ci-e2e-diffusion",
        num_gpus=4,
        script="tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh",
    ),
}

GROUP_PATTERNS = {
    "ci-core": (
        "tests/agent_loop/**",
        "tests/reward_loop/**",
        "tests/workers/**",
        "tests/gpu_smoke/run_gpu_smoke_core.sh",
        "verl_omni/agent_loop/**",
        "verl_omni/reward_loop/**",
        "verl_omni/workers/**",
    ),
    "ci-e2e-omni": (
        "tests/gpu_smoke/run_gpu_smoke_omni_e2e.sh",
        "tests/special_e2e/*omni*",
        "verl_omni/models/transformers/qwen3_omni_thinker.py",
        "verl_omni/trainer/config/omni/**",
        "verl_omni/trainer/omni/**",
    ),
    "ci-e2e-diffusion": (
        "tests/gpu_smoke/run_gpu_smoke_diffusion_e2e.sh",
        "tests/special_e2e/*diffusion*",
        "tests/special_e2e/*dpo*",
        "tests/special_e2e/*flowgrpo*",
        "tests/special_e2e/*qwen_image*",
        "verl_omni/models/diffusers/**",
        "verl_omni/pipelines/**",
        "verl_omni/trainer/config/diffusion/**",
        "verl_omni/trainer/diffusion/**",
    ),
}

FULL_SUITE_PATTERNS = (
    ".github/actions/gpu-smoke-prepare/**",
    ".github/actions/gpu-smoke-upload-logs/**",
    ".github/workflows/gpu_smoke.yml",
    "pyproject.toml",
    "tests/gpu_smoke/lib_gpu_smoke.sh",
    "tests/gpu_smoke/run_gpu_smoke_tests.sh",
    "tests/gpu_smoke/select_gpu_smoke_groups.py",
    "verl_omni/trainer/config/data/**",
    "verl_omni/trainer/config/optim/**",
    "verl_omni/trainer/config/profiler/**",
    "verl_omni/trainer/main_*.py",
)


def normalize_path(path: str) -> str:
    return str(PurePosixPath(path.strip().replace("\\", "/")))


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def all_group_names() -> list[str]:
    return list(GROUP_ORDER)


def select_group_names(changed_files: Iterable[str]) -> list[str]:
    selected: set[str] = set()
    normalized_files = [normalize_path(path) for path in changed_files if path.strip()]

    if not normalized_files:
        return all_group_names()

    for path in normalized_files:
        if matches_any(path, FULL_SUITE_PATTERNS):
            return all_group_names()

        matched = False
        for group_name, patterns in GROUP_PATTERNS.items():
            if matches_any(path, patterns):
                selected.add(group_name)
                matched = True

        if not matched:
            return all_group_names()

    return [group_name for group_name in GROUP_ORDER if group_name in selected] or all_group_names()


def runner_size_for(total_gpus: int) -> str:
    if total_gpus <= 2:
        return "L20x2"
    if total_gpus <= 4:
        return "L20x4"
    return "L20x8"


def build_plan(group_names: Iterable[str]) -> dict[str, object]:
    offset = 0
    groups: list[dict[str, object]] = []

    for group_name in group_names:
        group = GROUPS[group_name]
        devices = ",".join(str(device_id) for device_id in range(offset, offset + group.num_gpus))
        groups.append(
            {
                "name": group.name,
                "num_gpus": group.num_gpus,
                "devices": devices,
                "script": group.script,
            }
        )
        offset += group.num_gpus

    if offset > MAX_GPUS:
        names = ", ".join(group["name"] for group in groups)
        raise ValueError(f"selected GPU smoke groups require {offset} GPUs, exceeding {MAX_GPUS}: {names}")

    return {
        "groups": groups,
        "group_count": len(groups),
        "total_gpus": offset,
        "runner_size": runner_size_for(offset),
    }


def read_changed_files(path: str) -> list[str]:
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--all", action="store_true", help="select all GPU smoke groups")
    source.add_argument("--groups", nargs="+", choices=GROUP_ORDER, help="select explicit GPU smoke groups")
    source.add_argument("--changed-files-file", help="newline-delimited changed file list")
    parser.add_argument("changed_files", nargs="*", help="changed files to classify")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.all:
        group_names = all_group_names()
    elif args.groups:
        group_names = [group_name for group_name in GROUP_ORDER if group_name in set(args.groups)]
    else:
        changed_files = read_changed_files(args.changed_files_file) if args.changed_files_file else args.changed_files
        group_names = select_group_names(changed_files)

    try:
        print(json.dumps(build_plan(group_names), separators=(",", ":")))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
