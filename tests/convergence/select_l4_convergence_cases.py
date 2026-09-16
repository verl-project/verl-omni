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
"""Select L4 convergence cases for CI and assign the 8-GPU runner size."""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

REQUIRED_GPUS_PER_CASE = 8
RUNNER_SIZE = "L20x8"
CASE_ORDER = ("qwen_image_ocr_lora_v1", "sd35_medium_ocr_lora_v1")


@dataclass(frozen=True)
class ConvergenceCase:
    name: str
    script: str
    required_gpus: int = REQUIRED_GPUS_PER_CASE


CASES = {
    "qwen_image_ocr_lora_v1": ConvergenceCase(
        name="qwen_image_ocr_lora_v1",
        script="tests/convergence/qwen_image_ocr_lora_v1/run_qwen_image_ocr_lora_v1.sh",
    ),
    "sd35_medium_ocr_lora_v1": ConvergenceCase(
        name="sd35_medium_ocr_lora_v1",
        script="tests/convergence/sd35_medium_ocr_lora_v1/run_sd35_medium_ocr_lora_v1.sh",
    ),
}

CASE_PATTERNS = {
    "qwen_image_ocr_lora_v1": (
        "tests/convergence/qwen_image_ocr_lora_v1/**",
        "examples/flowgrpo_trainer/qwen_image/**",
        "verl_omni/pipelines/qwen_image_flow_grpo/**",
        "verl_omni/pipelines/qwen_image_dual_grpo/**",
        "verl_omni/pipelines/qwen_image_edit_flow_grpo/**",
    ),
    "sd35_medium_ocr_lora_v1": (
        "tests/convergence/sd35_medium_ocr_lora_v1/**",
        "examples/flowgrpo_trainer/sd35/**",
        "examples/flowgrpo_trainer/data_process/sd3_ocr.py",
        "verl_omni/pipelines/sd3_flow_grpo/**",
    ),
}

FULL_SUITE_PATTERNS = (
    ".github/actions/gpu-smoke-prepare/**",
    ".github/actions/l4-convergence-upload-artifacts/**",
    ".github/workflows/l4_convergence.yml",
    "pyproject.toml",
    "tests/convergence/select_l4_convergence_cases.py",
    "verl_omni/trainer/config/data/**",
    "verl_omni/trainer/config/optim/**",
    "verl_omni/trainer/config/profiler/**",
    "verl_omni/trainer/diffusion/**",
    "verl_omni/trainer/main_diffusion_v1.py",
    "verl_omni/utils/reward_score/genrm_ocr.py",
)


def normalize_path(path: str) -> str:
    return str(PurePosixPath(path.strip().replace("\\", "/")))


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def all_case_names() -> list[str]:
    return list(CASE_ORDER)


def select_case_names(changed_files: Iterable[str]) -> list[str]:
    selected: set[str] = set()
    normalized_files = [normalize_path(path) for path in changed_files if path.strip()]

    if not normalized_files:
        return all_case_names()

    for path in normalized_files:
        if matches_any(path, FULL_SUITE_PATTERNS):
            return all_case_names()

        matched = False
        for case_name, patterns in CASE_PATTERNS.items():
            if matches_any(path, patterns):
                selected.add(case_name)
                matched = True

        if not matched:
            return all_case_names()

    return [case_name for case_name in CASE_ORDER if case_name in selected] or all_case_names()


def build_plan(case_names: Iterable[str]) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    for case_name in case_names:
        case = CASES[case_name]
        if case.required_gpus != REQUIRED_GPUS_PER_CASE:
            raise ValueError(
                f"L4 case {case_name} requires {case.required_gpus} GPUs; "
                f"CI runner must provide exactly {REQUIRED_GPUS_PER_CASE}."
            )
        cases.append(
            {
                "name": case.name,
                "script": case.script,
                "required_gpus": case.required_gpus,
            }
        )

    if not cases:
        raise ValueError("No L4 convergence cases selected")

    return {
        "cases": cases,
        "case_count": len(cases),
        "required_gpus": REQUIRED_GPUS_PER_CASE,
        "runner_size": RUNNER_SIZE,
    }


def read_changed_files(path: str) -> list[str]:
    with open(path, encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--all", action="store_true", help="select all L4 convergence cases")
    source.add_argument("--cases", nargs="+", choices=CASE_ORDER, help="select explicit L4 cases")
    source.add_argument("--changed-files-file", help="newline-delimited changed file list")
    parser.add_argument("changed_files", nargs="*", help="changed files to classify")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.all:
        case_names = all_case_names()
    elif args.cases:
        case_names = [case_name for case_name in CASE_ORDER if case_name in set(args.cases)]
    else:
        changed_files = read_changed_files(args.changed_files_file) if args.changed_files_file else args.changed_files
        case_names = select_case_names(changed_files)

    try:
        print(json.dumps(build_plan(case_names), separators=(",", ":")))
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
