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

"""Enforce that the git pins in pyproject.toml match the .github/ pin files.

The [gpu]/[train] extras carry vllm-omni and verl as direct-URL git pins,
while CI workflows and the ROCm/NPU Dockerfiles install the same packages
from .github/vllm_omni_pin.txt and .github/verl_pin.txt. A drift between
the two sources silently splits the stack; this check makes it a failure.
"""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PIN_FILES = {
    "vllm-omni": ".github/vllm_omni_pin.txt",
    "verl": ".github/verl_pin.txt",
}


def _toml_git_pins() -> dict[str, str]:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    pins: dict[str, str] = {}
    reqs: list[str] = list(pyproject["project"]["dependencies"])
    for deps in pyproject["project"]["optional-dependencies"].values():
        reqs.extend(deps)
    for dep in reqs:
        m = re.match(r"([A-Za-z0-9_.-]+)\s*@\s*git\+https://[^@]+@([0-9a-f]{40})$", dep)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def test_git_pins_match_pin_files() -> None:
    toml_pins = _toml_git_pins()
    for name, rel_path in PIN_FILES.items():
        file_pin = (REPO_ROOT / rel_path).read_text().strip()
        assert name in toml_pins, (
            f"{name} has no git pin in pyproject.toml optional-dependencies; "
            f"either add it next to the {rel_path} consumer or drop the pin file"
        )
        assert toml_pins[name] == file_pin, (
            f"{name} pin mismatch: pyproject.toml has {toml_pins[name]}, {rel_path} has {file_pin}"
        )
