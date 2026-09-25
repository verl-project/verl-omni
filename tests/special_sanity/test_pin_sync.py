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

The core dependencies carry vllm-omni and verl as
direct-URL git pins, while CI workflows and the ROCm/NPU Dockerfiles install
the same packages from .github/vllm_omni_pin.txt and .github/verl_pin.txt.
A drift between the two sources silently splits the stack; this check makes
it a failure. Also enforces that vllm cpu wheel URLs in CI/RTD carry the
toml's vllm version.
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
            f"{name} has no git pin in pyproject.toml; "
            f"either add it next to the {rel_path} consumer or drop the pin file"
        )
        assert toml_pins[name] == file_pin, (
            f"{name} pin mismatch: pyproject.toml has {toml_pins[name]}, {rel_path} has {file_pin}"
        )


def test_vllm_cpu_wheel_urls_match_toml() -> None:
    """The vllm cpu wheel URLs in CI/RTD must carry the toml's vllm version."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    reqs: list[str] = list(pyproject["project"]["dependencies"])
    for deps in pyproject["project"]["optional-dependencies"].values():
        reqs.extend(deps)
    m = next(r for r in reqs if re.fullmatch(r"vllm==[0-9.]+", r))
    toml_vllm = m.split("==")[1]

    targets = sorted(REPO_ROOT.glob(".github/workflows/*.yml")) + [
        REPO_ROOT / ".readthedocs.yaml",
    ]
    checked = 0
    for path in targets:
        for ver in re.findall(r"vllm-(\d+\.\d+\.\d+)\+cpu", path.read_text()):
            checked += 1
            assert ver == toml_vllm, (
                f"{path.relative_to(REPO_ROOT)} pins vllm-{ver}+cpu but pyproject pins vllm=={toml_vllm}"
            )
    assert checked > 0, "no vllm cpu wheel URLs found to check"
