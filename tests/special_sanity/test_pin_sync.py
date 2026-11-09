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
toml's vllm version, and that the nightly env pin and the NPU/ROCm Dockerfile
re-statements of pyproject pins stay inside pyproject's own bounds.
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


def _version(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in v.split("."))


def _spec_bounds(spec: str) -> tuple[tuple[int, ...], tuple[int, ...], bool]:
    """Return ``(lower, upper, upper_inclusive)`` for a ``>=X,<=?Y`` requirement."""
    lo = re.search(r">=\s*([0-9.]+)", spec)
    hi = re.search(r"(<=|<)\s*([0-9.]+)", spec)
    assert lo and hi, f"unparsable version spec: {spec}"
    upper_inclusive = hi.group(1) == "<="
    return _version(lo.group(1)), _version(hi.group(2)), upper_inclusive


def test_secondary_pin_surfaces_track_pyproject() -> None:
    """Restatements of pyproject pins (nightly env, NPU/ROCm Dockerfiles) must not drift outside them."""
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    reqs: list[str] = list(pyproject["project"]["dependencies"])
    for deps in pyproject["project"]["optional-dependencies"].values():
        reqs.extend(deps)
    tf_spec = next(r for r in reqs if r.startswith("transformers"))
    kernels_pin = next(r for r in reqs if r.startswith("kernels==")).split("==")[1]
    py_lo, py_hi, py_hi_incl = _spec_bounds(tf_spec)

    nightly = re.search(
        r'NIGHTLY_PIN_TRANSFORMERS:\s*"([0-9.]+)"',
        (REPO_ROOT / ".github/workflows/l3_nightly.yml").read_text(),
    ).group(1)
    v = _version(nightly)
    assert py_lo <= v and (v <= py_hi if py_hi_incl else v < py_hi), (
        f"l3_nightly.yml NIGHTLY_PIN_TRANSFORMERS {nightly} falls outside pyproject's {tf_spec}"
    )

    npu_spec = re.search(
        r"transformers\s*(>=\s*[0-9.]+\s*,\s*<=?\s*[0-9.]+)",
        (REPO_ROOT / "docker/Dockerfile.npu").read_text(),
    ).group(1)
    npu_lo, npu_hi, _ = _spec_bounds(npu_spec)
    assert py_lo <= npu_lo and npu_hi <= py_hi, (
        f"docker/Dockerfile.npu transformers '{npu_spec}' extends outside pyproject's {tf_spec}"
    )

    rocm_kernels = re.search(r"kernels==([0-9.]+)", (REPO_ROOT / "docker/Dockerfile.rocm").read_text()).group(1)
    assert rocm_kernels == kernels_pin, (
        f"docker/Dockerfile.rocm pins kernels=={rocm_kernels} but pyproject pins kernels=={kernels_pin}"
    )


def test_npu_stack_surfaces_agree() -> None:
    """The NPU image, in-job sync, and docs must restate one stack."""
    dockerfile = (REPO_ROOT / "docker/Dockerfile.npu").read_text()
    sync_script = (REPO_ROOT / "tests/npu_smoke/sync_npu_stack.sh").read_text()
    workflow = (REPO_ROOT / ".github/workflows/npu_smoke.yml").read_text()
    doc = (REPO_ROOT / "docs/contributing/npu_smoke_tests.md").read_text()

    torch_npu_docker = re.findall(r"torch-npu==([0-9a-z.]+)", dockerfile)
    torch_npu_sync = re.findall(r"TORCH_NPU_PIN=\"([0-9a-z.]+)\"", sync_script)
    assert torch_npu_docker and set(torch_npu_docker) == set(torch_npu_sync), (
        f"docker/Dockerfile.npu torch-npu pins {sorted(set(torch_npu_docker))} != "
        f"tests/npu_smoke/sync_npu_stack.sh pins {sorted(set(torch_npu_sync))}"
    )

    triton_docker = set(re.findall(r"triton-ascend==([0-9a-z.]+)", dockerfile))
    triton_sync = set(re.findall(r"TRITON_ASCEND_PIN=\"([0-9a-z.]+)\"", sync_script))
    assert triton_docker and triton_docker == triton_sync, (
        f"docker/Dockerfile.npu triton-ascend pins {sorted(triton_docker)} != "
        f"tests/npu_smoke/sync_npu_stack.sh pins {sorted(triton_sync)}"
    )

    # FLA NPU variant differs per SOC lane; the version must not.
    fla_docker = set(re.findall(r"flash-linear-attention-npu-a\d==([0-9.]+)", dockerfile))
    fla_sync = set(re.findall(r"flash-linear-attention-npu-a\d==([0-9.]+)", sync_script))
    assert fla_docker and fla_docker == fla_sync, (
        f"docker/Dockerfile.npu flash-linear-attention-npu versions {sorted(fla_docker)} != "
        f"tests/npu_smoke/sync_npu_stack.sh versions {sorted(fla_sync)}"
    )

    workflow_tag = re.search(r"image:\s*\S*pr-verl-omni-a2:(\S+)", workflow)
    doc_tag = re.search(r"pr-verl-omni-a2:(\S+)", doc)
    assert workflow_tag and doc_tag, "npu smoke image tag missing from workflow or docs"
    assert workflow_tag.group(1) == doc_tag.group(1), (
        f".github/workflows/npu_smoke.yml runs pr-verl-omni-a2:{workflow_tag.group(1)} but "
        f"docs/contributing/npu_smoke_tests.md documents pr-verl-omni-a2:{doc_tag.group(1)}"
    )
