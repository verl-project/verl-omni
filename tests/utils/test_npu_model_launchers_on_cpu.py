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
"""Check NPU wrappers at their GPU-launcher boundary without model execution."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CASES = [
    ("run_qwen3_tts_grpo_npu.sh", "run_qwen3_tts_grpo_smoke.sh"),
    ("run_crd_sd35_npu.sh", "run_crd_sd35.sh"),
]


@pytest.fixture
def launcher_runtime(tmp_path):
    """Stub device preflight and delegated workloads, keeping wrapper logic real."""
    root = tmp_path / "repo with spaces"
    scripts = root / "tests/special_e2e"
    scripts.mkdir(parents=True)
    for name in {case[0] for case in CASES} | {"lib_npu_e2e.sh"}:
        shutil.copy2(REPO / "tests/special_e2e" / name, scripts / name)
    for name in {case[1] for case in CASES}:
        (scripts / name).write_text(
            f"#!/bin/bash\nexec {sys.executable} - \"$@\" <<'PY'\n"
            "import json,os,pathlib,sys\n"
            "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps(dict(argv=sys.argv[1:],env=dict(os.environ))))\n"
            "raise SystemExit(int(os.environ['RAW_EXIT']))\nPY\n"
        )
    bins = tmp_path / "bin"
    bins.mkdir()
    python = bins / "python3"
    python.write_text(
        f"#!{sys.executable}\nimport sys\n"
        "if sys.argv[1:]==['-'] or (len(sys.argv)>1 and sys.argv[1]=='-'):\n"
        "    source=sys.stdin.read()\n"
        "    if 'import torch' not in source:\n"
        "        sys.argv=sys.argv[1:]\n"
        "        exec(compile(source,'<wrapper>','exec'))\n"
    )
    python.chmod(0o755)
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("ASCEND_", "RAY_", "NPU_", "NUM_", "MODEL_", "DATA_", "TOKENIZER_", "QWEN3_", "BOOGU_", "CRD_")
        )
    }
    env.update(PATH=f"{bins}:{env['PATH']}", NUM_NPUS="8", CAPTURE=str(tmp_path / "capture.json"))
    # Skip only the host-specific CANN setup, not lib_npu_e2e.sh itself.
    command = 'source() { case "$1" in */set_env.sh) return 0;; *) builtin source "$@";; esac; }; '
    command += 'export -f source; exec bash "$@"'
    return root, scripts, env, command


@pytest.mark.parametrize("script,base", CASES)
@pytest.mark.parametrize("raw_exit", [0, 7, 134])
def test_wrapper_preserves_delegation_paths_and_raw_exit(launcher_runtime, script, base, raw_exit):
    """Wrappers preserve caller overrides and every nonzero workload status."""
    root, scripts, env, command = launcher_runtime
    model = root / "owned model"
    model.mkdir()
    (model / "model_index.json").write_text("{}")
    env.update(
        RAW_EXIT=str(raw_exit),
        CRD_MODEL_PATH=str(model),
        WORK_DIR=str(root / "tts work"),
        RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="",
        RAY_MASTER_PORT_RANGE="30000,30100",
    )
    result = subprocess.run(
        ["bash", "-c", command, "fixture", str(scripts / script), "trainer.logger=console"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == raw_exit, result.stdout + result.stderr
    captured = json.loads(Path(env["CAPTURE"]).read_text())
    assert captured["argv"][-1] == "trainer.logger=console"
    assert "trainer.device=npu" in captured["argv"]
    assert captured["env"]["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert captured["env"]["NUM_GPUS"] == "8"
    if script != "run_qwen3_tts_grpo_npu.sh":
        assert captured["env"]["MODEL_PATH"] == str(model)


def test_explicit_missing_checkpoint_fails_before_delegation(launcher_runtime):
    """A typo in an explicit checkpoint path must not silently build another model."""
    root, scripts, env, command = launcher_runtime
    env.update(CRD_MODEL_PATH=str(root / "missing"), RAW_EXIT="0")
    result = subprocess.run(
        ["bash", "-c", command, "fixture", str(scripts / "run_crd_sd35_npu.sh")],
        env=env,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert not Path(env["CAPTURE"]).exists()


@pytest.mark.parametrize("script,base", CASES)
@pytest.mark.parametrize("count", ["0", "3", "invalid"])
def test_invalid_topology_fails_before_delegation(launcher_runtime, script, base, count):
    """Invalid device counts fail before building a model or starting the trainer."""
    _, scripts, env, command = launcher_runtime
    env.update(NUM_NPUS=count, RAW_EXIT="0")
    result = subprocess.run(
        ["bash", "-c", command, "fixture", str(scripts / script)], env=env, capture_output=True, timeout=10
    )
    assert result.returncode == 2
    assert not Path(env["CAPTURE"]).exists()
