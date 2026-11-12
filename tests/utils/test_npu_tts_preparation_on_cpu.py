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
"""Exercise offline TTS preparation without model packages or network access."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def preparation_runtime(tmp_path):
    """Run the actual shell helper with stub audio imports and a cached hub API."""
    root = tmp_path / "repo with spaces"
    scripts = root / "tests/npu_smoke"
    scripts.mkdir(parents=True)
    for name in ("prepare_tts_smoke.sh", "requirements_tts_smoke.txt"):
        shutil.copy2(REPO / "tests/npu_smoke" / name, scripts / name)
    (root / "pyproject.toml").write_text(
        '[project]\ndependencies = ["soundfile>=0.12.1"]\n'
        '[project.optional-dependencies]\naudio = ["audioread"]\n'
        'omni = ["librosa>=0.10.2", "onnxruntime>=1.20.0", "sox>=1.5.0"]\n'
    )
    bins = tmp_path / "bin"
    bins.mkdir()
    stub = bins / "python3"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json,os,pathlib,sys,types\n"
        "if sys.argv[1:] != ['-']: raise SystemExit(91)\n"
        "for name in ('librosa','onnxruntime','soundfile','sox','qwen_tts',"
        "'qwen_tts.core','qwen_tts.core.models','qwen_tts.core.models.modeling_qwen3_tts'):\n"
        "    sys.modules[name]=types.ModuleType(name)\n"
        "sys.modules['qwen_tts.core.models.modeling_qwen3_tts'].Qwen3TTSForConditionalGeneration=object\n"
        "hub=types.ModuleType('huggingface_hub')\n"
        "hub.snapshot_download=lambda *args,**kwargs:pathlib.Path(os.environ['CACHE_CAPTURE']).write_text("
        "json.dumps(dict(args=args,kwargs=kwargs)))\n"
        "sys.modules['huggingface_hub']=hub\n"
        "exec(compile(sys.stdin.read(),'<preparation>','exec'))\n"
    )
    stub.chmod(0o755)
    (bins / "sox").write_text("#!/bin/sh\nexit 0\n")
    (bins / "sox").chmod(0o755)
    env = dict(os.environ, PATH=f"{bins}:{os.environ['PATH']}", CACHE_CAPTURE=str(tmp_path / "cache.json"))
    return root, scripts, env


@pytest.mark.parametrize("check_only", [False, True])
def test_preparation_checks_without_installing_packages(preparation_runtime, check_only):
    """Preparation imports prerequisites and downloads only the pinned source."""
    _, scripts, env = preparation_runtime
    command = ["bash", str(scripts / "prepare_tts_smoke.sh")]
    if check_only:
        command.append("--check-only")
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    capture = Path(env["CACHE_CAPTURE"])
    assert capture.exists() != check_only
    if not check_only:
        data = json.loads(capture.read_text())
        assert data["args"] == ["optimum-intel-internal-testing/tiny-random-qwen3-tts"]
        assert data["kwargs"] == {"revision": "6374d605b31381cac6f9577f5e742af2f76ba79c"}


@pytest.mark.parametrize("requirement", ["librosa>=0.12.0", "librosa<0.11"])
def test_project_bounds_reject_incompatible_pins(preparation_runtime, requirement):
    """Changes to project extras cannot silently invalidate the CI audio lock."""
    root, scripts, env = preparation_runtime
    project = root / "pyproject.toml"
    project.write_text(project.read_text().replace("librosa>=0.10.2", requirement))
    result = subprocess.run(
        ["bash", str(scripts / "prepare_tts_smoke.sh")], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0
    assert "conflicts with pyproject.toml" in result.stderr
    assert not Path(env["CACHE_CAPTURE"]).exists()


def test_preparation_requires_exact_ci_pins(preparation_runtime):
    """A mutable audio requirement cannot replace a reproducible CI pin."""
    _, scripts, env = preparation_runtime
    lock = scripts / "requirements_tts_smoke.txt"
    lock.write_text(lock.read_text().replace("librosa==0.11.0", "librosa>=0.11.0"))
    result = subprocess.run(
        ["bash", str(scripts / "prepare_tts_smoke.sh")], env=env, capture_output=True, text=True, timeout=10
    )
    assert result.returncode != 0
    assert "Expected an exact audio smoke pin" in result.stderr
    assert not Path(env["CACHE_CAPTURE"]).exists()
