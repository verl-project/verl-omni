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
"""Check the tiny-only H3 encoder and conditioning width compatibility patch."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

_PATCH_DIR = Path(__file__).parents[1] / "special_e2e" / "minimax_h3_tiny_patch"
_TARGETS = {
    "vllm_omni.diffusion.models.minimax_h3.encoder": "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "vllm_omni.model_executor.models.minimax_h3.conditioning": "MINIMAX_H3_TEXT_HIDDEN_SIZE",
}


@pytest.fixture
def tiny_patch(monkeypatch):
    monkeypatch.delenv("VERL_OMNI_MINIMAX_H3_TINY_TEXT_CONFIG", raising=False)
    spec = importlib.util.spec_from_file_location("tiny_h3_sitecustomize_test", _PATCH_DIR / "sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name, constant", _TARGETS.items())
def test_tiny_patch_updates_both_width_contracts(tiny_patch, name, constant):
    module = ModuleType(name)
    loader = SimpleNamespace(exec_module=lambda target: setattr(target, constant, 5120))
    tiny_patch._PatchLoader(loader, 32).exec_module(module)
    assert getattr(module, constant) == 32


@pytest.mark.parametrize("name", _TARGETS)
def test_tiny_patch_fails_if_upstream_constant_disappears(tiny_patch, name):
    with pytest.raises(RuntimeError, match="update the tiny E2E patch"):
        tiny_patch._PatchLoader(SimpleNamespace(exec_module=lambda target: None), 32).exec_module(ModuleType(name))


def test_tiny_patch_is_inactive_without_explicit_config(tiny_patch, monkeypatch):
    before = list(sys.meta_path)
    tiny_patch._install()
    assert sys.meta_path == before


def test_tiny_patch_real_conditioning_preserves_validation(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"text_config": {"hidden_size": 32}}))
    env = {**os.environ, "VERL_OMNI_MINIMAX_H3_TINY_TEXT_CONFIG": str(config)}
    env["PYTHONPATH"] = os.pathsep.join([str(_PATCH_DIR), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import torch
from vllm_omni.model_executor.models.minimax_h3 import conditioning
from vllm_omni.diffusion.models.minimax_h3 import encoder
assert conditioning.MINIMAX_H3_TEXT_HIDDEN_SIZE == 32
assert encoder.MINIMAX_H3_QWEN3VL_HIDDEN_DIM == 32
payload = {"hidden_states": torch.zeros(2, 32, dtype=torch.bfloat16), "token_tags": torch.zeros(2, dtype=torch.int64)}
conditioning.MiniMaxH3TextConditioning.from_payload(payload)
payload["hidden_states"] = torch.zeros(2, 31, dtype=torch.bfloat16)
try:
    conditioning.MiniMaxH3TextConditioning.from_payload(payload)
except ValueError as exc:
    assert "[tokens, 32]" in str(exc)
else:
    raise AssertionError("width validation was disabled")
print("tiny conditioning validation: PASS")
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "tiny conditioning validation: PASS" in result.stdout
