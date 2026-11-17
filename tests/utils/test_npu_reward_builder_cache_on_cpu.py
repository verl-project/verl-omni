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
"""Validate cached reward fixtures without loading model frameworks."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("state", ["complete", "partial", "wrong_seed", "unowned", "symlink"])
def test_cached_reward_validation_needs_no_framework_import(tmp_path, state):
    """Reuse or reject existing assets without loading torch/Transformers just to inspect files."""
    checkpoint = tmp_path / "reward checkpoint"
    checkpoint.mkdir()
    for name in (
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
    ):
        (checkpoint / name).write_text("{}")
    marker = checkpoint / "tiny_reward_checkpoint_metadata.json"
    marker.write_text(json.dumps({"builder": "qwen25_vl_reward_npu", "format_version": 1, "seed": 42}))
    if state == "partial":
        (checkpoint / "model.safetensors").unlink()
    elif state == "wrong_seed":
        marker.write_text(marker.read_text().replace("42", "43"))
    elif state == "unowned":
        marker.unlink()
    elif state == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(checkpoint, target_is_directory=True)
        checkpoint = alias
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from tests.reward_loop.build_qwen25_vl_reward_tiny_random "
            "import ensure_tiny_qwen25_vl_reward_checkpoint as ensure; "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules; ensure(sys.argv[1]); "
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules",
            str(checkpoint),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == (0 if state == "complete" else 1), result.stdout + result.stderr
    if state != "complete":
        assert "ValueError" in result.stderr
        assert "ModuleNotFoundError" not in result.stderr
