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
"""Keep e2e reward construction independent of an active core checkpoint build."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "special_e2e"


@pytest.mark.parametrize(
    "wrapper,delegate",
    [
        ("run_flowgrpo_qwen_image_v1_npu.sh", "run_flowgrpo_qwen_image.sh"),
        ("run_online_dpo_qwen_image_npu.sh", "run_online_dpo_qwen_image.sh"),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_reward_launcher_does_not_write_the_core_checkpoint(tmp_path, wrapper, delegate, explicit):
    """Exercise real wrappers while only builder/trainer command boundaries are stubbed."""
    root = tmp_path / "fixture repo"
    scripts = root / "tests/special_e2e"
    scripts.mkdir(parents=True)
    shutil.copyfile(_SCRIPTS / wrapper, scripts / wrapper)
    (scripts / "lib_npu_e2e.sh").write_text('NUM_NPUS=8\nNPU_RUNTIME_ARGS=("trainer.device=npu")\n')
    (root / "tests/__init__.py").touch()
    reward_module = root / "tests/reward_loop"
    reward_module.mkdir()
    (reward_module / "__init__.py").touch()
    (reward_module / "build_qwen25_vl_reward_tiny_random.py").write_text(
        """import json, os
from pathlib import Path
def ensure_tiny_qwen25_vl_reward_checkpoint(path):
    output = Path(path)
    if (output / "core-build-in-progress").exists():
        raise RuntimeError("Core is still writing this checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    (output / "weights.bin").write_bytes(b"tiny reward weights")
    with Path(os.environ["EVENTS"]).open("a") as stream:
        stream.write(json.dumps({"kind": "builder", "path": str(output)}) + "\\n")
    return output
"""
    )
    (scripts / delegate).write_text(
        f"#!/usr/bin/env bash\n\"{sys.executable}\" - <<'PY'\n"
        "import json, os\nfrom pathlib import Path\n"
        "with Path(os.environ['EVENTS']).open('a') as stream:\n"
        "    stream.write(json.dumps({'kind': 'trainer', 'path': os.environ['REWARD_MODEL_PATH']}) + '\\n')\n"
        "PY\n"
    )
    home = tmp_path / "home with spaces"
    core = home / ".cache/verl-omni-npu-smoke/Qwen2.5-VL-Reward-NPU"
    core.mkdir(parents=True)
    marker = core / "core-build-in-progress"
    marker.write_bytes(b"unfinished core-owned weights")
    custom = tmp_path / "existing reward"
    custom.mkdir()
    weights = custom / "weights.bin"
    weights.write_bytes(b"user reward weights")
    events = tmp_path / "events.jsonl"
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"MODEL_PATH", "TOKENIZER_PATH", "QWEN_IMAGE_REWARD_MODEL_PATH", "REWARD_MODEL_PATH"}
    }
    env.update(
        HOME=str(home),
        PYTHONPATH=str(root),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        DATA_DIR=str(tmp_path / "data"),
        EVENTS=str(events),
    )
    if explicit:
        env["QWEN_IMAGE_REWARD_MODEL_PATH"] = str(custom)
    result = subprocess.run(
        ["bash", str(scripts / wrapper)], cwd=root, env=env, capture_output=True, text=True, timeout=20
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_bytes() == b"unfinished core-owned weights"
    assert list(core.iterdir()) == [marker]
    assert weights.read_bytes() == b"user reward weights"
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert [entry["kind"] for entry in recorded] == (["trainer"] if explicit else ["builder", "trainer"])
    if explicit:
        assert recorded[0]["path"] == str(custom)
    else:
        assert recorded[0]["path"] == recorded[1]["path"]
        assert Path(recorded[0]["path"]).is_dir()
        assert Path(recorded[0]["path"]) != core
