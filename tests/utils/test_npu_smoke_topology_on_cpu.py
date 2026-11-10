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
"""Keep inference smoke small without reducing training ranks or masking failures."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.utils.npu_test_topology import inference_npu_count


@pytest.mark.parametrize("available,requested", [(2, "2"), (4, "4"), (8, "8"), (8, "6"), (8, None)])
@pytest.mark.parametrize("full", ["0", "1"])
def test_inference_allocation_preserves_explicit_full_topology(monkeypatch, available, requested, full):
    """A smoke uses TP=2; an explicit full run consumes the validated allocation."""
    monkeypatch.delenv("NUM_NPUS", raising=False)
    if requested is not None:
        monkeypatch.setenv("NUM_NPUS", requested)
    monkeypatch.setenv("NPU_SMOKE_FULL_TOPOLOGY", full)
    assert inference_npu_count(available) == (int(requested or available) if full == "1" else 2)


@pytest.mark.parametrize(
    "available,requested,full",
    [(8, "0", "0"), (8, "1", "0"), (8, "3", "0"), (8, "10", "0"), (0, "2", "0"), (8, "bad", "0"), (8, "8", "yes")],
)
def test_invalid_allocation_fails_before_selecting_a_smaller_smoke(monkeypatch, available, requested, full):
    """Capping service replicas must not hide a missing or invalid allocation."""
    monkeypatch.setenv("NUM_NPUS", requested)
    monkeypatch.setenv("NPU_SMOKE_FULL_TOPOLOGY", full)
    with pytest.raises(ValueError):
        inference_npu_count(available)


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


@pytest.mark.parametrize(
    "wrapper,delegate",
    [
        ("run_flowgrpo_qwen_image_v1_npu.sh", "run_flowgrpo_qwen_image.sh"),
        ("run_online_dpo_qwen_image_npu.sh", "run_online_dpo_qwen_image.sh"),
        ("run_diffusionnft_qwen_image_npu.sh", "run_diffusionnft_qwen_image.sh"),
        ("run_flowgrpo_qwen_image_edit_npu.sh", "run_flowgrpo_qwen_image_edit.sh"),
    ],
)
@pytest.mark.parametrize("tp,raw_exit", [(None, 0), ("1", 0), (None, 134)])
def test_qwen_wrappers_keep_training_ranks_and_raw_exits(tmp_path, wrapper, delegate, tp, raw_exit):
    """Only inference TP defaults change; caller overrides and trainer failures survive delegation."""
    source = Path(__file__).resolve().parents[1] / "special_e2e"
    scripts = tmp_path / "repo/tests/special_e2e"
    scripts.mkdir(parents=True)
    shutil.copyfile(source / wrapper, scripts / wrapper)
    (scripts / "lib_npu_e2e.sh").write_text('NUM_NPUS=8\nNPU_RUNTIME_ARGS=("trainer.device=npu")\n')
    (scripts / delegate).write_text(
        f'"{sys.executable}" - "$@" <<\'PY\'\n'
        "import json,os,sys\nfrom pathlib import Path\n"
        'Path(os.environ["RESULT"]).write_text(json.dumps({"argv":sys.argv[1:], "env":dict(os.environ)}))\n'
        f"raise SystemExit({raw_exit})\nPY\n"
    )
    checkpoint = tmp_path / "caller model"
    checkpoint.mkdir()
    (checkpoint / "model_index.json").write_text("{}")
    result_path = tmp_path / "result.json"
    env = {k: v for k, v in os.environ.items() if k not in ("ROLLOUT_TP", "REWARD_TP")}
    env.update(
        MODEL_PATH=str(checkpoint),
        QWEN_IMAGE_REWARD_MODEL_PATH=str(checkpoint),
        DATA_DIR=str(tmp_path / "data"),
        RESULT=str(result_path),
    )
    if tp is not None:
        env.update(ROLLOUT_TP=tp, REWARD_TP=tp)
    final_override = "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    result = subprocess.run(
        ["bash", str(scripts / wrapper), final_override], env=env, capture_output=True, text=True, timeout=15
    )
    assert result.returncode == raw_exit, result.stdout + result.stderr
    captured = json.loads(result_path.read_text())
    assert captured["env"]["NUM_GPUS"] == "8"
    assert captured["argv"][-1] == final_override
    expected = tp or "2"
    if "edit" in wrapper:
        assert captured["env"]["ROLLOUT_TP"] == expected
    else:
        assert f"actor_rollout_ref.rollout.tensor_model_parallel_size={expected}" in captured["argv"][:-1]
    if "REWARD_MODEL_PATH" in captured["env"]:
        assert captured["env"]["REWARD_TP"] == expected
