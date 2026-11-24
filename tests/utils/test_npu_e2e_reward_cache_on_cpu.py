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
@pytest.mark.parametrize(
    "model_source",
    [
        "owned",
        "env",
        "cli",
        "cli+",
        "cli++",
        "parent",
        "parent++",
        "root",
        "root++",
        "models",
        "models++",
        "named",
        "named++",
    ],
)
@pytest.mark.parametrize("override", [False, True])
def test_reward_launcher_does_not_write_the_core_checkpoint(tmp_path, wrapper, delegate, model_source, override):
    """Exercise real wrappers while only builder/trainer command boundaries are stubbed."""
    root = tmp_path / "fixture repo"
    scripts = root / "tests/special_e2e"
    scripts.mkdir(parents=True)
    shutil.copyfile(_SCRIPTS / wrapper, scripts / wrapper)
    for helper in ("lib_npu_e2e.sh", "npu_runtime.py"):
        shutil.copyfile(_SCRIPTS / helper, scripts / helper)
    bash_env = tmp_path / "bash_env"
    bash_env.write_text(
        """source() {
    case "$1" in */set_env.sh) return 0 ;; *) builtin source "$@" ;; esac
}
python3() {
    if [[ "$#" -eq 1 && "$1" == - ]]; then
        cat >/dev/null
    else
        "$TEST_PYTHON" "$@"
    fi
}
"""
    )
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
        f'#!/usr/bin/env bash\n"{sys.executable}" - "$@" <<\'PY\'\n'
        "import json, os, sys\nfrom pathlib import Path\n"
        "with Path(os.environ['EVENTS']).open('a') as stream:\n"
        "    stream.write(json.dumps({'kind': 'trainer', 'path': os.environ['REWARD_MODEL_PATH'], "
        "'argv': sys.argv[1:]}) + '\\n')\n"
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
        BASH_ENV=str(bash_env),
        TEST_PYTHON=sys.executable,
        NUM_NPUS="8",
        TORCHNPU_PRECOMPILE_THREADS="2",
    )
    if model_source == "env":
        env["QWEN_IMAGE_REWARD_MODEL_PATH"] = str(custom)
    user_args = []
    if model_source.startswith("cli"):
        user_args.append(f"{model_source.removeprefix('cli')}reward.reward_model.model_path={custom}")
    elif model_source.startswith("parent"):
        user_args.append(
            f"{model_source.removeprefix('parent')}reward.reward_model={{model_path:{json.dumps(str(custom))}}}"
        )
    elif model_source.startswith("root"):
        user_args.append(
            f"{model_source.removeprefix('root')}reward={{reward_model:{{model_path:{json.dumps(str(custom))}}}}}"
        )
    elif model_source.startswith(("models", "named")):
        engine = f"{{backend:engine,model_path:{json.dumps(str(custom))}}}"
        selector = (
            f"{model_source.removeprefix('models') or '+'}reward.models={{ocr:{engine}}}"
            if model_source.startswith("models")
            else f"{model_source.removeprefix('named') or '+'}reward.models.ocr={engine}"
        )
        user_args.extend(["reward.reward_model.enable=False", selector])
    if override:
        user_args.append("reward.reward_model.rollout.max_num_seqs=4")
    result = subprocess.run(
        ["bash", str(scripts / wrapper), *user_args], cwd=root, env=env, capture_output=True, text=True, timeout=20
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_bytes() == b"unfinished core-owned weights"
    assert list(core.iterdir()) == [marker]
    assert weights.read_bytes() == b"user reward weights"
    recorded = [json.loads(line) for line in events.read_text().splitlines()]
    assert [entry["kind"] for entry in recorded] == (["trainer"] if model_source == "env" else ["builder", "trainer"])
    defaults = [
        "reward.reward_model.rollout.max_num_batched_tokens=512",
        "reward.reward_model.rollout.max_num_seqs=8",
        "++reward.reward_model.rollout.engine_kwargs.vllm.limit_mm_per_prompt={image:1,video:0}",
        "++reward.reward_model.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=67108864",
    ]
    argv = recorded[-1]["argv"]
    for default in defaults:
        assert (default in argv) is (model_source == "owned")
    if user_args:
        assert argv[-len(user_args) :] == user_args
    assert not any(key in arg for arg in argv for key in ("max_model_len", "mm_processor_kwargs", "skip_mm_profiling"))
    if model_source.startswith(("models", "named")):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(_SCRIPTS.parents[1] / "verl_omni/trainer/config"), version_base=None):
            config = compose(config_name="diffusion_trainer", overrides=argv)
        assert config.reward.reward_model.enable is False
        assert config.reward.models.ocr.model_path == str(custom)
        rollout = config.reward.reward_model.rollout
        assert rollout.max_num_batched_tokens == 8192
        assert rollout.max_num_seqs == (4 if override else 1024)
        assert "kv_cache_memory_bytes" not in rollout.engine_kwargs.vllm
        assert "limit_mm_per_prompt" not in rollout.engine_kwargs.vllm
    if model_source == "env":
        assert recorded[0]["path"] == str(custom)
    else:
        assert recorded[0]["path"] == recorded[1]["path"]
        assert Path(recorded[0]["path"]).is_dir()
        assert Path(recorded[0]["path"]) != core
