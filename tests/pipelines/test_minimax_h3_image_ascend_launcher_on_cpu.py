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
"""Check the Ascend launcher contract without starting Ray or loading models."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

RECIPE = Path(__file__).resolve().parents[2] / "examples/diffusionnft_trainer/minimax_h3_image"


@pytest.fixture
def launch(tmp_path):
    bash = os.environ.get("BASH_FOR_TEST") or shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to exercise the launcher")
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, executable in (("bash", bash), ("python3", sys.executable)):
        wrapper = tools / name
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(Path(executable).as_posix())} "$@"\n')
        wrapper.chmod(0o755)
    model = tmp_path / "model weights"
    actor = model / "transformer_diffusers"
    actor.mkdir(parents=True)
    (actor / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3ImageTransformer2DModel"}))
    for filename in ("pdd_config.json", "pdd_fused_heads.safetensors"):
        (model / filename).touch()
    capture = tmp_path / "capture.json"
    stub = tools / "trainer"
    stub.write_text(
        "#!/bin/sh\nexec python3 - \"$@\" <<'PY'\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE'], 'w') as handle:\n"
        "    json.dump({'args': sys.argv[1:], 'env': dict(os.environ)}, handle)\n"
        "sys.exit(int(os.environ.get('STUB_EXIT', '0')))\nPY\n"
    )
    stub.chmod(0o755)
    # Stub the device-node check so the real launcher can also run on a CPU-only host.
    bash_env = tmp_path / "devices.sh"
    bash_env.write_text(
        "test() { if [[ $1 == -c && $2 == /dev/davinci[0-7] ]]; then\n"
        '    [[ $2 != "${MISSING_DEVICE:-}" ]]; else builtin test "$@"; fi; }\n'
    )
    env = {name: os.environ[name] for name in ("PATH", "SystemRoot") if name in os.environ}
    env.update(
        HOME=tmp_path.as_posix(),
        BASH_ENV=bash_env.as_posix(),
        PATH=str(tools) + os.pathsep + env.get("PATH", ""),
        MODEL_PATH=model.as_posix(),
        DATA_DIR=(tmp_path / "data files").as_posix(),
        OUTPUT_DIR=(tmp_path / "output files").as_posix(),
        CAPTURE=capture.as_posix(),
        VERL_OMNI_ENTRY="trainer",
        PPOCR_MODEL_DIR="unused",
        PICKSCORE_PATH="unused",
        CLIP_H14_PATH="unused",
        CUSTOM_REWARD_MODEL_PATH="unused",
    )

    def run(*args, script="run_minimax_h3_image_ocr_lora_npu.sh", **updates):
        capture.unlink(missing_ok=True)
        result = subprocess.run(
            [bash, str(RECIPE / script), *args],
            env=env | updates,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        captured = json.loads(capture.read_text()) if capture.exists() else None
        return result, captured

    return run, actor


def test_native_eight_device_defaults_and_user_overrides(launch):
    run, _ = launch
    result, captured = run("trainer.total_training_steps=1", "trainer.experiment_name=space in name")
    assert result.returncode == 0, result.stderr
    args = captured["args"]
    assert args[-2:] == ["trainer.total_training_steps=1", "trainer.experiment_name=space in name"]
    settings = dict(arg.lstrip("+").split("=", 1) for arg in args)
    assert settings["actor_rollout_ref.actor.optim.lr"] == "1e-4"
    assert settings["actor_rollout_ref.actor.fsdp_config.fsdp_size"] == "4"
    assert settings["actor_rollout_ref.rollout.tensor_model_parallel_size"] == "2"
    assert settings["actor_rollout_ref.rollout.agent.num_workers"] == "2"
    assert settings["reward.reward_model.n_gpus_per_node"] == "3"
    assert settings["reward.num_workers"] == "3"
    assert settings["reward.reward_functions.hps.required"] == "true"
    assert settings["trainer.val_before_train"] == "True"
    assert settings["trainer.test_freq"] == "5"
    assert settings["trainer.save_freq"] == "50"
    assert "trainer.total_training_steps=100" in args
    assert "model weights" in settings["actor_rollout_ref.model.config_path"]
    assert captured["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6"
    assert captured["env"]["PREFERENCE_SCORER_GPU"] == "7"
    assert captured["env"]["VERL_OMNI_H3_IMAGE_ROLLOUT_KERNELS"] == "0"
    assert captured["env"]["VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH"] == "0"


def test_trainer_failure_status_propagates(launch):
    result, _ = launch[0](STUB_EXIT="23")
    assert result.returncode == 23, result.stderr


def test_base_gpu_recipe_keeps_its_defaults(launch):
    if shutil.which("seq") is None:
        pytest.skip("The existing GPU recipe requires coreutils seq")
    run, actor = launch
    result, captured = run(script="run_minimax_h3_image_ocr_lora.sh", ACTOR_PATH=actor.as_posix())
    assert result.returncode == 0, result.stderr
    assert "trainer.total_training_steps=50" in captured["args"]
    assert "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8" in captured["args"]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2"
    assert captured["env"]["VERL_OMNI_H3_IMAGE_PACK_MICRO_BATCH"] == "1"


def test_partial_allocation_is_rejected_before_training(launch):
    result, captured = launch[0](ASCEND_RT_VISIBLE_DEVICES="4,5,6,7")
    assert result.returncode != 0
    assert captured is None
    assert "physical NPUs 0 through 7" in result.stderr


@pytest.mark.parametrize("override", [None, "ray_kwargs.ray_init.num_gpus=8"])
def test_rocm_launcher_visibility_and_ray_allocation(launch, tmp_path, override):
    run, actor = launch
    (tmp_path / "torch.py").write_text("from types import SimpleNamespace\nversion = SimpleNamespace(hip='7.0')\n")
    args = [] if override is None else [override]
    result, captured = run(
        *args,
        script="run_minimax_h3_image_ocr_lora.sh",
        ACTOR_PATH=actor.as_posix(),
        PYTHONPATH=tmp_path.as_posix(),
        CUDA_VISIBLE_DEVICES="0,1,2",
        HIP_VISIBLE_DEVICES="0,1,2",
        ROCR_VISIBLE_DEVICES="0,1,2",
    )
    if override:
        assert result.returncode != 0 and captured is None
        assert "set N_GPUS instead" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    assert "++ray_kwargs.ray_init.address=local" in captured["args"]
    assert "++ray_kwargs.ray_init.num_gpus=3" in captured["args"]
    assert all(
        name not in captured["env"] for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
    )


def test_legacy_actor_weights_are_rejected_before_training(launch):
    run, actor = launch
    (actor / "config.json").write_text(json.dumps({"_class_name": "MiniMaxH3Transformer3DModel"}))
    result, captured = run()
    assert result.returncode != 0
    assert captured is None
    assert "MiniMaxH3ImageTransformer2DModel" in result.stderr


def test_unset_mask_requires_scorer_device(launch):
    result, captured = launch[0](MISSING_DEVICE="/dev/davinci7")
    assert result.returncode != 0 and captured is None
    assert "missing /dev/davinci7" in result.stderr
