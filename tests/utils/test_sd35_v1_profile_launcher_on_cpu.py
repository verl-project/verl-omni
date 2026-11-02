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
"""Run the SD3.5 profiling launcher without starting training."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples/flowgrpo_trainer/sd35/profile_sd35_medium_ocr_lora_v1.sh"


@pytest.fixture
def launch(tmp_path, monkeypatch):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to exercise the launcher")
    capture = tmp_path / "argv"
    trainer = tmp_path / "python3"
    trainer.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE_FILE"\nexit "${TRAINER_EXIT_CODE:-0}"\n')
    trainer.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("CAPTURE_FILE", capture.as_posix())
    monkeypatch.setenv("OCR_WORKSPACE", (tmp_path / "OCR data").as_posix())
    monkeypatch.delenv("PROFILE_DIR", raising=False)
    monkeypatch.delenv("MODEL_PATH", raising=False)

    def run(*overrides):
        result = subprocess.run([bash, str(SCRIPT), *overrides], cwd=ROOT, capture_output=True, text=True, timeout=30)
        argv = capture.read_bytes().decode().rstrip("\0").split("\0")
        assert argv[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]
        return result, dict(arg.split("=", 1) for arg in argv[2:])

    return run


def test_profile_defaults_select_v1_and_both_collectors(launch, tmp_path):
    result, settings = launch()
    assert result.returncode == 0, result.stderr
    assert settings["trainer.use_v1"] == "true"
    assert settings["trainer.v1.trainer_mode"] == "sync"
    assert settings["actor_rollout_ref.model.path"] == "stabilityai/stable-diffusion-3.5-medium"
    assert settings["data.train_files"] == (tmp_path / "OCR data/data/ocr/sd3/train.parquet").as_posix()
    assert settings["global_profiler.tool"] == "torch"
    assert settings["global_profiler.steps"] == "[1]"
    assert settings["global_profiler.profile_continuous_steps"] == "False"
    for role, discrete in [("actor", "False"), ("rollout", "True")]:
        prefix = f"actor_rollout_ref.{role}.profiler"
        assert settings[f"{prefix}.enable"] == "True"
        assert settings[f"{prefix}.all_ranks"] == "False"
        assert settings[f"{prefix}.ranks"] == "[0]"
        assert settings[f"{prefix}.tool_config.torch.discrete"] == discrete
    assert settings["trainer.total_training_steps"] == "1"
    assert settings["trainer.save_freq"] == settings["trainer.test_freq"] == "-1"
    assert settings["trainer.resume_mode"] == "disable"
    assert settings["trainer.logger"] == "[console]"
    samples_per_gpu = (
        int(settings["data.train_batch_size"])
        * int(settings["actor_rollout_ref.rollout.n"])
        // int(settings["trainer.n_gpus_per_node"])
    )
    assert samples_per_gpu % int(settings["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"]) == 0
    assert int(settings["data.train_batch_size"]) // int(settings["actor_rollout_ref.actor.ppo_mini_batch_size"]) == 2
    assert settings["actor_rollout_ref.rollout.algo.sde_window_range"] == "[0,4]"
    assert settings["actor_rollout_ref.rollout.pipeline.num_inference_steps"] == "4"


def test_environment_paths_and_caller_overrides_survive(launch, monkeypatch, tmp_path):
    output = (tmp_path / "profile traces").as_posix()
    model = (tmp_path / "SD3.5 Medium").as_posix()
    monkeypatch.setenv("PROFILE_DIR", output)
    monkeypatch.setenv("MODEL_PATH", model)
    result, settings = launch("global_profiler.steps=[2]", "trainer.total_training_steps=2")
    assert result.returncode == 0, result.stderr
    assert settings["global_profiler.save_path"] == output
    assert settings["actor_rollout_ref.model.path"] == model
    assert settings["global_profiler.steps"] == "[2]"
    assert settings["trainer.total_training_steps"] == "2"


def test_profile_launcher_preserves_training_failure(launch, monkeypatch):
    monkeypatch.setenv("TRAINER_EXIT_CODE", "17")
    result, _ = launch()
    assert result.returncode == 17
