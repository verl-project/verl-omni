# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exercise the Ascend launcher without starting Ray, loading models, or using NPUs."""

import json
import os
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir


def test_minimax_h3_ascend_recipe(tmp_path):
    root = Path(__file__).resolve().parents[2]
    toolkit = tmp_path / "ascend-toolkit"
    toolkit.mkdir()
    (toolkit / "set_env.sh").touch()
    atb = tmp_path / "nnal" / "atb"
    atb.mkdir(parents=True)
    (atb / "set_env.sh").touch()
    model = tmp_path / "model"
    (model / "FL2VA").mkdir(parents=True)
    (model / "transformer").mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    capture = tmp_path / "argv.json"
    python = bindir / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{bindir}:{os.environ['PATH']}",
        "ASCEND_HOME_PATH": str(toolkit),
        "MODEL_PATH": str(model),
        "DATA_DIR": str(tmp_path / "data"),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "IMAGEBIND_MODEL_PATH": str(tmp_path / "imagebind.pth"),
        "CAPTURE_ARGS": str(capture),
        "NUM_GPUS": "16",
        "FSDP_SIZE": "8",
        "ROLLOUT_TP": "4",
        "TEXT_ENCODER_TP": "4",
        "ROLLOUT_N": "8",
        "TOTAL_TRAINING_STEPS": "30",
        "TOTAL_EPOCHS": "30",
        "TIMESTEP_FRACTION": "0.34",
    }
    script = root / "examples/diffusionnft_trainer/minimax_h3/run_minimax_h3_t2va_lora_npu.sh"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    argv = json.loads(capture.read_text())
    assert argv[:2] == ["-m", "verl_omni.trainer.main_diffusion"]
    with initialize_config_dir(config_dir=str(root / "verl_omni/trainer/config"), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=argv[2:])
    assert config.trainer.device == "npu"
    assert config.trainer.n_gpus_per_node == 16
    assert config.trainer.total_epochs == config.trainer.total_training_steps == 30
    assert config.actor_rollout_ref.actor.fsdp_config.fsdp_size == 8
    assert config.actor_rollout_ref.actor.fsdp_config.offload_policy is True
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert config.actor_rollout_ref.model.attn_backend == "_native_npu"
    assert config.actor_rollout_ref.rollout.rollout_attn_backend == "TORCH_SDPA"
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.actor_rollout_ref.rollout.rollout_adapter == "old"
    assert config.algorithm.timestep_fraction == 0.34
    assert "clap" not in config.reward.reward_functions
    assert config.reward.reward_functions.imagebind.device == "npu:1"
    assert config.reward.reward_functions.imagebind.weights.text_video == 0.8

    invalid = subprocess.run(
        ["bash", str(script)], env={**env, "NUM_GPUS": "15"}, capture_output=True, text=True, timeout=30
    )
    assert invalid.returncode != 0
    assert "divisible" in invalid.stderr
