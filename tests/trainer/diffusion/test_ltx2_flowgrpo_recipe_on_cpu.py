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
"""CPU configuration coverage for the LTX-2.3 V1 FlowGRPO launchers."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.workers.config.reward import (
    normalize_reward_config,
    resolve_reward_model_name,
    validate_reward_model_terms,
)

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _ROOT / "verl_omni/trainer/config"
_EXAMPLES = _ROOT / "examples/flowgrpo_trainer/ltx2"


@pytest.mark.parametrize(("recipe", "device"), [("t2av", "cuda"), ("t2av", "npu"), ("ti2va", "cuda")])
def test_ltx2_v1_recipe_projects_explicit_scorers(tmp_path, recipe, device):
    """Pass the launcher's real Hydra arguments through the scorer normalizer."""
    commands = tmp_path / "bin"
    commands.mkdir()
    capture = tmp_path / "trainer-argv"
    for name, script in {
        "python3": 'printf "%s\\0" "$@" > "$CAPTURE"',
        "npu-smi": "exit 0" if device == "npu" else "exit 1",
        "nvidia-smi": "printf '%s\\n' GPU",
    }.items():
        command = commands / name
        command.write_text(f"#!/usr/bin/env bash\n{script}\n")
        command.chmod(0o755)
    toolkit = tmp_path / "ascend-toolkit"
    atb = tmp_path / "nnal/atb"
    for directory in (toolkit, atb):
        directory.mkdir(parents=True)
        (directory / "set_env.sh").touch()

    model_path = "test/clap-checkpoint"
    env = {
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORKSPACE": str(tmp_path),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "NUM_GPUS": "8" if device == "cuda" else "16",
        "ASCEND_HOME_PATH": str(toolkit),
        "CLAP_MODEL_PATH": model_path,
        "CAPTURE": str(capture),
    }
    subprocess.run(
        ["bash", str(_EXAMPLES / f"run_ltx2_3_{recipe}_lora_v1.sh")],
        cwd=_ROOT,
        env=env,
        check=True,
        capture_output=True,
        timeout=15,
    )
    argv = capture.read_bytes().rstrip(b"\x00").decode().split("\x00")
    assert argv[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=argv[2:])

    assert config.reward.custom_reward_function.path is None
    assert not config.reward.reward_functions
    normalized = normalize_reward_config(config)
    validate_reward_model_terms(normalized)
    expected_clap_kwargs = {
        "device": f"{device}:0",
        "model_name_or_path": model_path if recipe == "t2av" else "laion/larger_clap_general",
    }
    assert OmegaConf.to_container(normalized.reward.scorers.clap.kwargs) == expected_clap_kwargs
    projected = OmegaConf.to_container(normalized.reward.reward_functions)
    assert list(projected) == ["clap", "imagebind"]
    assert projected == {
        "clap": {
            "path": str(_ROOT / "verl_omni/utils/reward_score/clap.py"),
            "name": "compute_score",
            "weight": 1.0,
            **expected_clap_kwargs,
        },
        "imagebind": {
            "path": "pkg://verl_omni.utils.reward_score.imagebind",
            "name": "compute_score_native",
            "weight": 1.0,
            "model": "imagebind",
            "mode": "audio_video",
        },
    }
    assert resolve_reward_model_name("imagebind", projected["imagebind"], normalized.reward.models) == "imagebind"
    assert list(normalized.reward.models.imagebind.placement.devices) == [1]
    assert normalized.reward.models.imagebind.offload is False
    assert isinstance(load_reward_manager(normalized, tokenizer=None), MultiVisualRewardManager)
