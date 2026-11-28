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
"""CPU configuration coverage for the MiniMax H3 V1 FlowGRPO launchers."""

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
    parse_reward_model_config,
    resolve_reward_model_name,
    validate_reward_model_terms,
)

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _ROOT / "verl_omni/trainer/config"
_EXAMPLES = _ROOT / "examples/flowgrpo_trainer/minimax_h3"


@pytest.mark.parametrize(
    ("recipe", "device"),
    [
        ("t2va", "cuda"),
        ("t2va", "npu"),
        ("t2va", "mlu"),
        ("fl2va", "cuda"),
        ("fl2va", "npu"),
        ("ref2va", "cuda"),
        ("ref2va", "npu"),
    ],
)
@pytest.mark.parametrize("override_form", ["default", "explicit", "legacy"])
def test_minimax_h3_v1_recipe_reward_overrides(tmp_path, recipe, device, override_form):
    """Compose the launcher's Hydra arguments and instantiate the reward manager."""
    commands = tmp_path / "bin"
    commands.mkdir()
    capture = tmp_path / "trainer-argv"
    python = commands / "python3"
    python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE"\n')
    python.chmod(0o755)

    toolkit = tmp_path / "ascend-toolkit"
    atb = tmp_path / "nnal/atb"
    for directory in (toolkit, atb):
        directory.mkdir(parents=True)
        (directory / "set_env.sh").touch()
    model_root = tmp_path / "model"
    for component in ("Ref2VA", "transformer_ref"):
        (model_root / component).mkdir(parents=True)

    clap_path = str(tmp_path / "clap-checkpoint")
    imagebind_path = str(tmp_path / "imagebind.pth")
    env = {
        "PATH": f"{commands}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "WORKSPACE": str(tmp_path),
        "DATA_DIR": str(tmp_path / "data"),
        "MODEL_PATH": str(model_root),
        "OUTPUT_DIR": str(tmp_path / "output"),
        "ASCEND_HOME_PATH": str(toolkit),
        "CLAP_MODEL_PATH": clap_path,
        "IMAGEBIND_MODEL_PATH": imagebind_path,
        "CAPTURE": str(capture),
    }
    suffix = "" if device == "cuda" else f"_{device}"
    launcher = _EXAMPLES / f"run_minimax_h3_{recipe}_lora_v1{suffix}.sh"
    overrides = []
    if override_form == "explicit":
        overrides = [
            'reward.scorers.clap.kwargs.model_name_or_path="test/clap=override"',
            "reward.scorers.imagebind.kwargs.mode=text_video",
        ]
    elif override_form == "legacy":
        overrides = [
            'reward.reward_functions.clap.model_name_or_path="test/clap=override"',
            "++reward.reward_functions.imagebind.mode=text_video",
        ]
    subprocess.run(
        ["bash", str(launcher), *overrides, "trainer.logger=[console]"],
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

    if override_form == "legacy":
        assert config.reward.get("scorers") is None
        assert config.reward.custom_reward_function.name == "_multi_reward_placeholder"
    else:
        assert config.reward.scorers
        assert config.reward.custom_reward_function.path is None
        assert not config.reward.reward_functions
    normalized = normalize_reward_config(config)
    validate_reward_model_terms(normalized)
    projected = OmegaConf.to_container(normalized.reward.reward_functions)
    assert list(projected) == ["clap", "imagebind"]
    assert projected == {
        "clap": {
            "path": str(_ROOT / "verl_omni/utils/reward_score/clap.py"),
            "name": "compute_score",
            "weight": 1.0,
            "required": True,
            "device": f"{device}:0",
            "model_name_or_path": "test/clap=override" if overrides else clap_path,
        },
        "imagebind": {
            "path": "pkg://verl_omni.utils.reward_score.imagebind",
            "name": "compute_score_native",
            "weight": 1.0,
            "required": True,
            "model": "imagebind",
            "mode": "text_video" if overrides else "audio_video",
        },
    }
    assert resolve_reward_model_name("imagebind", projected["imagebind"], normalized.reward.models) == "imagebind"
    imagebind = parse_reward_model_config("imagebind", normalized.reward.models.imagebind)
    assert imagebind.backend == "native"
    assert imagebind.model_path == imagebind_path
    assert imagebind.placement.bundle_indices == [1]
    assert imagebind.resolved_offload is False
    assert normalized.reward.reward_model.enable is False
    assert normalized.reward.aggregation == "weighted_sum"
    assert normalized.trainer.use_v1 is True
    assert normalized.trainer.v1.trainer_mode == "sync"
    assert normalized.trainer.device == device
    assert list(normalized.trainer.logger) == ["console"]
    assert isinstance(load_reward_manager(normalized, tokenizer=None), MultiVisualRewardManager)
