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

"""CPU contract for the Qwen-Image-Edit NPU PickScore scorer recipe."""

import os
import subprocess
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from verl.trainer.ppo.reward import load_reward_manager

from verl_omni.reward_loop.reward_manager.multi import MultiVisualRewardManager
from verl_omni.workers.config.reward import (
    NativeRewardModelConfig,
    normalize_reward_config,
    parse_reward_model_config,
    resolve_reward_model_name,
    reward_pool_is_separate,
    reward_role_required,
    validate_reward_model_terms,
)

ROOT = Path(__file__).parents[3]
RECIPE = ROOT / "examples/flowgrpo_trainer/qwen_image_edit/run_qwen_image_edit_lora_v1_npu.sh"


@pytest.mark.parametrize("override_form", ["default", "explicit", "legacy"])
def test_qwen_image_edit_npu_recipe_composes_native_pickscore_scorer(tmp_path, override_form):
    """Compose the NPU launcher arguments without loading a model."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "trainer-argv"
    fake_python = bin_dir / "python3"
    fake_python.write_text('#!/usr/bin/env bash\nprintf "%s\\0" "$@" > "$CAPTURE_PATH"\n')
    fake_python.chmod(0o755)
    workspace = tmp_path / "recipe-workspace"
    policy_path = tmp_path / "policy-checkpoint"
    reward_path = tmp_path / "pickscore-checkpoint"
    trailing_override = "trainer.logger=[console]"
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "CAPTURE_PATH": str(capture),
        "WORKSPACE": str(workspace),
        "MODEL_PATH": str(policy_path),
        "PICKSCORE_MODEL_PATH": str(reward_path),
        "NUM_GPUS_ACTOR_ROLLOUT_REWARD": "6",
        "ROLLOUT_TP": "2",
        "NATIVE_REWARD_DEVICES": "[2,5]",
        "REWARD_OFFLOAD": "true",
    }

    scorer_path = "pkg://verl_omni.utils.reward_score.pickscore_reward"
    overrides = []
    if override_form != "default":
        prefix = "reward.reward_functions.pickscore" if override_form == "legacy" else "reward.scorers.pickscore"
        overrides = [f"{prefix}.path={scorer_path}", f"+{prefix}.weight=0.5"]
    subprocess.run(
        ["bash", str(RECIPE), *overrides, trailing_override],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    args = capture.read_bytes().rstrip(b"\x00").decode().split("\x00")
    assert args[:2] == ["-m", "verl_omni.trainer.main_diffusion_v1"]
    assert args[-1] == trailing_override

    config_dir = ROOT / "verl_omni/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="diffusion_trainer", overrides=args[2:])

    if override_form == "legacy":
        assert config.reward.get("scorers") is None
    else:
        assert config.reward.scorers
        assert not config.reward.reward_functions
    normalized = normalize_reward_config(config)
    term_config = normalized.reward.reward_functions.pickscore
    projected = OmegaConf.to_container(term_config, resolve=True)
    expected: dict[str, str | float] = {
        "path": scorer_path,
        "name": "compute_score_pickscore_native",
    }
    if overrides:
        expected["weight"] = 0.5
    assert projected == expected
    validate_reward_model_terms(normalized)

    model = parse_reward_model_config("pickscore", normalized.reward.models.pickscore)
    assert isinstance(model, NativeRewardModelConfig)
    assert model.model_path == str(reward_path)
    assert model.resolved_offload is True
    assert model.placement.devices == [2, 5]
    assert model.executor.model == "verl_omni.utils.reward_score.pickscore_reward:PickScoreNativeModel"
    assert resolve_reward_model_name("pickscore", term_config, normalized.reward.models) == "pickscore"
    assert reward_pool_is_separate(normalized) is False
    assert reward_role_required(normalized) is True
    manager = load_reward_manager(normalized, tokenizer=None)
    assert isinstance(manager, MultiVisualRewardManager)
    term = manager._sub_rewards[0]
    assert {key: term[key] for key in ("key", "weight", "required", "model", "use_rollout_sampling_params")} == {
        "key": "pickscore",
        "weight": 0.5 if overrides else 1.0,
        "required": False,
        "model": "pickscore",
        "use_rollout_sampling_params": False,
    }
    assert term["extra_args"] == {}

    assert normalized.reward.reward_model.enable is False
    assert normalized.reward.reward_model.enable_resource_pool is False
    assert normalized.reward.reward_model.rollout.enforce_eager is True
    assert normalized.reward.reward_manager.name == "MultiVisualRewardManager"
    assert normalized.reward.reward_manager.source == "importlib"
    assert normalized.reward.reward_manager.module.path == "pkg://verl_omni.reward_loop.reward_manager"
    assert normalized.actor_rollout_ref.model.path == str(policy_path)
    assert normalized.actor_rollout_ref.rollout.tensor_model_parallel_size == 2
    assert normalized.actor_rollout_ref.rollout.agent.num_workers == 3
    assert normalized.data.train_files == str(workspace / "data/qwen_image_edit/train.parquet")
    assert normalized.data.val_files == str(workspace / "data/qwen_image_edit/test.parquet")
    assert normalized.trainer.device == "npu"
    assert normalized.trainer.use_v1 is True
    assert normalized.trainer.n_gpus_per_node == 6
    assert list(normalized.trainer.logger) == ["console"]
