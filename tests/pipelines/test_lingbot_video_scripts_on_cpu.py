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

"""CPU checks for LingBot Dense T2V example script configuration guards."""

import os
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "examples" / "flowgrpo_trainer" / "lingbot_video"


def _script_env(**overrides):
    env = os.environ.copy()
    env.update(
        {
            "CHECK_CONFIG_ONLY": "True",
            "WORKSPACE": str(_REPO_ROOT),
            "REWARD_FUNCTION_PATH": "verl_omni/utils/reward_score/hpsv3_reward.py",
            "REWARD_FUNCTION_NAME": "compute_score_hpsv3",
        }
    )
    env.update(overrides)
    return env


def _run_script_config_check(script, **env_overrides):
    return subprocess.run(
        ["bash", str(_SCRIPT_DIR / script)],
        cwd=_REPO_ROOT,
        env=_script_env(**env_overrides),
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "script",
    ["run_lingbot_dense_t2v_lora.sh", "run_lingbot_dense_t2v_lora_fsdp2.sh"],
)
def test_lingbot_scripts_default_batch_config_is_self_consistent(script):
    result = _run_script_config_check(script)

    assert result.returncode == 0, result.stderr + result.stdout
    assert "batch settings are self-consistent" in result.stderr
    assert "global_rollout_batch=128" in result.stderr
    assert "rollout_batch_per_gpu=16" in result.stderr
    assert "actor_update_global_mini_batch=64" in result.stderr
    assert "actor_update_mini_batch_per_gpu=8" in result.stderr
    assert "ppo_micro_batch_size_per_gpu=2" in result.stderr
    assert "log_prob_micro_batch_size_per_gpu=2" in result.stderr
    assert "ref_log_prob_micro_batch_size_per_gpu=2" in result.stderr
    assert "enable_gradient_checkpointing=True" in result.stderr
    assert "rollout_noise_level=0.7, rollout_sde_type=dance_sde" in result.stderr


@pytest.mark.parametrize(
    "script",
    ["run_lingbot_dense_t2v_lora.sh", "run_lingbot_dense_t2v_lora_fsdp2.sh"],
)
def test_lingbot_scripts_reject_non_divisible_ppo_micro_batch(script):
    result = _run_script_config_check(script, PPO_MICRO_BATCH_SIZE_PER_GPU="3")

    assert result.returncode == 2
    assert "ACTOR_UPDATE_MINI_BATCH_PER_GPU=8" in result.stderr
    assert "PPO_MICRO_BATCH_SIZE_PER_GPU=3" in result.stderr
