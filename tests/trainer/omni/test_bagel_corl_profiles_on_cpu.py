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

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load():
    spec = importlib.util.spec_from_file_location(
        "profiles", ROOT / "examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("profile", ["reproduction", "B0", "B1", "B2-1e-6", "B2-1e-5", "B3", "B4"])
def test_profiles_require_explicit_provenance_and_isolation(profile):
    tool = load()
    with pytest.raises(ValueError):
        tool.build_profile(profile, 11, {})
    env, overrides = tool.build_profile(
        profile,
        11,
        {
            "START_CHECKPOINT": "/model/base",
            "UNICOT_TRAIN": "/data/train.parquet",
            "UNICOT_VAL": "/data/panel.parquet",
            "EXPERIMENT_PREFIX": "test",
        },
    )
    assert env["RESUME_MODE"] == "disable"
    assert env["NOISE_LEVEL"] == "0.8" and env["SDE_WINDOW_RANGE"] == "[0,2]"
    assert env["REBUILD_UNICOT"] == "0"
    assert env["DEFAULT_LOCAL_DIR"].endswith(f"test_{profile}_seed11")
    assert "data.seed=11" in overrides
    assert "actor_rollout_ref.actor.data_loader_seed=11" in overrides
    assert "actor_rollout_ref.actor.fsdp_config.seed=11" in overrides
    assert "actor_rollout_ref.rollout.seed=11" in overrides
    assert env["CFG_TEXT_SCALE"] == "4.0" and env["CFG_IMG_SCALE"] == "1.5"
    if profile == "B0":
        assert "+actor_rollout_ref.model.corl_no_optimizer_step=True" in overrides
    if profile == "B1":
        assert "+actor_rollout_ref.model.corl_freeze_und=True" in overrides
        assert "+actor_rollout_ref.rollout.agent.corl_fixed_gen_prompt=__user_request__" in overrides
        assert env["S"] == "2" and env["MAX_GENERATE_PASSES"] == "1"
    if profile.startswith("B2"):
        assert "+actor_rollout_ref.model.corl_freeze_gen=True" in overrides
    if profile == "B4":
        assert "++agentic_image_gen.corl_reward_version=v2" in overrides
        assert env["MAX_GENERATE_PASSES"] == "3"


def test_dry_run_cli_does_not_launch_or_write(tmp_path):
    env = dict(
        os.environ,
        START_CHECKPOINT="/model/base",
        UNICOT_TRAIN="/data/train.parquet",
        UNICOT_VAL="/data/panel.parquet",
        EXPERIMENT_PREFIX="dry",
        EXPERIMENT_ROOT=str(tmp_path),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/agenticllmgrpo_trainer/bagel/corl_convergence_profiles/run_profile.py"),
            "B0",
            "--seed",
            "11",
            "--dry-run",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(result.stdout)
    assert "corl_no_optimizer_step=True" in value["command"]
    assert list(tmp_path.iterdir()) == []
