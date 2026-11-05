# Copyright 2026 verl-omni contributors
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
"""Exercise the public recipe's real shell composition without starting a trainer."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = ROOT / "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_megatron_avqa_proximal_reuse.sh"


@pytest.fixture
def launch(tmp_path):
    capture = tmp_path / "calls.jsonl"
    environment = tmp_path / "environment.jsonl"
    python = tmp_path / "python stub"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'with open(os.environ["CAPTURE"], "a") as stream:\n'
        '    stream.write(json.dumps(sys.argv[1:]) + "\\n")\n'
        'with open(os.environ["ENV_CAPTURE"], "a") as stream:\n'
        '    stream.write(json.dumps(os.environ.get("NVTE_ALLOW_NONDETERMINISTIC_ALGO")) + "\\n")\n'
        'print("{}")\n'
        'phase = "config" if "--cfg" in sys.argv else "train"\n'
        'sys.exit(19 if os.environ.get("FAIL_PHASE") == phase else 0)\n'
    )
    python.chmod(0o755)
    train, val = tmp_path / "train set.parquet", tmp_path / "validation set.parquet"
    train.touch()
    val.touch()
    env = dict(
        os.environ,
        MODEL_PATH="/tmp/checkpoint with spaces",
        TRAIN_FILE=str(train),
        VAL_FILE=str(val),
        OUTPUT_DIR=str(tmp_path / "artifacts"),
        PYTHON=str(python),
        CAPTURE=str(capture),
        ENV_CAPTURE=str(environment),
    )
    env.pop("NVTE_ALLOW_NONDETERMINISTIC_ALGO", None)

    def run(*args, **settings):
        selected = dict(env, **settings)
        result = subprocess.run(
            ["bash", str(LAUNCHER), *args], cwd=tmp_path, env=selected, capture_output=True, text=True, timeout=30
        )
        calls = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        return result, calls, selected

    return run


def overrides(call):
    return dict(arg.split("=", 1) for arg in call if "=" in arg)


def test_ep1_recipe_and_paths_reach_both_native_invocations(launch):
    result, calls, env = launch()
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2 and "--cfg" in calls[0] and "--cfg" not in calls[1]
    for call in calls:
        values = overrides(call)
        assert values["data.train_files"] == env["TRAIN_FILE"]
        assert values["data.val_files"] == env["VAL_FILE"]
        assert values["actor_rollout_ref.model.path"] == "/tmp/checkpoint with spaces"
        assert values["reward.custom_reward_function.path"] == str(
            ROOT / "verl_omni/utils/reward_score/choice_reward.py"
        )
        assert values["trainer.v1.step_extension"] == "proximal_actor_reuse"
        assert values["actor_rollout_ref.actor.megatron.expert_model_parallel_size"] == "1"
        assert values["actor_rollout_ref.actor.megatron.expert_tensor_parallel_size"] == "null"
        assert values["data.train_batch_size"] == values["actor_rollout_ref.actor.ppo_mini_batch_size"] == "16"
        assert values["actor_rollout_ref.actor.ppo_epochs"] == "1"
        assert values["actor_rollout_ref.actor.shuffle"] == "false"
        assert values["trainer.v1.separate_async.parameter_sync_step"] == "1"
        assert values["actor_rollout_ref.actor.clip_ratio_low"] == "0.0003"
        assert values["actor_rollout_ref.actor.clip_ratio_high"] == "0.0004"


def test_caller_override_precedence_and_config_only(launch):
    result, calls, _ = launch(
        "trainer.v1.step_extension=null",
        "actor_rollout_ref.actor.megatron.expert_model_parallel_size=4",
        AVQA_CONFIG_ONLY="1",
    )
    assert result.returncode == 0, result.stderr
    assert len(calls) == 1
    values = overrides(calls[0])
    assert values["trainer.v1.step_extension"] == "null"
    assert values["actor_rollout_ref.actor.megatron.expert_model_parallel_size"] == "4"


@pytest.mark.parametrize("setting,expected", [(None, "0"), ("0", "0"), ("1", "1")])
def test_attention_environment_reaches_driver_and_ray(launch, setting, expected):
    settings = {} if setting is None else {"NVTE_ALLOW_NONDETERMINISTIC_ALGO": setting}
    result, calls, env = launch(**settings)
    assert result.returncode == 0, result.stderr
    assert [json.loads(line) for line in Path(env["ENV_CAPTURE"]).read_text().splitlines()] == [expected, expected]
    for call in calls:
        values = overrides(call)
        assert values["+ray_kwargs.ray_init.runtime_env.env_vars.NVTE_ALLOW_NONDETERMINISTIC_ALGO"] == (
            "${oc.env:NVTE_ALLOW_NONDETERMINISTIC_ALGO,0}"
        )


@pytest.mark.parametrize("phase,expected_calls", [("config", 1), ("train", 2)])
def test_native_failure_status_propagates_through_shell(launch, phase, expected_calls):
    result, calls, _ = launch(FAIL_PHASE=phase)
    assert result.returncode == 19
    assert len(calls) == expected_calls


def test_invalid_topology_fails_before_native_trainer(launch):
    result, calls, _ = launch(NUM_GPUS="7")
    assert result.returncode == 2
    assert calls == []


def test_missing_parquet_fails_before_native_trainer(launch):
    result, calls, _ = launch(TRAIN_FILE="/missing/pr777-train.parquet")
    assert result.returncode == 2
    assert calls == []
