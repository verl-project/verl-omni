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

import os
import shutil
import subprocess
from pathlib import Path

import pytest

LAUNCHER = (
    Path(__file__).resolve().parents[2] / "examples/gspo_trainer/qwen3_omni" / "run_qwen3_omni_thinker_gspo_avqa_v1.sh"
)


def test_avqa_npu_launcher_wires_v1_multimodal_training():
    avqa_launcher = LAUNCHER.read_text(encoding="utf-8")

    required_settings = (
        "python3 -m verl_omni.trainer.main_omni",
        "data.custom_cls.name=QwenOmniRLHFDataset",
        "++data.mm_processor_kwargs.sampling_rate=16000",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.rollout.name=vllm_omni",
        "engine_kwargs.vllm_omni.pipeline_name=qwen3_omni_moe",
        "reward.reward_manager.source=register",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py",
    )
    assert all(setting in avqa_launcher for setting in required_settings)
    assert "models.transformers" not in avqa_launcher


@pytest.mark.parametrize(
    ("model", "runner", "expected_runner", "graph_mode", "device_count", "exit_code"),
    [
        ("Ascend950DT", "unset", "0", "FULL_AND_PIECEWISE", 8, 0),
        ("Ascend910B", "unset", "unset", "PIECEWISE", 16, 0),
        ("", "1", "1", "PIECEWISE", 16, 0),
        ("Ascend950DT", "1", "0", "FULL_AND_PIECEWISE", 8, 7),
    ],
    ids=["ascend950dt", "other-npu", "gpu", "training-failure"],
)
def test_avqa_launcher_selects_runner_and_propagates_training_status(
    tmp_path, model, runner, expected_runner, graph_mode, device_count, exit_code
):
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("Bash is required to exercise the launcher")
    for env_script in (tmp_path / "cann/set_env.sh", tmp_path / "nnal/atb/set_env.sh"):
        env_script.parent.mkdir(parents=True)
        env_script.touch()

    stub = r"""
unset DEVICE NUM_GPUS_ACTOR_ROLLOUT_REWARD ROLLOUT_TP VLLM_OMNI_CUDAGRAPH_MODE VLLM_USE_V2_MODEL_RUNNER
export ASCEND_HOME_PATH=./cann MODEL_PATH=test-model TRAIN_FILE=test-train VAL_FILE=test-val
export TEST_NPU_MODEL="$2" TEST_TRAINER_EXIT="$4"
[[ "$3" == unset ]] || export VLLM_USE_V2_MODEL_RUNNER="$3"
npu-smi() {
    [[ "$TEST_NPU_MODEL" ]] || return 1
    for id in {0..7}; do printf '| %s | %s | OK |\n' "$id" "$TEST_NPU_MODEL"; done
}
python3() {
    printf 'runner=%s\n' "${VLLM_USE_V2_MODEL_RUNNER-unset}" > runner
    printf '%s\n' "$@" > trainer-argv
    printf 'trainer output\n'
    return "$TEST_TRAINER_EXIT"
}
export -f npu-smi python3
exec "$BASH" "$1"
"""
    env = {key: value for key, value in os.environ.items() if key not in {"BASH_ENV", "SHELLOPTS"}}
    result = subprocess.run(
        [bash, "-c", stub, "launcher", LAUNCHER.as_posix(), model, runner, str(exit_code)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stderr
    assert (tmp_path / "runner").read_text(encoding="utf-8").strip() == f"runner={expected_runner}"
    args = (tmp_path / "trainer-argv").read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["-m", "verl_omni.trainer.main_omni"]
    assert f"++actor_rollout_ref.rollout.engine_kwargs.vllm_omni.compilation_config.cudagraph_mode={graph_mode}" in args
    assert f"trainer.n_gpus_per_node={device_count}" in args
    assert "trainer output" in (tmp_path / "run_qwen3omni_npu_avqa_v1.log").read_text(encoding="utf-8").splitlines()
