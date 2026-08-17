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

"""CPU checks for LingBot Dense T2V example script launchers."""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "examples" / "flowgrpo_trainer" / "lingbot_video"


def _read_script(script: str) -> str:
    return (_SCRIPT_DIR / script).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script, experiment_name",
    [("run_lingbot_dense_t2v_lora_fsdp2.sh", "lingbot_dense_t2v_lora_fsdp2")],
)
def test_lingbot_training_scripts_use_simple_example_launcher_shape(script, experiment_name):
    text = _read_script(script)

    assert "set -x" in text
    assert "CHECK_CONFIG_ONLY" not in text
    assert "must_divide" not in text
    assert "python3 -m verl_omni.trainer.main_diffusion" in text
    assert f"experiment_name=${{EXPERIMENT_NAME:-{experiment_name}}}" in text
    assert "output_dir=${OUTPUT_DIR:-$WORKSPACE/outputs/$experiment_name}" in text
    assert "checkpoint_dir=${CHECKPOINT_DIR:-$output_dir/checkpoints}" in text
    assert 'exec > >(tee -a "$log_file") 2>&1' in text
    assert 'echo "Logging to $log_file"' in text


@pytest.mark.parametrize("script", ["run_lingbot_dense_t2v_lora_fsdp2.sh"])
def test_lingbot_training_scripts_keep_validated_defaults(script):
    text = _read_script(script)

    expected_snippets = [
        "data.train_batch_size=${TRAIN_BATCH_SIZE:-16}",
        "data.val_batch_size=${VAL_BATCH_SIZE:-16}",
        "actor_rollout_ref.rollout.n=${ROLLOUT_GROUP_SIZE:-8}",
        "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-2}",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=${ENABLE_GRADIENT_CHECKPOINTING:-True}",
        "actor_rollout_ref.rollout.pipeline.num_frames=${NUM_FRAMES:-81}",
        "actor_rollout_ref.rollout.algo.noise_level=${ROLLOUT_NOISE_LEVEL:-0.7}",
        "actor_rollout_ref.rollout.algo.sde_type=${ROLLOUT_SDE_TYPE:-dance_sde}",
        "actor_rollout_ref.rollout.pipeline.shift=${FLOW_SHIFT:-3.0}",
        "actor_rollout_ref.rollout.pipeline.guidance_scale=${GUIDANCE_SCALE:-3.0}",
        "actor_rollout_ref.rollout.pipeline.num_inference_steps=${NUM_INFERENCE_STEPS:-10}",
        "actor_rollout_ref.rollout.val_kwargs.pipeline.num_inference_steps=${VAL_NUM_INFERENCE_STEPS:-40}",
        "actor_rollout_ref.actor.optim.lr=${LR:-1e-5}",
        "actor_rollout_ref.model.lora_rank=${LORA_RANK:-64}",
        "actor_rollout_ref.model.lora_alpha=${LORA_ALPHA:-128}",
        'trainer.logger="${TRAINER_LOGGER:-',
        'actor_rollout_ref.rollout.algo.sde_window_range="${SDE_WINDOW_RANGE:-[0,5]}"',
        "reward.custom_reward_function.name=${REWARD_FUNCTION_NAME:-compute_score_hpsv3}",
        "trainer.rollout_data_dir=$rollout_data_dir",
        "trainer.validation_data_dir=$val_data_dir",
        "trainer.resume_mode=${RESUME_MODE:-auto}",
    ]
    for snippet in expected_snippets:
        assert snippet in text


def test_lingbot_fsdp2_script_sets_fsdp2_specific_knobs():
    text = _read_script("run_lingbot_dense_t2v_lora_fsdp2.sh")

    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP:-2}" in text
    assert "actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${ACTOR_SP:-1}" in text
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}" in text
    assert "actor_rollout_ref.actor.strategy=${ACTOR_STRATEGY:-fsdp2}" in text
