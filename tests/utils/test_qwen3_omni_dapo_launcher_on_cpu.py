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

from pathlib import Path

DAPO_WITHOUT_DYNAMIC_SAMPLING_SETTINGS = (
    "python3 -m verl_omni.trainer.main_omni",
    "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla",
    "actor_rollout_ref.actor.clip_ratio_low=0.2",
    "actor_rollout_ref.actor.clip_ratio_high=0.28",
    "actor_rollout_ref.actor.clip_ratio_c=10.0",
    "actor_rollout_ref.actor.loss_agg_mode=token-mean",
    "actor_rollout_ref.actor.use_kl_loss=false",
    "actor_rollout_ref.actor.entropy_coeff=0",
    "algorithm.trainer_type=policy_gradient",
    "algorithm.sample_source=online",
    "algorithm.adv_estimator=grpo",
    "algorithm.use_kl_in_reward=false",
    "algorithm.filter_groups.enable=false",
    "reward.reward_manager.source=register",
    '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"',
)


def _script_settings(script: str) -> set[str]:
    return {line.strip().removesuffix("\\").rstrip() for line in script.splitlines()}


def _assert_dapo_without_dynamic_sampling_contract(script: str) -> None:
    settings = _script_settings(script)
    assert set(DAPO_WITHOUT_DYNAMIC_SAMPLING_SETTINGS) <= settings
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" not in settings
    assert "algorithm.filter_groups.enable=true" not in settings


def test_dapo_example_launcher_has_phase_one_contract():
    repo_root = Path(__file__).parents[2]
    launcher = (repo_root / "examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh").read_text(
        encoding="utf-8"
    )

    _assert_dapo_without_dynamic_sampling_contract(launcher)
    settings = _script_settings(launcher)
    assert ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" in launcher
    assert "actor_rollout_ref.actor.freeze_vision_tower=true" in settings
    assert 'TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/avqa_r1_6k/train.parquet"}' in launcher
    assert 'VAL_FILE=${VAL_FILE:-"$HOME/data/avqa_r1_6k/validation.parquet"}' in launcher
    assert {
        "data.custom_cls.name=QwenOmniRLHFDataset",
        "data.seed=42",
        "data.val_max_samples=-1",
        "data.validation_shuffle=false",
        "reward.reward_manager.name=naive",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py",
        "reward.custom_reward_function.name=compute_score",
        "trainer.val_before_train=true",
        "trainer.test_freq=10",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.top_k=-1",
    } <= settings
    assert "actor_rollout_ref.rollout.val_kwargs.temperature=1.0" not in settings
    assert "actor_rollout_ref.rollout.val_kwargs.top_p=0.7" not in settings
    assert "data.val_max_samples=4" not in settings
    assert "data.validation_shuffle=true" not in settings
    assert "trainer.val_before_train=false" not in settings
    assert "overlong_buffer_cfg" not in launcher


def test_dapo_tiny_random_smoke_matches_example_contract():
    repo_root = Path(__file__).parents[2]
    smoke = (repo_root / "tests/special_e2e/run_dapo_qwen3_omni_thinker_lora_v1_smoke.sh").read_text(encoding="utf-8")

    _assert_dapo_without_dynamic_sampling_contract(smoke)
    settings = _script_settings(smoke)
    assert "reward.reward_manager.name=dapo" in settings
    assert "build_qwen3_omni_tiny_random.py" in smoke
    assert "SKIP_COMPAT_DEPS_INSTALL:-0" in smoke
    assert 'trainer.total_training_steps="${TOTAL_TRAIN_STEPS}"' in smoke

    assert "data.max_response_length=512" in settings
    assert {
        "reward.reward_kwargs.max_resp_len=512",
        "reward.reward_kwargs.overlong_buffer_cfg.enable=true",
        "reward.reward_kwargs.overlong_buffer_cfg.len=128",
        "reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0",
        "reward.reward_kwargs.overlong_buffer_cfg.log=true",
    } <= settings


def test_dapo_dynamic_sampling_example_launcher_has_phase_three_contract():
    repo_root = Path(__file__).parents[2]
    launcher = (
        repo_root
        / "examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_dynamic_sampling_lora_v1.sh"
    ).read_text(encoding="utf-8")

    settings = _script_settings(launcher)
    # Same token-level DAPO policy contract as Phase 1, minus filter_groups.enable=false.
    assert set(DAPO_WITHOUT_DYNAMIC_SAMPLING_SETTINGS) - {"algorithm.filter_groups.enable=false"} <= settings
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" not in settings
    assert ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" in launcher
    assert "actor_rollout_ref.actor.freeze_vision_tower=true" in settings
    assert {
        "algorithm.filter_groups.enable=true",
        "algorithm.filter_groups.metric=acc",
        "reward.reward_model.enable=false",
        "reward.reward_manager.name=dapo",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py",
        "reward.custom_reward_function.name=compute_score",
    } <= settings
    # filter_groups requires the streaming (non-colocated) reward path.
    assert "reward.reward_model.enable=true" not in settings


def test_dapo_mmk12_example_launcher_has_phase_four_contract():
    repo_root = Path(__file__).parents[2]
    launcher = (
        repo_root / "examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_mmk12_v1.sh"
    ).read_text(encoding="utf-8")

    settings = _script_settings(launcher)
    # Same token-level DAPO policy contract as the AVQA dynamic sampling launcher.
    assert set(DAPO_WITHOUT_DYNAMIC_SAMPLING_SETTINGS) - {"algorithm.filter_groups.enable=false"} <= settings
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" not in settings
    assert ".*talker.*|.*code2wav.*|.*code_predictor.*|.*visual.*|.*audio_tower.*" in launcher
    assert "actor_rollout_ref.actor.freeze_vision_tower=true" in settings
    assert {
        "algorithm.filter_groups.enable=true",
        "algorithm.filter_groups.metric=acc",
        "reward.reward_model.enable=false",
        "reward.reward_manager.name=dapo",
        "reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py",
        "reward.custom_reward_function.name=compute_score",
        "reward.reward_kwargs.overlong_buffer_cfg.enable=true",
        "reward.reward_kwargs.overlong_buffer_cfg.len=1024",
        "reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0",
        "reward.reward_kwargs.max_resp_len=12288",
    } <= settings
    assert 'TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/mmk12/train.parquet"}' in launcher
    assert 'VAL_FILE=${VAL_FILE:-"$HOME/data/mmk12/test.parquet"}' in launcher
