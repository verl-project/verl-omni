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
"""Launcher contract test for the MiniCPM-o 4.5 thinker GSPO AVQA recipe."""

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh"
)


def _script_settings(script: str) -> set[str]:
    return {line.strip().removesuffix("\\").rstrip() for line in script.splitlines()}


def test_minicpmo_gspo_launcher_contract():
    settings = _script_settings(_SCRIPT.read_text())

    # GSPO loss knobs copied verbatim from the Qwen3-Omni GSPO recipe.
    assert "actor_rollout_ref.actor.policy_loss.loss_mode=gspo" in settings
    assert "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean" in settings
    assert "actor_rollout_ref.actor.clip_ratio_low=3e-4" in settings
    assert "actor_rollout_ref.actor.clip_ratio_high=4e-4" in settings
    assert "actor_rollout_ref.actor.clip_ratio_c=10.0" in settings
    assert "actor_rollout_ref.actor.use_kl_loss=false" in settings
    assert "algorithm.adv_estimator=grpo" in settings
    assert "algorithm.use_kl_in_reward=False" in settings

    # rmpad stays ON: the packed path is implemented in the training adapter.
    assert "actor_rollout_ref.model.use_remove_padding=True" in settings

    # LoRA + weight-sync contract.
    assert "actor_rollout_ref.model.lora.merge=True" in settings
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings

    # MiniCPM-specific wiring.
    assert "actor_rollout_ref.model.trust_remote_code=True" in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5"' in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar"' in settings
    assert "data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets" in settings
    assert "data.custom_cls.name=MiniCPMORLHFDataset" in settings
    assert "+data.mm_processor_kwargs.sampling_rate=16000" in settings
    # OpenBMB's supported render mode: the template pre-fills an empty think
    # block instead of making the model close its own.
    assert "+data.apply_chat_template_kwargs.enable_thinking=false" in settings
    assert "+actor_rollout_ref.model.override_config.init_tts=false" in settings
    assert "+actor_rollout_ref.model.override_config.use_cache=false" in settings
    assert "+actor_rollout_ref.model.override_config.stream_input=false" in settings
    assert any(
        line.startswith("actor_rollout_ref.model.exclude_modules=") and ".*vpm.*" in line and ".*apm.*" in line
        for line in settings
    )

    # AVQA reward and proven hyperparameters from the Qwen3-Omni recipe.
    assert "data.max_response_length=12288" in settings  # 4096 CLI overrides truncate MiniCPM's think
    assert "actor_rollout_ref.rollout.temperature=0.6" in settings  # checkpoint think default, not Qwen's 1.0
    assert "reward.reward_manager.source=register" in settings
    assert "reward.reward_manager.name=naive" in settings
    assert "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py" in settings
    assert "reward.custom_reward_function.name=compute_score" in settings
    assert "data.train_batch_size=128" in settings
    assert "actor_rollout_ref.rollout.n=16" in settings
    assert "trainer.n_gpus_per_node=4" in settings


def test_minicpmo_gspo_launcher_keeps_flash_attention_default():
    # sdpa breaks train/rollout consistency; the recipe must not override the
    # flash_attention_2 default baked into OmniModelConfig.
    active_lines = [line for line in _SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    assert not any("attn_implementation" in line for line in active_lines)
    assert not any("sdpa" in line for line in active_lines)
