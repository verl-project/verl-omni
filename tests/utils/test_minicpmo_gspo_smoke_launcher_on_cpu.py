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
"""Launcher contract test for the MiniCPM-o 4.5 thinker GSPO collocated-sync smoke.

Pins the wiring the smoke shares with the production recipe
(examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh)
plus the deviations the GPU CI environment forces (the sdpa attention pin),
not the tunable hyperparameters.
"""

from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "tests/special_e2e/run_gspo_minicpmo_thinker_lora_v1_smoke.sh"


def _active_settings() -> set[str]:
    """Smoke settings, comments dropped."""
    lines = [line for line in _SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    return {line.strip().removesuffix("\\").rstrip() for line in lines}


def _setting_value(key: str) -> str:
    """Value of the single exact ``key=...`` setting line."""
    matches = [line for line in _active_settings() if line.startswith(f"{key}=")]
    assert len(matches) == 1, f"expected exactly one {key}= line, got {matches}"
    return matches[0].split("=", 1)[1]


def test_smoke_wires_the_minicpm_path():
    settings = _active_settings()

    # Model and rollout: remote-code load, the AR pipeline, and its engine stack.
    # GPU CI ships no flash-attn while OmniModelConfig defaults the actor attention
    # to flash_attention_2, so the smoke pins sdpa like the Qwen3-Omni smoke.
    assert "actor_rollout_ref.model.trust_remote_code=True" in settings
    assert "+actor_rollout_ref.model.override_config.attn_implementation=sdpa" in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5"' in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar"' in settings
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings
    assert "actor_rollout_ref.model.lora.merge=True" in settings
    assert "actor_rollout_ref.model.use_remove_padding=True" in settings
    # The packed path and rollout ids flow through the training adapter.
    assert "actor_rollout_ref.rollout.name=vllm_omni" in settings

    # Dataset and rendering: the AVQA media loader at the Whisper sampling rate.
    assert "data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets" in settings
    assert "data.custom_cls.name=MiniCPMORLHFDataset" in settings
    assert "+data.mm_processor_kwargs.sampling_rate=16000" in settings
    assert "+data.apply_chat_template_kwargs.enable_thinking=false" in settings

    # Reward: the answer-tag-preserving manager plus the choice reward.
    assert "reward.reward_manager.source=register" in settings
    assert "reward.reward_manager.name=minicpm_naive" in settings
    assert "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py" in settings
    assert "reward.custom_reward_function.name=compute_score" in settings

    # Frozen towers excluded by regex, as in the recipe (the regex lives in a
    # shell variable so the exclude_modules line stays readable).
    assert any(line.startswith("EXCLUDE_MODULES=") and ".*vpm.*" in line and ".*apm.*" in line for line in settings)
    assert any(
        line.startswith("actor_rollout_ref.model.exclude_modules=") and "${EXCLUDE_MODULES}" in line
        for line in settings
    )

    # Tiny random checkpoint + dummy AVQA data are built by this repo's helpers.
    assert any(
        line.startswith('python3 "${REPO_ROOT}/tests/special_e2e/build_minicpmo_tiny_random.py"') for line in settings
    )
    assert any(
        line.startswith('python3 "${REPO_ROOT}/tests/special_e2e/create_dummy_minicpmo_avqa_data.py"')
        for line in settings
    )
    # Both splits gate the rebuild: the trainer constructs the val dataset at
    # setup even with val_before_train=false.
    assert 'if [ ! -f "${DATA_DIR}/train.parquet" ] || [ ! -f "${DATA_DIR}/validation.parquet" ]; then' in settings


def test_smoke_keeps_upstream_sampling_defaults():
    """The smoke overrides no non-val sampling settings."""
    settings = _active_settings()

    # Upstream sampling defaults keep the training/engine logprob division bit-exact;
    # val_kwargs sampling stays free to differ.
    assert not any(
        ("rollout.temperature=" in line or "rollout.top_p=" in line or "rollout.top_k=" in line)
        and "val_kwargs" not in line
        for line in settings
    )


def test_smoke_scales_collocated_resources_consistently():
    settings = _active_settings()

    # One collocated pool: the FSDP2 actor shards over the same GPUs the TP rollout uses.
    assert _setting_value("actor_rollout_ref.rollout.tensor_model_parallel_size") == '"${NUM_GPUS}"'
    assert _setting_value("trainer.n_gpus_per_node") == '"${NUM_GPUS}"'
    assert "actor_rollout_ref.actor.strategy=fsdp2" in settings
    assert "actor_rollout_ref.ref.strategy=fsdp2" in settings

    # Smoke batch identity: mini-batches divide the sampled prompt batch evenly.
    train_batch = int(_setting_value("data.train_batch_size"))
    mini_batch = int(_setting_value("actor_rollout_ref.actor.ppo_mini_batch_size"))
    rollout_n = int(_setting_value("actor_rollout_ref.rollout.n"))
    assert train_batch % mini_batch == 0
    assert rollout_n >= 2  # GSPO needs a group to normalize over

    # The engine prompt budget covers the data prompt budget plus the image-slot
    # margin (query_num), mirroring the recipe's 4160 = 4096 + 64.
    prompt_length = int(_setting_value("actor_rollout_ref.rollout.prompt_length"))
    max_prompt = int(_setting_value("data.max_prompt_length"))
    response = int(_setting_value("data.max_response_length"))
    assert prompt_length == max_prompt + 64
    assert response >= 64  # room for an <answer> span sampled from a random policy

    # No checkpoints, no resume: the smoke only proves the training path.
    assert "trainer.save_freq=-1" in settings
    assert "trainer.resume_mode=disable" in settings
    assert "trainer.val_before_train=false" in settings
