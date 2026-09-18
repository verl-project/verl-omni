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
"""Launcher contract test for the MiniCPM-o 4.5 thinker GSPO AVQA recipe on the
separate-async trainer.

Pins the disaggregation wiring and the model wiring inherited from the colocated
recipe, not the tunable training hyperparameters.
"""

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_separate_async_v1.sh"
)


def _active_settings() -> set[str]:
    """Recipe settings, comments dropped."""
    lines = [line for line in _SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    return {line.strip().removesuffix("\\").rstrip() for line in lines}


def _setting_value(key: str) -> str:
    """Value of the single exact ``key=...`` setting line."""
    matches = [line for line in _active_settings() if line.startswith(f"{key}=")]
    assert len(matches) == 1, f"expected exactly one {key}= line, got {matches}"
    return matches[0].split("=", 1)[1]


def test_launcher_selects_the_separate_async_trainer():
    settings = _active_settings()

    assert "trainer.v1.trainer_mode=omni_separate_async" in settings
    assert "trainer.v1.separate_async.num_warmup_batches=1" in settings
    assert "trainer.v1.separate_async.parameter_sync_step=8" in settings
    assert "trainer.v1.sampler.max_off_policy_threshold=8" in settings
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl" in settings


def test_launcher_batch_identity_and_staleness_pin():
    # The upstream assert: 128 == parameter_sync_step * ppo_mini_batch_size, as in both parents.
    train_batch = int(_setting_value("data.train_batch_size"))
    sync_step = int(_setting_value("trainer.v1.separate_async.parameter_sync_step"))
    mini_batch = int(_setting_value("actor_rollout_ref.actor.ppo_mini_batch_size"))
    threshold = int(_setting_value("trainer.v1.sampler.max_off_policy_threshold"))

    assert train_batch == sync_step * mini_batch
    # At most one weight version of staleness: threshold == one sync cycle.
    assert threshold == sync_step


def test_launcher_splits_gpu_pools():
    settings = _active_settings()

    # Two standalone TP=1 replicas (one TP=2 replica is the config-only fallback).
    assert "actor_rollout_ref.rollout.nnodes=1" in settings
    assert "actor_rollout_ref.rollout.n_gpus_per_node=2" in settings
    assert "actor_rollout_ref.rollout.tensor_model_parallel_size=1" in settings
    # FSDP trainer pool: 2 GPUs (4 total).
    assert "trainer.n_gpus_per_node=2" in settings
    assert "trainer.nnodes=1" in settings
    # Rollout GPUs are dedicated, unlike the colocated recipe's 0.7 on shared GPUs.
    assert "actor_rollout_ref.rollout.gpu_memory_utilization=0.8" in settings


def test_launcher_ships_merged_lora_weights():
    settings = _active_settings()

    # Merged sync for accuracy parity with the colocated reference; the
    # merge=False adapter-delta flip is pinned by the naming test.
    assert "actor_rollout_ref.model.lora.merge=True" in settings
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings


def test_launcher_keeps_the_minicpm_model_wiring():
    settings = _active_settings()

    # Model and rollout: remote-code load, the AR pipeline, and its engine stack.
    assert "actor_rollout_ref.model.trust_remote_code=True" in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5"' in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar"' in settings
    assert "actor_rollout_ref.model.use_remove_padding=True" in settings

    # Dataset and rendering, inherited verbatim from the colocated recipe.
    assert "data.custom_cls.path=pkg://verl_omni.utils.dataset.omni_rl_datasets" in settings
    assert "data.custom_cls.name=MiniCPMORLHFDataset" in settings
    assert "+data.mm_processor_kwargs.sampling_rate=16000" in settings
    assert "+data.apply_chat_template_kwargs.enable_thinking=false" in settings
    # 4096-truncation would cut MiniCPM's think block short.
    assert "data.max_response_length=12288" in settings

    # The frozen towers are excluded by regex, not by a separate freeze flag.
    assert any(
        line.startswith("actor_rollout_ref.model.exclude_modules=") and ".*vpm.*" in line and ".*apm.*" in line
        for line in settings
    )

    # Rule-based reward: no GPU RM to trip the separate-async colocated-RM assert.
    assert "reward.reward_manager.name=minicpm_naive" in settings
    assert "reward.custom_reward_function.path=verl_omni/utils/reward_score/choice_reward.py" in settings

    # Standalone server actors import verl_omni through this export.
    assert "export VERL_USE_EXTERNAL_MODULES=verl_omni" in settings


def test_launcher_keeps_flash_attention_default():
    # sdpa broke train/rollout consistency in Qwen3-Omni experiments.
    assert not any("attn_implementation" in line or "sdpa" in line for line in _active_settings())


def test_launcher_keeps_upstream_sampling_default():
    # verl's training-side logprob path divides logits by the temperature in bf16
    # while the engine casts to fp32 first; inheriting the T=1.0 / top_p=1.0 /
    # top_k=-1 defaults keeps the division bit-exact. val_kwargs stays free.
    assert not any(
        ("rollout.temperature=" in line or "rollout.top_p=" in line or "rollout.top_k=" in line)
        and "val_kwargs" not in line
        for line in _active_settings()
    )


def test_launcher_sets_no_dead_config_overrides():
    # The config invariants (init_tts/use_cache/stream_input) live in the
    # training adapter's from_pretrained, so recipe-level override_config lines
    # would be dead config.
    assert not any("override_config" in line for line in _active_settings())
