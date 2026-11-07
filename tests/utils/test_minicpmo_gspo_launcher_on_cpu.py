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
"""Launcher contract test for the MiniCPM-o 4.5 thinker GSPO AVQA recipe.

Pins the wiring that would silently break the integration, not the training
hyperparameters (those track the Qwen3-Omni GSPO recipe and stay tunable).
"""

from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "examples/gspo_trainer/minicpm/run_minicpmo_4_5_thinker_gspo_lora_avqa_v1.sh"
)


def _active_settings() -> set[str]:
    """Recipe settings, comments dropped."""
    lines = [line for line in _SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    return {line.strip().removesuffix("\\").rstrip() for line in lines}


def test_minicpmo_gspo_launcher_wires_the_minicpm_path():
    settings = _active_settings()

    # Model and rollout: remote-code load, the AR pipeline, and its engine stack.
    assert "actor_rollout_ref.model.trust_remote_code=True" in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="minicpmo_4_5"' in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar"' in settings
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings
    # LoRA weight sync pushes the merged adapter into the engine.
    assert "actor_rollout_ref.model.lora.merge=True" in settings
    # The packed path is implemented by the training adapter.
    assert "actor_rollout_ref.model.use_remove_padding=True" in settings

    # Dataset and rendering.
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
    # The config invariants (init_tts/use_cache/stream_input) live in the adapter's
    # from_pretrained, so recipe-level override_config lines would be dead config.
    assert not any("override_config" in line for line in settings)


def test_minicpmo_gspo_launcher_keeps_flash_attention_default():
    # sdpa broke train/rollout consistency in Qwen3-Omni experiments.
    assert not any("attn_implementation" in line or "sdpa" in line for line in _active_settings())


def test_minicpmo_gspo_launcher_keeps_upstream_sampling_default():
    # verl's training-side logprob path divides logits by the temperature in bf16
    # while the engine casts to fp32 first, so any T != 1.0 shows the mismatch in
    # the parity metric; inheriting the T=1.0 / top_p=1.0 / top_k=-1 defaults keeps
    # the division bit-exact. val_kwargs sampling stays free to differ.
    assert not any(
        ("rollout.temperature=" in line or "rollout.top_p=" in line or "rollout.top_k=" in line)
        and "val_kwargs" not in line
        for line in _active_settings()
    )
