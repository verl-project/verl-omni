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
"""Launcher contract tests for the engine-side LoRA merge recipes.

Pins the wiring that selects engine-side merge, not the training
hyperparameters (those track the runtime-adapter / actor-merge recipes and
stay tunable).
"""

from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
_QWEN_IMAGE = _EXAMPLES / "flowgrpo_trainer/qwen_image/run_qwen_image_ocr_lora_v1.sh"
_QWEN3_OMNI = _EXAMPLES / "gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh"


def _active_settings(path: Path) -> set[str]:
    """Recipe settings, comments dropped."""
    lines = [line for line in path.read_text().splitlines() if not line.lstrip().startswith("#")]
    return {line.strip().removesuffix("\\").rstrip() for line in lines}


def test_qwen_image_launcher_selects_engine_merge():
    """Qwen-Image recipe selects engine merge with no hand-set engine kwargs."""
    settings = _active_settings(_QWEN_IMAGE)

    # The one knob that selects engine-side merge.
    assert "actor_rollout_ref.model.lora.merge=Engine" in settings
    # The engine loads real base weights (no dense bootstrap exists in engine mode).
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings
    # LoRA stays configured as in the runtime-adapter recipe.
    assert "actor_rollout_ref.model.lora_rank=64" in settings
    assert "actor_rollout_ref.model.lora_alpha=128" in settings
    # Engine-args selection is driven by the rollout, not hand-set engine kwargs.
    assert not any("weight_transfer_config" in line or "lora_application" in line for line in settings)


def test_qwen3_omni_launcher_selects_engine_merge():
    """Qwen3-Omni recipe flips actor-side merge to engine merge, no adapter flags."""
    settings = _active_settings(_QWEN3_OMNI)

    # Engine-side merge replaces the actor-side merge of the earlier recipe.
    assert "actor_rollout_ref.model.lora.merge=Engine" in settings
    assert "actor_rollout_ref.model.lora.merge=True" not in settings
    # The trainable AR stage must not carry runtime-adapter flags.
    assert not any("enable_lora" in line or "max_lora_rank" in line for line in settings)
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode="ar"' in settings
    assert '+actor_rollout_ref.rollout.engine_kwargs.vllm_omni.pipeline_name="qwen3_omni_moe"' in settings
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings
