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
"""Launcher contract for the Qwen3-Omni colocated-sync GPU LoRA recipes.

Pins adapter-only weight sync on rollout base weights loaded from the checkpoint, so
every sync, the first included, sends only adapter tensors.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RECIPES = [
    "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_v1.sh",
    "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_mmk12_v1.sh",
    "examples/gspo_trainer/qwen3_omni/run_qwen3_omni_thinker_gspo_lora_avqa_v1.sh",
    "examples/dapo_trainer/qwen3_omni/run_qwen3_omni_thinker_dapo_lora_v1.sh",
    "tests/special_e2e/run_gspo_qwen3_omni_thinker_lora_v1_smoke.sh",
]


@pytest.mark.parametrize("recipe", _RECIPES)
def test_recipe_syncs_adapter_tensors_onto_real_base_weights(recipe):
    lines = (_REPO_ROOT / recipe).read_text(encoding="utf-8").splitlines()
    settings = {line.strip().removesuffix("\\").rstrip().lower() for line in lines if not line.lstrip().startswith("#")}

    assert "actor_rollout_ref.model.lora.merge=false" in settings
    assert "actor_rollout_ref.rollout.load_format=safetensors" in settings
    assert not any(setting.startswith("trainer.v1.trainer_mode=") for setting in settings)
