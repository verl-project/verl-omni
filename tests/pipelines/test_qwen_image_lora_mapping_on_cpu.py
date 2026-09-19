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
"""CPU regressions for Qwen-Image live LoRA mapping and binding validation."""

from types import SimpleNamespace

import pytest
import torch


def test_qwen_lora_maps_output_projection_keys_and_targets_without_mutating_inputs():
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    tensor = torch.ones(2, 3)
    name = "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight"
    config = {"target_modules": ["to_out.0", "to_q", "transformer_blocks.0.attn.to_out.0"]}
    mapped, mapped_config = QwenImagePipelineWithLogProb.map_lora_update_to_engine({name: tensor}, config)
    assert list(mapped) == ["transformer.transformer_blocks.0.attn.to_out.lora_A.weight"]
    assert next(iter(mapped.values())) is tensor
    assert mapped_config["target_modules"] == ["to_out", "to_q", "transformer_blocks.0.attn.to_out"]
    assert config["target_modules"][0] == "to_out.0"


def test_qwen_lora_rejects_silent_partial_binding():
    from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

    model = SimpleNamespace(loras={"q": object(), "out": object()})
    with pytest.raises(ValueError, match="1 unbound modules"):
        QwenImagePipelineWithLogProb._validate_diffusion_lora_binding(
            lora_model=model, bound_lora_names=frozenset({"q"})
        )
    QwenImagePipelineWithLogProb._validate_diffusion_lora_binding(
        lora_model=model, bound_lora_names=frozenset({"q", "out"})
    )
