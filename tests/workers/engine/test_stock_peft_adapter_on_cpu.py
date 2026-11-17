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
"""The generic PEFT fallback must preserve the existing Diffusers adapter API."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from peft import PeftModel, get_peft_model_state_dict

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


@pytest.mark.parametrize("native_api,safe_serialization", [(True, True), (True, False), (False, True)])
def test_native_and_wrapped_lora_lifecycle_and_reload(tmp_path, native_api, safe_serialization):
    """Keep native Diffusers models native, and wrap plain modules with public PEFT."""

    def make_model():
        if native_api:
            return SD3Transformer2DModel(
                sample_size=4,
                patch_size=2,
                in_channels=4,
                num_layers=1,
                attention_head_dim=4,
                num_attention_heads=2,
                joint_attention_dim=8,
                caption_projection_dim=8,
                pooled_projection_dim=8,
            )
        return torch.nn.Sequential(torch.nn.Linear(4, 4))

    def make_engine(path=None):
        engine = LoRAAdapterMixin()
        engine.model_config = SimpleNamespace(
            lora_rank=2,
            lora_alpha=2,
            lora_init_weights=True,
            target_modules=["to_q"] if native_api else ["0"],
            target_parameters=None,
            exclude_modules=None,
            policy_state_adapters=["default", "old"],
            lora_adapter_path=path,
            use_shm=False,
        )
        engine.module = engine._build_lora_module(make_model().requires_grad_(False))
        return engine

    engine = make_engine()
    assert isinstance(engine.module, PeftModel) is not native_api
    assert set(engine.module.peft_config) == {"default", "old"}
    engine.copy_adapter()
    with engine.use_adapter("old"):
        assert any(p.requires_grad for n, p in engine.module.named_parameters() if ".old." in n)
        assert not any(p.requires_grad for n, p in engine.module.named_parameters() if ".default." in n)
    with engine.disable_adapter():
        assert all(m.disable_adapters for m in engine.module.modules() if hasattr(m, "lora_A"))
    assert all(not m.disable_adapters for m in engine.module.modules() if hasattr(m, "lora_A"))
    if native_api:
        StableDiffusion3Pipeline.save_lora_weights(
            tmp_path,
            transformer_lora_layers=get_peft_model_state_dict(engine.module),
            safe_serialization=safe_serialization,
        )
    else:
        engine.module.save_pretrained(tmp_path, selected_adapters=["default"])
    restored = make_engine(str(tmp_path))
    assert isinstance(restored.module, PeftModel) is not native_api
    assert set(restored.module.peft_config) == {"default", "old"}
    restored.copy_adapter()
