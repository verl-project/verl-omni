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
"""CPU tests for LoRAAdapterMixin without FSDP or Ray."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from verl_omni.workers.engine.lora_adapter_mixin import LoRAAdapterMixin


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


class _TestEngine(LoRAAdapterMixin):
    def __init__(self, model_config, module):
        self.model_config = model_config
        self.module = module


def _lora_config():
    return LoraConfig(r=4, lora_alpha=8, target_modules=["linear"], init_lora_weights="gaussian")


def _make_dual_adapter_module():
    module = get_peft_model(_TinyBackbone(), _lora_config())
    module.add_adapter("old", _lora_config())
    module.set_adapter("default")
    return module


def _make_engine(module=None):
    model_config = SimpleNamespace(
        lora_adapter_path=None,
        policy_state_adapters=("default", "old"),
        lora_rank=4,
        lora_alpha=8,
        lora_init_weights="gaussian",
        target_modules=["linear"],
        target_parameters=None,
        exclude_modules=None,
        use_shm=False,
    )
    module = module or _make_dual_adapter_module()
    return _TestEngine(model_config, module), module


def _adapter_param_tensors(module, adapter_name: str) -> list[torch.Tensor]:
    module.set_adapter(adapter_name)
    return [p.detach().clone() for p in module.parameters() if p.requires_grad]


class TestLoRAAdapterMixinCopy:
    def test_copy_adapter_syncs_params(self):
        engine, module = _make_engine()
        module.set_adapter("default")
        for param in module.parameters():
            if param.requires_grad:
                param.data.fill_(1.0)
        engine.copy_adapter(source="default", target="old")
        for param in _adapter_param_tensors(module, "old"):
            assert torch.all(param == 1.0)


class TestLoRAAdapterMixinEma:
    def test_ema_update_blends_params(self):
        engine, module = _make_engine()
        module.set_adapter("default")
        for param in module.parameters():
            if param.requires_grad:
                param.data.fill_(1.0)
        module.set_adapter("old")
        for param in module.parameters():
            if param.requires_grad:
                param.data.fill_(0.0)
        engine.ema_update_adapter(source="default", target="old", decay=0.5)
        for param in _adapter_param_tensors(module, "old"):
            assert torch.allclose(param, torch.full_like(param, 0.5))

    def test_ema_decay_out_of_range_raises(self):
        engine, _ = _make_engine()
        with pytest.raises(ValueError, match="Adapter EMA decay"):
            engine.ema_update_adapter(decay=1.1)


class TestLoRAAdapterMixinContext:
    def test_use_adapter_restores_default(self):
        engine, module = _make_engine()
        assert module.active_adapter == "default"
        with engine.use_adapter("old"):
            assert module.active_adapter == "old"
        assert module.active_adapter == "default"
