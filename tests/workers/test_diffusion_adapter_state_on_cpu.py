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
"""CPU tests for diffusion adapter state utilities."""

import torch

import verl_omni.workers.engine.fsdp.diffusers_impl as diffusers_impl
from verl_omni.workers.engine.fsdp.diffusers_impl import NFTDiffusersFSDPEngine


class FakeAdapterLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.ModuleDict(
            {
                "default": torch.nn.Linear(2, 2, bias=False),
                "old": torch.nn.Linear(2, 2, bias=False),
            }
        )
        self.lora_B = torch.nn.ModuleDict(
            {
                "default": torch.nn.Linear(2, 2, bias=False),
                "old": torch.nn.Linear(2, 2, bias=False),
            }
        )


class FakeAdapterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = FakeAdapterLayer()
        self.active_adapter = "default"

    def set_adapter(self, name: str):
        self.active_adapter = name
        for param_name, param in self.named_parameters():
            param.requires_grad_(f".{name}." in param_name)

    def forward(self, x):
        layer = self.layer.lora_A[self.active_adapter]
        return layer(x)


def _engine() -> NFTDiffusersFSDPEngine:
    engine = object.__new__(NFTDiffusersFSDPEngine)
    engine.module = FakeAdapterModule()
    return engine


def test_copy_adapter_copies_default_to_old_and_freezes_target() -> None:
    engine = _engine()
    with torch.no_grad():
        for name, param in engine.module.named_parameters():
            fill_value = 3.0 if ".default." in name else -1.0
            param.fill_(fill_value)

    engine.copy_adapter(source="default", target="old")

    params = dict(engine.module.named_parameters())
    for name, param in params.items():
        if ".old." not in name:
            continue
        source_name = name.replace(".old.", ".default.")
        torch.testing.assert_close(param, params[source_name])
        assert param.requires_grad is False


def test_ema_update_adapter_blends_old_toward_default() -> None:
    engine = _engine()
    with torch.no_grad():
        for name, param in engine.module.named_parameters():
            fill_value = 10.0 if ".default." in name else 2.0
            param.fill_(fill_value)

    engine.ema_update_adapter(source="default", target="old", decay=0.25)

    for name, param in engine.module.named_parameters():
        if ".old." in name:
            torch.testing.assert_close(param, torch.full_like(param, 8.0))
            assert param.requires_grad is False


def test_use_adapter_restores_default() -> None:
    engine = _engine()

    with engine.use_adapter("old"):
        assert engine.module.active_adapter == "old"

    assert engine.module.active_adapter == "default"


def test_old_adapter_stays_frozen_after_old_reference_default_forward_sequence() -> None:
    engine = _engine()
    engine.copy_adapter(source="default", target="old")

    x = torch.ones(1, 2)
    with engine.use_adapter("old"):
        with torch.no_grad():
            engine.module(x)
    engine.module.disable_adapters = lambda: engine.module.set_adapter("default")
    engine.module.enable_adapters = lambda: engine.module.set_adapter("default")
    engine.module(x).sum().backward()

    for name, param in engine.module.named_parameters():
        if ".old." in name:
            assert param.requires_grad is False
            assert param.grad is None, name


def test_get_per_tensor_param_uses_selected_adapter_for_lora_sync(monkeypatch) -> None:
    class FakePeftConfig:
        def to_dict(self):
            return {"r": 2}

    engine = _engine()
    engine.module.peft_config = {"default": FakePeftConfig()}
    engine._is_offload_param = False
    calls = {}

    def fake_collect_lora_params(module, layered_summon, base_sync_done, is_diffusers, adapter_name):
        calls["adapter_name"] = adapter_name
        calls["active_adapter"] = module.active_adapter
        return {"layer.weight": torch.ones(1)}

    monkeypatch.setattr(diffusers_impl, "load_fsdp_model_to_gpu", lambda module: None)
    monkeypatch.setattr(diffusers_impl, "convert_weight_keys", lambda params, module: params)
    monkeypatch.setattr(diffusers_impl, "collect_lora_params", fake_collect_lora_params)

    per_tensor_param, peft_config = engine.get_per_tensor_param(base_sync_done=True, adapter_name="old")

    assert calls == {"adapter_name": "old", "active_adapter": "old"}
    assert peft_config == {"r": 2}
    assert list(per_tensor_param)[0][0] == "transformer.layer.weight"
    assert engine.module.active_adapter == "default"
