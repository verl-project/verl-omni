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
"""CPU tests for loading a pre-trained LoRA adapter into a non-diffusers model."""

import logging

import pytest
import torch
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file
from torch import nn

from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase


class _TinyModel(NonDiffusersModelBase):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)

    @classmethod
    def from_pretrained(cls, model_path, torch_dtype=torch.float32, **kwargs):
        return cls()

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _ThreeLayerModel(_TinyModel):
    def __init__(self):
        super().__init__()
        self.other = nn.Linear(4, 4)
        self.lora_head = nn.Linear(4, 4)


def _lora_config(target_modules=("proj",)) -> LoraConfig:
    return LoraConfig(r=2, lora_alpha=2, target_modules=list(target_modules))


def _trained_adapter_state(model_cls=_TinyModel, config=None) -> dict[str, torch.Tensor]:
    donor = model_cls()
    donor.add_adapter(config or _lora_config())
    return {key: torch.full_like(value, 0.5) for key, value in get_peft_model_state_dict(donor).items()}


def _write_adapter(adapter_dir, state_dict: dict[str, torch.Tensor], config=None) -> str:
    (config or _lora_config()).save_pretrained(str(adapter_dir))
    save_file(state_dict, str(adapter_dir / "adapter_model.safetensors"))
    return str(adapter_dir)


def test_load_lora_adapter_copies_matching_weights(tmp_path):
    state = _trained_adapter_state()
    model = _TinyModel()

    model.load_lora_adapter(_write_adapter(tmp_path / "adapter", state))

    loaded = get_peft_model_state_dict(model)
    assert loaded.keys() == state.keys()
    for key, value in state.items():
        torch.testing.assert_close(loaded[key], value)


def test_load_lora_adapter_rejects_checkpoint_keys_the_model_lacks(tmp_path):
    state = _trained_adapter_state() | {"other.lora_A.weight": torch.zeros(2, 4)}

    with pytest.raises(ValueError, match=r"other\.lora_A\.weight"):
        _TinyModel().load_lora_adapter(_write_adapter(tmp_path / "adapter", state))


def test_rejected_checkpoint_leaves_no_adapter_behind(tmp_path):
    state = _trained_adapter_state()
    model = _TinyModel()

    with pytest.raises(ValueError):
        model.load_lora_adapter(_write_adapter(tmp_path / "bad", state | {"other.lora_A.weight": torch.zeros(2, 4)}))

    assert "default" not in model.peft_config
    assert type(model.proj) is nn.Linear
    assert all(param.requires_grad for param in model.parameters())
    model.load_lora_adapter(_write_adapter(tmp_path / "good", state))
    loaded = get_peft_model_state_dict(model)
    for key, value in state.items():
        torch.testing.assert_close(loaded[key], value)


def test_rejected_checkpoint_restores_existing_adapters_and_layers(tmp_path):
    model = _ThreeLayerModel()
    model.add_adapter(_lora_config(), adapter_name="a")
    model.add_adapter(_lora_config(), adapter_name="b")
    model.set_adapter("b")
    keys_before = set(model.state_dict())
    config = _lora_config(["proj", "other", "lora_head"])
    state = _trained_adapter_state(_ThreeLayerModel, config) | {"missing.lora_A.weight": torch.zeros(2, 4)}

    with pytest.raises(ValueError):
        model.load_lora_adapter(_write_adapter(tmp_path / "bad", state, config), adapter_name="c")

    assert model.proj.active_adapters == ["b"]
    assert type(model.other) is nn.Linear and type(model.lora_head) is nn.Linear
    assert set(model.state_dict()) == keys_before
    assert set(model.peft_config) == {"a", "b"}


def test_tensor_copy_failure_restores_existing_adapters_and_layers(tmp_path):
    model = _ThreeLayerModel()
    model.add_adapter(_lora_config(), adapter_name="a")
    model.add_adapter(_lora_config(), adapter_name="b")
    model.set_adapter("b")
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    requires_grad_before = {name: param.requires_grad for name, param in model.named_parameters()}
    config = _lora_config(["proj", "other", "lora_head"])
    state = _trained_adapter_state(_ThreeLayerModel, config)
    state[next(iter(state))] = torch.zeros(3, 3)

    with pytest.raises(RuntimeError):
        model.load_lora_adapter(_write_adapter(tmp_path / "bad", state, config), adapter_name="c")

    assert model.proj.active_adapters == ["b"]
    assert type(model.other) is nn.Linear and type(model.lora_head) is nn.Linear
    assert set(model.peft_config) == {"a", "b"}
    assert {name: param.requires_grad for name, param in model.named_parameters()} == requires_grad_before
    assert model.state_dict().keys() == state_before.keys()
    for key, value in state_before.items():
        torch.testing.assert_close(model.state_dict()[key], value)


def test_load_lora_adapter_keeps_initial_values_for_keys_missing_from_checkpoint(tmp_path, caplog):
    state = _trained_adapter_state()
    missing_key = next(key for key in state if "lora_B" in key)
    del state[missing_key]
    model = _TinyModel()

    with caplog.at_level(logging.WARNING):
        model.load_lora_adapter(_write_adapter(tmp_path / "adapter", state))

    loaded = get_peft_model_state_dict(model)
    torch.testing.assert_close(loaded[missing_key], torch.zeros_like(loaded[missing_key]))
    assert "1 keys in model but not in checkpoint" in caplog.text
