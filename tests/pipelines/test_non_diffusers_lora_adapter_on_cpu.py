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


def _lora_config() -> LoraConfig:
    return LoraConfig(r=2, lora_alpha=2, target_modules=["proj"])


def _trained_adapter_state() -> dict[str, torch.Tensor]:
    donor = _TinyModel()
    donor.add_adapter(_lora_config())
    return {key: torch.full_like(value, 0.5) for key, value in get_peft_model_state_dict(donor).items()}


def _write_adapter(adapter_dir, state_dict: dict[str, torch.Tensor]) -> str:
    _lora_config().save_pretrained(str(adapter_dir))
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
    assert "default" not in model.proj.lora_A
    assert all(param.requires_grad for param in model.parameters())
    model.load_lora_adapter(_write_adapter(tmp_path / "good", state))
    loaded = get_peft_model_state_dict(model)
    for key, value in state.items():
        torch.testing.assert_close(loaded[key], value)


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
