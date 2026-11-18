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
"""Layered export must retain native-image LoRA weights behind the PEFT wrapper."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel
from verl.utils import device as device_utils

from verl_omni.utils import fsdp_utils


@pytest.mark.parametrize("adapter_name", ["default", "old"])
def test_layered_export_matches_native_peft_state(monkeypatch, adapter_name):
    """Collect both block stacks and only the requested adapter using real model parameters."""
    native = pytest.importorskip("diffusers.models.transformers.transformer_minimax_h3_image")
    actor = native.MiniMaxH3ImageTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=8,
        hidden_size=16,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        time_embed_hidden_dim=16,
        time_embed_dim=16,
        rope_freq_dim=2,
    ).requires_grad_(False)
    config = LoraConfig(r=2, lora_alpha=4, target_modules=["to_q", "to_v"])
    model = get_peft_model(actor, config, autocast_adapter_dtype=False)
    model.add_adapter("old", config)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                parameter.fill_(0.125 if ".default." in name else 0.25)

    blocks = set(model.transformer_blocks) | set(model.token_refiner.refiner_blocks)
    monkeypatch.setattr(fsdp_utils, "fsdp_version", lambda module: 2 if module in blocks else 0)
    monkeypatch.setattr(FullyShardedDataParallel, "summon_full_params", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(device_utils, "get_torch_device", lambda: SimpleNamespace(empty_cache=lambda: None))
    actual = fsdp_utils._layered_summon_lora_params_diffusers(
        model,
        adapter_name=adapter_name,
        layer_prefixes=["transformer_blocks.", "token_refiner.refiner_blocks."],
    )
    expected = get_peft_model_state_dict(model, adapter_name=adapter_name)
    assert actual and set(actual) == set(expected)
    assert any("transformer_blocks." in key for key in actual)
    assert any("token_refiner.refiner_blocks." in key for key in actual)
    for name, tensor in actual.items():
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)
