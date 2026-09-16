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
"""CPU tests for the frozen-adapter LoRA state dict extraction fallback."""

import torch

from verl_omni.utils.fsdp_utils import _extract_adapter_lora_state_dict


def test_extracts_only_the_requested_adapters_lora_tensors():
    raw_state = {
        "attn.to_q.base_layer.weight": torch.zeros(2, 2),
        "attn.to_q.lora_A.default.weight": torch.ones(2, 2),
        "attn.to_q.lora_A.old.weight": torch.full((2, 2), 2.0),
        "attn.to_q.lora_B.default.weight": torch.full((2, 2), 3.0),
        "attn.to_q.lora_B.old.weight": torch.full((2, 2), 4.0),
    }

    result = _extract_adapter_lora_state_dict(raw_state, "old")

    assert set(result) == {"attn.to_q.lora_A.weight", "attn.to_q.lora_B.weight"}
    assert torch.equal(result["attn.to_q.lora_A.weight"], raw_state["attn.to_q.lora_A.old.weight"])
    assert torch.equal(result["attn.to_q.lora_B.weight"], raw_state["attn.to_q.lora_B.old.weight"])


def test_returns_empty_when_adapter_has_no_lora_tensors():
    raw_state = {
        "attn.to_q.base_layer.weight": torch.zeros(2, 2),
        "attn.to_q.lora_A.default.weight": torch.ones(2, 2),
    }

    assert _extract_adapter_lora_state_dict(raw_state, "old") == {}
