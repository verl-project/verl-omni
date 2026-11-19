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
"""Ascend grouped matmul for Qwen3-Omni Thinker experts."""

from copy import copy

import torch
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextExperts,
    Qwen3OmniMoeThinkerTextModel,
)
from transformers.utils import is_torch_npu_available


def _ascend_grouped_mm(
    experts: Qwen3OmniMoeThinkerTextExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
        raise ValueError("ascend_grouped_mm requires BF16 Thinker activations on an Ascend NPU.")
    if hidden_states.shape[0] == 0:
        return torch.zeros_like(hidden_states)

    import torch_npu
    from verl.models.transformers.npu_patch import NPUGmmFunction

    # As in verl-omni #819, keep the packed parameters and pass transposed views to GMM.
    gate_up_weight = experts.gate_up_proj.transpose(1, 2)
    down_weight = experts.down_proj.transpose(1, 2)
    permuted, row_ids = torch_npu.npu_moe_token_permute(hidden_states, top_k_index.to(torch.int32))
    counts = torch.histc(top_k_index, bins=experts.num_experts, min=0, max=experts.num_experts)
    gate_up = NPUGmmFunction.apply(permuted, gate_up_weight, counts)
    activated = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = NPUGmmFunction.apply(activated, down_weight, counts)
    unpermuted = torch_npu.npu_moe_token_unpermute(output.to(top_k_weights.dtype), row_ids, probs=top_k_weights)
    return unpermuted.to(hidden_states.dtype)


def enable_ascend_grouped_mm(text_model: Qwen3OmniMoeThinkerTextModel) -> None:
    """Select verl's Ascend experts backend for the loaded Thinker text model."""
    if not is_torch_npu_available():
        raise ValueError("ascend_grouped_mm requires an available Ascend NPU.")
    if text_model.config.hidden_act != "silu":
        raise ValueError("ascend_grouped_mm requires SiLU Thinker experts.")

    ALL_EXPERTS_FUNCTIONS.register("ascend_grouped_mm", _ascend_grouped_mm)
    # FSDP meta initialization can reuse the caller's HF config.
    text_model.config = copy(text_model.config)
    text_model.set_experts_implementation("ascend_grouped_mm")
    for experts in text_model.modules():
        if isinstance(experts, Qwen3OmniMoeThinkerTextExperts):
            experts.config = text_model.config
