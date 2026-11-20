# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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
"""Thinker NPU patches adapted from https://github.com/verl-project/verl/pull/7960."""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def qwen3_omni_experts_forward_npu(
    self,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    """Run Thinker experts using verl's NPU GMM autograd and fused SwiGLU.

    Args:
        hidden_states: Floating-point token states of shape (tokens, hidden_size).
        top_k_index: Integer expert indices of shape (tokens, top_k).
        top_k_weights: Floating-point routing weights of shape (tokens, top_k).

    Returns:
        Expert outputs with the input shape and dtype.
    """
    import torch_npu
    from verl.models.transformers.npu_patch import NPUGmmFunction

    # Keep HF parameters intact for FSDP/checkpoints; transpose views avoid copying
    # all expert weights on each forward or caching stale weights after an update.
    gate_up_proj = self.gate_up_proj.transpose(1, 2)
    down_proj = self.down_proj.transpose(1, 2)
    permuted_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, top_k_index.to(torch.int32))
    tokens_per_expert = torch.histc(top_k_index, bins=self.num_experts, min=0, max=self.num_experts)
    gate_up = NPUGmmFunction.apply(permuted_states, gate_up_proj, tokens_per_expert, 1)
    intermediate_states = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = NPUGmmFunction.apply(intermediate_states, down_proj, tokens_per_expert, 1)
    output = torch_npu.npu_moe_token_unpermute(output.to(top_k_weights.dtype), row_ids_map, probs=top_k_weights)
    return output.to(hidden_states.dtype)


def apply_qwen3_omni_thinker_npu_patch(text_config: Any) -> None:
    """Install Thinker NPU patches, including grouped expert forward and backward.

    Args:
        text_config: The loaded Thinker's text configuration.

    Raises:
        ValueError: The configured activation is incompatible with SwiGLU.
    """
    if text_config.hidden_act != "silu":
        raise ValueError(f"Qwen3-Omni Thinker NPU patches require hidden_act='silu', got {text_config.hidden_act!r}.")

    from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as modeling
    from verl.models.transformers.npu_patch import (
        apply_rotary_pos_emb_npu,
        rms_norm_forward_npu,
        silu_forward_npu,
    )

    patches = (
        (modeling.Qwen3OmniMoeThinkerTextExperts, qwen3_omni_experts_forward_npu),
        (modeling.Qwen3OmniMoeThinkerTextRMSNorm, rms_norm_forward_npu),
        (modeling.Qwen3OmniMoeTextRMSNorm, rms_norm_forward_npu),
        (modeling.Qwen3OmniMoeThinkerTextMLP, silu_forward_npu),
    )

    def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
        # Transformers' fifth positional argument is unsqueeze_dim, not position_ids.
        return apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    for module_cls, forward in patches:
        module_cls.forward = forward
    modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
    logger.info("Applied Qwen3-Omni Thinker NPU patches: MoE GMM, RMSNorm, MLP and RoPE")
