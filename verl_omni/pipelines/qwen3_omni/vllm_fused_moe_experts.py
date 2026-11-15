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
"""vLLM fused MoE experts registered with Transformers.

The vLLM-Omni rollout runs the experts through vLLM's ``fused_experts`` kernel; calling the same kernel
in the actor makes the MoE output bitwise equal on both sides. The kernel has no backward, so the gradient
is taken from Transformers' ``grouped_mm`` experts evaluated on the same inputs.
"""

import torch
from transformers.integrations.moe import ExpertsInterface, grouped_mm_experts_forward

VLLM_FUSED_MOE_EXPERTS = "vllm_fused_moe"


def fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Run vLLM's fused experts on HF-layout weights.

    Args:
        hidden_states: ``(N, H)`` routed tokens.
        gate_up_proj: ``(E, 2I, H)`` gate and up projections, gate first.
        down_proj: ``(E, H, I)`` down projections.
        top_k_index: ``(N, K)`` selected experts.
        top_k_weights: ``(N, K)`` routing weights in the activation dtype.

    Returns:
        ``(N, H)`` combined expert output in the activation dtype.
    """
    from vllm.model_executor.layers.fused_moe import fused_experts as vllm_fused_experts

    return vllm_fused_experts(
        hidden_states, gate_up_proj, down_proj, top_k_weights.float(), top_k_index.to(torch.int32)
    )


class _ValueWithGradientOf(torch.autograd.Function):
    """Returns ``value`` and sends its gradient to ``reference``."""

    @staticmethod
    def forward(ctx, reference, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def vllm_fused_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Experts forward for ``ExpertsInterface``: vLLM ``fused_experts`` output with the ``grouped_mm`` gradient."""
    with torch.no_grad():
        output = fused_experts(hidden_states, self.gate_up_proj, self.down_proj, top_k_index, top_k_weights)
    if not torch.is_grad_enabled():
        return output
    reference = grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    return _ValueWithGradientOf.apply(reference, output)


def register_vllm_fused_moe_experts() -> str:
    """Register :func:`vllm_fused_moe_experts_forward` under ``vllm_fused_moe``."""
    ExpertsInterface.register(VLLM_FUSED_MOE_EXPERTS, vllm_fused_moe_experts_forward)
    return VLLM_FUSED_MOE_EXPERTS
