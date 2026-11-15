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
"""CPU tests for the registered vLLM fused-MoE experts implementation (no kernel call)."""

import torch
from transformers import Qwen3OmniMoeTextConfig
from transformers.integrations.moe import ExpertsInterface, grouped_mm_experts_forward
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts

from verl_omni.pipelines.qwen3_omni import vllm_fused_moe_experts

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, TOKENS = 16, 8, 4, 2, 6


def _experts(experts_implementation: str) -> Qwen3OmniMoeThinkerTextExperts:
    config = Qwen3OmniMoeTextConfig(
        hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, num_experts=NUM_EXPERTS, hidden_act="silu"
    )
    config._experts_implementation = experts_implementation
    experts = Qwen3OmniMoeThinkerTextExperts(config)
    torch.manual_seed(0)
    for p in experts.parameters():
        p.data.normal_(0, 0.2)
    return experts


def _routing():
    torch.manual_seed(1)
    hidden_states = torch.randn(TOKENS, HIDDEN, requires_grad=True)
    top_k_index = torch.stack([torch.randperm(NUM_EXPERTS)[:TOP_K] for _ in range(TOKENS)])
    top_k_weights = torch.rand(TOKENS, TOP_K, requires_grad=True)
    return hidden_states, top_k_index, top_k_weights


def test_register_adds_experts_implementation():
    name = vllm_fused_moe_experts.register_vllm_fused_moe_experts()

    assert name == "vllm_fused_moe"
    assert ExpertsInterface()["vllm_fused_moe"] is vllm_fused_moe_experts.vllm_fused_moe_experts_forward


def test_forward_runs_the_kernel_and_backward_differentiates_grouped_mm(monkeypatch):
    calls = []

    def fake_fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights):
        calls.append((hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights))
        return torch.full_like(hidden_states, 7.0)

    monkeypatch.setattr(vllm_fused_moe_experts, "fused_experts", fake_fused_experts)
    vllm_fused_moe_experts.register_vllm_fused_moe_experts()
    experts, reference = _experts("vllm_fused_moe"), _experts("grouped_mm")
    hidden_states, top_k_index, top_k_weights = _routing()
    ref_hidden_states = hidden_states.detach().requires_grad_()
    ref_top_k_weights = top_k_weights.detach().requires_grad_()
    torch.manual_seed(2)
    grad_output = torch.randn(TOKENS, HIDDEN)

    out = experts(hidden_states, top_k_index, top_k_weights)
    out.backward(grad_output)
    ref = grouped_mm_experts_forward(reference, ref_hidden_states, top_k_index, ref_top_k_weights)
    ref.backward(grad_output)

    assert len(calls) == 1
    assert calls[0][1] is experts.gate_up_proj and calls[0][2] is experts.down_proj
    assert torch.equal(out, torch.full_like(out, 7.0))
    assert torch.equal(hidden_states.grad, ref_hidden_states.grad)
    assert torch.equal(top_k_weights.grad, ref_top_k_weights.grad)
    assert torch.equal(experts.gate_up_proj.grad, reference.gate_up_proj.grad)
    assert torch.equal(experts.down_proj.grad, reference.down_proj.grad)
    assert hidden_states.grad.abs().sum() > 0 and experts.gate_up_proj.grad.abs().sum() > 0


def test_no_grad_forward_skips_autograd(monkeypatch):
    monkeypatch.setattr(vllm_fused_moe_experts, "fused_experts", lambda h, *_args: h + 1)
    vllm_fused_moe_experts.register_vllm_fused_moe_experts()
    experts = _experts("vllm_fused_moe")
    hidden_states, top_k_index, top_k_weights = _routing()

    with torch.no_grad():
        out = experts(hidden_states, top_k_index, top_k_weights)

    assert not out.requires_grad
    assert torch.equal(out, hidden_states + 1)
