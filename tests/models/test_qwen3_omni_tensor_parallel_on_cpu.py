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
"""CPU tests for the actor-side reproduction of a tensor-parallel rollout's row-parallel layers."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3OmniMoeTextConfig
from transformers.integrations.moe import ExpertsInterface
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts

from verl_omni.models.transformers import qwen3_omni
from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter


def test_tensor_parallel_sum_adds_in_fp32_and_rounds_once():
    # 3 * 2**-10 is below half a bf16 step at 1.0, so adding the partials one by one in bf16 would stay at 1.0.
    partials = [torch.tensor([value], dtype=torch.bfloat16) for value in (1.0, 3 * 2**-10, 3 * 2**-10)]

    total = qwen3_omni._tensor_parallel_sum(partials)

    assert total.dtype == torch.bfloat16
    assert total.item() == 1.0 + 2**-7


def test_tensor_parallel_linear_adds_the_partial_products_of_each_rank():
    torch.manual_seed(0)
    linear = nn.Linear(8, 6, bias=False, dtype=torch.bfloat16)
    hidden_states = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
    partials = [
        F.linear(hidden_states[:, :4], linear.weight[:, :4]),
        F.linear(hidden_states[:, 4:], linear.weight[:, 4:]),
    ]
    expected = (partials[0].float() + partials[1].float()).to(torch.bfloat16)

    qwen3_omni.use_tensor_parallel_linear(linear, 2)
    output = linear(hidden_states)

    assert torch.equal(output, expected)
    output.float().sum().backward()
    assert hidden_states.grad.abs().sum() > 0 and linear.weight.grad.abs().sum() > 0


def test_experts_forward_runs_the_kernel_on_each_rank_slice(monkeypatch):
    hidden, intermediate, num_experts = 8, 6, 4
    config = Qwen3OmniMoeTextConfig(
        hidden_size=hidden, moe_intermediate_size=intermediate, num_experts=num_experts, hidden_act="silu"
    )
    config._experts_implementation = "vllm_fused_moe"
    experts = Qwen3OmniMoeThinkerTextExperts(config)
    for param in experts.parameters():
        param.data.normal_()  # the module leaves them uninitialized, and NaNs never compare equal
    calls = []

    def fake_fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights):
        calls.append((gate_up_proj, down_proj))
        return torch.full_like(hidden_states, float(len(calls)))

    monkeypatch.setattr(qwen3_omni, "fused_experts", fake_fused_experts)
    assert qwen3_omni.register_vllm_fused_moe_experts(2) == "vllm_fused_moe"
    hidden_states = torch.randn(3, hidden)
    top_k_index = torch.zeros(3, 2, dtype=torch.long)

    with torch.no_grad():
        output = experts(hidden_states, top_k_index, torch.rand(3, 2))

    assert len(calls) == 2
    for rank, (gate_up_proj, down_proj) in enumerate(calls):
        rows = slice(rank * 3, rank * 3 + 3)
        gate, up = experts.gate_up_proj[:, :intermediate], experts.gate_up_proj[:, intermediate:]
        assert torch.equal(gate_up_proj, torch.cat([gate[:, rows], up[:, rows]], dim=1))
        assert torch.equal(down_proj, experts.down_proj[..., rows])
    assert torch.equal(output, torch.full_like(hidden_states, 3.0))
    qwen3_omni.register_vllm_fused_moe_experts()
    assert ExpertsInterface()["vllm_fused_moe"].keywords == {"tp_size": 1}


class _Thinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.rotary_emb = SimpleNamespace(inv_freq=torch.ones(4), mrope_section=[2, 1, 1], attention_scaling=1.0)
        self.model.layers = nn.ModuleList([nn.Module() for _ in range(2)])
        for layer in self.model.layers:
            layer.self_attn = nn.Module()
            layer.self_attn.o_proj = nn.Linear(8, 4, bias=False)

    def forward(self, *args, **kwargs):
        return None

    def get_input_embeddings(self):
        return None

    def set_input_embeddings(self, value):
        pass


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.thinker = _Thinker()


@pytest.mark.parametrize("true_on_policy, tp_size, split", [(True, 2, True), (True, 1, False), (False, 2, False)])
def test_configure_model_splits_o_proj_only_for_a_tensor_parallel_true_on_policy_rollout(
    true_on_policy, tp_size, split
):
    model_config = SimpleNamespace(true_on_policy=true_on_policy, rollout_tensor_parallel_size=tp_size)

    module = Qwen3OmniThinkerAdapter.configure_model(_Model(), model_config)

    hidden_states = torch.randn(3, 8)
    for layer in module.thinker.model.layers:
        o_proj = layer.self_attn.o_proj
        halves = F.linear(hidden_states[:, :4], o_proj.weight[:, :4]) + F.linear(
            hidden_states[:, 4:], o_proj.weight[:, 4:]
        )
        whole = F.linear(hidden_states, o_proj.weight)
        assert torch.equal(o_proj(hidden_states), halves if split else whole)


def test_adapter_registers_the_experts_for_the_rollout_tensor_parallel_size():
    model_config = SimpleNamespace(rollout_tensor_parallel_size=2)

    implementations = Qwen3OmniThinkerAdapter.register_true_on_policy_implementations(model_config)

    assert implementations == {"attn_implementation": "hub_fa3", "experts_implementation": "vllm_fused_moe"}
    assert ExpertsInterface()["vllm_fused_moe"].keywords == {"tp_size": 2}
    qwen3_omni.register_vllm_fused_moe_experts()
