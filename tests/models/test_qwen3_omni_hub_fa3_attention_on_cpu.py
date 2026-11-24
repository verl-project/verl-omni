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
"""CPU tests for the registered hub FlashAttention-3 attention (no kernel load)."""

from types import SimpleNamespace

import pytest
import torch
from transformers import AttentionInterface
from transformers.integrations.moe import ExpertsInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS

from verl_omni.models.transformers import qwen3_omni
from verl_omni.pipelines.model_base import OmniModelBase
from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter


def test_packed_cu_seqlens_splits_on_position_restarts():
    position_ids = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])

    cu_seqlens, max_seqlen = qwen3_omni.packed_cu_seqlens(position_ids)

    assert cu_seqlens.dtype == torch.int32
    assert cu_seqlens.tolist() == [0, 3, 5, 9]
    assert max_seqlen == 4


def test_register_adds_attention_and_flash_attention_2_mask():
    name = qwen3_omni.register_hub_fa3_attention()

    assert name == "hub_fa3"
    assert AttentionInterface()["hub_fa3"] is qwen3_omni.hub_fa3_attention
    assert ALL_MASK_ATTENTION_FUNCTIONS["hub_fa3"] is ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"]


@pytest.mark.parametrize("position_shape", ["2d", "3d"])
def test_attention_calls_varlen_kernel_on_packed_layout(monkeypatch, position_shape):
    calls = []

    def fake_varlen(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal):
        calls.append((q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale, causal))
        return q + 1

    monkeypatch.setattr(qwen3_omni, "_flash_attn_varlen_func", lambda: fake_varlen)
    heads, tokens, dim = 2, 5, 4
    query, key, value = (torch.randn(1, heads, tokens, dim) for _ in range(3))
    positions = torch.tensor([[0, 1, 2, 0, 1]])
    if position_shape == "3d":
        positions = positions[None].expand(3, 1, tokens)

    out, weights = qwen3_omni.hub_fa3_attention(None, query, key, value, None, scaling=0.5, position_ids=positions)

    q, k, v, cu_q, cu_k, max_q, max_k, scale, causal = calls[0]
    assert q.shape == (tokens, heads, dim)
    assert torch.equal(q, query[0].transpose(0, 1))
    assert torch.equal(k, key[0].transpose(0, 1))
    assert torch.equal(v, value[0].transpose(0, 1))
    assert cu_q.tolist() == cu_k.tolist() == [0, 3, 5]
    assert (max_q, max_k, scale, causal) == (3, 3, 0.5, True)
    assert out.shape == (1, tokens, heads, dim)
    assert torch.equal(out[0], q + 1)
    assert weights is None


def test_thinker_adapter_registers_hub_fa3_and_fused_moe_for_true_on_policy():
    implementations = Qwen3OmniThinkerAdapter.register_true_on_policy_implementations(
        SimpleNamespace(rollout_tensor_parallel_size=1)
    )

    assert implementations == {"attn_implementation": "hub_fa3", "experts_implementation": "vllm_fused_moe"}
    assert AttentionInterface()["hub_fa3"] is qwen3_omni.hub_fa3_attention
    assert ExpertsInterface()["vllm_fused_moe"].func is qwen3_omni.vllm_fused_moe_experts_forward


def test_base_adapter_refuses_true_on_policy():
    with pytest.raises(NotImplementedError, match="OmniModelBase does not support true_on_policy"):
        OmniModelBase.register_true_on_policy_implementations(SimpleNamespace())
