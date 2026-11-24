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
"""CPU contract tests for the true on-policy mode against the pinned Transformers, vLLM, vLLM-Omni and verl.

The rollout-side HF-numerics ops are plain torch, so their op order and rounding can be compared bitwise with
the actor's Transformers modules without a GPU. Kernel-level equality (attention, experts, GEMMs) is covered
by tests/special_e2e/run_qwen3_omni_true_on_policy_gate.sh.
"""

import dataclasses
import inspect

import pytest
import torch
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextRMSNorm,
    Qwen3OmniMoeThinkerTextRotaryEmbedding,
    Qwen3OmniMoeThinkerTextTopKRouter,
    apply_rotary_pos_emb,
)
from verl.protocol import DataProto
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.workers.config import FSDPEngineConfig
from vllm.model_executor.layers.fused_moe import fused_experts
from vllm_omni.model_executor.models.qwen3_omni import hf_numerics, qwen3_omni_moe_thinker

from verl_omni.models.transformers.qwen3_omni import ExactMRotaryEmbedding
from verl_omni.pipelines.qwen3_omni.omni_rollout_adapter import Qwen3OmniRolloutAdapter

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, TOKENS, NUM_EXPERTS, TOP_K = 64, 4, 2, 16, 37, 8, 2
DTYPE = torch.bfloat16


def _text_config() -> Qwen3OmniMoeTextConfig:
    return Qwen3OmniMoeTextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        norm_topk_prob=True,
        rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [4, 2, 2]},
    )


def _hf_norm() -> Qwen3OmniMoeThinkerTextRMSNorm:
    torch.manual_seed(0)
    norm = Qwen3OmniMoeThinkerTextRMSNorm(HIDDEN, eps=1e-6).to(DTYPE)
    norm.weight.data.normal_(1.0, 0.5)
    return norm


def test_rollout_rms_norm_matches_the_hf_module_bitwise():
    norm = _hf_norm()
    hidden_states = 3 * torch.randn(TOKENS, HIDDEN, dtype=DTYPE)

    out = hf_numerics.hf_rms_norm(hidden_states, norm.weight, norm.variance_epsilon)

    assert torch.equal(out, norm(hidden_states))


def test_rollout_residual_add_and_norm_match_hf_bitwise():
    norm = _hf_norm()
    hidden_states = 3 * torch.randn(TOKENS, HIDDEN, dtype=DTYPE)
    residual = 3 * torch.randn(TOKENS, HIDDEN, dtype=DTYPE)

    out, stream = hf_numerics.hf_add_rms_norm(hidden_states, residual, norm.weight, norm.variance_epsilon)

    assert torch.equal(stream, hidden_states + residual)
    assert torch.equal(out, norm(hidden_states + residual))


@pytest.mark.parametrize("mrope_positions", [False, True])
def test_rollout_rope_matches_the_actor_rotary_bitwise(mrope_positions):
    config = _text_config()
    hf_rotary = Qwen3OmniMoeThinkerTextRotaryEmbedding(config)
    actor_rotary = ExactMRotaryEmbedding.from_hf(hf_rotary)
    rollout_rotary = hf_numerics.HFNumericsRotaryEmbedding(config)
    torch.manual_seed(0)
    query = torch.randn(TOKENS, HEADS * HEAD_DIM, dtype=DTYPE)
    key = torch.randn(TOKENS, KV_HEADS * HEAD_DIM, dtype=DTYPE)
    text = torch.arange(3000, 3000 + TOKENS)
    positions = torch.stack([text, text // 2, text % 5]) if mrope_positions else text

    rollout_query, rollout_key = rollout_rotary(positions, query, key)
    position_ids = positions[:, None] if mrope_positions else positions[None]
    cos, sin = actor_rotary(torch.empty(1, TOKENS, HIDDEN, dtype=DTYPE), position_ids)
    actor_query, actor_key = apply_rotary_pos_emb(
        query.view(TOKENS, HEADS, HEAD_DIM).transpose(0, 1)[None],
        key.view(TOKENS, KV_HEADS, HEAD_DIM).transpose(0, 1)[None],
        cos,
        sin,
    )

    assert torch.equal(rollout_rotary.inv_freq, hf_rotary.inv_freq)
    assert torch.equal(rollout_query, actor_query[0].transpose(0, 1).reshape(TOKENS, -1))
    assert torch.equal(rollout_key, actor_key[0].transpose(0, 1).reshape(TOKENS, -1))


def test_rollout_moe_routing_matches_the_hf_router_bitwise(monkeypatch):
    router = Qwen3OmniMoeThinkerTextTopKRouter(_text_config()).to(DTYPE)
    torch.manual_seed(0)
    router.weight.data.normal_(0, 0.5)
    hidden_states = torch.randn(TOKENS, HIDDEN, dtype=DTYPE)
    calls = []

    def fake_fused_experts(hidden_states, w13, w2, topk_weights, topk_ids):
        calls.append((topk_weights, topk_ids))
        return torch.zeros_like(hidden_states)

    monkeypatch.setattr(hf_numerics, "fused_experts", fake_fused_experts)
    hf_numerics.hf_moe(hidden_states, router.weight, torch.empty(0), torch.empty(0), TOP_K, True)
    _, routing_weights, selected_experts = router(hidden_states)

    topk_weights, topk_ids = calls[0]
    assert torch.equal(topk_weights, routing_weights.float())
    assert torch.equal(topk_ids, selected_experts.to(torch.int32))


def test_vllm_fused_experts_takes_the_arguments_the_actor_passes():
    assert list(inspect.signature(fused_experts).parameters)[:5] == [
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
    ]


def test_vllm_omni_thinker_reads_the_option_the_rollout_adapter_sets():
    (option,) = Qwen3OmniRolloutAdapter.get_true_on_policy_engine_extras(0)["additional_config"]

    assert f'additional_config.get("{option}"' in inspect.getsource(qwen3_omni_moe_thinker)


def test_verl_provides_the_batch_invariant_engine_option():
    assert "batch_invariant" in {field.name for field in dataclasses.fields(FSDPEngineConfig)}


def test_verl_counts_raw_log_prob_mismatches():
    rollout_log_probs = torch.tensor([[-1.0, -2.0, -3.0, -4.0]])
    old_log_probs = rollout_log_probs.clone()
    old_log_probs[0, 1] = torch.nextafter(old_log_probs[0, 1], torch.tensor(0.0))
    data = DataProto.from_dict(
        {
            "rollout_log_probs": rollout_log_probs,
            "old_log_probs": old_log_probs,
            "response_mask": torch.tensor([[1, 1, 1, 0]]),
            "responses": torch.zeros(1, 4, dtype=torch.long),
        }
    )

    assert calculate_debug_metrics(data)["training/rollout_logprobs_mismatch_count"] == 1
