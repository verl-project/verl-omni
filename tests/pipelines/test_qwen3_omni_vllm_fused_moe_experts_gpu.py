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
"""GPU tests for the vLLM fused-MoE experts implementation at the Qwen3-Omni thinker shapes."""

import pytest
import torch
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts
from verl.utils.device import get_device_name

from tests.utils.gpu_test_topology import enabled_gpu_count
from verl_omni.pipelines.qwen3_omni import vllm_fused_moe_experts

pytestmark = pytest.mark.skipif(not enabled_gpu_count(), reason="requires a GPU")

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, TOKENS = 2048, 768, 128, 8, 512
DEVICE = get_device_name()


def _experts(experts_implementation: str, dtype: torch.dtype) -> Qwen3OmniMoeThinkerTextExperts:
    config = Qwen3OmniMoeTextConfig(
        hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, num_experts=NUM_EXPERTS, hidden_act="silu"
    )
    config._experts_implementation = experts_implementation
    experts = Qwen3OmniMoeThinkerTextExperts(config).to(device=DEVICE, dtype=dtype)
    torch.manual_seed(0)
    for p in experts.parameters():
        p.data.normal_(0, 0.02)
    return experts


def _routing(dtype: torch.dtype):
    torch.manual_seed(1)
    hidden_states = torch.randn(TOKENS, HIDDEN, device=DEVICE, dtype=dtype, requires_grad=True)
    logits = torch.randn(TOKENS, NUM_EXPERTS, device=DEVICE)
    top_k_weights, top_k_index = torch.topk(logits.softmax(-1), TOP_K, dim=-1)
    top_k_weights = (top_k_weights / top_k_weights.sum(-1, keepdim=True)).to(dtype).requires_grad_()
    return hidden_states, top_k_index, top_k_weights


def _run(experts, hidden_states, top_k_index, top_k_weights, grad_output):
    out = experts(hidden_states, top_k_index, top_k_weights)
    out.backward(grad_output)
    return out, (hidden_states.grad, top_k_weights.grad, experts.gate_up_proj.grad, experts.down_proj.grad)


def test_forward_is_the_vllm_kernel_and_backward_is_grouped_mm():
    from vllm.model_executor.layers.fused_moe import fused_experts

    vllm_fused_moe_experts.register_vllm_fused_moe_experts()
    experts, reference = _experts("vllm_fused_moe", torch.bfloat16), _experts("grouped_mm", torch.bfloat16)
    hidden_states, top_k_index, top_k_weights = _routing(torch.bfloat16)
    ref_inputs = (hidden_states.detach().requires_grad_(), top_k_index, top_k_weights.detach().requires_grad_())
    torch.manual_seed(2)
    grad_output = torch.randn(TOKENS, HIDDEN, device=DEVICE, dtype=torch.bfloat16)

    out, grads = _run(experts, hidden_states, top_k_index, top_k_weights, grad_output)
    ref, ref_grads = _run(reference, *ref_inputs, grad_output)
    kernel = fused_experts(
        hidden_states.detach(),
        experts.gate_up_proj.detach(),
        experts.down_proj.detach(),
        top_k_weights.detach().float(),
        top_k_index.to(torch.int32),
    )

    assert torch.equal(out, kernel)
    assert not torch.equal(out, ref)  # the two kernels round differently, so the comparison is sensitive
    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        assert torch.isfinite(grad).all()
        assert torch.equal(grad, ref_grad)


def test_backward_tracks_the_fp32_reference():
    vllm_fused_moe_experts.register_vllm_fused_moe_experts()
    experts, reference = _experts("vllm_fused_moe", torch.bfloat16), _experts("grouped_mm", torch.float32)
    hidden_states, top_k_index, top_k_weights = _routing(torch.bfloat16)
    ref_inputs = (
        hidden_states.detach().float().requires_grad_(),
        top_k_index,
        top_k_weights.detach().float().requires_grad_(),
    )
    torch.manual_seed(2)
    grad_output = torch.randn(TOKENS, HIDDEN, device=DEVICE)

    _, grads = _run(experts, hidden_states, top_k_index, top_k_weights, grad_output.to(torch.bfloat16))
    _, ref_grads = _run(reference, *ref_inputs, grad_output)

    for grad, ref_grad in zip(grads, ref_grads, strict=True):
        cosine = torch.nn.functional.cosine_similarity(grad.float().flatten(), ref_grad.flatten(), dim=0)
        assert cosine > 0.99, cosine


def test_rows_do_not_depend_on_the_batch(monkeypatch):
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    from vllm.model_executor.layers.fused_moe import fused_moe

    fused_moe.get_moe_configs.cache_clear()
    experts = _experts("vllm_fused_moe", torch.bfloat16)
    hidden_states, top_k_index, top_k_weights = _routing(torch.bfloat16)

    with torch.no_grad():
        whole = vllm_fused_moe_experts.vllm_fused_moe_experts_forward(
            experts, hidden_states, top_k_index, top_k_weights
        )
        for chunk in (1, 8, 32, 256):
            parts = [
                vllm_fused_moe_experts.vllm_fused_moe_experts_forward(
                    experts, hidden_states[i : i + chunk], top_k_index[i : i + chunk], top_k_weights[i : i + chunk]
                )
                for i in range(0, TOKENS, chunk)
            ]
            assert torch.equal(torch.cat(parts), whole), chunk
