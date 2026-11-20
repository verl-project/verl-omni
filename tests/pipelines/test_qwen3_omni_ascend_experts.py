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

import math

import pytest
import torch


def _assert_bf16_close(actual, expected):
    error = (actual.detach().cpu().float() - expected.detach()).abs()
    assert error.norm() <= 0.01 * expected.norm()
    assert error.max() <= 0.015 * expected.abs().max()


@pytest.mark.parametrize("top_k", [1, 2])
def test_ascend_grouped_mm_matches_eager_values_and_gradients(top_k):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("Requires an Ascend NPU")

    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts

    from verl_omni.models.transformers.qwen3_omni import _ascend_grouped_mm

    config = Qwen3OmniMoeTextConfig(
        hidden_size=128, moe_intermediate_size=256, num_experts=8, num_experts_per_tok=top_k, hidden_act="silu"
    )
    config._experts_implementation = "eager"
    reference = Qwen3OmniMoeThinkerTextExperts(config)
    candidate = Qwen3OmniMoeThinkerTextExperts(config).to(device="npu", dtype=torch.bfloat16)

    generator = torch.Generator().manual_seed(20261119)
    with torch.no_grad():
        for param in reference.parameters():
            weight = torch.randn(param.shape, generator=generator) / math.sqrt(param.shape[-1])
            param.copy_(weight.bfloat16())
    candidate.load_state_dict(reference.state_dict())

    route_ids = torch.tensor([[0, 7], [7, 3], [3, 0], [0, 3]] * 8)[:29, :top_k].contiguous()
    inputs = torch.randn(29, 128, generator=generator).bfloat16()
    route_weights = torch.randn(29, top_k, generator=generator).softmax(-1).bfloat16()
    output_gradient = torch.randn(29, 128, generator=generator).bfloat16()
    reference_inputs = inputs.float().requires_grad_()
    reference_weights = route_weights.float().requires_grad_()
    candidate_inputs = inputs.to("npu").requires_grad_()
    candidate_weights = route_weights.to("npu").requires_grad_()

    expected = reference(reference_inputs, route_ids, reference_weights)
    actual = _ascend_grouped_mm(candidate, candidate_inputs, route_ids.to("npu"), candidate_weights)
    expected.backward(output_gradient.float())
    actual.backward(output_gradient.to("npu"))

    _assert_bf16_close(actual, expected)
    pairs = [(candidate_inputs.grad, reference_inputs.grad), (candidate_weights.grad, reference_weights.grad)]
    for name, param in candidate.named_parameters():
        grad = param.grad.cpu().float()
        torch.testing.assert_close(grad[[1, 2, 4, 5, 6]], torch.zeros_like(grad[[1, 2, 4, 5, 6]]))
        pairs.append((grad, reference.get_parameter(name).grad))
    for actual_gradient, expected_gradient in pairs:
        _assert_bf16_close(actual_gradient, expected_gradient)
