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
"""Run on Ascend: pytest -q tests/models/test_qwen3_omni_npu_numerics.py."""

import copy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu")

from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as modeling
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTextConfig

from verl_omni.models.transformers.qwen3_omni_npu import apply_qwen3_omni_thinker_npu_patch


@pytest.fixture
def install_patch(monkeypatch):
    # Restore global forwards so every reference run uses Transformers' implementation.
    for name in (
        "Qwen3OmniMoeThinkerTextExperts",
        "Qwen3OmniMoeThinkerTextRMSNorm",
        "Qwen3OmniMoeTextRMSNorm",
        "Qwen3OmniMoeThinkerTextMLP",
    ):
        cls = getattr(modeling, name)
        monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(modeling, "apply_rotary_pos_emb", modeling.apply_rotary_pos_emb)
    return lambda: apply_qwen3_omni_thinker_npu_patch(SimpleNamespace(hidden_act="silu"))


def assert_bf16_matches_fp32(actual, reference):
    """Bound aggregate and peak BF16 error against an independent FP32 reference."""
    actual = actual.detach().cpu().float()
    reference = reference.detach().cpu().float()
    assert torch.isfinite(actual).all()
    difference = actual - reference
    epsilon = torch.finfo(torch.bfloat16).eps
    assert difference.norm() <= 2 * epsilon * reference.norm() + 1e-6
    assert difference.abs().max() <= 4 * epsilon * reference.abs().max() + 1e-6


@pytest.mark.parametrize("seed", [7960, 42, 2026])
@pytest.mark.parametrize("kind", ["experts", "decoder_norm", "final_norm", "mlp"])
def test_forward_and_backward_match_transformers(install_patch, kind, seed):
    torch.manual_seed(seed)
    config = Qwen3OmniMoeTextConfig(
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        hidden_act="silu",
    )
    config._experts_implementation = "eager"
    if kind == "experts":
        reference = modeling.Qwen3OmniMoeThinkerTextExperts(config)
        with torch.no_grad():
            for param in reference.parameters():
                param.normal_(std=0.02)
    elif kind == "mlp":
        reference = modeling.Qwen3OmniMoeThinkerTextMLP(config)
    elif kind == "decoder_norm":
        reference = modeling.Qwen3OmniMoeThinkerTextRMSNorm(128)
    else:
        reference = modeling.Qwen3OmniMoeTextRMSNorm(128)
    reference = reference.to(device="npu", dtype=torch.bfloat16)
    patched = copy.deepcopy(reference)
    # Match quantized inputs/weights exactly; do not use one BF16 rounding path as truth.
    oracle = copy.deepcopy(reference).cpu().float()
    inputs = torch.randn(16, 128, device="npu", dtype=torch.bfloat16, requires_grad=True)
    patched_inputs = inputs.detach().clone().requires_grad_()
    oracle_inputs = inputs.detach().cpu().float().requires_grad_()
    weights = torch.randn(16, 2, device="npu").softmax(-1).to(torch.bfloat16).detach().requires_grad_()
    patched_weights = weights.detach().clone().requires_grad_()
    oracle_weights = weights.detach().cpu().float().requires_grad_()
    # Leave one expert empty and include repeated assignments across tokens.
    indices = torch.tensor([[0, 1], [1, 2]] * 8, device="npu", dtype=torch.int64)
    grad_output = torch.randn_like(inputs)
    args = (indices, weights) if kind == "experts" else ()
    expected = reference(inputs, *args)
    # Use a common upstream gradient to isolate the Jacobian from output rounding.
    expected.backward(grad_output)
    oracle_args = (indices.cpu(), oracle_weights) if kind == "experts" else ()
    precise = oracle(oracle_inputs, *oracle_args)
    precise.backward(grad_output.cpu().float())
    install_patch()
    args = (indices, patched_weights) if kind == "experts" else ()
    actual = patched(patched_inputs, *args)
    actual.backward(grad_output)
    for output in (actual, expected):
        assert_bf16_matches_fp32(output, precise)
    for grad in (patched_inputs.grad, inputs.grad):
        assert_bf16_matches_fp32(grad, oracle_inputs.grad)
    assert patched.state_dict().keys() == reference.state_dict().keys()
    for name, param in patched.named_parameters():
        assert_bf16_matches_fp32(param.grad, oracle.get_parameter(name).grad)
        assert_bf16_matches_fp32(reference.get_parameter(name).grad, oracle.get_parameter(name).grad)
    if kind == "experts":
        for grad in (patched_weights.grad, weights.grad):
            assert_bf16_matches_fp32(grad, oracle_weights.grad)


@pytest.mark.parametrize("seed", [7960, 42, 2026])
@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
def test_rope_forward_and_backward_match_transformers(install_patch, unsqueeze_dim, seed):
    torch.manual_seed(seed)
    shape = (2, 4, 8, 128) if unsqueeze_dim == 1 else (2, 8, 4, 128)
    query = torch.randn(shape, device="npu", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(shape, device="npu", dtype=torch.bfloat16, requires_grad=True)
    patched_query = query.detach().clone().requires_grad_()
    patched_key = key.detach().clone().requires_grad_()
    oracle_query = query.detach().cpu().float().requires_grad_()
    oracle_key = key.detach().cpu().float().requires_grad_()
    angles = torch.randn(2, 8, 64, device="npu", dtype=torch.bfloat16).repeat(1, 1, 2)
    cos, sin = angles.cos(), angles.sin()
    grad_outputs = (torch.randn_like(query), torch.randn_like(key))
    expected = modeling.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim)
    torch.autograd.backward(expected, grad_outputs)
    precise = modeling.apply_rotary_pos_emb(
        oracle_query, oracle_key, cos.cpu().float(), sin.cpu().float(), unsqueeze_dim
    )
    torch.autograd.backward(precise, tuple(grad.cpu().float() for grad in grad_outputs))
    install_patch()
    actual = modeling.apply_rotary_pos_emb(patched_query, patched_key, cos, sin, unsqueeze_dim)
    torch.autograd.backward(actual, grad_outputs)
    for results in (actual, expected):
        for result, target in zip(results, precise, strict=True):
            assert_bf16_matches_fp32(result, target)
    for grad in (patched_query.grad, query.grad):
        assert_bf16_matches_fp32(grad, oracle_query.grad)
    for grad in (patched_key.grad, key.grad):
        assert_bf16_matches_fp32(grad, oracle_key.grad)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("routing_dtype", [torch.bfloat16, torch.float32])
def test_gmm_dispatch_and_parameter_updates(install_patch, monkeypatch, training, routing_dtype):
    """Exercise empty experts, both routing dtypes and live weights after an update."""
    import torch_npu

    from verl_omni.models.transformers.qwen3_omni_npu import qwen3_omni_experts_forward_npu

    torch.manual_seed(7960)
    config = Qwen3OmniMoeTextConfig(
        hidden_size=128, moe_intermediate_size=256, num_experts=8, num_experts_per_tok=2, hidden_act="silu"
    )
    config._experts_implementation = "eager"
    reference = modeling.Qwen3OmniMoeThinkerTextExperts(config).to(device="npu", dtype=torch.bfloat16)
    with torch.no_grad():
        for param in reference.parameters():
            param.normal_(std=0.02)
    patched = copy.deepcopy(reference)
    reference.train(training)
    patched.train(training)
    original_forward = type(reference).forward
    parameters = dict(patched.named_parameters())
    install_patch()
    assert type(patched).forward is qwen3_omni_experts_forward_npu

    gmm = torch_npu.npu_grouped_matmul
    calls = []

    def tracked_gmm(inputs, weights, **kwargs):
        # Record metadata only: retaining tensors would change memory lifetime.
        calls.append((weights[0].data_ptr(), weights[0].stride(), kwargs["group_type"]))
        return gmm(inputs, weights, **kwargs)

    monkeypatch.setattr(torch_npu, "npu_grouped_matmul", tracked_gmm)
    inputs = torch.randn(31, 128, device="npu", dtype=torch.bfloat16)
    indices = torch.tensor([[0, 7], [7, 2], [2, 7]] * 10 + [[0, 2]], device="npu")
    weights = torch.randn(31, 2, device="npu").softmax(-1).to(routing_dtype)
    grad_output = torch.randn_like(inputs)
    optimizers = [torch.optim.SGD(layer.parameters(), lr=0.1) for layer in (reference, patched)]
    for _ in range(2):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        ref_inputs = inputs.detach().clone().requires_grad_(training)
        npu_inputs = inputs.detach().clone().requires_grad_(training)
        ref_weights = weights.detach().clone().requires_grad_(training)
        npu_weights = weights.detach().clone().requires_grad_(training)
        with torch.set_grad_enabled(training):
            expected = original_forward(reference, ref_inputs, indices, ref_weights)
            calls.clear()
            actual = patched(npu_inputs, indices, npu_weights)
            assert len(calls) == 2
            for call, param in zip(calls, (patched.gate_up_proj, patched.down_proj), strict=True):
                assert call == (param.data_ptr(), param.transpose(1, 2).stride(), 0)
            if training:
                expected.backward(grad_output)
                actual.backward(grad_output)
                assert len(calls) == 6
                assert sum(call[2] == 2 for call in calls) == 2  # Weight-gradient GMMs.

        def assert_bf16_close(result, target):
            # Fused SwiGLU changes BF16 rounding; bound both aggregate and peak error.
            difference = (result.float() - target.float()).abs()
            torch.testing.assert_close(
                difference.norm() / target.float().norm().clamp_min(1e-8),
                torch.zeros((), device=result.device),
                rtol=0,
                atol=0.02,
            )
            assert difference.max() <= 0.02 * target.float().abs().max() + 1e-5

        assert_bf16_close(actual, expected)
        if training:
            assert_bf16_close(npu_inputs.grad, ref_inputs.grad)
            assert_bf16_close(npu_weights.grad, ref_weights.grad)
            for name, param in patched.named_parameters():
                assert param is parameters[name]
                assert_bf16_close(param.grad, reference.get_parameter(name).grad)
                torch.testing.assert_close(param.grad[1], torch.zeros_like(param.grad[1]), rtol=0, atol=0)
            for optimizer in optimizers:
                optimizer.step()
        else:
            assert not actual.requires_grad
            # Check that a subsequent no-grad call also reads current parameters.
            with torch.no_grad():
                reference.down_proj.mul_(0.5)
                patched.down_proj.mul_(0.5)
