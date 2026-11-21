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
"""Numerical parity for the MiniMax H3 actor's fused Ascend RMSNorm."""

import pytest
import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT


@pytest.mark.parametrize("shape", [(1, 3191, 5376), (1, 3191, 56, 128)])
@pytest.mark.parametrize("trainable_weight", [False, True])
def test_h3_fused_rms_norm_matches_pytorch_on_npu(shape, trainable_weight, monkeypatch):
    """Compare outputs and gradients through the real CANN kernel."""
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU is not available")

    generator = torch.Generator().manual_seed(42)
    hidden_states = torch.randn(shape, generator=generator).to(device="npu:0", dtype=torch.bfloat16).requires_grad_()
    norm = torch.nn.RMSNorm(shape[-1], eps=1e-5, device="npu:0", dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(0.9 + 0.2 * torch.rand(shape[-1], generator=generator))
    norm.weight.requires_grad_(trainable_weight)
    grad_output = torch.randn(shape, generator=generator).to(hidden_states)
    inputs = (hidden_states, norm.weight) if trainable_weight else (hidden_states,)

    expected = norm(hidden_states)
    expected_grads = torch.autograd.grad(expected, inputs, grad_output)
    original = torch_npu.npu_rms_norm
    calls = []

    def record_kernel(x, weight, *, epsilon):
        calls.append((x.shape, epsilon))
        return original(x, weight, epsilon=epsilon)

    monkeypatch.setattr(torch_npu, "npu_rms_norm", record_kernel)
    MiniMaxH3DiffusionNFT.configure_train_mode(norm)
    actual = norm(hidden_states)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)

    assert calls == [(torch.Size(shape), norm.eps)]
    assert actual.dtype == expected.dtype
    for value, reference in zip((actual, *actual_grads), (expected, *expected_grads), strict=True):
        assert torch.isfinite(value).all()
        value, reference = value.float(), reference.float()
        relative_l2 = (value - reference).norm() / reference.norm().clamp_min(1e-10)
        assert relative_l2.item() < 1e-3
