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
"""CPU tests for the diffusers NPU affine-less RMSNorm patch.

LTX-2.x transformers set ``norm_elementwise_affine=false``, so every block norm has
``weight=None``; diffusers' NPU branch feeds that ``None`` into ``npu_rms_norm`` and
crashes. The patch reproduces the upstream CPU/CUDA ``weight is None`` behavior (plain
RMS normalization) only on NPU; every other path must delegate to the upstream forward
untouched.
"""

import torch
from diffusers.models import normalization
from diffusers.models.normalization import RMSNorm

import verl_omni.utils.npu_rmsnorm_patch as npu_patch


def _reset_and_patch(monkeypatch):
    monkeypatch.setattr(npu_patch, "_patched", False)
    npu_patch.patch_diffusers_npu_rmsnorm()


def test_npu_affineless_rmsnorm_matches_upstream_cpu_reference(monkeypatch):
    """NPU + weight=None computes the upstream CPU-branch plain-RMS normalization."""
    monkeypatch.setattr("diffusers.utils.is_torch_npu_available", lambda: True)
    _reset_and_patch(monkeypatch)

    norm = RMSNorm(16, eps=1e-6, elementwise_affine=False)
    assert norm.weight is None

    x = torch.randn(2, 4, 5, 16, dtype=torch.bfloat16) * 3
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    expected = (x * torch.rsqrt(variance + norm.eps)).to(x.dtype)
    torch.testing.assert_close(norm(x), expected)


def test_patch_is_idempotent(monkeypatch):
    """A second call leaves the installed wrapper untouched."""
    _reset_and_patch(monkeypatch)
    first_wrapper = normalization.RMSNorm.forward

    npu_patch.patch_diffusers_npu_rmsnorm()

    assert normalization.RMSNorm.forward is first_wrapper


def test_weightless_rmsnorm_delegates_without_npu(monkeypatch):
    """Non-NPU + weight=None must delegate, i.e. no behavior change on CUDA/CPU."""
    # Explicitly force the non-NPU path: torch_npu may be importable in the test
    # env even for "CPU" tests.
    monkeypatch.setattr("diffusers.utils.is_torch_npu_available", lambda: False)
    sentinel = object()

    def fake_original(self, x):
        return sentinel

    monkeypatch.setattr(normalization.RMSNorm, "forward", fake_original)
    _reset_and_patch(monkeypatch)

    norm = RMSNorm(16, eps=1e-6, elementwise_affine=False)
    assert norm(torch.randn(2, 16)) is sentinel


def test_affine_rmsnorm_delegates_on_npu(monkeypatch):
    """NPU + weight present must not take the manual path."""
    monkeypatch.setattr("diffusers.utils.is_torch_npu_available", lambda: True)
    sentinel = object()
    calls = []

    def fake_original(self, x):
        calls.append(x)
        return sentinel

    monkeypatch.setattr(normalization.RMSNorm, "forward", fake_original)
    _reset_and_patch(monkeypatch)

    norm = RMSNorm(16, eps=1e-6, elementwise_affine=True)
    assert norm(torch.randn(2, 16)) is sentinel
    assert len(calls) == 1
