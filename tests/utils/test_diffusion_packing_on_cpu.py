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
"""Shared packed self-attention contracts independent of any diffusion model."""

import pytest
import torch
from torch.nn import functional as F

from verl_omni.utils import diffusion_packing
from verl_omni.utils.diffusion_packing import PackedSequenceLayout


def test_native_packing_matches_independent_samples_and_gradients():
    torch.manual_seed(3)
    lengths = [2, 5, 3]
    layout = PackedSequenceLayout.from_lengths(lengths, torch.device("cpu"))
    qkv = [torch.randn(1, sum(lengths), 2, 8, requires_grad=True) for _ in range(3)]
    expected = torch.cat(
        [
            F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            for q, k, v in zip(*(tensor.split(lengths, dim=1) for tensor in qkv), strict=True)
        ],
        dim=1,
    )
    actual = layout.attention(*qkv, backend="native")
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual.square().sum(), qkv)
    expected_grad = torch.autograd.grad(expected.square().sum(), qkv)
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual_value, expected_value)
    changed = [tensor.detach().clone() for tensor in qkv]
    changed[2][:, 2:7] += 10
    isolated = layout.attention(*changed, backend="native")
    torch.testing.assert_close(isolated[:, :2], actual[:, :2])
    torch.testing.assert_close(isolated[:, 7:], actual[:, 7:])


def test_fa3_uses_cumulative_boundaries_without_dense_padding(monkeypatch):
    layout = PackedSequenceLayout.from_lengths([2, 5], torch.device("cpu"))
    query = torch.randn(1, 7, 2, 8)
    calls = []

    def kernel(q, k, v, **kwargs):
        calls.append(kwargs)
        assert q.shape == (7, 2, 8)
        assert kwargs["cu_seqlens_q"].tolist() == [0, 2, 7]
        assert kwargs["cu_seqlens_q"].dtype == torch.int32
        assert kwargs["cu_seqlens_k"] is kwargs["cu_seqlens_q"]
        assert kwargs["max_seqlen_q"] == kwargs["max_seqlen_k"] == 5
        assert kwargs["causal"] is False
        return v

    monkeypatch.setattr(diffusion_packing, "get_fa3_varlen", lambda: kernel)
    torch.testing.assert_close(layout.attention(query, query, query, "_flash_3_varlen_hub"), query)
    assert len(calls) == 1
    assert "valid_mask" not in vars(layout)
    assert "padded_indices" not in vars(layout)


@pytest.mark.parametrize("lengths", [[], [0], [-1], [2, 0]])
def test_invalid_sample_boundaries_fail_closed(lengths):
    with pytest.raises(ValueError, match="positive lengths"):
        PackedSequenceLayout.from_lengths(lengths, torch.device("cpu"))


def test_unavailable_fa3_fails_without_a_native_fallback(monkeypatch):
    def unavailable():
        raise RuntimeError("FA3 kernel unavailable")

    monkeypatch.setattr(diffusion_packing, "get_fa3_varlen", unavailable)
    layout = PackedSequenceLayout.from_lengths([2], torch.device("cpu"))
    query = torch.randn(1, 2, 2, 8)
    with pytest.raises(RuntimeError, match="FA3 kernel unavailable"):
        layout.attention(query, query, query, "_flash_3_varlen_hub")


def test_unsupported_backend_fails_closed():
    layout = PackedSequenceLayout.from_lengths([2], torch.device("cpu"))
    query = torch.randn(1, 2, 2, 8)
    with pytest.raises(ValueError, match="Unsupported packed attention backend"):
        layout.attention(query, query, query, "flash_varlen_hub")
