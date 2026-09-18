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
"""CPU correctness tests for the omni nitrobrew chunked-vocab KL kernels.

Compares the chunked online-softmax forward (and autograd backward) against a
naive reference that materializes the full [N, V] teacher logits, and checks
the multi-teacher grouping path against per-group references.

Ports verl PR 6194's test structure (Tilde Research) and adds a moped
grouping case.
"""

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

import pytest
import torch
import torch.nn.functional as F

from verl_omni.trainer.distillation.nitrobrew_loss import (
    _chunked_kl_forward,
    _chunked_reverse_kl_forward,
    _NitrobrewKL,
    _NitrobrewReverseKL,
    compute_nitrobrew_multi_kl,
    compute_nitrobrew_multi_reverse_kl,
)

# Re-export single-teacher internals needed for reference comparisons.


def _naive_forward_kl(z, w, s, temperature=1.0):
    zt = (z @ w.T).float() / temperature
    s = s.float() / temperature
    log_pt = F.log_softmax(zt, dim=-1)
    log_ps = F.log_softmax(s, dim=-1)
    pt = log_pt.exp()
    return (pt * (log_pt - log_ps)).sum(dim=-1)


def _naive_reverse_kl(z, w, s, temperature=1.0):
    zt = (z @ w.T).float() / temperature
    s = s.float() / temperature
    log_pt = F.log_softmax(zt, dim=-1)
    log_ps = F.log_softmax(s, dim=-1)
    ps = log_ps.exp()
    return (ps * (log_ps - log_pt)).sum(dim=-1)


def _make_inputs(seed=0, n=4, v=32, d=8):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    w = torch.randn(v, d, generator=g) * 0.1
    s = torch.randn(n, v, generator=g)
    return z, w, s


class TestSingleTeacher:
    def test_forward_matches_naive(self):
        z, w, s = _make_inputs()
        kl_chunked, _, _ = _chunked_kl_forward(z, w, s, chunk_V=4)
        kl_naive = _naive_forward_kl(z, w, s)
        assert torch.allclose(kl_chunked, kl_naive, atol=1e-5, rtol=1e-4)

    def test_reverse_matches_naive(self):
        z, w, s = _make_inputs(seed=1)
        kl_chunked, _, _ = _chunked_reverse_kl_forward(z, w, s, chunk_V=4)
        kl_naive = _naive_reverse_kl(z, w, s)
        assert torch.allclose(kl_chunked, kl_naive, atol=1e-5, rtol=1e-4)

    def test_forward_with_temperature(self):
        z, w, s = _make_inputs(seed=2)
        kl_chunked, _, _ = _chunked_kl_forward(z, w, s, chunk_V=8, temperature=0.7)
        kl_naive = _naive_forward_kl(z, w, s, temperature=0.7)
        assert torch.allclose(kl_chunked, kl_naive, atol=1e-5, rtol=1e-4)

    def test_forward_backward_matches_autograd(self):
        z, w, s = _make_inputs(seed=3)
        s_chunked = s.clone().requires_grad_(True)
        s_naive = s.clone().requires_grad_(True)

        kl_chunked = _NitrobrewKL.apply(z, w, s_chunked, 4, 1.0, None)
        kl_naive = _naive_forward_kl(z, w, s_naive)
        grad_out = torch.randn_like(kl_chunked)
        kl_chunked.backward(grad_out)
        kl_naive.backward(grad_out)
        assert torch.allclose(s_chunked.grad, s_naive.grad, atol=1e-5, rtol=1e-4)

    def test_reverse_backward_matches_autograd(self):
        z, w, s = _make_inputs(seed=4)
        s_chunked = s.clone().requires_grad_(True)
        s_naive = s.clone().requires_grad_(True)
        kl_chunked = _NitrobrewReverseKL.apply(z, w, s_chunked, 4, 1.0)
        kl_naive = _naive_reverse_kl(z, w, s_naive)
        grad_out = torch.randn_like(kl_chunked)
        kl_chunked.backward(grad_out)
        kl_naive.backward(grad_out)
        assert torch.allclose(s_chunked.grad, s_naive.grad, atol=1e-5, rtol=1e-4)


class _LossConfig:
    def __init__(self, temperature=1.0, log_prob_min_clamp=-10.0):
        self.kd_temperature = temperature
        self.log_prob_min_clamp = log_prob_min_clamp


class TestMultiTeacherGrouping:
    def _config(self):
        return type("C", (), {"distillation_loss": _LossConfig()})()

    def test_forward_groups_by_teacher(self):
        # Two teachers; tokens 0,1 -> t0; tokens 2,3 -> t1.
        z, w, s = _make_inputs(seed=7, n=4)
        w0, w1 = w, w * 0.9  # distinct unembeddings
        key_ids = torch.tensor([0, 0, 1, 1])
        unembeds = {0: w0, 1: w1}

        out = compute_nitrobrew_multi_kl(s[None], z[None], key_ids, unembeds, self._config(), "thd")
        kl = out["distillation_losses"][0]

        expected0 = _naive_forward_kl(z[:2], w0, s[:2])
        expected1 = _naive_forward_kl(z[2:], w1, s[2:])
        assert torch.allclose(kl[:2], expected0, atol=1e-5, rtol=1e-4)
        assert torch.allclose(kl[2:], expected1, atol=1e-5, rtol=1e-4)
        assert not torch.isnan(kl).any()

    def test_reverse_groups_by_teacher(self):
        z, w, s = _make_inputs(seed=8, n=4)
        w0, w1 = w, w * 1.1
        key_ids = torch.tensor([0, 1, 0, 1])
        unembeds = {0: w0, 1: w1}
        out = compute_nitrobrew_multi_reverse_kl(s[None], z[None], key_ids, unembeds, self._config(), "thd")
        kl = out["distillation_losses"][0]
        assert torch.allclose(kl[[0, 2]], _naive_reverse_kl(z[[0, 2]], w0, s[[0, 2]]), atol=1e-5, rtol=1e-4)
        assert torch.allclose(kl[[1, 3]], _naive_reverse_kl(z[[1, 3]], w1, s[[1, 3]]), atol=1e-5, rtol=1e-4)

    def test_single_teacher_equivalent(self):
        z, w, s = _make_inputs(seed=9, n=4)
        key_ids = torch.zeros(4, dtype=torch.long)
        out = compute_nitrobrew_multi_kl(s[None], z[None], key_ids, {0: w}, self._config(), "thd")
        expected = _naive_forward_kl(z, w, s)
        assert torch.allclose(out["distillation_losses"][0], expected, atol=1e-5, rtol=1e-4)

    def test_missing_teacher_raises(self):
        z, w, s = _make_inputs(seed=10, n=4)
        key_ids = torch.tensor([0, 0, 5, 5])  # no unembed for id 5
        with pytest.raises(RuntimeError, match="no registered teacher unembedding"):
            compute_nitrobrew_multi_kl(s[None], z[None], key_ids, {0: w}, self._config(), "thd")

    def test_kd_temperature_applied(self):
        z, w, s = _make_inputs(seed=11, n=4)
        key_ids = torch.zeros(4, dtype=torch.long)
        cfg = type("C", (), {"distillation_loss": _LossConfig(temperature=0.7)})()
        out = compute_nitrobrew_multi_kl(s[None], z[None], key_ids, {0: w}, cfg, "thd")
        expected = _naive_forward_kl(z, w, s, temperature=0.7)
        assert torch.allclose(out["distillation_losses"][0], expected, atol=1e-5, rtol=1e-4)
