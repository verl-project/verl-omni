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
"""CPU tests for the exact M-RoPE frequencies used by the true on-policy actor."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3OmniMoeTextConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextRotaryEmbedding

from verl_omni.models.transformers.qwen3_omni import ExactMRotaryEmbedding
from verl_omni.pipelines.qwen3_omni.thinker_training_adapter import Qwen3OmniThinkerAdapter

HEAD_DIM, HIDDEN, SEQ = 16, 32, 6


def _hf_rotary() -> Qwen3OmniMoeThinkerTextRotaryEmbedding:
    config = Qwen3OmniMoeTextConfig(
        hidden_size=HIDDEN,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [3, 3, 2]},
    )
    return Qwen3OmniMoeThinkerTextRotaryEmbedding(config)


def _positions(dim: int, start: int = 0) -> torch.Tensor:
    text = torch.arange(start, start + SEQ)
    if dim == 2:
        return text[None]
    return torch.stack([text, text // 2, text % 3])[:, None, :]


class _TF32Matmul(torch.overrides.TorchFunctionMode):
    """Rounds fp32 matmul operands to TF32, as vLLM's batch-invariant bmm kernel does."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if func in (torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.bmm):
            args = tuple(
                (a.view(torch.int32) & ~0x1FFF).view(torch.float32)
                if isinstance(a, torch.Tensor) and a.dtype == torch.float32
                else a
                for a in args
            )
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("positions_dim", [2, 3])
def test_matches_hf_rotary_bitwise(dtype, positions_dim):
    hf = _hf_rotary()
    exact = ExactMRotaryEmbedding.from_hf(hf)
    x = torch.randn(1, SEQ, HIDDEN, dtype=dtype)
    positions = _positions(positions_dim)

    cos_hf, sin_hf = hf(x, positions)
    cos, sin = exact(x, positions)

    assert cos.dtype == sin.dtype == dtype
    assert cos.shape == cos_hf.shape == (1, SEQ, HEAD_DIM)
    assert torch.equal(cos, cos_hf) and torch.equal(sin, sin_hf)


def test_frequencies_do_not_go_through_fp32_matmul():
    exact = ExactMRotaryEmbedding.from_hf(_hf_rotary())
    x = torch.randn(1, SEQ, HIDDEN, dtype=torch.bfloat16)
    positions = _positions(3, start=3000)
    cos_ref, sin_ref = exact(x, positions)
    matmul_freqs = exact.inv_freq[:, None] @ positions[0].float()

    with _TF32Matmul():
        cos, sin = exact(x, positions)
        assert not torch.equal(exact.inv_freq[:, None] @ positions[0].float(), matmul_freqs)

    assert torch.equal(cos, cos_ref) and torch.equal(sin, sin_ref)


def test_frequencies_survive_module_dtype_cast():
    exact = ExactMRotaryEmbedding.from_hf(_hf_rotary())
    x = torch.randn(1, SEQ, HIDDEN, dtype=torch.bfloat16)
    positions = _positions(3, start=3000)
    cos_ref, sin_ref = exact(x, positions)

    exact.to(torch.bfloat16)
    cos, sin = exact(x, positions)

    assert torch.equal(cos, cos_ref) and torch.equal(sin, sin_ref)


class _Thinker(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.rotary_emb = _hf_rotary()

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
        self.talker = nn.Linear(2, 2)


@pytest.mark.parametrize("true_on_policy", [True, False])
def test_configure_model_swaps_the_text_rotary_only_for_true_on_policy(true_on_policy):
    module = _Model()
    original = module.thinker.model.rotary_emb

    module = Qwen3OmniThinkerAdapter.configure_model(module, SimpleNamespace(true_on_policy=true_on_policy))

    rotary = module.thinker.model.rotary_emb
    assert not hasattr(module, "talker")
    if true_on_policy:
        assert isinstance(rotary, ExactMRotaryEmbedding)
        assert torch.equal(rotary.inv_freq, original.inv_freq)
        assert rotary.mrope_section == original.mrope_section
    else:
        assert rotary is original
