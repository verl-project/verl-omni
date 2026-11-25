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
"""CPU tests for the two-stage duplex policy measure (``policy_measure``)."""

import math

import pytest
import torch
from torch.nn.functional import log_softmax, softmax

from verl_omni.utils.dataset.duplex.policy_measure import (
    resolve_forbidden_token_ids,
    two_stage_logprob,
    two_stage_logprobs,
)
from verl_omni.utils.dataset.duplex.records import RecordSetError

_VOCAB = 8
_CHUNK_EOS = 5
_TTS_PAD = 6
_BAD = 7
_FORBIDDEN = torch.tensor([_TTS_PAD, _BAD, _CHUNK_EOS])  # the resolver's composition order
_ROW = torch.tensor([0.5, 0.2, 0.0, -0.5, -1.0, 2.0, -5.0, -5.0])


def _joint(logits: torch.Tensor, token_id: int, temperature: float) -> torch.Tensor:
    return two_stage_logprobs(
        logits,
        torch.tensor([token_id]),
        chunk_eos_id=_CHUNK_EOS,
        forbidden_index=_FORBIDDEN,
        temperature=torch.tensor([temperature]),
    )


def _masked(row: torch.Tensor) -> torch.Tensor:
    return row.index_fill(0, _FORBIDDEN, float("-inf"))


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.3])
def test_boundary_row_is_the_raw_softmax_logprob_whatever_the_temperature(temperature):
    value = _joint(_ROW[None], _CHUNK_EOS, temperature)
    torch.testing.assert_close(value, torch.log(softmax(_ROW, dim=-1))[_CHUNK_EOS : _CHUNK_EOS + 1])


@pytest.mark.parametrize("temperature", [0.7, 1.3])
def test_content_row_is_the_masked_stage2_joint(temperature):
    value = _joint(_ROW[None], 4, temperature)
    p0 = softmax(_ROW, dim=-1)[_CHUNK_EOS]
    expected = torch.log1p(-p0) + log_softmax(_masked(_ROW) / temperature, dim=-1)[4]
    torch.testing.assert_close(value, expected[None])
    # A forbidden token drawn through stage 2 has zero mass under the engine.
    assert _joint(_ROW[None], _TTS_PAD, temperature).item() == float("-inf")


def test_greedy_temperature_compares_at_one():
    torch.testing.assert_close(_joint(_ROW[None], 4, 0.0), _joint(_ROW[None], 4, 1.0))
    torch.testing.assert_close(_joint(_ROW[None], 4, 1e-6), _joint(_ROW[None], 4, 1.0))


def test_saturated_boundary_probability_stays_finite():
    # chunk_eos up by 40 nats: p0 rounds to 1 in fp32; down by 40: p0 underflows to 0.
    saturated = torch.full((_VOCAB,), -40.0)
    saturated[_CHUNK_EOS] = 40.0
    content = _joint(saturated[None], 3, 0.7).item()
    assert math.isfinite(content) and content < -10
    assert _joint(saturated[None], _CHUNK_EOS, 0.7).item() == pytest.approx(0.0, abs=1e-6)
    vanishing = torch.full((_VOCAB,), 40.0)
    vanishing[_CHUNK_EOS] = -40.0
    boundary = _joint(vanishing[None], _CHUNK_EOS, 0.7).item()
    assert math.isfinite(boundary) and boundary < -80
    assert math.isfinite(_joint(vanishing[None], 3, 0.7).item())
    for row, token_id in ((saturated, 3), (saturated, _CHUNK_EOS), (vanishing, _CHUNK_EOS)):
        scalar = two_stage_logprob(row, token_id, chunk_eos_id=_CHUNK_EOS, forbidden_index=_FORBIDDEN, temperature=0.7)
        assert math.isfinite(scalar) and scalar == pytest.approx(_joint(row[None], token_id, 0.7).item(), abs=1e-6)


def test_scalar_and_vectorized_agree_on_every_branch():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(6, _VOCAB, generator=generator)
    token_ids = torch.tensor([_CHUNK_EOS, _CHUNK_EOS, 0, 3, 4, _TTS_PAD])
    temperatures = torch.tensor([0.0, 0.7, 1.0, 1.3, 0.7, 0.7])
    vectorized = two_stage_logprobs(
        logits, token_ids, chunk_eos_id=_CHUNK_EOS, forbidden_index=_FORBIDDEN, temperature=temperatures
    )
    for index, (row, token_id, temperature) in enumerate(zip(logits, token_ids, temperatures.tolist(), strict=True)):
        scalar = two_stage_logprob(
            row, int(token_id), chunk_eos_id=_CHUNK_EOS, forbidden_index=_FORBIDDEN, temperature=temperature
        )
        vector_value = vectorized[index].item()
        assert scalar == vector_value or abs(scalar - vector_value) < 1e-6  # -inf == -inf on the forbidden draw


def test_temperature_broadcasts_a_single_value_and_rejects_a_misaligned_vector():
    logits = torch.stack([_ROW, _ROW])
    per_row = two_stage_logprobs(
        logits,
        torch.tensor([4, 3]),
        chunk_eos_id=_CHUNK_EOS,
        forbidden_index=_FORBIDDEN,
        temperature=torch.tensor([0.7, 0.7]),
    )
    torch.testing.assert_close(
        two_stage_logprobs(
            logits,
            torch.tensor([4, 3]),
            chunk_eos_id=_CHUNK_EOS,
            forbidden_index=_FORBIDDEN,
            temperature=torch.tensor([0.7]),
        ),
        per_row,
    )
    torch.testing.assert_close(
        two_stage_logprobs(
            logits, torch.tensor([4, 3]), chunk_eos_id=_CHUNK_EOS, forbidden_index=_FORBIDDEN, temperature=0.7
        ),
        per_row,
    )
    with pytest.raises(ValueError, match="one value per logit row"):
        two_stage_logprobs(
            logits,
            torch.tensor([4, 3]),
            chunk_eos_id=_CHUNK_EOS,
            forbidden_index=_FORBIDDEN,
            temperature=torch.tensor([0.7, 0.7, 0.7]),
        )
    with pytest.raises(ValueError, match=r"logits \[N, V\] and token_ids \[N\]"):
        two_stage_logprobs(
            logits, torch.tensor([[4, 3]]), chunk_eos_id=_CHUNK_EOS, forbidden_index=_FORBIDDEN, temperature=0.7
        )


def test_stage2_gradient_is_exactly_zero_on_forbidden_logits():
    logits = _ROW.clone().requires_grad_(True)[None]
    drawn = 4
    temperature = 0.7
    joint = _joint(logits, drawn, temperature)
    p0 = softmax(logits, dim=-1)[0, _CHUNK_EOS]
    stage2_only = joint - torch.log1p(-p0)
    (gradient,) = torch.autograd.grad(stage2_only, logits)
    gradient = gradient[0]
    assert gradient[_FORBIDDEN].abs().max().item() == 0.0
    one_hot = torch.zeros(_VOCAB)
    one_hot[drawn] = 1.0
    # d log_softmax(z / T)[a] / dz_j = (δ_aj − softmax(mask(z) / T)[j]) / T
    torch.testing.assert_close(gradient, (one_hot - softmax(_masked(logits[0]) / temperature, dim=-1)) / temperature)


def test_content_row_gradient_matches_the_joint_analytic_form():
    logits = _ROW.clone().requires_grad_(True)[None]
    drawn = 4
    temperature = 0.7
    (gradient,) = torch.autograd.grad(_joint(logits, drawn, temperature), logits)
    gradient = gradient[0]
    assert torch.isfinite(gradient).all()
    raw, truncated = softmax(logits, dim=-1)[0], softmax(_masked(logits[0]) / temperature, dim=-1)
    p0 = raw[_CHUNK_EOS]
    # Stage 0 reaches a non-boundary logit j through p0 alone: d log(1 − p0) / dz_j = p0 · softmax(z)[j] / (1 − p0).
    for other in (0, 3):  # two non-special, non-drawn logits
        assert gradient[other].item() == pytest.approx(
            (p0 * raw[other] / (1 - p0) - truncated[other] / temperature).item(), rel=1e-5
        )
    one_hot = torch.zeros(_VOCAB)
    one_hot[drawn] = 1.0
    assert gradient[drawn].item() == pytest.approx(
        (p0 * raw[drawn] / (1 - p0) + 1.0 / temperature - truncated[drawn] / temperature).item(), rel=1e-5
    )
    # Every logit, the forbidden ones included, moves a content row through the stage-0 term.
    assert gradient[_FORBIDDEN].abs().min().item() > 0.0


def test_boundary_row_gradient_is_delta_eos_minus_softmax():
    logits = _ROW.clone().requires_grad_(True)[None]
    (gradient,) = torch.autograd.grad(_joint(logits, _CHUNK_EOS, 0.7), logits)
    gradient = gradient[0]
    expected = -softmax(_ROW, dim=-1)
    expected[_CHUNK_EOS] += 1.0
    torch.testing.assert_close(gradient, expected, atol=1e-7, rtol=0.0)


class _StubTokenizer:
    """Special-token lookups only; unknown tokens encode to nothing (never one id)."""

    unk_token_id = 0

    def __init__(self, *, bad_token_ids: list[int] | None = None, **special_ids: int):
        self.ids = {"<|tts_pad|>": _TTS_PAD, "<|chunk_eos|>": _CHUNK_EOS} | special_ids
        if bad_token_ids is not None:
            self.bad_token_ids = bad_token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.ids.get(token, self.unk_token_id)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        return []


def test_forbidden_set_is_tts_pad_then_bad_token_ids_then_chunk_eos():
    assert resolve_forbidden_token_ids(_StubTokenizer(bad_token_ids=[_BAD, 9])) == [_TTS_PAD, _BAD, 9, _CHUNK_EOS]
    assert resolve_forbidden_token_ids(_StubTokenizer()) == [_TTS_PAD, _CHUNK_EOS]


@pytest.mark.parametrize("missing", ["<|tts_pad|>", "<|chunk_eos|>"])
def test_forbidden_set_fails_closed_on_an_unresolvable_special_token(missing):
    stub = _StubTokenizer(bad_token_ids=[_BAD], **{missing: 0})  # resolves to unk
    with pytest.raises(RecordSetError, match="cannot resolve the engine's forbidden set"):
        resolve_forbidden_token_ids(stub)
