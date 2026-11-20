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
"""CPU tests for the exact-distribution logprob recomputation.

The truncation math is checked against an independent brute-force nucleus
reference (enumerate the ordered support, keep tokens whose preceding mass is
within top_p, renormalize) rather than against itself.
"""

import math

import pytest
import torch

from verl_omni.utils.dataset.duplex.sampling_logprobs import (
    DuplexSamplingContext,
    DuplexUnitInputs,
    build_sampling_context,
    recompute_action_logprobs,
    top_k_top_p_candidate_mass,
    top_k_top_p_filter,
    unit_inputs_from_trajectory,
)
from verl_omni.utils.dataset.duplex.trajectory import AXIS_TURN, DuplexEpisodeEvent, DuplexTrajectory

VOCAB = 64
CHUNK_EOS = 7
TTS_PAD = 3


def brute_force_mass(logits: torch.Tensor, token_id: int, *, top_k: int, top_p: float) -> float:
    """Reference: enumerate the ordered nucleus support and renormalize."""
    logits = logits.reshape(-1).float()
    order_by_value = torch.argsort(logits, descending=True, stable=True)
    kth = logits[order_by_value[top_k - 1]].item() if 0 < top_k < logits.numel() else float("-inf")
    regular = [int(i) for i in order_by_value if logits[i] > kth]
    ties = sorted(int(i) for i in range(logits.numel()) if logits[i] == kth)
    max_v = max(logits[i].item() for i in regular + ties)
    weights = {i: math.exp(logits[i].item() - max_v) for i in regular + ties}
    total = sum(weights.values())
    ordered = sorted(regular, key=lambda i: (-logits[i].item(), i)) + ties
    kept, cumulative = [], 0.0
    for position, token in enumerate(ordered):
        if position == 0 or cumulative <= top_p:
            kept.append(token)
        cumulative += weights[token] / total
    kept_mass = sum(weights[token] for token in kept)
    if token_id not in kept:
        return math.log(torch.finfo(torch.float32).tiny)
    return math.log(weights[token_id] / kept_mass)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("top_k,top_p", [(8, 0.8), (16, 0.95), (0, 0.9), (64, 0.0), (8, 0.0)])
def test_candidate_mass_matches_brute_force_reference(seed, top_k, top_p):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(VOCAB, generator=generator) * 2
    for token_id in (0, int(logits.argmax()), VOCAB - 1):
        engine_style = top_k_top_p_candidate_mass(logits, token_id, top_k=top_k or VOCAB, top_p=top_p or 1.0)
        reference = brute_force_mass(logits, token_id, top_k=top_k or VOCAB, top_p=top_p or 1.0)
        assert engine_style == pytest.approx(reference, rel=1e-4, abs=1e-5), (token_id, top_k, top_p)


@pytest.mark.parametrize("seed", [3, 4])
def test_candidate_mass_handles_kth_value_ties(seed):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(VOCAB, generator=generator)
    # Quantize so several tokens share the k-th value.
    logits = (logits * 2).round() / 2
    top_k = 6
    for token_id in range(0, VOCAB, 7):
        engine_style = top_k_top_p_candidate_mass(logits, token_id, top_k=top_k, top_p=0.85)
        reference = brute_force_mass(logits, token_id, top_k=top_k, top_p=0.85)
        assert engine_style == pytest.approx(reference, rel=2e-3, abs=1e-5), token_id


def test_identity_settings_reduce_to_plain_softmax():
    generator = torch.Generator().manual_seed(5)
    logits = torch.randn(VOCAB, generator=generator)
    token_id = 11
    mass = top_k_top_p_candidate_mass(logits, token_id, top_k=VOCAB, top_p=1.0)
    assert mass == pytest.approx(torch.log_softmax(logits.float(), dim=-1)[token_id].item(), rel=1e-4, abs=1e-6)


def test_greedy_filter_keeps_kth_ties_and_nucleus():
    generator = torch.Generator().manual_seed(6)
    logits = torch.randn(VOCAB, generator=generator)
    logits = (logits * 2).round() / 2  # force ties, including at the k-th value
    filtered = top_k_top_p_filter(logits, top_k=8, top_p=0.9)
    probabilities = torch.softmax(filtered.float(), dim=-1)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1e-6)
    # Every token strictly above the k-th value survives; the nucleus drops
    # at least one tail token; the support is a subset of the top-k ties ∪ above.
    kth = torch.topk(logits, 8).values[-1]
    above = logits > kth
    assert (filtered[above] > float("-inf")).all()
    assert (filtered == float("-inf")).any()
    assert ((filtered > float("-inf")) <= (logits >= kth)).all()


def _context(penalty: float = 1.05) -> DuplexSamplingContext:
    return DuplexSamplingContext(
        chunk_eos_token_id=CHUNK_EOS,
        tts_pad_token_id=TTS_PAD,
        bad_token_ids=(21, 22),
        repetition_penalty=penalty,
    )


def test_recompute_boundary_hit_is_the_raw_softmax():
    generator = torch.Generator().manual_seed(7)
    row = torch.randn(1, VOCAB, generator=generator)
    inputs = [
        DuplexUnitInputs(
            token_id=CHUNK_EOS,
            action_kind="sampled",
            stream="boundary",
            temperature=0.7,
            top_k=100,
            top_p=0.8,
            stage0_prob_chunk_eos=0.3,
        )
    ]
    values, valid = recompute_action_logprobs(row, inputs, _context())
    assert valid[0]
    assert values[0].item() == pytest.approx(torch.log_softmax(row.float(), dim=-1)[0, CHUNK_EOS].item(), rel=1e-5)


def test_recompute_content_joint_combines_stages():
    generator = torch.Generator().manual_seed(8)
    row = torch.randn(1, VOCAB, generator=generator)
    target = 30
    inputs = [
        DuplexUnitInputs(token_id=target, action_kind="sampled", stream="content", temperature=0.7, top_k=16, top_p=0.9)
    ]
    values, _ = recompute_action_logprobs(row, inputs, _context())
    prob_eos = float(torch.softmax(row.float(), dim=-1)[0, CHUNK_EOS])
    stage2_row = row.clone()
    stage2_row[0, [TTS_PAD, 21, 22, CHUNK_EOS]] = float("-inf")
    stage2_row = stage2_row / 0.7
    expected = math.log(1.0 - prob_eos) + brute_force_mass(stage2_row, target, top_k=16, top_p=0.9)
    assert values[0].item() == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_recompute_identity_transform_equals_plain_softmax():
    generator = torch.Generator().manual_seed(9)
    row = torch.randn(1, VOCAB, generator=generator)
    target = 40
    inputs = [
        DuplexUnitInputs(
            token_id=target, action_kind="sampled", stream="content", temperature=1.0, top_k=VOCAB, top_p=1.0
        )
    ]
    values, _ = recompute_action_logprobs(row, inputs, _context(penalty=1.0))
    # No truncation and no penalty: the joint is the plain softmax renormalized
    # over everything the forbidden mask keeps (eos, pad, and bad ids drop out).
    probabilities = torch.softmax(row.float(), dim=-1)
    forbidden = _context().forbidden_token_ids()
    kept_mass = 1.0 - sum(float(probabilities[0, token_id]) for token_id in forbidden)
    expected = math.log(float(probabilities[0, target]) / kept_mass)
    # The candidate path normalizes in its own float32 order; allow its rounding.
    assert values[0].item() == pytest.approx(expected, rel=2e-3, abs=1e-5)


def test_recompute_repetition_penalty_divides_history_logits():
    generator = torch.Generator().manual_seed(10)
    row = torch.randn(1, VOCAB, generator=generator)
    target = 12
    row[0, target] = 4.0  # dominant target token
    inputs = [
        DuplexUnitInputs(
            token_id=target, action_kind="sampled", stream="content", temperature=1.0, top_k=VOCAB, top_p=1.0
        )
    ]
    penalized, _ = recompute_action_logprobs(row, inputs, _context(penalty=1.05), action_history=[[target]])
    clean, _ = recompute_action_logprobs(row, inputs, _context(penalty=1.0), action_history=[[]])
    # Dividing the dominant logit by the penalty must lower its probability.
    assert penalized[0].item() < clean[0].item() - 0.01


def test_forced_rows_are_not_recomputed():
    # The mid-turn listen rewrite is forced (masked out of the loss); the
    # recompute utility recomputes sampled draws only.
    generator = torch.Generator().manual_seed(11)
    row = torch.randn(1, VOCAB, generator=generator)
    inputs = [
        DuplexUnitInputs(
            token_id=50,
            action_kind="forced",
            stream="content",
            temperature=1.0,
            top_k=VOCAB,
            top_p=1.0,
            sampled_token=51,
        )
    ]
    values, valid = recompute_action_logprobs(row, inputs, _context(penalty=1.0))
    assert not valid[0]
    assert values[0].item() != values[0].item()  # NaN: no draw to recompute


def test_forbidden_targets_get_tiny_mass():
    generator = torch.Generator().manual_seed(12)
    row = torch.randn(1, VOCAB, generator=generator)
    inputs = [
        DuplexUnitInputs(token_id=21, action_kind="sampled", stream="content", temperature=0.7, top_k=16, top_p=0.8)
    ]
    values, _ = recompute_action_logprobs(row, inputs, _context())
    assert values[0].item() <= math.log(torch.finfo(torch.float32).tiny) + 1e-6


def test_shape_mismatch_is_rejected():
    inputs = [
        DuplexUnitInputs(token_id=1, action_kind="sampled", stream="content", temperature=1.0, top_k=0, top_p=1.0)
    ] * 3
    with pytest.raises(Exception, match="logits for"):
        recompute_action_logprobs(torch.randn(2, VOCAB), inputs, _context())


class _Tokenizer:
    bad_token_ids = [21, 22]
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return {"<|chunk_eos|>": CHUNK_EOS, "<|tts_pad|>": TTS_PAD, "<|listen|>": 5}.get(token)

    def encode(self, text, add_special_tokens=False):
        return [CHUNK_EOS]


class _Vocabulary:
    listen_token_id = 5


def test_build_sampling_context_resolves_and_validates():
    context = build_sampling_context(
        _Tokenizer(),
        _Vocabulary(),
        {"minicpmo_duplex": {"forbidden_tokens": "tts_pad+bad_token_ids+chunk_eos", "repetition_penalty": 1.05}},
    )
    assert context.forbidden_token_ids() == (3, 7, 21, 22)
    with pytest.raises(Exception, match="unknown forbidden-token identity"):
        build_sampling_context(_Tokenizer(), _Vocabulary(), {"minicpmo_duplex": {"forbidden_tokens": "custom"}})


def test_unit_inputs_from_trajectory_carries_stage_quantities():
    import numpy as np

    from tests.utils.dataset.duplex._duplex_fixtures import listening_then_speaking_wires

    wires = listening_then_speaking_wires()
    wires[1]["model"] = {
        "minicpmo_duplex": {
            "stage0_prob_chunk_eos": 0.25,
            "stage2_logprob": -1.1,
            "forbidden_tokens": "tts_pad+bad_token_ids+chunk_eos",
        }
    }
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=2.0),
        record_set_wires=wires,
        client_audio_chunks=[np.zeros(16000, dtype=np.float32) for _ in range(5)],
    )
    inputs = unit_inputs_from_trajectory(trajectory)
    assert len(inputs) == 5
    assert all(unit.temperature == 0.7 and unit.top_k == 100 and unit.top_p == 0.8 for unit in inputs)
    # The response scope carries the per-stage quantities; the listening scope does not.
    assert inputs[-1].stage0_prob_chunk_eos == pytest.approx(0.25)
    assert inputs[-1].stage2_logprob == pytest.approx(-1.1)
    assert inputs[0].stage0_prob_chunk_eos is None
