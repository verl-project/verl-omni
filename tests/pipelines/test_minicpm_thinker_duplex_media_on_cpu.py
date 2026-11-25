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
"""CPU tests for the thinker adapter's duplex media branch (explicit bounds) and
its two-stage log-probability hook on verl's dense output preparation."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData, NonTensorStack
from verl.utils import tensordict_utils as tu
from verl.utils import torch_functional as verl_F
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter
from verl_omni.utils.dataset.duplex.policy_measure import two_stage_logprob
from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY
from verl_omni.workers.engine.fsdp.omni_impl import OmniFSDPEngine

PLACEHOLDER = 9004  # the duplex tests' <|audio|> id
UNIT = 9001
UNIT_END = 9002


def _payload(bounds: list[list[int]], mels: int | None = None, *, reference: bool = False) -> dict:
    payload = {
        "audio_bounds": [list(span) for span in bounds],
        "unit_mels": [np.ones((span[1] - span[0], 80), dtype=np.float32) for span in bounds]
        if mels is None
        else [np.ones((mels, 80), dtype=np.float32) for _ in bounds],
        "unit_mel_lens": [span[1] - span[0] if mels is None else mels for span in bounds],
    }
    if reference:
        # What to_extra_fields keeps of the reference block without its PCM.
        payload["ref_audio"] = {"sample_rate": 16000, "num_audio_tokens": bounds[0][1] - bounds[0][0], "source": ""}
    return payload


def _padded_ids(num_samples: int, prompt_len: int, spans_per_sample: list[list[tuple[int, int]]]):
    rows = []
    for spans in spans_per_sample:
        ids = [1] * prompt_len
        for start, end in spans:
            ids.extend([UNIT] + [PLACEHOLDER] * (end - start) + [UNIT_END])
        rows.append(ids)
    width = max(len(row) for row in rows)
    return torch.tensor([row + [0] * (width - len(row)) for row in rows], dtype=torch.long)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_padded_duplex_batch_receives_explicit_bounds(batch_size):
    prompt_len = 4
    spans = [[(5, 15)]] * batch_size
    input_ids = _padded_ids(batch_size, prompt_len, spans)
    # Full-row coordinates: the span in ids starts at the <unit> token + 1.
    payloads = [_payload([[prompt_len + 1, prompt_len + 11]]) for _ in range(batch_size)]
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(batch_size, -1).contiguous()},
        {DUPLEX_REPLAY_KEY: payloads},
        model_config=None,
    )
    data = result["data"]
    assert len(data["audio_features"]) == batch_size
    assert len(data["audio_feature_lens"][0]) == 1
    assert data["audio_bounds"] == [[[prompt_len + 1, prompt_len + 11]]] * batch_size
    # The placeholder run the bounds cover is exactly the audio span in the ids.
    for sample in range(batch_size):
        start, end = data["audio_bounds"][sample][0]
        assert set(data["input_ids"][sample][start:end].tolist()) == {PLACEHOLDER}


def test_packed_duplex_batch_shifts_bounds_by_packed_offsets():
    prompt_len = 2
    spans = [[(3, 5)], [(3, 5)]]
    input_ids_rows = []
    for spans_one in spans:
        ids = [1] * prompt_len
        for start, end in spans_one:
            ids.extend([UNIT] + [PLACEHOLDER] * (end - start) + [UNIT_END])
        input_ids_rows.append(ids)
    packed = torch.tensor([sum(input_ids_rows, [])], dtype=torch.long)
    # rmpad: position ids reset per sample make _is_packed_batch true.
    position_ids = torch.tensor([list(range(len(input_ids_rows[0]))) + list(range(len(input_ids_rows[1])))])
    row_len = len(input_ids_rows[0])
    payloads = [
        _payload([[prompt_len + 1, prompt_len + 3]]),
        _payload([[prompt_len + 1, prompt_len + 3]]),
    ]
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": packed, "position_ids": position_ids},
        {DUPLEX_REPLAY_KEY: payloads},
        model_config=None,
    )
    data = result["data"]
    # After packing, both samples' media fold into one pseudo-row.
    assert len(data["audio_bounds"]) == 1
    first, second = data["audio_bounds"][0]
    assert first == [prompt_len + 1, prompt_len + 3]
    assert second == [row_len + prompt_len + 1, row_len + prompt_len + 3]
    # The folded features keep both samples' clips in order.
    assert len(data["audio_feature_lens"][0]) == 2
    # The episode boundaries survive the fold for the per-episode audio-cache reset.
    assert data["packed_audio_clips"] == [1, 1]
    assert data["audio_clip_kinds"] == ["unit", "unit"]


def test_duplex_batch_stamps_clip_kinds_per_row_in_clip_order():
    # Each row: the offline reference block, then three streaming units.
    spans = [[(2, 15), (17, 27), (29, 39), (41, 51)]] * 2
    input_ids = _padded_ids(2, 2, spans)
    payloads = [_payload([[3, 16], [18, 28], [30, 40], [42, 52]], reference=True) for _ in range(2)]
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(2, -1).contiguous()},
        {DUPLEX_REPLAY_KEY: payloads},
        model_config=None,
    )
    data = result["data"]
    assert data["audio_clip_kinds"] == ["reference", "unit", "unit", "unit"] * 2
    assert sum(len(sample) for sample in data["audio_feature_lens"]) == len(data["audio_clip_kinds"])
    # A reference-less replay streams every clip, its first unit included.
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(2, -1).contiguous()},
        {DUPLEX_REPLAY_KEY: [_payload([[3, 16], [18, 28], [30, 40], [42, 52]]) for _ in range(2)]},
        model_config=None,
    )
    assert result["data"]["audio_clip_kinds"] == ["unit"] * 8


def test_nonduplex_batch_carries_no_clip_kinds():
    input_ids = torch.ones(2, 6, dtype=torch.long)
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(6).expand(2, -1).contiguous()},
        {},
        model_config=None,
    )
    assert "audio_clip_kinds" not in result["data"]
    assert "packed_audio_clips" not in result["data"]


def test_missing_replay_mels_fail_closed():
    input_ids = _padded_ids(1, 2, [[(3, 5)]])
    with pytest.raises(ValueError, match="mel features but carries audio bounds"):
        MiniCPMThinkerAdapter.prepare_model_inputs(
            {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1])[None]},
            {DUPLEX_REPLAY_KEY: [{"audio_bounds": [[3, 5]]}]},
            model_config=None,
        )


def test_missing_replay_bounds_fail_closed():
    input_ids = _padded_ids(1, 2, [[(3, 5)]])
    with pytest.raises(ValueError, match="mel features but no audio bounds"):
        MiniCPMThinkerAdapter.prepare_model_inputs(
            {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1])[None]},
            {DUPLEX_REPLAY_KEY: [{"unit_mels": [np.zeros((80, 10), np.float32)], "unit_mel_lens": [10]}]},
            model_config=None,
        )


def test_degenerate_zero_unit_payload_passes_through():
    # The failure-tolerance path's zero-unit payload (no mels, no bounds)
    # contributes empty media containers instead of failing the micro-batch.
    input_ids = _padded_ids(1, 2, [[(3, 5)]])
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1])[None]},
        {DUPLEX_REPLAY_KEY: [{"audio_bounds": []}]},
        model_config=None,
    )
    data = result["data"]
    assert data["audio_features"] == [[]]
    assert data["audio_feature_lens"] == [[]]
    assert data["audio_bounds"] == [[]]


def test_degenerate_row_keeps_the_good_rows_media():
    input_ids = _padded_ids(2, 2, [[(3, 5)], [(3, 5)]])
    payloads = [_payload([[3, 5]]), {"audio_bounds": []}]
    result = MiniCPMThinkerAdapter.prepare_model_inputs(
        {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(2, -1).contiguous()},
        {DUPLEX_REPLAY_KEY: payloads},
        model_config=None,
    )
    data = result["data"]
    assert data["audio_bounds"][0] == [[3, 5]]  # the good row keeps its bounds
    assert data["audio_bounds"][1] == []  # the degenerate row is empty
    assert len(data["audio_feature_lens"][0]) == 1


def test_count_mismatch_fails_closed():
    input_ids = _padded_ids(2, 2, [[(3, 5)], [(3, 5)]])
    with pytest.raises(ValueError, match="replay payloads"):
        MiniCPMThinkerAdapter.prepare_model_inputs(
            {"input_ids": input_ids, "position_ids": torch.arange(input_ids.shape[1]).expand(2, -1).contiguous()},
            {DUPLEX_REPLAY_KEY: [_payload([[3, 5]])]},
            model_config=None,
        )


# ---------------------------------------------------------------------------
# prepare_model_outputs: the two-stage log-probability hook
# ---------------------------------------------------------------------------

_VOCAB = 16
_CHUNK_EOS, _TTS_PAD, _BAD = 5, 6, 7
_FORBIDDEN = torch.tensor([_TTS_PAD, _BAD, _CHUNK_EOS])
# Two dense samples of different lengths: (prompt, response) token counts, the per-position
# response mask, and each row's sampling temperature (a sampled row and a greedy row).
_PROMPT_RESPONSE = [(3, 4), (2, 3)]
_RESPONSE_MASK = [[1, 0, 1, 0], [0, 1, 0]]
_TEMPERATURES = [0.7, 0.0]


class _DuplexTokenizer:
    """Checkpoint-tokenizer stub: the duplex specials resolve; ``bad_token_ids`` only when given."""

    unk_token_id = 0

    def __init__(self, bad_token_ids: list[int] | None = None) -> None:
        if bad_token_ids is not None:
            self.bad_token_ids = bad_token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|tts_pad|>": _TTS_PAD, "<|chunk_eos|>": _CHUNK_EOS}.get(token, self.unk_token_id)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        return []


_STUB_TOKENIZER = _DuplexTokenizer(bad_token_ids=[_BAD])


def _model_config(tokenizer=_STUB_TOKENIZER):
    """An ``OmniModelConfig`` stand-in carrying the processor (``tokenizer=None``: none loaded)."""
    return SimpleNamespace(processor=None if tokenizer is None else SimpleNamespace(tokenizer=tokenizer))


def _duplex_micro_batch(seed: int, *, use_remove_padding: bool | None = False) -> tuple[TensorDict, list[torch.Tensor]]:
    """A dense two-sample duplex batch: sample 0's response holds a boundary hit, a forced
    ``tts_pad`` (a forbidden token at a mask-0 position) and a content token; sample 1 a
    content token drawn greedily. ``use_remove_padding=None`` leaves the flag unset."""
    generator = torch.Generator().manual_seed(seed)
    tokens = [
        torch.randint(0, _VOCAB, (prompt + response,), generator=generator) for prompt, response in _PROMPT_RESPONSE
    ]
    tokens[0][3:7] = torch.tensor([_CHUNK_EOS, _TTS_PAD, 2, 4])
    lengths = [row.numel() for row in tokens]
    input_ids = torch.nested.nested_tensor_from_jagged(torch.cat(tokens), torch.tensor([0, lengths[0], sum(lengths)]))
    response_mask = torch.nested.nested_tensor_from_jagged(
        torch.tensor(sum(_RESPONSE_MASK, []), dtype=torch.float32), torch.tensor([0, 4, 7])
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "response_mask": response_mask, "temperature": torch.tensor(_TEMPERATURES)},
        batch_size=[2],
    )
    if use_remove_padding is not None:
        tu.assign_non_tensor(micro_batch, use_remove_padding=use_remove_padding)
    micro_batch[DUPLEX_REPLAY_KEY] = NonTensorStack.from_list([NonTensorData({"audio_bounds": []})] * 2)
    return micro_batch, tokens


def _dense_output_args(micro_batch: TensorDict) -> dict:
    """The three keys verl's dense ``prepare_model_inputs`` branch stores."""
    return {
        "temperature_is_one": False,
        "temperature": micro_batch["temperature"],
        "input_ids_rmpad_rolled": torch.roll(micro_batch["input_ids"].values(), shifts=-1, dims=0),
    }


def _verl_model_output(logits: torch.Tensor, micro_batch: TensorDict, *, calculate_entropy: bool) -> dict:
    """verl's dense-branch ``prepare_model_outputs``, unmodified, on a bare engine."""
    tu.assign_non_tensor(micro_batch, calculate_entropy=calculate_entropy)
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.engine_config = SimpleNamespace(
        batch_invariant=False, entropy_checkpointing=False, entropy_from_logits_with_chunking=False
    )
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits
    return engine.prepare_model_outputs(
        SimpleNamespace(logits=logits), _dense_output_args(micro_batch), micro_batch, None
    )


def _hook(logits, verl_output, micro_batch, model_config):
    return MiniCPMThinkerAdapter.prepare_model_outputs(
        dict(verl_output), SimpleNamespace(logits=logits), _dense_output_args(micro_batch), micro_batch, model_config
    )


def _batch_logits(requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(2, 7, _VOCAB, generator=torch.Generator().manual_seed(1), requires_grad=requires_grad)


class TestPrepareModelOutputs:
    def test_masked_positions_take_the_joint_and_every_other_position_keeps_verl(self):
        micro_batch, tokens = _duplex_micro_batch(seed=0)
        logits = _batch_logits()
        with torch.no_grad():
            verl_output = _verl_model_output(logits, micro_batch, calculate_entropy=True)
        output = _hook(logits, verl_output, micro_batch, _model_config())
        new_log_probs, verl_log_probs = output["log_probs"].values(), verl_output["log_probs"].values()

        # Brute force over every response position: sample i's position r is scored by the logit one
        # token earlier, packed row off_i + seq_len_i − resp_len_i − 1 + r (response_from_nested's shift).
        replaced_rows, kept_rows, offset = [], [], 0
        for sample, (prompt, response) in enumerate(_PROMPT_RESPONSE):
            seq_len = prompt + response
            for r, mask_bit in enumerate(_RESPONSE_MASK[sample]):
                row = offset + seq_len - response - 1 + r
                action = tokens[sample][prompt + r].item()
                if mask_bit:
                    expected = two_stage_logprob(
                        logits[sample, prompt + r - 1],
                        action,
                        chunk_eos_id=_CHUNK_EOS,
                        forbidden_index=_FORBIDDEN,
                        temperature=_TEMPERATURES[sample],
                    )
                    assert new_log_probs[row].item() == pytest.approx(expected, abs=1e-5)
                    replaced_rows.append(row)
                else:
                    assert new_log_probs[row].item() == verl_log_probs[row].item()
                    kept_rows.append(row)
            offset += seq_len
        # Boundary (row 2), content at T=0.7 (row 4), content at T=0 (row 9): exactly the mask-1 rows changed.
        assert replaced_rows == [2, 4, 9]
        assert kept_rows == [3, 5, 8, 10]  # the forced tts_pad row (3) included
        assert (new_log_probs != verl_log_probs).nonzero().flatten().tolist() == replaced_rows
        assert output["log_probs"].is_nested and output["log_probs"].offsets().tolist() == [0, 7, 12]
        assert output["entropy"] is verl_output["entropy"]

    def test_old_ref_and_update_forwards_score_identically_under_the_same_weights(self):
        micro_batch, _ = _duplex_micro_batch(seed=0)
        logits = _batch_logits()
        scored = {}
        # verl's three actor forwards differ in the entropy flag: old_log_probs computes it, the
        # reference never does, the update forward only with entropy_coeff != 0.
        for forward, calculate_entropy in (("old", True), ("ref", False), ("update", False), ("update+entropy", True)):
            with torch.no_grad():
                verl_output = _verl_model_output(logits, micro_batch, calculate_entropy=calculate_entropy)
            assert ("entropy" in verl_output) is calculate_entropy
            scored[forward] = _hook(logits, verl_output, micro_batch, _model_config())["log_probs"].values()
        torch.testing.assert_close(scored["old"], scored["ref"], rtol=0, atol=0)
        torch.testing.assert_close(scored["update"], scored["ref"], rtol=0, atol=0)
        torch.testing.assert_close(scored["update+entropy"], scored["ref"], rtol=0, atol=0)

    def test_training_forward_backpropagates_through_the_joint_at_the_replaced_rows_only(self):
        micro_batch, _ = _duplex_micro_batch(seed=0)
        logits = _batch_logits(requires_grad=True)
        verl_output = _verl_model_output(logits, micro_batch, calculate_entropy=False)
        output = _hook(logits, verl_output, micro_batch, _model_config())
        output["log_probs"].values()[[2, 4, 9]].sum().backward()
        gradient = logits.grad
        # Finite despite the forced tts_pad row, whose unapplied joint is -inf.
        assert torch.isfinite(gradient).all()
        moved = {tuple(position) for position in gradient.abs().sum(-1).nonzero().tolist()}
        assert moved == {(0, 2), (0, 4), (1, 2)}  # padded (sample, prompt + r − 1)

    def test_forbidden_set_is_resolved_from_the_checkpoint_tokenizer_on_every_forward(self):
        micro_batch, _ = _duplex_micro_batch(seed=0)
        logits = _batch_logits()
        verl_output = _verl_model_output(logits, micro_batch, calculate_entropy=False)
        with pytest.raises(RuntimeError, match="processor"):
            _hook(logits, verl_output, micro_batch, _model_config(tokenizer=None))
        with pytest.raises(ValueError, match="bad_token_ids"):
            _hook(logits, verl_output, micro_batch, _model_config(_DuplexTokenizer()))
        with pytest.raises(RuntimeError, match="processor"):
            _hook(logits, verl_output, micro_batch, None)
        torch.testing.assert_close(
            _hook(logits, verl_output, micro_batch, _model_config())["log_probs"].values(),
            _hook(logits, verl_output, micro_batch, _model_config())["log_probs"].values(),
        )


def _omni_engine() -> OmniFSDPEngine:
    """A bare omni engine whose adapter is the real MiniCPM thinker; its verl base is faked per test."""
    engine = object.__new__(OmniFSDPEngine)
    engine.model_config = _model_config()
    engine.model_adapter_cls = MiniCPMThinkerAdapter
    return engine


def test_engine_dispatch_returns_verl_output_untouched_without_a_duplex_replay():
    micro_batch, _ = _duplex_micro_batch(seed=0)
    del micro_batch[DUPLEX_REPLAY_KEY]
    verl_output = {"log_probs": "verl"}
    with patch.object(FSDPEngineWithLMHead, "prepare_model_outputs", return_value=verl_output):
        assert _omni_engine().prepare_model_outputs(object(), {}, micro_batch, None) is verl_output


@pytest.mark.parametrize("packed_flag", [True, None], ids=["use_remove_padding=true", "flag-absent"])
def test_engine_dispatch_fails_closed_on_a_packed_duplex_batch(packed_flag):
    # Sequence parallelism requires remove-padding in verl, so the packed check covers it.
    micro_batch, _ = _duplex_micro_batch(seed=0, use_remove_padding=packed_flag)
    with patch.object(FSDPEngineWithLMHead, "prepare_model_outputs", return_value={"log_probs": "verl"}):
        with pytest.raises(NotImplementedError, match="use_remove_padding=false"):
            _omni_engine().prepare_model_outputs(SimpleNamespace(logits=_batch_logits()), {}, micro_batch, None)
