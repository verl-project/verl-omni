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
"""Dependency-light Qwen3-TTS actor and rollout contract tests."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _load(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[2] / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


forward = _load("qwen3_tts_forward_test", "verl_omni/pipelines/qwen3_tts/talker_forward.py")
rollout = _load("qwen3_tts_rollout_test", "verl_omni/pipelines/qwen3_tts/rollout_utils.py")

TOKENS = forward.TalkerTokens(900, 901, 902, 4196, 4197, 4198, 4203, 4204, 4205)


def test_talker_batch_matches_auto_language_teacher_forcing_layout():
    text_ids = torch.tensor([1, 2, 3, 4, 5, 6])
    codes = torch.arange(3 * 16, dtype=torch.long).reshape(3, 16) % 128
    codes[:, 0] = torch.tensor([10, 11, 12])

    batch = forward.build_talker_batch([text_ids], [codes], TOKENS, sub_codebook_vocab=2048)

    speaker_slot = 6
    codec_start = 8 + text_ids.numel() - 1
    assert batch.input_ids[0, 3:speaker_slot, 1].tolist() == [4203, 4204, 4205]
    assert not batch.codec_embedding_mask[0, speaker_slot]
    assert batch.input_ids[0, speaker_slot + 1, 1] == TOKENS.codec_pad
    torch.testing.assert_close(batch.codec_ids[0, codec_start : codec_start + 3], codes)
    assert batch.logit_start == [codec_start - 1]
    assert batch.codec_lens == [3]


def test_codec0_mask_matches_rollout_vocabulary():
    masked = forward.mask_codec0_logits(torch.zeros((1, 2, 4300)), 2048, TOKENS.codec_eos)

    assert (masked[..., 0] < -1e3).all()
    assert torch.isfinite(masked[..., 1:2048]).all()
    assert (masked[..., 2048 : TOKENS.codec_eos] < -1e3).all()
    assert torch.isfinite(masked[..., TOKENS.codec_eos]).all()
    assert (masked[..., TOKENS.codec_eos + 1 :] < -1e3).all()


def test_only_validated_auto_language_layout_is_accepted():
    assert forward.require_auto_language("auto") == "Auto"
    with pytest.raises(ValueError, match="supports only tts_language=Auto"):
        forward.require_auto_language("Chinese")
    with pytest.raises(ValueError, match="supports only tts_language=Auto"):
        forward.require_auto_language(None)


def test_actor_logits_align_to_effective_codec0_response(monkeypatch):
    class CodePredictor:
        @staticmethod
        def get_input_embeddings():
            return [torch.nn.Embedding(2048, 4)]

    talker = SimpleNamespace(code_predictor=CodePredictor())
    model = SimpleNamespace(
        talker=talker,
        config=SimpleNamespace(
            tts_pad_token_id=TOKENS.tts_pad,
            tts_bos_token_id=TOKENS.tts_bos,
            tts_eos_token_id=TOKENS.tts_eos,
            talker_config=SimpleNamespace(
                codec_pad_id=TOKENS.codec_pad,
                codec_bos_id=TOKENS.codec_bos,
                codec_eos_token_id=TOKENS.codec_eos,
                codec_nothink_id=TOKENS.codec_nothink,
                codec_think_bos_id=TOKENS.codec_think_bos,
                codec_think_eos_id=TOKENS.codec_think_eos,
            ),
        ),
    )

    def fake_logits(_talker, batch, _speaker):
        vocab = torch.arange(1, 4301, dtype=torch.float32).reshape(1, 1, -1)
        return vocab.expand(1, batch.input_ids.shape[1] - 1, -1)

    monkeypatch.setattr(forward, "codec0_logits", fake_logits)
    input_ids = torch.zeros((1, 9), dtype=torch.long)
    input_ids[0, -3:] = torch.tensor([10, 11, 12])
    codes = torch.zeros((1, 6, 16), dtype=torch.long)
    codes[0, :3, 0] = torch.tensor([10, 11, 12])

    logits = forward.tts_actor_logits(
        model,
        input_ids,
        torch.ones_like(input_ids),
        torch.tensor([[1, 2, 3, 4, 5, 6]]),
        codes,
        torch.tensor([3]),
        torch.tensor([6]),
        torch.zeros((1, 4)),
    )

    assert torch.nonzero(logits.abs().sum(dim=-1)[0], as_tuple=False).reshape(-1).tolist() == [5, 6, 7]
    assert logits[0, 5, 0] == -1e4
    assert logits[0, 5, 1] == 2
    assert logits[0, 5, 2047] == 2048
    assert logits[0, 5, 2048] == -1e4
    assert logits[0, 5, TOKENS.codec_eos] == TOKENS.codec_eos + 1


@pytest.mark.parametrize("response_length", [2, 14, 15, 16, 17, 32])
def test_codec_alignment_recovers_exact_prefix_without_final_residual_row(response_length):
    token_ids = list(range(100, 100 + response_length - 1)) + [2150]
    generated = torch.arange((response_length - 1) * 16, dtype=torch.long).reshape(response_length - 1, 16) + 1
    generated[:, 0] = torch.tensor(token_ids[:-1])
    raw = torch.cat((torch.zeros(12, 16, dtype=torch.long), generated))

    aligned = rollout.align_audio_codes(raw, token_ids)

    assert aligned[:, 0].tolist() == token_ids
    torch.testing.assert_close(aligned[:-1, 1:], generated[:, 1:])
    assert not aligned[-1, 1:].any()


def test_codec_alignment_preserves_final_row_and_rejects_heuristic_match():
    token_ids = [101, 102, 103, 2150]
    generated = torch.arange(4 * 16, dtype=torch.long).reshape(4, 16) + 1
    generated[:, 0] = torch.tensor(token_ids)
    aligned = rollout.align_audio_codes(torch.cat((torch.zeros(12, 16, dtype=torch.long), generated)), token_ids)
    torch.testing.assert_close(aligned, generated)

    malformed = torch.zeros(15, 16, dtype=torch.long)
    malformed[12:, 0] = torch.tensor([101, 999, 103])
    with pytest.raises(RuntimeError, match="Could not exactly align"):
        rollout.align_audio_codes(malformed, token_ids)


def test_codec_alignment_rejects_ambiguous_exact_matches():
    raw = torch.zeros(2, 16, dtype=torch.long)
    raw[:, 0] = 101

    with pytest.raises(RuntimeError, match="Ambiguous Qwen3-TTS codec alignment"):
        rollout.align_audio_codes(raw, [101, 2150])


def test_talker_batch_and_logit_mask_reject_contract_mismatches():
    with pytest.raises(ValueError, match="matching non-empty"):
        forward.build_talker_batch([], [], TOKENS, sub_codebook_vocab=2048)
    with pytest.raises(ValueError, match="codebook_vocab"):
        forward.mask_codec0_logits(torch.zeros((1, 2, 100)), 101, TOKENS.codec_eos)
