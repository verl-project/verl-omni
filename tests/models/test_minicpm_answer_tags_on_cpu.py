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
"""CPU tests: MiniCPM answer-tag specialness vs verl's reward decode.

MiniCPM-o 4.5 registers <answer>/</answer> as special tokens while
<think>/</think> are plain; verl's reward manager decodes responses with
skip_special_tokens=True, which strips exactly the answer tags and zeros
choice_reward. keep_answer_tags_when_decoding demotes the two tokens in the
fast backend so the tags survive the scored string.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from verl_omni.models.transformers.minicpm_o import keep_answer_tags_when_decoding
from verl_omni.utils.reward_score.choice_reward import compute_score


def _minicpm_style_tokenizer():
    """Scratch tokenizer mirroring MiniCPM-o 4.5's tag specialness."""
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "A": 4, "B": 5, "[unk]": 3}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf.add_tokens(["<think>", "</think>"], special_tokens=False)
    hf.add_tokens(["<answer>", "</answer>"], special_tokens=True)  # the checkpoint's asymmetry
    return hf


def _response_ids(tokenizer):
    return tokenizer.encode("<think>compare a and b</think><answer>A</answer>", add_special_tokens=False)


def test_naive_decode_strips_answer_tags_and_zeroes_reward():
    # This is the bug: exactly the mechanism verl's naive reward manager hits.
    tokenizer = _minicpm_style_tokenizer()
    assert tokenizer.added_tokens_decoder[tokenizer.convert_tokens_to_ids("<answer>")].special is True

    response_str = tokenizer.decode(_response_ids(tokenizer), skip_special_tokens=True)
    assert "<think>" in response_str  # think tags are NOT special — they survive
    assert "<answer>" not in response_str  # answer tags are special — stripped

    assert compute_score(data_source="avqa_r1_6k", solution_str=response_str, ground_truth="<answer>A</answer>") == {
        "score": 0.0,
        "accuracy": 0.0,
    }


def test_keep_answer_tags_when_decoding_restores_reward():
    tokenizer = _minicpm_style_tokenizer()
    answer_id = tokenizer.convert_tokens_to_ids("<answer>")
    think_id = tokenizer.convert_tokens_to_ids("<think>")

    assert keep_answer_tags_when_decoding(tokenizer) is True

    # Flags demoted; ids, atomic encoding, and other specials untouched.
    assert tokenizer.added_tokens_decoder[answer_id].special is False
    assert tokenizer.convert_tokens_to_ids("<answer>") == answer_id
    assert tokenizer.convert_tokens_to_ids("<think>") == think_id
    b_id = tokenizer.convert_tokens_to_ids("B")
    close_id = tokenizer.convert_tokens_to_ids("</answer>")
    assert tokenizer.encode("<answer>B</answer>", add_special_tokens=False) == [answer_id, b_id, close_id]

    # verl's exact decode path now keeps the tags, and the reward scores.
    response_str = tokenizer.decode(_response_ids(tokenizer), skip_special_tokens=True)
    assert "<answer>" in response_str and "<" + "/answer>" in response_str and "A" in response_str
    assert compute_score(data_source="avqa_r1_6k", solution_str=response_str, ground_truth="<answer>A</answer>") == {
        "score": 1.0,
        "accuracy": 1.0,
    }


def test_keep_answer_tags_is_noop_when_already_plain():
    tokenizer = _minicpm_style_tokenizer()
    keep_answer_tags_when_decoding(tokenizer)
    assert keep_answer_tags_when_decoding(tokenizer) is False  # second call: nothing to demote

    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "[unk]": 1}, unk_token="[unk]"))
    plain = PreTrainedTokenizerFast(tokenizer_object=tok)
    plain.add_tokens(["<answer>", "</answer>"], special_tokens=False)
    assert keep_answer_tags_when_decoding(plain) is False


def test_configure_tokenizer_applies_demotion(monkeypatch):
    from transformers import AutoTokenizer

    from verl_omni.pipelines.minicpm.thinker_training_adapter import MiniCPMThinkerAdapter

    stub = _minicpm_style_tokenizer()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: stub)
    model_config = type("Cfg", (), {"trust_remote_code": True})()
    returned = MiniCPMThinkerAdapter.configure_tokenizer("/fake/minicpm", model_config)
    assert returned is stub
    assert stub.added_tokens_decoder[stub.convert_tokens_to_ids("<answer>")].special is False


def test_keep_answer_tags_raises_when_unfixable():
    class _NoBackend:
        added_tokens_decoder = {}

        def convert_tokens_to_ids(self, token):
            return 0

    # No tag is special -> no-op path (no backend needed).
    assert keep_answer_tags_when_decoding(_NoBackend()) is False

    class _SpecialNoBackend(_NoBackend):
        pass

    # Make the decoder report a special tag while no backend exists.
    class _Entry:
        special = True

    _SpecialNoBackend.added_tokens_decoder = {0: _Entry()}
    with pytest.raises(RuntimeError, match="no fast backend"):
        keep_answer_tags_when_decoding(_SpecialNoBackend())
