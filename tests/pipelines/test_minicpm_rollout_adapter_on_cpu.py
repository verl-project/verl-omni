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
"""CPU tests for the MiniCPM rollout adapter's sampling-ban hook.

MiniCPM-o 4.5 packs the Talker/Code2Wav vocabularies as added tokens above
the chat-control anchors (everything after ``<|im_end|>``). A thinker-only
policy occasionally samples them mid-response and corrupts the RL training
signal, so ``policy_logit_bias`` bans every added id above the anchor except
the two answer tags the choice reward decodes.
"""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from verl_omni.pipelines.minicpm.omni_rollout_adapter import MiniCPMORolloutAdapter


def _minicpm_style_tokenizer():
    """Scratch tokenizer mirroring MiniCPM-o 4.5's token-block layout."""
    # No vocab holes: a gap desyncs added-token id assignment.
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "d": 3, "[unk]": 4}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf.add_tokens(["<|im_start|>", "<|im_end|>"], special_tokens=True)  # chat-control anchors
    hf.add_tokens(["<answer>", "</answer>"], special_tokens=True)
    hf.add_tokens(["<|tts_eos|>", "<|tts_pad|>", "<|code_0|>"], special_tokens=True)  # talker/codec block
    return hf


def test_policy_logit_bias_blocks_talkers_keeps_answer_tags():
    tokenizer = _minicpm_style_tokenizer()
    anchor = tokenizer.convert_tokens_to_ids("<|im_end|>")

    bias = MiniCPMORolloutAdapter.policy_logit_bias(tokenizer)

    assert bias is not None
    talker_ids = {tokenizer.convert_tokens_to_ids(token) for token in ("<|tts_eos|>", "<|tts_pad|>", "<|code_0|>")}
    assert set(bias) == talker_ids
    assert all(value == float("-inf") for value in bias.values())
    # The anchor itself and everything below it stay samplable.
    assert anchor not in bias
    assert tokenizer.convert_tokens_to_ids("<|im_start|>") not in bias
    assert tokenizer.convert_tokens_to_ids("a") not in bias
    # The reward-decoded answer tags stay samplable.
    assert tokenizer.convert_tokens_to_ids("<answer>") not in bias
    assert tokenizer.convert_tokens_to_ids("</answer>") not in bias


def test_policy_logit_bias_fails_closed_without_the_anchor():
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "[unk]": 1}, unk_token="[unk]"))
    plain = PreTrainedTokenizerFast(tokenizer_object=tok)
    plain.add_tokens(["<|im_start|>"], special_tokens=True)

    with pytest.raises(ValueError, match="anchor"):
        MiniCPMORolloutAdapter.policy_logit_bias(plain)
