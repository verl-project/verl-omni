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
"""CPU tests for the MiniCPM answer-tag reward-decode patch.

The MiniCPM package wraps the stock ``NaiveRewardManager`` once at import
to demote the checkpoint's special ``<answer>`` tags on the tokenizer the
reward loop hands it — verl's reward loop builds its own tokenizer and
decodes with ``skip_special_tokens=True``, which strips the tags and
zeroes every choice-reward score. The wrap is a no-op for tokenizers
without special answer tags, so non-MiniCPM runs keep stock behavior.
"""

from __future__ import annotations

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

import verl_omni  # noqa: F401  # installs the patch via pipelines.minicpm
from verl_omni.utils.reward_score.choice_reward import compute_score


def _minicpm_style_tokenizer():
    """Scratch tokenizer mirroring MiniCPM-o 4.5's tag specialness."""
    # No vocab holes: a gap desyncs added-token id assignment.
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "b": 1, "c": 2, "A": 4, "B": 5, "[unk]": 3}, unk_token="[unk]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tok)
    hf.add_tokens(["<answer>", "</answer>"], special_tokens=True)  # the checkpoint's asymmetry
    return hf


def test_patch_installed_at_package_import_and_idempotent():
    from verl_omni.pipelines.minicpm.reward_decode import install_reward_decode_patch

    assert getattr(NaiveRewardManager, "_minicpm_keeps_answer_tags", False)
    before = NaiveRewardManager.__init__
    install_reward_decode_patch()
    assert NaiveRewardManager.__init__ is before  # never double-wrapped


def test_stock_naive_manager_keeps_answer_tags():
    tokenizer = _minicpm_style_tokenizer()
    ids = tokenizer.encode("a b c <answer>A</answer>", add_special_tokens=False)
    assert "<answer>" not in tokenizer.decode(ids, skip_special_tokens=True)  # the bug

    manager = NaiveRewardManager(config={}, tokenizer=tokenizer, compute_score=compute_score)

    assert manager.tokenizer is tokenizer
    assert tokenizer.added_tokens_decoder[tokenizer.convert_tokens_to_ids("<answer>")].special is False
    response_str = tokenizer.decode(ids, skip_special_tokens=True)  # verl's decode path
    assert "<answer>" in response_str and "<" + "/answer>" in response_str
    assert compute_score(data_source="avqa_r1_6k", solution_str=response_str, ground_truth="<answer>A</answer>") == {
        "score": 1.0,
        "accuracy": 1.0,
    }


def test_stock_naive_manager_untouched_for_tagless_tokenizers():
    tok = Tokenizer(models.WordLevel(vocab={"a": 0, "[unk]": 1}, unk_token="[unk]"))
    plain = PreTrainedTokenizerFast(tokenizer_object=tok)
    plain.add_tokens(["<answer>", "</answer>"], special_tokens=False)

    manager = NaiveRewardManager(config={}, tokenizer=plain, compute_score=compute_score)

    assert manager.tokenizer is plain  # constructed unchanged, no raise
