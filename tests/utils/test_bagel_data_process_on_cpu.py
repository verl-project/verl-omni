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

import pytest

from examples.flowgrpo_trainer.data_process import bagel_ocr, bagel_pickscore


class Tokenizer:
    def convert_tokens_to_ids(self, name):
        return {"<|im_start|>": 1, "<|im_end|>": 2}[name]

    def encode(self, text, add_special_tokens=False):
        return [ord(char) + 3 for char in text]


@pytest.fixture(
    params=[bagel_ocr.tokenize_bagel_prompt, bagel_pickscore.tokenize_bagel_prompt], ids=["ocr", "pickscore"]
)
def tokenize_prompt(request):
    return request.param


@pytest.mark.parametrize("content_length", [0, 16, 254, 255, 256, 1024])
def test_prompt_truncation_preserves_framing_and_content_prefix(tokenize_prompt, content_length):
    tokenizer = Tokenizer()
    caption = ("abcdefghijklmnopqrstuvwxyz" * 40)[:content_length]
    tokens = tokenize_prompt(tokenizer, caption, max_length=256)

    assert tokens[0] == 1
    assert tokens[-1] == 2
    assert tokens[1:-1] == tokenizer.encode(caption)[:254]
    assert len(tokens) == min(content_length + 2, 256)


def test_minimum_prompt_limit_keeps_both_markers(tokenize_prompt):
    assert tokenize_prompt(Tokenizer(), "a cat", max_length=2) == [1, 2]


@pytest.mark.parametrize("max_length", [-1, 0, 1])
def test_prompt_limit_must_fit_both_markers(tokenize_prompt, max_length):
    with pytest.raises(ValueError, match="start and end markers"):
        tokenize_prompt(Tokenizer(), "a cat", max_length=max_length)
