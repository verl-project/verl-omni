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

from examples.flowgrpo_trainer.data_process.bagel_dvreward import prepare_record


class Tokenizer:
    def convert_tokens_to_ids(self, name):
        return {"<|im_start|>": 1, "<|im_end|>": 2}[name]

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(char) + 3 for char in text]


@pytest.mark.parametrize("dictionary", [False, True])
def test_questions_caption_and_exact_token_limit(dictionary):
    questions = [{"question": "Is it red?", "is_valid": False}] if dictionary else ["Is it red?"]
    record = {"prompt": " red cat ", "semantic_questions": questions, "quality_questions": ["Is it clear?"]}
    result = prepare_record(record, Tokenizer(), 9, "train", 7)
    assert result["prompt"] == [{"role": "user", "content": "red cat"}]
    assert result["reward_model"]["ground_truth"] == "red cat"
    assert result["prompt_token_ids"] == [1, *Tokenizer().encode("red cat"), 2]
    assert result["extra_info"] == {
        "split": "train",
        "index": 7,
        "semantic_questions": ["Is it red?"],
        "quality_questions": ["Is it clear?"],
    }
    assert record["semantic_questions"] == questions
    assert record["prompt"] == " red cat "
    with pytest.raises(ValueError, match="exceeds max_prompt_length"):
        prepare_record(record, Tokenizer(), 8, "train", 7)


@pytest.mark.parametrize("group", ["semantic_questions", "quality_questions"])
def test_empty_question_group_is_rejected(group):
    record = {"prompt": "cat", "semantic_questions": ["Cat?"], "quality_questions": ["Clear?"], group: []}
    with pytest.raises(ValueError, match="both semantic and quality"):
        prepare_record(record, Tokenizer(), 5, "test", 0)
