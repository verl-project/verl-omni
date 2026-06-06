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
"""CPU tests for single-turn diffusion agent helpers."""

import os

os.environ["VERL_OMNI_LIGHTWEIGHT_IMPORT"] = "1"

import pytest

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop, _messages_to_text


def test_messages_to_text_prefers_user_content_over_system_prompt():
    messages = [
        {"role": "system", "content": "You are an image generator."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a red cabin"},
                {"type": "image", "image": "ignored"},
                {"type": "text", "text": "in fresh snow"},
            ],
        },
    ]

    assert _messages_to_text(messages) == "a red cabin\nin fresh snow"


def test_messages_to_text_falls_back_to_all_text_when_no_user_role():
    assert _messages_to_text([{"role": "assistant", "content": "fallback text"}]) == "fallback text"


def test_messages_to_text_accepts_plain_string():
    assert _messages_to_text("plain prompt") == "plain prompt"


@pytest.mark.asyncio
async def test_encode_prompt_ids_falls_back_to_plain_tokenizer_without_chat_template():
    class _Tokenizer:
        chat_template = None

        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is True
            assert text == "a red cabin"
            return [101, 102]

    loop = object.__new__(DiffusionSingleTurnAgentLoop)
    loop.tokenizer = _Tokenizer()

    prompt_ids = await loop._encode_prompt_ids([{"role": "user", "content": "a red cabin"}])

    assert prompt_ids == [101, 102]
