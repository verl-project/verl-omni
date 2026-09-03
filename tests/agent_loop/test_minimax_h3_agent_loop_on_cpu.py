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
"""CPU tests for MiniMax H3 token-id-native prompt preparation."""

import asyncio
from types import SimpleNamespace

import pytest

from verl_omni.agent_loop.single_turn_agent_loop import DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.minimax_h3_diffusion_nft.agent_loop import MiniMaxH3DiffusionSingleTurnAgentLoop
from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    MINIMAX_H3_TOKEN_ID_NATIVE_KEY,
    messages_to_text,
)


def test_messages_to_text_ignores_structured_media_items():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": object()},
                {"type": "video", "video": object()},
                {"type": "text", "text": "A campfire under the stars."},
            ],
        }
    ]

    assert messages_to_text(messages) == "A campfire under the stars."


def test_h3_agent_loop_preserves_reference_media_paths():
    agent = object.__new__(MiniMaxH3DiffusionSingleTurnAgentLoop)
    image = object()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "video", "video": "/tmp/reference.mp4", "start_time_seconds": 1.5},
                {"type": "audio", "audio": "/tmp/reference.wav"},
                {"type": "text", "text": "Animate the references."},
            ],
        }
    ]

    media = asyncio.run(agent.process_multi_modal_info(messages))

    assert media == {
        "images": [image],
        "videos": [{"path": "/tmp/reference.mp4", "start_time_seconds": 1.5}],
        "audios": ["/tmp/reference.wav"],
    }


def test_h3_agent_loop_marks_token_ids_as_native(monkeypatch):
    async def capture_sampling_params(self, sampling_params, **kwargs):
        del self, kwargs
        return sampling_params

    monkeypatch.setattr(DiffusionSingleTurnAgentLoop, "run", capture_sampling_params)
    agent = object.__new__(MiniMaxH3DiffusionSingleTurnAgentLoop)

    result = asyncio.run(agent.run({"temperature": 1.0}, raw_prompt=[]))

    assert result == {"temperature": 1.0, MINIMAX_H3_TOKEN_ID_NATIVE_KEY: True}


def test_h3_agent_loop_inherits_base_initialization_without_tokenization():
    """The inherited base initializer supplies CT state without tokenization."""

    class ExplodingTokenizer:
        def __call__(self, *args, **kwargs):
            raise AssertionError("H3 initialization must not tokenize")

        def apply_chat_template(self, *args, **kwargs):
            raise AssertionError("H3 initialization must not apply a chat template")

    rollout = SimpleNamespace(prompt_length=64)
    trainer_config = SimpleNamespace(config=SimpleNamespace(actor_rollout_ref=SimpleNamespace(rollout=rollout)))
    tokenizer = ExplodingTokenizer()

    agent = MiniMaxH3DiffusionSingleTurnAgentLoop(
        trainer_config,
        server_manager=object(),
        tokenizer=tokenizer,
        processor=None,
        dataset_cls=None,
        data_config=SimpleNamespace(config={}),
    )

    assert agent.tokenizer is tokenizer
    assert agent.rollout_config is rollout
    assert agent.extra_tokenizer_map == {}
    assert agent.continuous_token_builder is not None


@pytest.mark.parametrize("method_name", ["ct_build_initial_tokens", "apply_chat_template"])
def test_h3_agent_loop_tokenizes_raw_text_without_special_tokens(method_name):
    calls = []

    class Tokenizer:
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return {"input_ids": [11, 12, 13]}

    async def run():
        agent = object.__new__(MiniMaxH3DiffusionSingleTurnAgentLoop)
        agent.tokenizer = Tokenizer()
        agent.rollout_config = SimpleNamespace(prompt_length=128)
        agent.loop = asyncio.get_running_loop()
        return await getattr(agent, method_name)(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": object()},
                        {"type": "text", "text": "Raw H3 prompt"},
                    ],
                }
            ],
            images=[object()],
        )

    assert asyncio.run(run()) == [11, 12, 13]
    assert calls == [
        (
            "Raw H3 prompt",
            {
                "padding": False,
                "truncation": True,
                "max_length": 128,
                "add_special_tokens": False,
            },
        )
    ]


@pytest.mark.parametrize("method_name", ["ct_build_initial_tokens", "apply_chat_template"])
def test_h3_agent_loop_rejects_empty_text(method_name):
    async def run():
        agent = object.__new__(MiniMaxH3DiffusionSingleTurnAgentLoop)
        agent.rollout_config = SimpleNamespace(prompt_length=128)
        return await getattr(agent, method_name)([{"role": "user", "content": [{"type": "image"}]}])

    with pytest.raises(ValueError, match="non-empty text prompt"):
        asyncio.run(run())
