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
"""CPU tests for the UnifiedReward parser and prompt selection."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.reward_score.unified_reward import (
    _build_unified_reward_prompt,
    _parse_unified_reward_scores,
    compute_score_unified_reward,
)

TEST_PROMPT_BUILDER = "pkg://tests.utils.reward_score.test_unified_reward_on_cpu.build_test_prompt"
MODEL_OUTPUT = "Alignment Score (1-5): 4\nCoherence Score (1-5): 5\nStyle Score (1-5): 3"


def build_test_prompt(caption: str) -> str:
    """Return a prompt loaded through pkg:// for ``caption``."""
    return f"Judge the image against: {caption}"


def _capture_chat_requests(monkeypatch):
    from verl_omni.utils.reward_score import unified_reward

    requests = []

    async def chat_complete(router_address, chat_complete_request, session=None):
        requests.append(chat_complete_request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=MODEL_OUTPUT))])

    monkeypatch.setattr(unified_reward, "_chat_complete", chat_complete)
    return requests


async def _score(*, user_prompt: str | None = None):
    image = torch.zeros(3, 8, 8, dtype=torch.uint8)
    kwargs = {} if user_prompt is None else {"user_prompt": user_prompt}
    return await compute_score_unified_reward(
        data_source="test",
        solution_image=image,
        ground_truth="a cat",
        extra_info={"media_kind": "image"},
        reward_router_address="judge",
        **kwargs,
    )


def test_unified_reward_requires_all_labeled_axes():
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5") == {}
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5\nStyle Score: 3") == {
        "alignment": 4.0,
        "coherence": 5.0,
        "style": 3.0,
    }


@pytest.mark.asyncio
async def test_default_prompt_matches_existing_request(monkeypatch):
    requests = _capture_chat_requests(monkeypatch)

    await _score()

    assert [message["role"] for message in requests[0]["messages"]] == ["user"]
    content = requests[0]["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": _build_unified_reward_prompt("a cat")}


@pytest.mark.asyncio
async def test_literal_user_prompt_is_used_verbatim(monkeypatch):
    requests = _capture_chat_requests(monkeypatch)

    await _score(user_prompt="Rate the alignment only.")

    assert requests[0]["messages"][0]["content"][1]["text"] == "Rate the alignment only."


@pytest.mark.asyncio
async def test_literal_user_prompt_replaces_caption_placeholder(monkeypatch):
    requests = _capture_chat_requests(monkeypatch)

    await _score(user_prompt="Score {caption} against the image. Caption: {caption}")

    assert requests[0]["messages"][0]["content"][1]["text"] == ("Score a cat against the image. Caption: a cat")


@pytest.mark.asyncio
async def test_pkg_user_prompt_builder_receives_caption(monkeypatch):
    requests = _capture_chat_requests(monkeypatch)

    await _score(user_prompt=TEST_PROMPT_BUILDER)

    assert requests[0]["messages"][0]["content"][1]["text"] == "Judge the image against: a cat"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "spec",
    [
        "pkg://missing_dot",
        "pkg://tests.utils.reward_score.test_unified_reward_on_cpu.missing_builder",
    ],
)
async def test_invalid_pkg_user_prompt_spec_raises(monkeypatch, spec):
    requests = _capture_chat_requests(monkeypatch)

    with pytest.raises(ValueError, match="builder"):
        await _score(user_prompt=spec)

    assert requests == []
