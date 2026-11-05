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

import asyncio
from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.reward_score import unified_reward
from verl_omni.utils.reward_score.unified_reward import (
    _build_unified_reward_prompt,
    _parse_unified_reward_scores,
    compute_score_unified_reward,
)

TEST_PROMPT_BUILDER = "pkg://tests.utils.reward_score.test_unified_reward_on_cpu.build_test_prompt"
MODEL_OUTPUT = "Alignment Score (1-5): 4\nCoherence Score (1-5): 5\nStyle Score (1-5): 3"

_IN_RANGE_RESPONSE = "Alignment Score (1-5): 4\nCoherence Score (1-5): 5\nStyle Score (1-5): 3"
_OUT_OF_RANGE_RESPONSE = (
    "Alignment Score (1-5): 0.34040001034736633\n"
    "Coherence Score (1-5): 2.91129994392395\n"
    "Style Score (1-5): 2.4103000164031982"
)
_RESULT_KEYS = {"score", "raw_score", "response", "malformed_frames"}


def test_unified_reward_requires_all_labeled_axes_inside_one_to_five():
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5") == {}
    assert _parse_unified_reward_scores(_OUT_OF_RANGE_RESPONSE) == {}
    assert _parse_unified_reward_scores("Alignment Score: 4\nCoherence Score: 5\nStyle Score: 6") == {}
    assert _parse_unified_reward_scores(_IN_RANGE_RESPONSE) == {
        "alignment": 4.0,
        "coherence": 5.0,
        "style": 3.0,
    }


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


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Completion:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


async def _score_with_response(monkeypatch, content: str) -> dict:
    async def fake_chat(**kwargs):
        del kwargs
        return _Completion(content)

    monkeypatch.setattr(unified_reward, "_chat_complete", fake_chat)
    image = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
    return await unified_reward.compute_score_unified_reward(
        data_source="test",
        solution_image=image,
        ground_truth="a cat",
        extra_info={"media_kind": "image", "prompt": "a cat"},
        reward_router_address="127.0.0.1:1",
    )


def test_in_range_scores_normalize_and_keep_the_reward_schema(monkeypatch):
    result = asyncio.run(_score_with_response(monkeypatch, _IN_RANGE_RESPONSE))
    raw_score = (4.0 + 5.0 + 3.0) / 3.0
    assert set(result) == _RESULT_KEYS
    assert result["malformed_frames"] == 0
    assert result["response"] == _IN_RANGE_RESPONSE
    assert result["raw_score"] == pytest.approx(raw_score)
    assert result["score"] == pytest.approx((raw_score - 1.0) / 4.0)


def test_out_of_range_alignment_returns_zero_with_the_same_keys(monkeypatch):
    result = asyncio.run(_score_with_response(monkeypatch, _OUT_OF_RANGE_RESPONSE))
    assert set(result) == _RESULT_KEYS
    assert result["score"] == 0.0
    assert result["raw_score"] == 0.0
    assert result["response"] == ""
    assert result["malformed_frames"] == 1


def test_unparseable_response_returns_zero_with_the_same_keys(monkeypatch):
    result = asyncio.run(_score_with_response(monkeypatch, "sorry, no scores"))
    assert set(result) == _RESULT_KEYS
    assert result["score"] == 0.0
    assert result["raw_score"] == 0.0
    assert result["response"] == ""
    assert result["malformed_frames"] == 1
