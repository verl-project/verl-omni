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
"""CPU tests for the UnifiedReward parser."""

import asyncio

import pytest
import torch

from verl_omni.utils.reward_score import unified_reward
from verl_omni.utils.reward_score.unified_reward import _parse_unified_reward_scores

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
        extra_info={"prompt": "a cat"},
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
