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
"""Output budget of the GenRM OCR judge: scorer default, environment and explicit overrides."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.utils.reward_score import genrm_ocr


@pytest.fixture
def captured_request(monkeypatch):
    requests = []

    async def fake_chat_complete(router_address, chat_complete_request):
        requests.append(chat_complete_request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

    monkeypatch.setattr(genrm_ocr, "_chat_complete", fake_chat_complete)
    return requests


class _WordTokenizer:
    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return text.split()


async def _score(ground_truth="hello", **kwargs):
    image = torch.zeros((3, 8, 8), dtype=torch.uint8)
    return await genrm_ocr.compute_score_ocr(
        data_source="ocr",
        solution_image=image,
        ground_truth=ground_truth,
        extra_info={"media_kind": "image"},
        reward_router_address="127.0.0.1:0",
        model_name="judge",
        **kwargs,
    )


def test_default_budget_is_a_short_transcription(monkeypatch):
    monkeypatch.delenv("GENRM_OCR_MAX_TOKENS", raising=False)
    assert genrm_ocr._sampling_params()["max_tokens"] == 256


def test_environment_overrides_the_default_budget(monkeypatch):
    monkeypatch.setenv("GENRM_OCR_MAX_TOKENS", "64")
    assert genrm_ocr._sampling_params()["max_tokens"] == 64


@pytest.mark.asyncio
async def test_explicit_sampling_params_override_the_budget(monkeypatch, captured_request):
    monkeypatch.setenv("GENRM_OCR_MAX_TOKENS", "64")
    result = await _score(sampling_params={"max_tokens": 32})
    assert result["score"] == 1.0
    assert captured_request[0]["max_tokens"] == 32
    assert captured_request[0]["temperature"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_request_uses_the_default_budget_without_overrides(monkeypatch, captured_request):
    monkeypatch.delenv("GENRM_OCR_MAX_TOKENS", raising=False)
    await _score()
    assert captured_request[0]["max_tokens"] == 256


@pytest.mark.asyncio
async def test_ground_truth_over_the_budget_is_rejected_before_any_request(captured_request):
    with pytest.raises(ValueError, match="3 tokens.*max_tokens=2"):
        await _score("one two three", sampling_params={"max_tokens": 2}, reward_model_tokenizer=_WordTokenizer())
    assert captured_request == []


@pytest.mark.asyncio
async def test_ground_truth_at_the_budget_is_scored(captured_request):
    await _score("one two three", sampling_params={"max_tokens": 3}, reward_model_tokenizer=_WordTokenizer())
    assert len(captured_request) == 1


@pytest.mark.asyncio
async def test_ground_truth_is_checked_against_the_default_budget(monkeypatch, captured_request):
    monkeypatch.delenv("GENRM_OCR_MAX_TOKENS", raising=False)
    with pytest.raises(ValueError, match="257 tokens.*max_tokens=256"):
        await _score(" ".join(["word"] * 257), reward_model_tokenizer=_WordTokenizer())
    assert captured_request == []
