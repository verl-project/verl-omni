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
"""CPU tests for the generic HTTP reward client."""

import asyncio
import importlib.util
import pickle
from pathlib import Path

import pytest
import torch


def _load_client_module():
    module_path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/http_scorer_client.py"
    spec = importlib.util.spec_from_file_location("http_scorer_client_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load_client_module()


class _FakeResponse:
    def __init__(self, result, status=200, detail="error"):
        self.result = result
        self.status = status
        self.detail = detail

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def read(self):
        return pickle.dumps(self.result)

    async def text(self):
        return self.detail


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.closed = False

    def post(self, url, **kwargs):
        return self.response


def _compute_with_response(monkeypatch, response):
    session = _FakeSession(response)
    monkeypatch.setattr(client.compute_score, "_session", session, raising=False)
    monkeypatch.setattr(client, "_prepare_image_bytes", lambda image: b"jpeg")
    return asyncio.run(
        client.compute_score(
            solution_image=torch.zeros(3, 2, 2),
            ground_truth="prompt",
            server_url="http://scorer.test/v1/score",
        )
    )


def test_compute_score_accepts_a_real_zero_reward(monkeypatch):
    result = _compute_with_response(monkeypatch, _FakeResponse({"scores": [0.0]}))

    assert result == {"score": 0.0}


def test_compute_score_rejects_http_errors(monkeypatch):
    with pytest.raises(RuntimeError, match="HTTP 500: unavailable"):
        _compute_with_response(monkeypatch, _FakeResponse({}, status=500, detail="unavailable"))


def test_compute_score_rejects_service_errors(monkeypatch):
    with pytest.raises(RuntimeError, match="server error: model failed"):
        _compute_with_response(monkeypatch, _FakeResponse({"error": "model failed"}))


def test_compute_score_rejects_empty_scores(monkeypatch):
    with pytest.raises(RuntimeError, match="returned no scores"):
        _compute_with_response(monkeypatch, _FakeResponse({"scores": []}))
