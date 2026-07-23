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
"""CPU tests for the latent DRM safetensors HTTP client."""

import asyncio
import importlib.util
from pathlib import Path

import aiohttp
import pytest
import torch
from safetensors.torch import load as load_tensors


def _load_client_module():
    module_path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/latent_http_scorer_client.py"
    spec = importlib.util.spec_from_file_location("latent_http_scorer_client_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load_client_module()


def _reward_inputs():
    return {
        "solution_image": torch.zeros(16, 2, 2),
        "ground_truth": "",
        "extra_info": {
            "prompt_embeds": torch.arange(32, dtype=torch.float32).reshape(4, 8),
            "pooled_prompt_embeds": torch.arange(8, dtype=torch.float32),
        },
        "server_url": "http://drm.test/v1/score",
    }


def test_serialize_request_uses_solution_latent_and_protocol_fields():
    inputs = _reward_inputs()
    payload = client._serialize_request(
        inputs["solution_image"],
        inputs["extra_info"],
        noise_level=0.4,
        noise_seed=7,
    )
    tensors = load_tensors(payload)

    assert set(tensors) == {"latents", "prompt_embeds", "pooled_prompt_embeds", "u", "seeds"}
    assert tensors["latents"].shape == (1, 16, 2, 2)
    assert tensors["prompt_embeds"].shape == (1, 4, 8)
    assert tensors["pooled_prompt_embeds"].shape == (1, 8)
    torch.testing.assert_close(tensors["u"], torch.tensor([0.4]))
    torch.testing.assert_close(tensors["seeds"], torch.tensor([7]))


def test_serialize_request_prefers_explicit_clean_latent():
    inputs = _reward_inputs()
    explicit_latent = torch.ones(16, 2, 2)
    inputs["extra_info"]["latents_clean"] = explicit_latent

    tensors = load_tensors(
        client._serialize_request(
            inputs["solution_image"],
            inputs["extra_info"],
            noise_level=0.0,
            noise_seed=None,
        )
    )

    torch.testing.assert_close(tensors["latents"], explicit_latent.unsqueeze(0))
    assert "seeds" not in tensors


def test_batched_tensor_rejects_invalid_rank_and_batch_size():
    with pytest.raises(ValueError, match="rank"):
        client._batched_tensor(torch.zeros(2, 2), "latents", expected_rank=4)
    with pytest.raises(ValueError, match="batch size 2"):
        client._batched_tensor(torch.zeros(2, 16, 2, 2), "latents", expected_rank=4)


class _FakeResponse:
    def __init__(self, result, status=200, detail="error"):
        self.result = result
        self.status = status
        self.detail = detail

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self):
        return self.result

    async def text(self):
        return self.detail


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.response


def test_request_score_sends_protocol_header_and_parses_score(monkeypatch):
    response = _FakeResponse({"protocol_version": client.PROTOCOL_VERSION, "raw_scores": [2.5]})
    session = _FakeSession(response)

    async def fake_session():
        return session

    monkeypatch.setattr(client, "_session", fake_session)
    score = asyncio.run(client._request_score("http://drm.test/v1/score", b"payload", timeout=3.0))

    assert score == pytest.approx(2.5)
    url, request = session.requests[0]
    assert url == "http://drm.test/v1/score"
    assert request["data"] == b"payload"
    assert request["headers"][client.PROTOCOL_HEADER] == client.PROTOCOL_VERSION


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ({"protocol_version": "wrong", "raw_scores": [1.0]}, "protocol mismatch"),
        ({"protocol_version": client.PROTOCOL_VERSION, "raw_scores": []}, "one raw score"),
        ({"protocol_version": client.PROTOCOL_VERSION, "raw_scores": [float("nan")]}, "non-finite"),
    ],
)
def test_request_score_rejects_invalid_responses(monkeypatch, result, message):
    async def fake_session():
        return _FakeSession(_FakeResponse(result))

    monkeypatch.setattr(client, "_session", fake_session)
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(client._request_score("http://drm.test/v1/score", b"payload", timeout=3.0))


def test_compute_score_retries_and_applies_scale_and_bias(monkeypatch):
    attempts = 0

    async def fake_request_score(server_url, payload, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise aiohttp.ClientConnectionError("temporary failure")
        return 3.0

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(client, "_request_score", fake_request_score)
    monkeypatch.setattr(client.asyncio, "sleep", no_sleep)

    result = asyncio.run(
        client.compute_score(
            **_reward_inputs(),
            score_scale=0.1,
            score_bias=1.0,
            max_retries=2,
        )
    )

    assert attempts == 3
    assert result == {"score": pytest.approx(1.3), "drm_raw_score": pytest.approx(3.0)}
