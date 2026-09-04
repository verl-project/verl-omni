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
"""CPU contracts for the external audio reward client."""

import asyncio
import base64
import importlib.util
import socket
import threading
from pathlib import Path

import numpy as np
import pytest
from aiohttp import web


def _load_client_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/audio_http_scorer_client.py"
    spec = importlib.util.spec_from_file_location("audio_http_scorer_client_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


client = _load_client_module()


def test_request_serialization_preserves_waveform_prompt_and_scalar_metadata():
    waveform = np.array([0.0, -0.25, 0.5], dtype=np.float32)
    payload = client._serialize_request(
        (waveform, 24_000),
        "target text",
        {"id": "sample-1", "seed": np.array(7), "audio": waveform, "ignored": [1, 2]},
    )

    decoded = np.frombuffer(base64.b64decode(payload["waveform_f32_base64"]), dtype="<f4")
    np.testing.assert_array_equal(decoded, waveform)
    assert payload == {
        "protocol_version": "1",
        "waveform_f32_base64": payload["waveform_f32_base64"],
        "num_samples": 3,
        "sample_rate": 24_000,
        "prompt": "target text",
        "metadata": {"id": "sample-1", "seed": 7},
    }


@pytest.mark.parametrize(
    ("solution_audio", "message"),
    [
        (None, "expects solution_audio"),
        (([], 24_000), "empty waveform"),
        (([float("inf")], 24_000), "non-finite waveform"),
        (([0.0], 0), "positive integer"),
    ],
)
def test_invalid_request_is_rejected(solution_audio, message):
    with pytest.raises((TypeError, ValueError), match=message):
        client._serialize_request(solution_audio, "text", {})


async def _run_retry_e2e():
    state = {"attempts": 0, "always_fail": False, "invalid_body_once": False}

    async def score(request):
        payload = await request.json()
        assert payload["prompt"] == "target text"
        assert payload["sample_rate"] == 24_000
        state["attempts"] += 1
        if state["invalid_body_once"] and state["attempts"] == 1:
            return web.Response(body=b"\xff", status=503, content_type="text/plain")
        if state["always_fail"] or (not state["invalid_body_once"] and state["attempts"] <= 2):
            return web.json_response({"error": "temporary"}, status=503)
        return web.json_response({"score": 3.5, "raw_score": 3.5})

    app = web.Application()
    app.router.add_post("/score", score)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    site = web.SockSite(runner, sock)
    await site.start()
    server_url = f"http://127.0.0.1:{sock.getsockname()[1]}/score"
    kwargs = {
        "solution_audio": (np.zeros(32, dtype=np.float32), 24_000),
        "ground_truth": "target text",
        "data_source": "tts_reward",
        "server_url": server_url,
        "max_retries": 2,
        "retry_backoff": 0,
    }
    try:
        result = await client.compute_score(**kwargs)
        assert result == {"score": 3.5, "raw_score": 3.5}
        assert state["attempts"] == 3

        state.update(attempts=0, invalid_body_once=True)
        result = await client.compute_score(**kwargs)
        assert result == {"score": 3.5, "raw_score": 3.5}
        assert state["attempts"] == 2

        state.update(attempts=0, always_fail=True, invalid_body_once=False)
        with pytest.raises(RuntimeError, match="failed after 3 attempts"):
            await client.compute_score(**kwargs)
        assert state["attempts"] == 3
    finally:
        session = getattr(client.compute_score, "_session", None)
        if session is not None and not session.closed:
            await session.close()
        for name in ("_session", "_session_loop"):
            if hasattr(client.compute_score, name):
                delattr(client.compute_score, name)
        await runner.cleanup()


def test_retryable_http_errors_are_bounded():
    asyncio.run(_run_retry_e2e())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({"error": "model failed"}, "model failed"),
        ({}, "missing 'score'"),
        ({"score": True}, "invalid score"),
        ({"score": "1.25"}, "invalid score"),
        ({"score": "not-a-number"}, "invalid score"),
        ({"score": float("nan")}, "non-finite score"),
        ({"score": 1.0, "metric": float("inf")}, "finite JSON scalar"),
        ({"score": 1.0, "metric": [1.0]}, "finite JSON scalar"),
    ],
)
def test_response_validation_fails_closed(payload, message):
    with pytest.raises(RuntimeError, match=message):
        client._validate_response(payload)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout": float("nan")}, "finite number"),
        ({"max_retries": 1.5}, "integer"),
        ({"retry_backoff": float("inf")}, "finite number"),
    ],
)
def test_invalid_retry_configuration_is_rejected(kwargs, message):
    call = client.compute_score(
        solution_audio=(np.zeros(1, dtype=np.float32), 24_000),
        ground_truth="text",
        server_url="http://127.0.0.1:1/score",
        **kwargs,
    )

    with pytest.raises(ValueError, match=message):
        asyncio.run(call)


def test_unknown_reward_configuration_is_not_silently_swallowed():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        client.compute_score(
            solution_audio=(np.zeros(1, dtype=np.float32), 24_000),
            ground_truth="text",
            server_url="http://127.0.0.1:1/score",
            timeout_s=1.0,
        )


def test_open_session_from_another_event_loop_fails_closed():
    class Session:
        closed = False

    client.compute_score._session = Session()
    client.compute_score._session_loop = object()
    try:

        async def get_session():
            with pytest.raises(RuntimeError, match="cannot be shared across event loops"):
                await client._session()

        asyncio.run(get_session())
    finally:
        del client.compute_score._session
        del client.compute_score._session_loop


def test_compute_score_serializes_waveform_off_the_event_loop(monkeypatch):
    event_loop_thread = threading.get_ident()
    serialization_threads = []
    original_serialize_request = client._serialize_request

    def serialize_request(*args, **kwargs):
        serialization_threads.append(threading.get_ident())
        return original_serialize_request(*args, **kwargs)

    async def request_score(*args, **kwargs):
        return {"score": 1.0}

    monkeypatch.setattr(client, "_serialize_request", serialize_request)
    monkeypatch.setattr(client, "_request_score", request_score)
    result = asyncio.run(
        client.compute_score(
            solution_audio=(np.zeros(32, dtype=np.float32), 24_000),
            ground_truth="text",
            server_url="http://127.0.0.1:1/score",
        )
    )

    assert result == {"score": 1.0}
    assert serialization_threads and serialization_threads[0] != event_loop_thread
