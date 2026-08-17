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

"""End-to-end CPU test for HTTP scorer retries."""

import asyncio
import importlib.util
import pickle
import socket
from pathlib import Path

import pytest
import torch
from aiohttp import web


def _load_client_module():
    module_path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/http_scorer_client.py"
    spec = importlib.util.spec_from_file_location("http_scorer_client_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = _load_client_module()


async def _run_retry_e2e():
    if hasattr(client.compute_score, "_session"):
        del client.compute_score._session

    state = {"attempts": 0, "always_fail": False}

    async def score(request):
        payload = pickle.loads(await request.read())
        assert payload["prompts"] == ["prompt"]
        assert len(payload["images"]) == 1

        state["attempts"] += 1
        if state["always_fail"] or state["attempts"] <= 2:
            body = pickle.dumps({"error": "temporary scorer failure"})
            return web.Response(body=body, status=503)
        return web.Response(body=pickle.dumps({"scores": [0.75]}))

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
        "solution_image": torch.zeros(3, 2, 2),
        "ground_truth": "prompt",
        "server_url": server_url,
        "max_retries": 2,
        "retry_backoff": 0,
    }
    try:
        assert await client.compute_score(**kwargs) == {"score": 0.75}
        assert state["attempts"] == 3

        state.update(attempts=0, always_fail=True)
        with pytest.raises(RuntimeError, match="failed after 3 attempts:.*temporary scorer failure"):
            await client.compute_score(**kwargs)
        assert state["attempts"] == 3
    finally:
        session = getattr(client.compute_score, "_session", None)
        if session is not None and not session.closed:
            await session.close()
        await runner.cleanup()


def test_http_scorer_retry_e2e():
    """Verify recovery after two 503s and failure after retry exhaustion."""
    asyncio.run(_run_retry_e2e())
