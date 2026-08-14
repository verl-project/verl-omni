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

from unittest.mock import AsyncMock, Mock

import pytest

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer


@pytest.mark.asyncio
async def test_wake_up_restores_every_sleep_tag():
    server = object.__new__(vLLMOmniHttpServer)
    server.node_rank = 0
    server.engine = Mock()
    server.engine.collective_rpc = AsyncMock()
    server._invalidate_lora_request_cache = Mock()

    await server.wake_up()

    server.engine.collective_rpc.assert_awaited_once_with(
        "wake_up",
        kwargs={"tags": ["kv_cache", "weights"]},
    )
    server._invalidate_lora_request_cache.assert_called_once_with()


@pytest.mark.asyncio
async def test_wake_up_preserves_explicit_tags():
    server = object.__new__(vLLMOmniHttpServer)
    server.node_rank = 0
    server.engine = Mock()
    server.engine.collective_rpc = AsyncMock()
    server._invalidate_lora_request_cache = Mock()

    await server.wake_up(tags=["weights"])

    server.engine.collective_rpc.assert_awaited_once_with(
        "wake_up",
        kwargs={"tags": ["weights"]},
    )

