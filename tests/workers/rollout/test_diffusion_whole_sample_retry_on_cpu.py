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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch
from omegaconf import OmegaConf
from verl.workers.rollout.llm_server import LLMServerClient

from verl_omni.workers.rollout.diffusion_llm_server import DiffusionWholeSampleRetryLLMServerClient

pytestmark = pytest.mark.asyncio


def _client(**kwargs):
    return DiffusionWholeSampleRetryLLMServerClient(OmegaConf.create({}), retry_wait_s=0, **kwargs)


async def test_retry_discards_partial_sample_and_preserves_original_inputs(monkeypatch):
    aborted = SimpleNamespace(stop_reason="abort", extra_fields={"global_steps": 1}, diffusion_output=torch.ones(2))
    completed = SimpleNamespace(
        stop_reason="completed",
        extra_fields={"global_steps": 3, "min_global_steps": 3, "max_global_steps": 3},
        diffusion_output=torch.zeros(3),
    )
    calls = []

    async def generate(self, **kwargs):
        calls.append((list(kwargs["prompt_ids"]), dict(kwargs["sampling_params"]), kwargs["extra_prompt_ids"]))
        kwargs["prompt_ids"].append(99)
        kwargs["sampling_params"]["seed"] = -1
        return completed if len(calls) == 2 else aborted

    monkeypatch.setattr(LLMServerClient, "generate", generate)
    prompt_ids, params = [7, 8], {"seed": 42, "num_inference_steps": 4}
    output = await _client(max_retries=2).generate(
        "request", prompt_ids=prompt_ids, sampling_params=params, extra_prompt_ids={"clip": [9]}
    )
    assert calls == [(prompt_ids, params, {"clip": [9]})] * 2
    assert prompt_ids == [7, 8] and params["seed"] == 42
    assert output is completed
    assert output.extra_fields["min_global_steps"] == output.extra_fields["max_global_steps"] == 3


async def test_retry_exhaustion_fails_instead_of_returning_empty_success(monkeypatch):
    generate = AsyncMock(return_value=SimpleNamespace(stop_reason="aborted", extra_fields={"global_steps": 1}))
    monkeypatch.setattr(LLMServerClient, "generate", generate)
    with pytest.raises(RuntimeError, match="aborted after 2 attempts"):
        await _client(max_retries=2).generate("request", prompt_ids=[1], sampling_params={})
    assert generate.await_count == 2


@pytest.mark.parametrize("params", [{}, {"seed": None}])
async def test_unseeded_sample_chooses_one_seed_for_all_attempts(monkeypatch, params):
    choose_seed = Mock(return_value=123456)
    monkeypatch.setattr("verl_omni.workers.rollout.diffusion_llm_server.secrets.randbits", choose_seed)
    generate = AsyncMock(side_effect=[SimpleNamespace(stop_reason="aborted"), SimpleNamespace(stop_reason="completed")])
    monkeypatch.setattr(LLMServerClient, "generate", generate)
    original_params = dict(params)

    await _client(max_retries=2).generate("request", prompt_ids=[1], sampling_params=params)

    choose_seed.assert_called_once_with(32)
    assert [call.kwargs["sampling_params"]["seed"] for call in generate.await_args_list] == [123456, 123456]
    assert params == original_params


async def test_routing_pause_waits_without_consuming_retry_budget(monkeypatch):
    reopened = asyncio.Event()

    async def acquire(self, request_id):
        if not reopened.is_set():
            raise RuntimeError("No available servers in load balancer")
        return "server", "handle"

    monkeypatch.setattr(LLMServerClient, "_acquire_server", acquire)
    task = asyncio.create_task(_client(max_retries=1, only_hybrid=True)._acquire_server("request"))
    for _ in range(60):
        await asyncio.sleep(0)
    assert not task.done()
    reopened.set()
    assert await task == ("server", "handle")


@pytest.mark.parametrize("only_hybrid", [False, True])
async def test_unrelated_routing_failure_propagates(monkeypatch, only_hybrid):
    monkeypatch.setattr(LLMServerClient, "_acquire_server", AsyncMock(side_effect=RuntimeError("worker disconnected")))
    with pytest.raises(RuntimeError, match="worker disconnected"):
        await _client(only_hybrid=only_hybrid)._acquire_server("request")
