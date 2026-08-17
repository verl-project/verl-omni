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
from types import MethodType, SimpleNamespace

import pytest
import torch
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics
from verl.protocol import DataProto

from verl_omni.agent_loop import diffusion_agent_loop_tq
from verl_omni.agent_loop.diffusion_agent_loop import (
    DiffusionAgentLoopOutput,
    DiffusionAgentLoopWorker,
    _pad_prompt_extra_field,
)
from verl_omni.agent_loop.diffusion_agent_loop_tq import DiffusionAgentLoopWorkerTQ


class _FakeRemoteComputeScore:
    def __init__(self):
        self.received_data: DataProto | None = None

    async def remote(self, data: DataProto) -> dict:
        self.received_data = data
        return {"reward_score": 1.0, "reward_extra_info": {}}


class _FakeRewardLoopWorkerHandle:
    def __init__(self):
        self.compute_score = _FakeRemoteComputeScore()


class _DummyDiffusionAgentLoopWorker:
    _compute_score = DiffusionAgentLoopWorker._compute_score

    def __init__(self, reward_loop_worker_handle: _FakeRewardLoopWorkerHandle):
        self.reward_loop_worker_handles = [reward_loop_worker_handle]


@pytest.mark.parametrize(
    ("key", "value", "expected_shape"),
    [
        ("prompt_embeds", torch.ones(2, 4), (3, 4)),
        ("negative_prompt_embeds", torch.ones(2, 4), (3, 4)),
        ("prompt_embeds_mask", torch.ones(2), (3,)),
        ("negative_prompt_embeds_mask", torch.ones(2), (3,)),
    ],
)
def test_pad_prompt_extra_field_pads_without_truncation(key, value, expected_shape):
    padded = _pad_prompt_extra_field(key, value, target_length=3)

    assert padded.shape == expected_shape
    assert torch.equal(padded[:2], value)
    assert torch.count_nonzero(padded[2:]) == 0


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("prompt_embeds", torch.ones(4, 2)),
        ("prompt_embeds_mask", torch.ones(4)),
    ],
)
def test_pad_prompt_extra_field_rejects_truncation(key, value):
    with pytest.raises(ValueError, match="exceeds max_prompt_embed_length=3"):
        _pad_prompt_extra_field(key, value, target_length=3)


@pytest.mark.asyncio
async def test_run_prompt_publishes_failure_after_siblings_settle(monkeypatch):
    worker_cls = DiffusionAgentLoopWorkerTQ.__ray_metadata__.modified_class
    worker = object.__new__(worker_cls)
    worker.rollout_config = SimpleNamespace(n=2, val_kwargs=SimpleNamespace(n=2))
    lifecycle = []

    async def fake_kv_put(*, key, partition_id, tag):
        assert key == "sample"
        assert partition_id == "train"
        lifecycle.append(tag["status"])

    async def fake_run_agent_loop(self, sampling_params, *, session_id, **kwargs):
        del self, sampling_params, kwargs
        if session_id == 0:
            raise RuntimeError("session failed")
        await asyncio.sleep(0.01)
        lifecycle.append("sibling_settled")

    monkeypatch.setattr(diffusion_agent_loop_tq.tq, "async_kv_put", fake_kv_put)
    worker._run_agent_loop = MethodType(fake_run_agent_loop, worker)

    await worker._run_prompt(
        prompt={"uid": "sample", "agent_name": "diffusion_single_turn_agent"},
        sampling_params={},
        trajectory={"validate": False},
        sample_index=0,
    )

    assert lifecycle == ["running", "sibling_settled", "failure"]


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [False, True])
async def test_async_reward_data_proto_preserves_validate_meta_info(validate: bool):
    reward_loop_worker_handle = _FakeRewardLoopWorkerHandle()
    worker = _DummyDiffusionAgentLoopWorker(reward_loop_worker_handle)
    output = DiffusionAgentLoopOutput(
        prompt_ids=[1, 2],
        response_diffusion_output=torch.zeros(3, 2, 2),
        metrics=AgentLoopMetrics(),
    )

    await worker._compute_score(
        output,
        prompts=torch.tensor([[1, 2]]),
        responses=torch.zeros(1, 3, 2, 2),
        kwargs={},
        validate=validate,
    )

    received_data = reward_loop_worker_handle.compute_score.received_data
    assert received_data is not None
    assert received_data.meta_info == {"validate": validate}
