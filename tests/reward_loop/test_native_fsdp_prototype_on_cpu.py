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
"""CPU contract tests; actual sharding is checked by validate_native_fsdp.py."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import torch

from verl_omni.reward_loop.native_fsdp import (
    FSDPRewardReplica,
    NativeFSDPExecutor,
    NativeFSDPGroupError,
    terminate_ranks,
)


def test_partial_submission_and_rank_failure_terminate_whole_group(monkeypatch):
    calls = []
    ranks = [object(), object()]

    def partial(*args):
        raise RuntimeError("rank failed")

    group = SimpleNamespace(world_size=2, workers=ranks, wake_up=lambda: [None, None], infer=partial)
    model = FSDPRewardReplica(group, lambda **kwargs: {})
    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.get", lambda refs, **kwargs: refs)
    monkeypatch.setattr(
        "verl_omni.reward_loop.native_fsdp.terminate_ranks", lambda workers, timeout: calls.extend(workers)
    )
    model.wake_up()
    with pytest.raises(NativeFSDPGroupError):
        model.infer(["prompt"], [object()])
    assert calls == ranks
    with pytest.raises(RuntimeError, match="closed"):
        model.wake_up()
    model.close()


def test_unconfirmed_rank_death_is_an_error_and_other_ranks_are_attempted(monkeypatch):
    calls = []
    actors = [SimpleNamespace(status=SimpleNamespace(remote=lambda: "pending")) for _ in range(2)]
    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.kill", lambda actor, **kwargs: calls.append(actor))

    def timeout(*args, **kwargs):
        raise TimeoutError("rank did not exit")

    monkeypatch.setattr("verl_omni.reward_loop.native_fsdp.ray.get", timeout)
    with pytest.raises(BaseExceptionGroup, match="job termination") as error:
        terminate_ranks(actors, 0.1)
    assert calls == actors and len(error.value.exceptions) == 2


@pytest.mark.asyncio
async def test_cancelled_native_handle_retains_accepted_work_until_settlement():
    entered, release = threading.Event(), threading.Event()
    finished = threading.Event()

    def infer(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        finished.set()
        return {
            "image_embeddings": torch.ones(1, 2),
            "text_embeddings": torch.ones(1, 2),
            "logit_scale": torch.ones(()),
        }

    handle = NativeFSDPExecutor(SimpleNamespace(infer=infer))
    task = asyncio.create_task(handle.infer(["prompt"], [object()]))
    assert await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert finished.is_set()
