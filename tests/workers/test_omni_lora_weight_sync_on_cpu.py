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
"""Regression pin: the colocated LoRA fast path must stamp the weight version.

The direct-IPC fast path bypasses ``ServerAdapter.update_weights``, whose tail
calls ``clear_kv_cache`` and ``set_global_steps``. Skipping the stamp leaves
hybrid servers at ``global_steps=None``, and the async trainers crash on the
None ``min/max_global_steps`` sample tags in ``_compute_metrics``.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import verl_omni.workers.engine_workers as ew


def _fast_path_worker(rollout_rank=0):
    worker = object.__new__(ew.ActorRolloutRefWorker)

    engine = MagicMock(spec=["module", "get_per_tensor_param", "is_param_offload_enabled"])
    engine.is_param_offload_enabled = False
    engine.module = SimpleNamespace(peft_config={"default": SimpleNamespace(to_dict=lambda: {"r": 8})})
    engine.get_per_tensor_param.return_value = (iter([("w", torch.zeros(1))]), {"r": 8})
    worker.actor = SimpleNamespace(engine=engine)

    server_handle = MagicMock()
    server_handle.clear_kv_cache.remote = AsyncMock()
    server_handle.set_global_steps.remote = AsyncMock()
    rollout = MagicMock()
    rollout.zmq_handle = "ipc:///tmp/test-lora-sync.sock"
    rollout.use_shm = False
    rollout.rollout_rank = rollout_rank
    rollout._ensure_server_handle.return_value = True
    rollout.server_handle = server_handle
    rollout._execute_method = AsyncMock(return_value=None)
    rollout.resume = AsyncMock()
    worker.rollout = rollout

    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            free_cache_engine=False,
            checkpoint_engine=SimpleNamespace(backend="naive", update_weights_bucket_megabytes=16),
        )
    )
    worker._rank = 0
    worker.peft_merge = False
    worker.base_sync_done = True
    worker.layered_summon = False
    worker.rollout_adapter = "default"
    worker._zmq_update_seq = 0
    worker._offload_actor_and_empty_cache = lambda timings=None: None
    return worker


def _run_update(worker, *, entry=None, **kwargs):
    # This regression test covers weight-version stamping, not background thread scheduling.
    async def run_in_event_loop(function, *args, **kwargs):
        return function(*args, **kwargs)

    sender = MagicMock()
    sender.async_send_weights = AsyncMock()
    with (
        patch.object(ew, "BucketedWeightSender", return_value=sender),
        patch.object(ew.asyncio, "to_thread", run_in_event_loop),
        patch.object(ew, "log_gpu_memory_usage", MagicMock()),
        patch.object(ew, "set_expandable_segments", MagicMock()),
        patch.object(ew, "get_torch_device", return_value=MagicMock(current_device=MagicMock(return_value=0))),
    ):
        update = entry or ew.ActorRolloutRefWorker.update_weights
        asyncio.run(update(worker, mode="naive", **kwargs))
    return sender


@pytest.mark.parametrize("has_lora,merge", [(False, False), (True, False), (True, True)])
def test_non_naive_sync_preserves_peft_and_full_weight_paths(has_lora, merge):
    worker = _fast_path_worker()
    if not has_lora:
        worker.actor.engine.module = torch.nn.Linear(2, 2)
    worker.peft_merge = merge
    worker.checkpoint_engine = SimpleNamespace(send_weights=AsyncMock())
    asyncio.run(worker.update_weights(mode="nccl", global_steps=3))
    expected = {"adapter_name": "default"}
    if has_lora and not merge:
        expected["base_sync_done"] = True
    worker.actor.engine.get_per_tensor_param.assert_called_once_with(**expected)
    worker.checkpoint_engine.send_weights.assert_awaited_once()


def test_lora_fast_path_stamps_global_steps_and_clears_kv_cache():
    worker = _fast_path_worker()
    sender = _run_update(worker, global_steps=7)

    sender.async_send_weights.assert_awaited_once()  # fast path actually ran
    worker.rollout.server_handle.clear_kv_cache.remote.assert_awaited_once()
    worker.rollout.server_handle.set_global_steps.remote.assert_awaited_once_with(7)


def test_lora_fast_path_skips_stamp_when_global_steps_none():
    # Mirrors ServerAdapter.update_weights: cache reset still happens, but no
    # stamp is written for an unknown step (never write a fabricated version).
    worker = _fast_path_worker()
    _run_update(worker, global_steps=None)

    worker.rollout.server_handle.clear_kv_cache.remote.assert_awaited_once()
    worker.rollout.server_handle.set_global_steps.remote.assert_not_awaited()


def test_lora_fast_path_stamp_gated_to_rollout_rank_zero():
    worker = _fast_path_worker(rollout_rank=1)
    _run_update(worker, global_steps=7)

    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()
    worker.rollout.server_handle.set_global_steps.remote.assert_not_awaited()


@pytest.mark.parametrize(
    "reply,stage_ids",
    [
        ([{"supported": False, "error": "receiver failed"}], [0]),
        ([], [0]),
        ([[]], [0]),
        ([[None], [{"error": "second replica failed"}]], [0]),
        ([None, None], [0, 1]),
    ],
)
def test_lora_fast_path_rejected_rpc_does_not_publish_weight_version(reply, stage_ids):
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
        vLLMOmniHttpServer,
        vLLMOmniServerAdapter,
    )

    worker = _fast_path_worker()
    server = object.__new__(vLLMOmniHttpServer)
    server.engine = SimpleNamespace(
        num_stages=2,
        collective_rpc=AsyncMock(side_effect=lambda **kwargs: reply if kwargs["stage_ids"] == [0] else []),
    )
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: stage_ids)

    rollout = object.__new__(vLLMOmniServerAdapter)
    rollout.zmq_handle = worker.rollout.zmq_handle
    rollout.use_shm = worker.rollout.use_shm
    rollout.rollout_rank = worker.rollout.rollout_rank
    rollout._ensure_server_handle = MagicMock(return_value=True)
    rpc_calls = []

    def remote(method, **kwargs):
        rpc_calls.append((method, kwargs))
        return asyncio.create_task(server.collective_rpc(method, **kwargs))

    rollout.server_handle = SimpleNamespace(
        collective_rpc=SimpleNamespace(remote=remote),
        clear_kv_cache=SimpleNamespace(remote=AsyncMock()),
        set_global_steps=SimpleNamespace(remote=AsyncMock()),
    )
    worker.rollout = rollout
    with pytest.raises(RuntimeError, match="update_weights_from_ipc failed"):
        _run_update(worker, global_steps=7)

    assert len(rpc_calls) == 1
    assert rpc_calls[0][0] == "update_weights_from_ipc"
    assert rpc_calls[0][1]["require_ack"] is True
    assert server.engine.collective_rpc.await_count == len(stage_ids)
    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()
    worker.rollout.server_handle.set_global_steps.remote.assert_not_awaited()


def test_lora_fast_path_cancelled_rpc_does_not_publish_weight_version():
    worker = _fast_path_worker()

    async def execute(method, non_block, kwargs):
        async def cancelled_rpc():
            raise asyncio.CancelledError

        return asyncio.create_task(cancelled_rpc())

    worker.rollout._execute_method = AsyncMock(side_effect=execute)
    with pytest.raises(asyncio.CancelledError):
        _run_update(worker, global_steps=7)

    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()
    worker.rollout.server_handle.set_global_steps.remote.assert_not_awaited()


def test_lora_fast_path_active_caller_cancellation_does_not_publish_weight_version():
    worker = _fast_path_worker()

    async def cancel_active_update(worker, **kwargs):
        entered = asyncio.Event()
        pending = []

        async def execute(method, non_block, kwargs):
            async def blocked_rpc():
                entered.set()
                await asyncio.Event().wait()

            future = asyncio.create_task(blocked_rpc())
            pending.append(future)
            return future

        worker.rollout._execute_method = AsyncMock(side_effect=execute)
        task = asyncio.create_task(ew.ActorRolloutRefWorker.update_weights(worker, **kwargs))
        await entered.wait()
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(pending) == 1
        assert pending[0].cancelled()

    _run_update(worker, entry=cancel_active_update, global_steps=7)
    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()
    worker.rollout.server_handle.set_global_steps.remote.assert_not_awaited()
