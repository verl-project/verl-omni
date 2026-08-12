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

import torch

import verl_omni.workers.engine_workers as ew


def _fast_path_worker(rollout_rank=0):
    worker = object.__new__(ew.ActorRolloutRefWorker)

    engine = MagicMock()
    engine.module = SimpleNamespace(peft_config={"default": object()})  # actor_has_lora
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
    worker.peft_merge = False
    worker.base_sync_done = True
    worker.layered_summon = False
    worker.rollout_adapter = "default"
    worker._zmq_update_seq = 0
    worker._offload_actor_and_empty_cache = lambda timings=None: None
    return worker


def _run_update(worker, **kwargs):
    sender = MagicMock()
    sender.async_send_weights = AsyncMock()
    with (
        patch.object(ew, "BucketedWeightSender", return_value=sender),
        patch.object(ew, "log_gpu_memory_usage", MagicMock()),
        patch.object(ew, "set_expandable_segments", MagicMock()),
    ):
        asyncio.run(ew.ActorRolloutRefWorker.update_weights(worker, mode="naive", **kwargs))
    return sender


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
