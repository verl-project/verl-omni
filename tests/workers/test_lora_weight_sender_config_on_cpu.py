# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.mark.parametrize(("regional_compile", "expected_gc_override"), [(False, None), (True, 1)])
@pytest.mark.asyncio
async def test_lora_fast_path_uses_regional_compile_gc_policy(monkeypatch, regional_compile, expected_gc_override):
    from verl_omni.workers import engine_workers

    sender_args = {}

    class FakeSender:
        def __init__(self, **kwargs):
            sender_args.update(kwargs)

        async def async_send_weights(self, weights):
            assert list(weights) == [("adapter", "weight")]

    async def execute_method(*args, **kwargs):
        return None

    async def run_inline(func, *args):
        return func(*args)

    rollout_config = SimpleNamespace(
        free_cache_engine=False,
        rollout_adapter="default",
        checkpoint_engine=SimpleNamespace(
            backend="naive",
            update_weights_bucket_megabytes=256,
        ),
    )
    rollout = SimpleNamespace(
        sleep_level=None,
        zmq_handle="ipc:///tmp/base.sock",
        use_shm=False,
        rollout_rank=1,
        _execute_method=execute_method,
    )
    worker = SimpleNamespace(
        config=SimpleNamespace(
            model={"use_regional_compile": regional_compile},
            rollout=rollout_config,
        ),
        gc_diagnostics=True,
        actor=SimpleNamespace(engine=SimpleNamespace(module=SimpleNamespace(peft_config={}))),
        rollout=rollout,
        peft_merge=False,
        base_sync_done=True,
        rank=0,
        _zmq_update_seq=0,
        _gather_lora_weights=lambda timings: ({"adapter": "weight"}, {}),
        _offload_actor_and_empty_cache=lambda timings: None,
    )
    monkeypatch.setattr(engine_workers, "BucketedWeightSender", FakeSender)
    monkeypatch.setattr(engine_workers, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_workers, "set_expandable_segments", lambda enabled: None)
    monkeypatch.setattr(engine_workers.asyncio, "to_thread", run_inline)

    update_weights = inspect.unwrap(engine_workers.ActorRolloutRefWorker.update_weights)
    await update_weights(worker, global_steps=3, mode="naive")

    if expected_gc_override is None:
        assert "gc_on_cleanup" not in sender_args
    else:
        assert sender_args["gc_on_cleanup"] == expected_gc_override
        assert type(sender_args["gc_on_cleanup"]) is type(expected_gc_override)
    assert sender_args["gc_diagnostics"] is True


@pytest.mark.parametrize(("regional_compile", "expected_gc_override"), [(False, None), (True, 1)])
@pytest.mark.asyncio
async def test_standard_weight_sync_uses_regional_compile_gc_policy(
    monkeypatch, regional_compile, expected_gc_override
):
    from verl_omni.workers import engine_workers

    actor_engine = SimpleNamespace(
        module=SimpleNamespace(),
        get_per_tensor_param=Mock(return_value=([("weight", "value")], None)),
    )
    rollout = SimpleNamespace(update_weights=AsyncMock())
    worker = SimpleNamespace(
        config=SimpleNamespace(
            model={"use_regional_compile": regional_compile},
            rollout=SimpleNamespace(
                checkpoint_engine=SimpleNamespace(backend="naive"),
                free_cache_engine=False,
            ),
        ),
        gc_diagnostics=True,
        actor=SimpleNamespace(engine=actor_engine),
        rollout=rollout,
        rollout_adapter="default",
        layered_summon=False,
        peft_merge=False,
        base_sync_done=True,
        rank=0,
        _offload_actor_and_empty_cache=lambda timings: None,
    )
    monkeypatch.setattr(engine_workers, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_workers, "set_expandable_segments", lambda enabled: None)

    update_weights = inspect.unwrap(engine_workers.ActorRolloutRefWorker.update_weights)
    await update_weights(worker, global_steps=3, mode="naive")

    expected_kwargs = {
        "peft_config": None,
        "base_sync_done": True,
        "global_steps": 3,
        "gc_diagnostics": True,
    }
    if expected_gc_override is not None:
        expected_kwargs["gc_on_cleanup"] = expected_gc_override
    rollout.update_weights.assert_awaited_once_with([("weight", "value")], **expected_kwargs)
