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
"""Behavioral regressions for the colocated LoRA capacity boundary.

Exercise the production weight-sync method, not an extracted copy of its logic.
The one-unit budget represents an actor or resumed rollout weight allocation.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import verl_omni.workers.engine_workers as ew


def _worker(*, offload=True, free_cache=True, lora=True, base_synced=True, merged=False):
    events = []
    adapter = torch.tensor([1.25, -2.5])
    peft_config = {"r": 8}
    occupancy = {"actor": int(offload), "rollout": 0}
    capacity_enforced = offload and lora and base_synced and not merged

    worker = object.__new__(ew.ActorRolloutRefWorker)
    engine = MagicMock()
    engine.is_param_offload_enabled = offload
    engine.module = SimpleNamespace(peft_config={"default": object()}) if lora else SimpleNamespace()

    def gather(**kwargs):
        events.append(("gather", kwargs))
        return iter([("adapter", adapter)]), peft_config if lora else None

    def move(device, **kwargs):
        events.append(("offload", device, kwargs))
        occupancy["actor"] = 0

    engine.get_per_tensor_param.side_effect = gather
    engine.to.side_effect = move
    worker.actor = SimpleNamespace(engine=engine)

    rollout = MagicMock()
    rollout.zmq_handle = "ipc:///tmp/test-colocated-capacity.sock"
    rollout.use_shm = False
    rollout.rollout_rank = 0
    rollout._ensure_server_handle.return_value = True
    rollout.server_handle.clear_kv_cache.remote = AsyncMock()
    rollout.server_handle.set_global_steps.remote = AsyncMock()

    async def resume(*, tags):
        events.append(("resume", tuple(tags)))
        if tags == ["weights"]:
            occupancy["rollout"] = 1
            if capacity_enforced:
                assert sum(occupancy.values()) <= 1, "actor and rollout weights overlap"

    async def execute(method, **kwargs):
        events.append(("execute", method, kwargs))
        return None

    rollout.resume = AsyncMock(side_effect=resume)
    rollout._execute_method = AsyncMock(side_effect=execute)
    rollout.update_weights = AsyncMock()
    worker.rollout = rollout
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            free_cache_engine=free_cache,
            checkpoint_engine=SimpleNamespace(backend="naive", update_weights_bucket_megabytes=16),
        )
    )
    worker._rank = 0
    worker.peft_merge = merged
    worker.base_sync_done = base_synced
    worker.layered_summon = False
    worker.rollout_adapter = "default"
    worker._zmq_update_seq = 0
    worker.checkpoint_engine = SimpleNamespace(send_weights=AsyncMock())
    return worker, events, adapter, peft_config


def _run(worker, events, *, mode="naive", global_steps=7, fail_send=False, device=None):
    sender = MagicMock()
    sent = []

    async def send(weights):
        sent.extend(weights)
        events.append(("send", tuple(sent)))
        if fail_send:
            raise RuntimeError("send failed")

    sender.async_send_weights = AsyncMock(side_effect=send)

    async def invoke():
        try:
            await ew.ActorRolloutRefWorker.update_weights(worker, mode=mode, global_steps=global_steps)
        finally:
            # The capacity-safe path must not strand gather/offload tasks on
            # either success or failure.
            assert asyncio.all_tasks() == {asyncio.current_task()}

    with (
        patch.object(ew, "BucketedWeightSender", return_value=sender),
        patch.object(ew, "log_gpu_memory_usage"),
        patch.object(ew, "set_expandable_segments"),
        patch.object(ew, "get_torch_device", return_value=device or _device(events)),
    ):
        asyncio.run(invoke())
    return sender, sent


def _device(events):
    device = MagicMock()
    device.current_device.return_value = 0
    device.synchronize.side_effect = lambda: events.append(("synchronize",))
    device.empty_cache.side_effect = lambda: events.append(("empty_cache",))
    return device


def _names(events):
    return [event[0] for event in events]


async def _wait_for_thread_event(event):
    while not event.is_set():
        await asyncio.sleep(0.001)


@pytest.mark.parametrize("free_cache", [True, False])
def test_offloaded_lora_releases_actor_before_rollout_and_forwards_exact_adapter(free_cache):
    worker, events, adapter, peft_config = _worker(free_cache=free_cache)
    sender, sent = _run(worker, events)

    names = _names(events)
    assert (
        names.index("gather")
        < names.index("offload")
        < names.index("synchronize")
        < names.index("empty_cache")
        < names.index("execute")
        < names.index("send")
    )
    assert names.count("offload") == 1
    assert names.count("empty_cache") == 1
    assert events[names.index("empty_cache")] == ("empty_cache",)
    if free_cache:
        assert names.index("empty_cache") < names.index("resume") < names.index("execute")
        assert events[-1] == ("resume", ("kv_cache",))
    else:
        assert "resume" not in names
    assert len(sent) == 1 and sent[0][0] == "adapter"
    assert sent[0][1] is adapter
    torch.testing.assert_close(sent[0][1], torch.tensor([1.25, -2.5]))
    assert sender.async_send_weights.await_count == 1
    assert worker.rollout._execute_method.await_args.kwargs["kwargs"]["peft_config"] is peft_config
    assert worker.rollout._execute_method.await_args.kwargs["kwargs"]["base_sync_done"] is True
    assert worker.rollout.server_handle.set_global_steps.remote.await_args.args == (7,)


@pytest.mark.parametrize("failure", ["gather", "offload", "resume"])
def test_offloaded_lora_failure_stops_later_stages(failure):
    worker, events, _, _ = _worker()
    if failure == "gather":
        worker.actor.engine.get_per_tensor_param.side_effect = RuntimeError("gather failed")
    elif failure == "offload":
        worker.actor.engine.to.side_effect = RuntimeError("offload failed")
    else:
        worker.rollout.resume.side_effect = RuntimeError("resume failed")

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        _run(worker, events)
    worker.rollout._execute_method.assert_not_awaited()
    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()
    if failure in ("gather", "offload"):
        worker.rollout.resume.assert_not_awaited()
    if failure == "gather":
        assert _names(events).count("offload") == 1


def test_offloaded_lora_ipc_failure_does_not_resume_kv_cache():
    worker, events, _, _ = _worker()
    with pytest.raises(RuntimeError, match="send failed"):
        _run(worker, events, fail_send=True)

    assert [event for event in events if event[0] == "resume"] == [("resume", ("weights",))]
    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()


def test_offloaded_lora_worker_uses_callers_device():
    worker, events, _, _ = _worker()
    device_local = threading.local()
    device_local.index = 3
    device = _device(events)
    device.current_device.side_effect = lambda: getattr(device_local, "index", None)
    device.set_device.side_effect = lambda index: setattr(device_local, "index", index)
    gather = worker.actor.engine.get_per_tensor_param.side_effect

    def check_device(**kwargs):
        assert threading.current_thread() is not threading.main_thread()
        assert device_local.index == 3
        return gather(**kwargs)

    worker.actor.engine.get_per_tensor_param.side_effect = check_device
    _run(worker, events, device=device)

    device.set_device.assert_called_once_with(3)
    assert _names(events).index("gather") < _names(events).index("offload")


@pytest.mark.parametrize("blocked_stage", ["gather", "offload"])
def test_offloaded_lora_blocking_work_keeps_event_loop_responsive(blocked_stage):
    worker, events, _, _ = _worker()
    entered = threading.Event()
    release = threading.Event()
    target = worker.actor.engine.get_per_tensor_param if blocked_stage == "gather" else worker.actor.engine.to
    original = target.side_effect

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    target.side_effect = blocked

    async def drive():
        task = asyncio.create_task(ew.ActorRolloutRefWorker.update_weights(worker, mode="naive", global_steps=7))
        try:
            await asyncio.wait_for(_wait_for_thread_event(entered), timeout=2)
            heartbeat = asyncio.Event()
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), timeout=1)
            assert not task.done()
        finally:
            release.set()
        await task
        assert asyncio.all_tasks() == {asyncio.current_task()}

    with (
        patch.object(ew, "BucketedWeightSender") as sender_type,
        patch.object(ew, "log_gpu_memory_usage"),
        patch.object(ew, "set_expandable_segments"),
        patch.object(ew, "get_torch_device", return_value=_device(events)),
    ):
        sender_type.return_value.async_send_weights = AsyncMock()
        asyncio.run(drive())

    assert _names(events).index("offload") < _names(events).index("resume")
    worker.rollout._execute_method.assert_awaited_once()


@pytest.mark.parametrize("blocked_stage", ["gather", "offload"])
@pytest.mark.parametrize("worker_fails", [False, True])
def test_offloaded_lora_cancellation_drains_repeatedly_cancelled_worker(blocked_stage, worker_fails):
    worker, events, _, _ = _worker()
    entered = threading.Event()
    release = threading.Event()
    target = worker.actor.engine.get_per_tensor_param if blocked_stage == "gather" else worker.actor.engine.to
    original = target.side_effect

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        if worker_fails:
            raise RuntimeError(f"{blocked_stage} failed")
        return original(*args, **kwargs)

    target.side_effect = blocked

    async def drive():
        task = asyncio.create_task(ew.ActorRolloutRefWorker.update_weights(worker, mode="naive", global_steps=7))
        try:
            await asyncio.wait_for(_wait_for_thread_event(entered), timeout=2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        if worker_fails:
            assert isinstance(caught.value.__cause__, RuntimeError)
            assert str(caught.value.__cause__) == f"{blocked_stage} failed"
        assert asyncio.all_tasks() == {asyncio.current_task()}

    with (
        patch.object(ew, "log_gpu_memory_usage"),
        patch.object(ew, "set_expandable_segments"),
        patch.object(ew, "get_torch_device", return_value=_device(events)),
    ):
        asyncio.run(drive())

    if blocked_stage == "gather":
        assert _names(events).count("offload") == 1
    worker.rollout.resume.assert_not_awaited()
    worker.rollout._execute_method.assert_not_awaited()


@pytest.mark.parametrize("blocked_stage", ["resume", "send"])
def test_offloaded_lora_cancellation_has_no_owned_task_or_kv_resume(blocked_stage):
    worker, events, _, _ = _worker()
    sender = MagicMock()

    async def drive():
        entered = asyncio.Event()

        async def resume(*, tags):
            events.append(("resume", tuple(tags)))
            if blocked_stage == "resume" and tags == ["weights"]:
                entered.set()
                await asyncio.Future()

        async def send(weights):
            events.append(("send", tuple(weights)))
            if blocked_stage == "send":
                entered.set()
                await asyncio.Future()

        worker.rollout.resume.side_effect = resume
        sender.async_send_weights = AsyncMock(side_effect=send)
        task = asyncio.create_task(ew.ActorRolloutRefWorker.update_weights(worker, mode="naive", global_steps=7))
        await asyncio.wait_for(entered.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert asyncio.all_tasks() == {asyncio.current_task()}

    with (
        patch.object(ew, "BucketedWeightSender", return_value=sender),
        patch.object(ew, "log_gpu_memory_usage"),
        patch.object(ew, "set_expandable_segments"),
        patch.object(ew, "get_torch_device", return_value=_device(events)),
    ):
        asyncio.run(drive())

    assert ("resume", ("kv_cache",)) not in events
    worker.rollout.server_handle.clear_kv_cache.remote.assert_not_awaited()
    assert ("execute" in _names(events)) is (blocked_stage == "send")


def test_non_offloaded_lora_keeps_fast_ipc_path():
    worker, events, adapter, _ = _worker(offload=False)
    _, sent = _run(worker, events)

    assert sent[0][1] is adapter
    assert _names(events).count("offload") == 0
    assert worker.rollout.resume.await_count == 2
    worker.rollout.update_weights.assert_not_awaited()
    assert worker.base_sync_done is True


def test_non_offloaded_lora_threads_use_callers_device():
    worker, events, _, _ = _worker(offload=False)
    device_local = threading.local()
    device_local.index = 3
    device = _device(events)
    device.current_device.side_effect = lambda: getattr(device_local, "index", None)
    device.set_device.side_effect = lambda index: setattr(device_local, "index", index)
    gather = worker.actor.engine.get_per_tensor_param.side_effect

    def check_gather(**kwargs):
        assert threading.current_thread() is not threading.main_thread()
        assert device_local.index == 3
        return gather(**kwargs)

    def check_offload():
        assert threading.current_thread() is not threading.main_thread()
        assert device_local.index == 3
        events.append(("synchronize",))

    worker.actor.engine.get_per_tensor_param.side_effect = check_gather
    device.synchronize.side_effect = check_offload
    _run(worker, events, device=device)

    assert device.set_device.call_count == 2
    assert all(call.args == (3,) for call in device.set_device.call_args_list)
    assert _names(events).index("gather") < _names(events).index("synchronize")


def test_non_naive_backend_is_unchanged():
    worker, events, _, _ = _worker()
    _run(worker, events, mode="disaggregated")

    worker.checkpoint_engine.send_weights.assert_awaited_once()
    worker.rollout.resume.assert_not_awaited()
    worker.rollout._execute_method.assert_not_awaited()
    worker.actor.engine.to.assert_not_called()
    assert _names(events) == ["gather"]


def test_first_base_sync_stays_on_standard_path():
    worker, events, _, _ = _worker(base_synced=False)
    _run(worker, events)

    worker.rollout.update_weights.assert_awaited()
    worker.rollout._execute_method.assert_not_awaited()
    assert _names(events).count("offload") == 1
    assert worker.base_sync_done is True


@pytest.mark.parametrize("lora,merged", [(False, False), (True, True)])
def test_non_lora_and_merged_lora_stay_on_standard_path(lora, merged):
    worker, events, _, _ = _worker(lora=lora, merged=merged)
    _run(worker, events)

    worker.rollout.update_weights.assert_awaited_once()
    worker.rollout._execute_method.assert_not_awaited()
    assert _names(events).count("offload") == 1


def test_repeated_offloaded_updates_do_not_keep_old_tasks_or_weights():
    worker, events, _, _ = _worker()
    _run(worker, events, global_steps=7)
    _run(worker, events, global_steps=8)

    assert _names(events).count("gather") == 2
    assert _names(events).count("offload") == 2
    assert _names(events).count("send") == 2
    assert worker._zmq_update_seq == 2
    assert worker.rollout.server_handle.set_global_steps.remote.await_args.args == (8,)
