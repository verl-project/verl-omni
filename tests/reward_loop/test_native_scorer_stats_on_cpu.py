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
"""Cumulative, read-only native scorer runtime statistics."""

import asyncio
import threading
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tests.reward_loop.test_native_scorer_executor_on_cpu import _TEST_MODULE, _executor
from verl_omni.reward_loop import reward_loop as loop_module
from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager, OmniRewardLoopWorker
from verl_omni.reward_loop.reward_model_executor import ScorerMicrobatcher


class Clock:
    def __init__(self):
        self.now = 0.0

    def read(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(executor_module, "perf_counter", clock.read)
    return clock


async def _until(predicate):
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0)


@pytest_asyncio.fixture
async def batcher_factory():
    owned = []
    async_releases = []
    thread_releases = []

    def create(*args, **kwargs):
        batcher = ScorerMicrobatcher(*args, **kwargs)
        owned.append(batcher)
        return batcher

    def async_release():
        event = asyncio.Event()
        async_releases.append(event)
        return event

    def thread_release():
        event = threading.Event()
        thread_releases.append(event)
        return event

    create.async_release = async_release
    create.thread_release = thread_release
    yield create
    for event in async_releases + thread_releases:
        event.set()
    for batcher in reversed(owned):
        await asyncio.wait_for(batcher.close(), 5)


@pytest.mark.asyncio
async def test_packed_and_single_batches_have_exact_host_and_queue_times(clock, batcher_factory):
    entered = asyncio.Event()
    release = batcher_factory.async_release()

    async def score(requests):
        entered.set()
        await release.wait()
        return [{"score": request["i"]} for request in requests]

    batcher = batcher_factory(score, max_batch_size=3)
    await batcher.start()
    first = asyncio.create_task(batcher.submit({"i": 0}))
    second = asyncio.create_task(batcher.submit({"i": 1}))
    await asyncio.wait_for(entered.wait(), 5)
    assert batcher.get_runtime_stats()["accepted_requests"] == 2
    clock.advance(3)
    release.set()
    assert await asyncio.gather(first, second) == [{"score": 0}, {"score": 1}]
    third = await batcher.submit({"i": 2})
    assert third == {"score": 2}
    await batcher.close()
    stats = batcher.get_runtime_stats()
    assert {key: stats[key] for key in ("enqueued_requests", "accepted_requests", "batches", "batch_requests")} == {
        "enqueued_requests": 3,
        "accepted_requests": 3,
        "batches": 2,
        "batch_requests": 3,
    }
    assert stats["peak_batch_size"] == 2
    assert stats["peak_queue_size"] == 2
    assert stats["queue_wait_seconds"] == 0
    assert stats["forward_seconds"] == 3
    stats["batches"] = -1
    assert batcher.get_runtime_stats()["batches"] == 2


@pytest.mark.asyncio
async def test_queue_wait_counts_only_accepted_request(clock, batcher_factory):
    entered = asyncio.Event()
    release = batcher_factory.async_release()

    async def score(requests):
        entered.set()
        await release.wait()
        return [{"score": requests[0]["i"]}]

    batcher = batcher_factory(score)
    await batcher.start()
    first = asyncio.create_task(batcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    second = asyncio.create_task(batcher.submit({"i": 1}))
    await _until(lambda: bool(batcher._queue))
    clock.advance(7)
    release.set()
    assert await asyncio.gather(first, second) == [{"score": 0}, {"score": 1}]
    assert batcher.get_runtime_stats()["queue_wait_seconds"] == 7
    await batcher.close()


@pytest.mark.asyncio
async def test_capacity_wait_withdrawal_and_pause_discard_are_separate(clock, batcher_factory):
    entered = asyncio.Event()
    release = batcher_factory.async_release()

    async def score(requests):
        entered.set()
        await release.wait()
        return [{"score": requests[0]["i"]}]

    batcher = batcher_factory(score, max_inflight=1)
    await batcher.start()
    accepted = asyncio.create_task(batcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    queued = asyncio.create_task(batcher.submit({"i": 1}))
    await _until(lambda: len(batcher._queue) == 1)
    blocked = asyncio.create_task(batcher.submit({"i": 2}))
    await _until(lambda: batcher.get_runtime_stats()["capacity_wait_requests"] == 1)
    clock.advance(2)
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    # The blocked call now occupies the queue; its capacity wait has ended.
    await _until(lambda: len(batcher._queue) == 1)
    stats = batcher.get_runtime_stats()
    assert stats["withdrawn_requests"] == 1
    assert stats["capacity_wait_seconds"] == 2
    clock.advance(3)
    pausing = asyncio.create_task(batcher.pause())
    with pytest.raises(RuntimeError, match="retry after wake_up"):
        await blocked
    release.set()
    assert await accepted == {"score": 0}
    await pausing
    stats = batcher.get_runtime_stats()
    assert stats["enqueued_requests"] == 3
    assert stats["accepted_requests"] == 1
    assert stats["withdrawn_requests"] == 1
    assert stats["discarded_requests"] == 1
    assert stats["queue_wait_seconds"] == 0
    assert stats["failed_requests"] == 0
    await batcher.close()


@pytest.mark.asyncio
async def test_cancelled_capacity_wait_and_close_account_for_rejection(clock, batcher_factory):
    entered = asyncio.Event()
    release = batcher_factory.async_release()

    async def score(requests):
        entered.set()
        await release.wait()
        return [{"score": 1}]

    batcher = batcher_factory(score, max_inflight=1)
    await batcher.start()
    first = asyncio.create_task(batcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    queued = asyncio.create_task(batcher.submit({"i": 1}))
    await _until(lambda: bool(batcher._queue))
    blocked = asyncio.create_task(batcher.submit({"i": 2}))
    await _until(lambda: batcher.get_runtime_stats()["capacity_wait_requests"] == 1)
    clock.advance(4)
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    closing = asyncio.create_task(batcher.close())
    with pytest.raises(RuntimeError, match="closed before"):
        await queued
    release.set()
    await first
    await closing
    stats = batcher.get_runtime_stats()
    assert stats["capacity_wait_seconds"] == 4
    assert stats["enqueued_requests"] == 2
    assert stats["discarded_requests"] == 1
    assert stats["failed_requests"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["raised", "slot", "shape"])
async def test_failures_count_batch_once_and_only_exception_slots(result, batcher_factory):
    def score(requests):
        if result == "raised":
            raise OSError("forward failed")
        if result == "shape":
            return []
        return [ValueError("bad first"), {"score": 1}, ValueError("bad third")]

    batcher = batcher_factory(score, max_batch_size=3)
    await batcher.start()
    outcomes = await asyncio.gather(*(batcher.submit({"i": i}) for i in range(3)), return_exceptions=True)
    await batcher.close()
    if result == "slot":
        assert isinstance(outcomes[0], ValueError)
        assert outcomes[1] == {"score": 1}
        assert isinstance(outcomes[2], ValueError)
    else:
        assert all(isinstance(outcome, OSError if result == "raised" else ValueError) for outcome in outcomes)
    stats = batcher.get_runtime_stats()
    assert stats["failed_batches"] == 1
    assert stats["failed_requests"] == (2 if result == "slot" else 3)
    assert stats["batches"] == 1


@pytest.mark.asyncio
async def test_consumer_cancellation_counts_accepted_failure_and_queued_discard(clock, batcher_factory):
    entered = asyncio.Event()
    release = batcher_factory.async_release()

    async def score(requests):
        entered.set()
        await release.wait()
        return [{"score": 1}]

    batcher = batcher_factory(score)
    await batcher.start()
    first = asyncio.create_task(batcher.submit({"i": 0}))
    await asyncio.wait_for(entered.wait(), 5)
    queued = asyncio.create_task(batcher.submit({"i": 1}))
    await _until(lambda: bool(batcher._queue))
    clock.advance(5)
    batcher._consumer.cancel()
    outcomes = await asyncio.gather(first, queued, return_exceptions=True)
    assert all(isinstance(outcome, RuntimeError) for outcome in outcomes)
    stats = batcher.get_runtime_stats()
    assert stats["accepted_requests"] == 1
    assert stats["failed_batches"] == 1
    assert stats["failed_requests"] == 1
    assert stats["discarded_requests"] == 1
    assert stats["forward_seconds"] == 5
    await batcher.close()


@pytest.mark.asyncio
async def test_accepted_caller_cancellation_waits_for_success_without_failure_count(clock, batcher_factory):
    entered = asyncio.Event()
    release = batcher_factory.async_release()

    async def score(requests):
        entered.set()
        await release.wait()
        return [{"score": requests[0]["i"]}]

    batcher = batcher_factory(score)
    await batcher.start()
    caller = asyncio.create_task(batcher.submit({"i": 4}))
    await asyncio.wait_for(entered.wait(), 5)
    caller.cancel()
    await asyncio.sleep(0)
    assert not caller.done()
    clock.advance(6)
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    await batcher.close()
    stats = batcher.get_runtime_stats()
    assert stats["accepted_requests"] == 1
    assert stats["batches"] == 1
    assert stats["withdrawn_requests"] == 0
    assert stats["failed_requests"] == 0
    assert stats["failed_batches"] == 0
    assert stats["forward_seconds"] == 6


@pytest.mark.asyncio
async def test_consumer_cancelled_before_first_step_does_not_invent_requests(batcher_factory):
    batcher = batcher_factory(lambda requests: [{"score": 1}])
    await batcher.start()
    consumer = batcher._consumer
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    with pytest.raises(RuntimeError, match="consumer was cancelled"):
        await batcher.submit({"i": 0})
    await batcher.close()
    stats = batcher.get_runtime_stats()
    assert stats["enqueued_requests"] == 0
    assert stats["accepted_requests"] == 0
    assert stats["discarded_requests"] == 0
    assert stats["batches"] == 0
    assert stats["failed_requests"] == 0


@pytest.mark.asyncio
async def test_cancelled_consumer_keeps_sync_forward_owned_until_thread_exits(clock, batcher_factory):
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = batcher_factory.thread_release()
    finished = threading.Event()

    def score(requests):
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(10), "scorer thread was not released"
            return [{"score": requests[0]["i"]}]
        finally:
            finished.set()

    batcher = batcher_factory(score)
    await batcher.start()
    caller = asyncio.create_task(batcher.submit({"i": 5}))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        consumer = batcher._consumer
        consumer.cancel()
        await asyncio.sleep(0)
        assert not consumer.done()
        assert not finished.is_set()
        clock.advance(8)
    finally:
        release.set()
    outcome = await asyncio.wait_for(asyncio.gather(caller, return_exceptions=True), 5)
    assert isinstance(outcome[0], RuntimeError)
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await batcher.close()
    stats = batcher.get_runtime_stats()
    assert finished.is_set()
    assert stats["forward_seconds"] == 8
    assert stats["failed_batches"] == 1
    assert stats["failed_requests"] == 1


@pytest.mark.asyncio
async def test_executor_counters_survive_no_offload_reload_and_remain_isolated(monkeypatch):
    first = _executor(monkeypatch, f"{_TEST_MODULE}:NonOffloadScorer")
    other = _executor(monkeypatch, f"{_TEST_MODULE}:NonOffloadScorer")
    legacy = _executor(monkeypatch, f"{_TEST_MODULE}:LegacyModel")
    assert legacy.get_runtime_stats() == {}
    await first.wake_up()
    old_batcher = first._microbatcher
    assert await first.score({"i": 1}) == {"score": 1}
    await first.sleep()
    assert first._microbatcher is None
    await first.wake_up()
    assert first._microbatcher is not old_batcher
    assert await first.score({"i": 2}) == {"score": 2}
    await first.close()
    assert first.get_runtime_stats()["batches"] == 2
    assert other.get_runtime_stats() == {}
    await other.wake_up()
    assert other.get_runtime_stats()["batches"] == 0
    await other.close()
    snapshot = first.get_runtime_stats()
    snapshot["enqueued_requests"] = 999
    assert first.get_runtime_stats()["enqueued_requests"] == 2


@pytest.mark.asyncio
async def test_offload_reuses_batcher_and_retains_counters(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:FakeScorer")
    await executor.wake_up()
    batcher = executor._microbatcher
    await executor.score({"i": 0})
    await executor.sleep()
    await executor.wake_up()
    assert executor._microbatcher is batcher
    await executor.score({"i": 1})
    await executor.close()
    assert executor.get_runtime_stats()["enqueued_requests"] == 2


def test_worker_getter_skips_non_scorer_owners_without_waking(monkeypatch):
    scorer = _executor(monkeypatch, f"{_TEST_MODULE}:FakeScorer")
    scorer._is_scorer = True
    legacy = _executor(monkeypatch, f"{_TEST_MODULE}:LegacyModel")
    worker = OmniRewardLoopWorker.__new__(OmniRewardLoopWorker)
    worker.native_reward_executors = {"scorer": scorer, "legacy": legacy, "fsdp": object()}
    assert worker.get_native_scorer_stats() == {"scorer": scorer.get_runtime_stats()}
    assert scorer._model is None


def test_manager_getter_preserves_group_worker_and_model_identity(monkeypatch):
    calls = []

    class RemoteGetter:
        def __init__(self, name, payload):
            self.name = name
            self.payload = payload

        def remote(self):
            calls.append(self.name)
            return self.payload

    def worker(name, payload):
        return SimpleNamespace(get_native_scorer_stats=RemoteGetter(name, payload))

    monkeypatch.setattr(loop_module.ray, "get", lambda refs: refs)
    manager = OmniRewardLoopManager.__new__(OmniRewardLoopManager)
    assert manager.get_native_scorer_stats() == {}
    manager._reward_worker_groups = {
        "native": [worker("w0", {"a": {"batches": 1}, "b": {"batches": 2}}), worker("w1", {"a": {"batches": 3}})],
        "shared": [worker("w2", {})],
    }
    assert manager.get_native_scorer_stats() == {
        "native": [{"a": {"batches": 1}, "b": {"batches": 2}}, {"a": {"batches": 3}}],
        "shared": [{}],
    }
    assert calls == ["w0", "w1", "w2"]
