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
"""Controller lifecycle serialization and accepted-RPC settlement tests."""

import asyncio
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl_omni.reward_loop.reward_model import NativeManagedRewardModel


class _ObservedLock(asyncio.Lock):
    def __init__(self):
        super().__init__()
        self.waiting = asyncio.Event()

    async def acquire(self):
        if self.locked():
            self.waiting.set()
        return await super().acquire()


def _model(workers, offload=True):
    model = NativeManagedRewardModel(
        "native",
        OmegaConf.create(
            {
                "backend": "native",
                "offload": offload,
                "placement": {"devices": [0]},
                "executor": {"model": "tests.fake:Model"},
            }
        ),
    )
    model.bind_workers(workers)
    model._lifecycle_lock = _ObservedLock()
    return model


def _worker(remote):
    return SimpleNamespace(
        **{
            f"{method}_reward_model": SimpleNamespace(remote=lambda name, method=method: remote(method))
            for method in ("wake_up", "sleep", "close")
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("wake_up", "sleep"),
        ("wake_up", "close"),
        ("sleep", "wake_up"),
        ("sleep", "close"),
        ("close", "wake_up"),
        ("close", "sleep"),
        ("close", "close"),
    ],
)
async def test_manager_lifecycle_calls_hold_lock_until_all_workers_settle(first, second):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    finished = [asyncio.Event(), asyncio.Event()]
    calls = []

    def remote(index, method):
        calls.append((index, method))

        async def run():
            if method == first and not finished[index].is_set():
                entered[index].set()
                await release[index].wait()
                finished[index].set()

        return run()

    model = _model([_worker(lambda method, i=i: remote(i, method)) for i in range(2)])
    first_call = asyncio.create_task(getattr(model, first)())
    second_call = None
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered)), 2)
        assert model._lifecycle_lock.locked(), "accepted worker RPCs must retain manager lifecycle ownership"
        second_call = asyncio.create_task(getattr(model, second)())
        await asyncio.wait_for(model._lifecycle_lock.waiting.wait(), 2)
        assert calls == [(0, first), (1, first)]
        release[0].set()
        await asyncio.wait_for(finished[0].wait(), 2)
        assert not first_call.done() and not second_call.done()
        assert model._lifecycle_lock.locked()
        release[1].set()
        await asyncio.wait_for(first_call, 2)
        if first == "close" and second == "wake_up":
            with pytest.raises(RuntimeError, match="closed"):
                await asyncio.wait_for(second_call, 2)
        else:
            await asyncio.wait_for(second_call, 2)
        expected = [(0, first), (1, first)]
        if first != "close":
            expected += [(0, second), (1, second)]
        assert calls == expected
        assert model._closed == ("close" in (first, second))
        assert model._resident == (first != "close" and second == "wake_up")
        assert not model._lifecycle_lock.locked()
    finally:
        for event in release:
            event.set()
        await asyncio.gather(*(task for task in (first_call, second_call) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "sleep", "close"])
async def test_repeated_caller_cancellation_keeps_manager_lock_until_rpc_settles(method):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    def remote(operation):
        calls.append(operation)

        async def run():
            if operation == method and len(calls) == 1:
                entered.set()
                await release.wait()

        return run()

    model = _model([_worker(remote)])
    active = asyncio.create_task(getattr(model, method)())
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        active.cancel()
        closing = asyncio.create_task(model.close())
        await asyncio.wait_for(model._lifecycle_lock.waiting.wait(), 2)
        active.cancel()
        assert not active.done() and not closing.done()
        assert model._lifecycle_lock.locked()
        assert calls == [method]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(active, 2)
        await asyncio.wait_for(closing, 2)
        assert model._closed and not model._resident
        assert calls == ([method] if method == "close" else [method, "close"])
    finally:
        release.set()
        await asyncio.gather(*(task for task in (active, closing) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_concurrent_resident_wakes_skip_second_worker_rpc():
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    def remote(method):
        calls.append(method)

        async def run():
            entered.set()
            await release.wait()

        return run()

    model = _model([_worker(remote)], offload=False)
    first = asyncio.create_task(model.wake_up())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = asyncio.create_task(model.wake_up())
        await asyncio.wait_for(model._lifecycle_lock.waiting.wait(), 2)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), 2)
        assert calls == ["wake_up"] and model._resident
    finally:
        release.set()
        await asyncio.gather(*(task for task in (first, second) if task is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("synchronous_failure", [False, True])
async def test_failed_lifecycle_drains_accepted_peer_before_unlocking(synchronous_failure):
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    def delayed(method):
        async def run():
            entered.set()
            await release.wait()
            finished.set()

        return run()

    def fail(method):
        if synchronous_failure:
            raise OSError("submission failed")

        async def run():
            raise OSError("worker failed")

        return run()

    model = _model([_worker(delayed), _worker(fail)])
    waking = asyncio.create_task(model.wake_up())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert model._lifecycle_lock.locked()
        assert not model._resident and not waking.done()
        release.set()
        with pytest.raises(OSError, match="failed"):
            await asyncio.wait_for(waking, 2)
        assert finished.is_set()
        assert not model._resident and not model._lifecycle_lock.locked()
    finally:
        release.set()
        await asyncio.gather(waking, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "sleep", "close"])
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_submission_failure_stops_admission_and_drains_before_next_lifecycle(method, cancel_caller):
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    calls = []

    def remote(index, operation):
        calls.append((index, operation))
        if index == 1 and operation == method:
            raise OSError("submission failed")

        async def run():
            if index == 0 and operation == method:
                entered.set()
                await release.wait()
                finished.set()
                raise ValueError("accepted worker failed")

        return run()

    model = _model([_worker(lambda operation, i=i: remote(i, operation)) for i in range(3)])
    active = asyncio.create_task(getattr(model, method)())
    following = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        following = asyncio.create_task(model.close())
        await asyncio.wait_for(model._lifecycle_lock.waiting.wait(), 2)
        if cancel_caller:
            active.cancel()
            active.cancel()
        assert calls == [(0, method), (1, method)]
        assert model._lifecycle_lock.locked()
        assert not active.done() and not following.done()
        release.set()
        if cancel_caller:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(active, 2)
        else:
            with pytest.raises(OSError, match="submission failed"):
                await asyncio.wait_for(active, 2)
        assert finished.is_set()
        if method == "close":
            with pytest.raises(OSError, match="submission failed"):
                await asyncio.wait_for(following, 2)
        else:
            await asyncio.wait_for(following, 2)
        assert not model._lifecycle_lock.locked()
    finally:
        release.set()
        await asyncio.gather(*(task for task in (active, following) if task is not None), return_exceptions=True)
