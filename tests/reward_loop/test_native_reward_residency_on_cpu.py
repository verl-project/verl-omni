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
"""State and cancellation tests for native reward residency."""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopWorker
from verl_omni.reward_loop.reward_model import MultiRewardModelManager, NativeManagedRewardModel
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.workers.config.reward import RewardModelSpec


def _executor(monkeypatch, model_class):
    monkeypatch.setattr(executor_module, "_load_native_model", lambda _: model_class)
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(executor_module, "get_device_id", lambda: 0)
    return NativeRewardExecutor(
        RewardModelSpec(
            name="native",
            backend="native",
            model_path=None,
            router_address=None,
            executor_config={"model": "tests.fake:Model"},
        )
    )


@pytest.mark.asyncio
async def test_owned_failure_after_repeated_cancellation_is_consumed():
    started = asyncio.Event()
    release = asyncio.Event()
    settled = asyncio.Event()

    async def fail():
        started.set()
        try:
            await release.wait()
            raise OSError("owned failure after cancellation")
        finally:
            settled.set()

    caller = asyncio.create_task(executor_module._await_owned(fail()))
    await started.wait()
    caller.cancel()
    await asyncio.sleep(0)
    caller.cancel()
    assert not caller.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert settled.is_set()


@pytest.mark.asyncio
async def test_retained_model_drains_cancelled_inference_and_preserves_identity(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    class Model:
        constructions = 0
        supports_cpu_offload = True

        def __init__(self, **kwargs):
            Model.constructions += 1

        async def infer(self):
            started.set()
            await release.wait()
            return 1

        async def sleep(self):
            self.asleep = True

        async def wake_up(self):
            self.asleep = False

        async def close(self):
            self.closed = True

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    original = executor._model
    inference = asyncio.create_task(executor.infer())
    await started.wait()
    inference.cancel()
    sleep = asyncio.create_task(executor.sleep())

    async def wait_for_fence():
        while executor._awake:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_for_fence(), 2)
        assert not sleep.done()
        with pytest.raises(RuntimeError, match="not awake"):
            await executor.infer()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await inference
    await sleep
    assert original.asleep
    await executor.wake_up()
    assert executor._model is original
    assert await executor.infer() == 1
    await executor.sleep()
    await executor.close()
    assert original.closed
    assert Model.constructions == 1
    with pytest.raises(RuntimeError, match="closed"):
        await executor.wake_up()


@pytest.mark.asyncio
async def test_transfer_failure_fails_closed(monkeypatch):
    class Model:
        supports_cpu_offload = True

        def __init__(self, **kwargs):
            pass

        async def sleep(self):
            raise OSError("move failed")

        async def wake_up(self):
            pass

        async def close(self):
            self.closed = True

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    model = executor._model
    with pytest.raises(OSError, match="move failed"):
        await executor.sleep()
    assert model.closed and executor._closed
    with pytest.raises(RuntimeError, match="not awake"):
        await executor.infer()


@pytest.mark.asyncio
async def test_sync_thread_is_drained_after_repeated_cancellation(monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    class Model:
        def __init__(self, **kwargs):
            self.closed = False

        def infer(self):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test did not release inference")
            return 2

        def close(self):
            self.closed = True

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    model = executor._model
    inference = asyncio.create_task(executor.infer())
    assert await asyncio.to_thread(entered.wait, 2)
    inference.cancel()
    await asyncio.sleep(0)
    inference.cancel()
    sleep = asyncio.create_task(executor.sleep())
    await asyncio.sleep(0)
    assert not sleep.done() and not model.closed
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await inference
    await sleep
    assert model.closed


@pytest.mark.asyncio
async def test_default_sleep_reconstructs_and_final_close(monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            self.closed = False

        def close(self):
            self.closed = True

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    first = executor._model
    await executor.sleep()
    assert first.closed and executor._model is None
    await executor.wake_up()
    assert executor._model is not first
    await executor.close()
    assert executor._closed


@pytest.mark.asyncio
async def test_worker_and_manager_forward_final_close_to_native_only():
    worker = object.__new__(OmniRewardLoopWorker)
    native = SimpleNamespace(close=AsyncMock())
    worker.native_reward_executors = {"native": native}
    await worker.close_reward_model()
    native.close.assert_awaited_once()

    model = object.__new__(NativeManagedRewardModel)
    model.spec = SimpleNamespace(name="native")
    model._workers = [SimpleNamespace(close_reward_model=SimpleNamespace(remote=lambda name: asyncio.sleep(0)))]
    model._closed = False
    model._resident = True
    model._lifecycle_lock = asyncio.Lock()
    manager = object.__new__(MultiRewardModelManager)
    manager.models = {"native": model, "engine": SimpleNamespace(close=AsyncMock())}
    await manager.close_native_models()
    assert model._closed and not model._resident
    manager.models["engine"].close.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_final_close_waits_for_all_native_executors():
    settled = asyncio.Event()

    async def fail():
        raise OSError("close failed")

    async def finish():
        await asyncio.sleep(0)
        settled.set()

    worker = object.__new__(OmniRewardLoopWorker)
    worker.native_reward_executors = {
        "first": SimpleNamespace(close=fail),
        "second": SimpleNamespace(close=finish),
    }
    with pytest.raises(OSError, match="close failed"):
        await worker.close_reward_model()
    assert settled.is_set()


@pytest.mark.asyncio
async def test_controller_waits_for_all_worker_refs_before_reporting_failure():
    release = asyncio.Event()
    settled = asyncio.Event()

    async def fail():
        raise OSError("first worker failed")

    async def delayed():
        await release.wait()
        settled.set()

    model = object.__new__(NativeManagedRewardModel)
    model.spec = SimpleNamespace(name="native")
    model._workers = [
        SimpleNamespace(close_reward_model=SimpleNamespace(remote=lambda name: fail())),
        SimpleNamespace(close_reward_model=SimpleNamespace(remote=lambda name: delayed())),
    ]
    model._resident = True
    model._closed = False
    model._lifecycle_lock = asyncio.Lock()
    closing = asyncio.create_task(model.close())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(OSError, match="first worker failed"):
        await closing
    assert settled.is_set()


@pytest.mark.asyncio
async def test_repeated_manager_close_cancellation_waits_for_worker_refs():
    release = asyncio.Event()
    started = asyncio.Event()
    settled = []

    async def delayed(index):
        started.set()
        await release.wait()
        settled.append(index)

    model = object.__new__(NativeManagedRewardModel)
    model.spec = SimpleNamespace(name="native")
    model._workers = [
        SimpleNamespace(close_reward_model=SimpleNamespace(remote=lambda name, i=i: delayed(i))) for i in range(2)
    ]
    model._resident = True
    model._closed = False
    model._lifecycle_lock = asyncio.Lock()
    manager = object.__new__(MultiRewardModelManager)
    manager.models = {"native": model}
    closing = asyncio.create_task(manager.close_native_models())
    await started.wait()
    closing.cancel()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done() and not model._closed
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert sorted(settled) == [0, 1]
    assert model._closed and not model._resident


@pytest.mark.asyncio
async def test_cancelled_worker_close_all_waits_for_every_executor():
    release = asyncio.Event()
    started = asyncio.Event()
    settled = []

    async def delayed(index):
        started.set()
        await release.wait()
        settled.append(index)

    worker = object.__new__(OmniRewardLoopWorker)
    worker.native_reward_executors = {str(i): SimpleNamespace(close=lambda i=i: delayed(i)) for i in range(2)}
    closing = asyncio.create_task(worker.close_reward_model())
    await started.wait()
    closing.cancel()
    closing.cancel()
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert sorted(settled) == [0, 1]


@pytest.mark.asyncio
async def test_failed_final_close_cannot_be_reported_as_success_by_controller(monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1
            raise OSError("close failed")

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    original = executor._model
    model = object.__new__(NativeManagedRewardModel)
    model.spec = SimpleNamespace(name="native")
    model._workers = [SimpleNamespace(close_reward_model=SimpleNamespace(remote=lambda name: executor.close()))]
    model._resident = True
    model._closed = False
    model._lifecycle_lock = asyncio.Lock()
    with pytest.raises(OSError, match="close failed"):
        await model.close()
    with pytest.raises(RuntimeError, match="cleanup previously failed"):
        await model.close()
    assert not model._closed and model._resident
    assert executor._closed and executor._model is None
    assert original.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        await executor.wake_up()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["wake_up", "sleep"])
async def test_transfer_error_survives_failed_cleanup(monkeypatch, caplog, method):
    class Model:
        supports_cpu_offload = True

        def __init__(self, **kwargs):
            self.close_calls = 0

        async def sleep(self):
            if method == "sleep":
                raise OSError("transfer failed")

        async def wake_up(self):
            raise OSError("transfer failed")

        async def close(self):
            self.close_calls += 1
            raise ValueError("cleanup failed")

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    original = executor._model
    if method == "wake_up":
        await executor.sleep()
    with pytest.raises(OSError, match="transfer failed"):
        await getattr(executor, method)()
    assert "cleanup failed" in caplog.text
    with pytest.raises(RuntimeError, match="cleanup previously failed"):
        await executor.close()
    assert original.close_calls == 1
    assert executor._closed and not executor._awake


@pytest.mark.asyncio
async def test_reconstruction_sleep_preserves_failed_disposal_state(monkeypatch):
    class Model:
        def __init__(self, **kwargs):
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1
            raise OSError("disposal failed")

    executor = _executor(monkeypatch, Model)
    await executor.wake_up()
    original = executor._model
    with pytest.raises(OSError, match="disposal failed"):
        await executor.sleep()
    with pytest.raises(RuntimeError, match="cleanup previously failed"):
        await executor.close()
    assert executor._closed and original.close_calls == 1
