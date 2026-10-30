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
"""NativeRewardExecutor hosting of class-based scorers (protocol + microbatcher)."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_loop import OmniRewardLoopManager
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import register_scorer
from verl_omni.workers.config.reward import RewardModelSpec

_TEST_MODULE = "tests.reward_loop.test_native_scorer_executor_on_cpu"


@register_scorer("cpu-fake-scorer")
class FakeScorer(RewardScorer):
    """Records lifecycle calls; scores request kwarg ``i`` back as ``score``."""

    max_batch_size = 4
    supports_cpu_offload = True

    instances = 0

    def __init__(self, fail_load: bool = False):
        self.fail_load = fail_load
        self.calls: list[str] = []
        self.batches: list[list[int]] = []
        FakeScorer.instances += 1

    def load(self, device):
        self.calls.append(f"load:{device}")
        if self.fail_load:
            raise OSError("weights unavailable")

    def score_batch(self, batch):
        self.batches.append([request["i"] for request in batch])
        return [{"score": request["i"]} for request in batch]

    def wake_up(self):
        self.calls.append("wake_up")

    def sleep(self):
        self.calls.append("sleep")

    def close(self):
        self.calls.append("close")


class NonOffloadScorer(FakeScorer):
    supports_cpu_offload = False


class ModelPathScorer(FakeScorer):
    """Accepts the top-level ``model_path`` injection like a weights-backed scorer."""

    def __init__(self, model_path: str | None = None):
        super().__init__()
        self.model_path = model_path


class GatedScorer(FakeScorer):
    def __init__(self):
        super().__init__()
        self._loop = asyncio.get_running_loop()
        self.entered = asyncio.Event()
        self.release = threading.Event()

    def score_batch(self, batch):
        self.batches.append([request["i"] for request in batch])
        self._loop.call_soon_threadsafe(self.entered.set)
        # Sync score_batch runs in an executor thread; block it, not the loop.
        if not self.release.wait(timeout=30):
            raise TimeoutError("gated forward was never released")
        return [{"score": request["i"]} for request in batch]


class LoopGatedScorer(FakeScorer):
    """Gated forward synchronized only by threading primitives, so the same
    scorer instance can gate calls across repeated sync-entrypoint invocations."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def score_batch(self, batch):
        self.batches.append([request["i"] for request in batch])
        self.entered.set()
        if not self.release.wait(timeout=30):
            raise TimeoutError("gated forward was never released")
        return [{"score": request["i"]} for request in batch]


class AsyncGatedScorer(FakeScorer):
    """Async gated forward — cancellation propagates into the coroutine itself."""

    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def score_batch(self, batch):
        self.batches.append([request["i"] for request in batch])
        self.entered.set()
        await self.release.wait()
        return [{"score": request["i"]} for request in batch]


class LegacyModel:
    """Legacy native model shape: infer() + opt-in CPU offload pair."""

    supports_cpu_offload = True

    def __init__(self, device=None):
        self.device = device

    async def infer(self, prompts, images):
        return [{"score": len(prompts)}]

    async def sleep(self):
        pass

    async def wake_up(self):
        pass

    async def close(self):
        pass


class LoadFailingScorer(RewardScorer):
    """Fails load() after construction; close() must still dispose the half-built scorer."""

    max_batch_size = 4
    instances: list["LoadFailingScorer"] = []

    def __init__(self):
        self.closed = False
        LoadFailingScorer.instances.append(self)

    def load(self, device):
        raise OSError("weights unavailable")

    def score_batch(self, batch):
        return [{"score": request["i"]} for request in batch]

    def close(self):
        self.closed = True


def _executor(
    monkeypatch, model_ref: str, kwargs: dict | None = None, model_path: str | None = None
) -> NativeRewardExecutor:
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(executor_module, "get_device_id", lambda: 0)
    return NativeRewardExecutor(
        RewardModelSpec(
            name="fake",
            backend="native",
            model_path=model_path,
            executor_config={"model": model_ref, "kwargs": kwargs or {}},
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model_ref", ["cpu-fake-scorer", f"{_TEST_MODULE}:FakeScorer"])
async def test_wake_loads_scorer_and_scores_through_the_microbatcher(monkeypatch, model_ref):
    FakeScorer.instances = 0
    executor = _executor(monkeypatch, model_ref)
    await executor.wake_up()
    try:
        assert FakeScorer.instances == 1
        scorer = executor._model
        assert scorer.calls[0].startswith("load:cpu")
        assert executor.reward_kwargs() == {"reward_model": executor}

        results = await asyncio.gather(*(executor.score({"i": i}) for i in range(3)))
        assert results == [{"score": i} for i in range(3)]
        assert scorer.batches == [[0, 1, 2]]
    finally:
        await executor.close()
    assert scorer.calls[-1] == "close"


@pytest.mark.asyncio
async def test_spec_model_path_forwarding_precedence_and_loud_rejection(monkeypatch):
    forwarded = _executor(monkeypatch, f"{_TEST_MODULE}:ModelPathScorer", model_path="/weights/from-spec")
    await forwarded.wake_up()
    assert forwarded._model.model_path == "/weights/from-spec"
    await forwarded.close()

    overridden = _executor(
        monkeypatch,
        f"{_TEST_MODULE}:ModelPathScorer",
        kwargs={"model_path": "/weights/from-kwargs"},
        model_path="/weights/from-spec",
    )
    await overridden.wake_up()
    assert overridden._model.model_path == "/weights/from-kwargs"
    await overridden.close()

    # FakeScorer takes no model_path; the injection must fail loudly.
    rejected = _executor(monkeypatch, "cpu-fake-scorer", model_path="/weights/from-spec")
    with pytest.raises(TypeError, match="model_path"):
        await rejected.wake_up()


@pytest.mark.asyncio
async def test_scoring_requires_awake_scorer_and_infer_is_rejected(monkeypatch):
    executor = _executor(monkeypatch, "cpu-fake-scorer")
    with pytest.raises(RuntimeError, match="not awake"):
        await executor.score({"i": 0})
    await executor.wake_up()
    try:
        with pytest.raises(TypeError, match="score"):
            await executor.infer(prompts=[], images=[])
        assert await executor.score({"i": 1}) == {"score": 1}
    finally:
        await executor.close()
    with pytest.raises(RuntimeError, match="not awake"):
        await executor.score({"i": 0})


@pytest.mark.asyncio
async def test_offload_sleep_pauses_consumer_and_reuses_instance_on_wake(monkeypatch):
    FakeScorer.instances = 0
    executor = _executor(monkeypatch, "cpu-fake-scorer")
    await executor.wake_up()
    scorer = executor._model
    assert await executor.score({"i": 0}) == {"score": 0}

    await executor.sleep()
    assert scorer.calls == [scorer.calls[0], "sleep"]
    assert executor._microbatcher._state == "paused"
    with pytest.raises(RuntimeError, match="not awake"):
        await executor.score({"i": 1})

    await executor.wake_up()
    assert scorer.calls == [scorer.calls[0], "sleep", "wake_up"]
    assert FakeScorer.instances == 1
    assert await executor.score({"i": 1}) == {"score": 1}
    await executor.close()


@pytest.mark.asyncio
async def test_non_offload_sleep_disposes_scorer_and_wake_rebuilds(monkeypatch):
    FakeScorer.instances = 0
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:NonOffloadScorer")
    await executor.wake_up()
    first = executor._model
    assert await executor.score({"i": 0}) == {"score": 0}

    await executor.sleep()
    assert first.calls[-1] == "close"
    assert executor._model is None

    await executor.wake_up()
    assert FakeScorer.instances == 2
    assert await executor.score({"i": 1}) == {"score": 1}
    await executor.close()


@pytest.mark.asyncio
async def test_failed_load_fails_closed(monkeypatch):
    executor = _executor(monkeypatch, "cpu-fake-scorer", kwargs={"fail_load": True})
    with pytest.raises(OSError, match="weights unavailable"):
        await executor.wake_up()
    assert executor._closed
    with pytest.raises(RuntimeError, match="closed"):
        await executor.wake_up()


@pytest.mark.asyncio
async def test_failed_load_disposes_the_half_built_scorer(monkeypatch):
    LoadFailingScorer.instances = []
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:LoadFailingScorer")
    with pytest.raises(OSError, match="weights unavailable"):
        await executor.wake_up()
    assert executor._closed
    assert LoadFailingScorer.instances[0].closed


@pytest.mark.asyncio
async def test_sleep_waits_for_the_in_flight_forward_before_offload(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:GatedScorer")
    await executor.wake_up()
    scorer = executor._model
    scoring = asyncio.create_task(executor.score({"i": 0}))
    await asyncio.wait_for(scorer.entered.wait(), 5)
    sleeping = asyncio.create_task(executor.sleep())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not sleeping.done(), "sleep must wait for the in-flight forward, not abandon it"

    scorer.release.set()
    assert await scoring == {"score": 0}
    await sleeping
    assert scorer.calls[-1] == "sleep"
    await executor.close()


@pytest.mark.asyncio
async def test_sleep_fails_queued_requests_retryably_and_offloads_after_the_accepted_forward(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:GatedScorer")
    await executor.wake_up()
    scorer = executor._model
    accepted = asyncio.create_task(executor.score({"i": 0}))
    await asyncio.wait_for(scorer.entered.wait(), 5)
    queued = [asyncio.create_task(executor.score({"i": i})) for i in (1, 2)]

    sleeping = asyncio.create_task(executor.sleep())
    outcomes = await asyncio.gather(*queued, return_exceptions=True)
    assert all(isinstance(outcome, RuntimeError) and "retry after wake_up" in str(outcome) for outcome in outcomes)
    for _ in range(10):
        await asyncio.sleep(0)
    assert not sleeping.done(), "weights must not offload while the accepted forward runs"
    assert "sleep" not in scorer.calls

    scorer.release.set()
    assert await accepted == {"score": 0}
    await sleeping
    assert scorer.batches == [[0]], "queued requests must fail, not score"
    assert scorer.calls[-1] == "sleep"
    await executor.close()


@pytest.mark.asyncio
async def test_close_fails_queued_requests_and_closes_after_the_accepted_forward(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:GatedScorer")
    await executor.wake_up()
    scorer = executor._model
    accepted = asyncio.create_task(executor.score({"i": 0}))
    await asyncio.wait_for(scorer.entered.wait(), 5)
    queued = [asyncio.create_task(executor.score({"i": i})) for i in (1, 2)]

    closing = asyncio.create_task(executor.close())
    outcomes = await asyncio.gather(*queued, return_exceptions=True)
    assert all(
        isinstance(outcome, RuntimeError) and "closed before this request was accepted" in str(outcome)
        for outcome in outcomes
    )
    for _ in range(10):
        await asyncio.sleep(0)
    assert not closing.done(), "teardown must join the accepted forward before disposing weights"

    scorer.release.set()
    assert await accepted == {"score": 0}
    await closing
    assert scorer.batches == [[0]], "queued requests must fail, not score"
    assert scorer.calls[-1] == "close"


@pytest.mark.asyncio
async def test_consumer_cancellation_keeps_the_sync_forward_owned_until_the_thread_settles(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:GatedScorer")
    await executor.wake_up()
    scorer = executor._model
    accepted = asyncio.create_task(executor.score({"i": 0}))
    await asyncio.wait_for(scorer.entered.wait(), 5)
    consumer = executor._microbatcher._consumer
    sleeping = asyncio.create_task(executor.sleep())

    consumer.cancel()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not accepted.done(), "the forward thread must stay owned, not failed early"
    assert not sleeping.done(), "weights must not offload while the forward thread still runs"
    assert "sleep" not in scorer.calls

    scorer.release.set()
    with pytest.raises(RuntimeError, match="consumer was cancelled"):
        await accepted
    await sleeping
    assert scorer.calls[-1] == "sleep"
    await executor.close()


@pytest.mark.asyncio
async def test_close_still_disposes_when_its_consumer_join_is_cancelled(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:AsyncGatedScorer")
    await executor.wake_up()
    scorer = executor._model
    accepted = asyncio.create_task(executor.score({"i": 0}))
    await asyncio.wait_for(scorer.entered.wait(), 5)
    consumer = executor._microbatcher._consumer

    closing = asyncio.create_task(executor.close())
    for _ in range(10):
        await asyncio.sleep(0)
    assert not closing.done(), "close must not finish while the accepted forward runs"

    consumer.cancel()
    with pytest.raises(RuntimeError, match="consumer was cancelled"):
        await accepted
    await closing
    assert scorer.calls[-1] == "close", "the scorer must be disposed despite the cancelled join"
    await executor.close(), "repeat close after successful cleanup returns cleanly"


@pytest.mark.asyncio
async def test_legacy_native_model_still_serves_infer(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:LegacyModel")
    await executor.wake_up()
    try:
        with pytest.raises(TypeError, match="infer"):
            await executor.score({"i": 0})
        assert await executor.infer(prompts=["p"], images=["i"]) == [{"score": 1}]
    finally:
        await executor.close()


def test_sync_entrypoint_pins_one_loop_and_survives_sleep_overlapping_a_forward(monkeypatch):
    """compute_rm_score() runs every call on one pinned loop, so state bound while
    a sleep overlaps an in-flight forward stays valid on the next call instead of
    raising ``bound to a different event loop``."""
    FakeScorer.instances = 0
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:LoopGatedScorer")
    manager = object.__new__(OmniRewardLoopManager)
    manager.multi_reward_model_manager = SimpleNamespace(models=["fake"])
    manager._score_lock = asyncio.Lock()
    manager._sync_loop = None

    async def async_compute_rm_score(i: int) -> dict:
        async with manager._score_lock:
            await executor.wake_up()
            scorer = executor._model
            scorer.entered.clear()
            scorer.release.clear()
            scoring = asyncio.create_task(executor.score({"i": i}))
            assert await asyncio.to_thread(scorer.entered.wait, 5), "forward never started"
            sleeping = asyncio.create_task(executor.sleep())
            for _ in range(20):
                await asyncio.sleep(0)
            assert not sleeping.done(), "sleep must join the accepted forward before offloading weights"
            scorer.release.set()
            assert await scoring == {"score": i}
            await sleeping
            return {"score": i}

    manager.async_compute_rm_score = async_compute_rm_score
    assert manager.compute_rm_score(0) == {"score": 0}
    assert manager.compute_rm_score(1) == {"score": 1}

    pinned = manager._sync_loop
    assert pinned is not None and not pinned.is_closed()
    pinned.run_until_complete(executor.close())
    assert manager._sync_loop is pinned, "the pinned loop must be reused, not recreated"
    assert FakeScorer.instances == 1


def test_empty_accelerator_cache_only_runs_on_an_available_accelerator(monkeypatch):
    calls = []

    def backend(*, has_cache: bool, available: bool):
        namespace = SimpleNamespace(is_available=lambda: available)
        if has_cache:
            namespace.empty_cache = lambda: calls.append("flushed")
        return namespace

    monkeypatch.setattr(
        executor_module,
        "torch",
        SimpleNamespace(
            cuda=backend(has_cache=True, available=True),
            cpu=backend(has_cache=False, available=False),
            mps=backend(has_cache=True, available=False),
        ),
    )
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cuda")
    executor_module._empty_accelerator_cache()
    assert calls == ["flushed"]

    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cpu")
    executor_module._empty_accelerator_cache()
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "mps")
    executor_module._empty_accelerator_cache()
    assert calls == ["flushed"], "backends without a cache or unavailable must be skipped"
