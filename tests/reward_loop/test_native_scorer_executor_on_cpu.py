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

import pytest

from verl_omni.reward_loop import reward_model_executor as executor_module
from verl_omni.reward_loop.reward_model_executor import NativeRewardExecutor
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import register_scorer
from verl_omni.workers.config.reward import RewardModelSpec

_TEST_MODULE = "tests.reward_loop.test_native_scorer_executor_on_cpu"


@register_scorer("cpu-fake-scorer", modality="test")
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


def _executor(monkeypatch, model_ref: str, kwargs: dict | None = None) -> NativeRewardExecutor:
    monkeypatch.setattr(executor_module, "get_device_name", lambda: "cpu")
    monkeypatch.setattr(executor_module, "get_device_id", lambda: 0)
    return NativeRewardExecutor(
        RewardModelSpec(name="fake", backend="native", executor_config={"model": model_ref, "kwargs": kwargs or {}})
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
    assert executor._microbatcher.state == "paused"
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
async def test_legacy_native_model_still_serves_infer(monkeypatch):
    executor = _executor(monkeypatch, f"{_TEST_MODULE}:LegacyModel")
    await executor.wake_up()
    try:
        with pytest.raises(TypeError, match="infer"):
            await executor.score({"i": 0})
        assert await executor.infer(prompts=["p"], images=["i"]) == [{"score": 1}]
    finally:
        await executor.close()


@pytest.mark.asyncio
async def test_executors_on_the_same_device_share_one_forward_lock(monkeypatch):
    first = _executor(monkeypatch, "cpu-fake-scorer")
    second = _executor(monkeypatch, "cpu-fake-scorer")
    await asyncio.gather(first.wake_up(), second.wake_up())
    try:
        assert first._forward_lock is second._forward_lock
    finally:
        await asyncio.gather(first.close(), second.close())
