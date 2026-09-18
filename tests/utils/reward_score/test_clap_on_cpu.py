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
"""CPU tests for CLAP burst batching."""

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _load_scorer_module():
    module_path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/clap.py"
    spec = importlib.util.spec_from_file_location("clap_reward_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


clap = _load_scorer_module()


def test_get_audio_normalizes_batch_and_channels():
    audio, sample_rate = clap._get_audio(
        {
            "audio": torch.ones(1, 2, 16),
            "audio_sample_rate": torch.tensor(48_000),
        }
    )

    assert audio.shape == (16,)
    assert sample_rate == 48_000


class _FakeProcessor:
    def __init__(self):
        self.batches = []

    def __call__(self, *, text, audio, sampling_rate, **kwargs):
        assert sampling_rate == 48_000
        self.batches.append((list(text), [waveform.copy() for waveform in audio]))
        prompt_features = {"right": [1.0, 0.0], "up": [0.0, 1.0]}
        return {
            "audio_values": torch.stack([torch.from_numpy(waveform[:2]) for waveform in audio]),
            "text_values": torch.tensor([prompt_features[prompt] for prompt in text]),
        }


class _FakeModel:
    def __call__(self, audio_values, text_values):
        return SimpleNamespace(audio_embeds=audio_values, text_embeds=text_values)


def _install_fake_torchaudio(monkeypatch):
    resample_calls = []

    def resample(waveform, *, orig_freq, new_freq):
        resample_calls.append((orig_freq, new_freq))
        return waveform

    functional = ModuleType("torchaudio.functional")
    functional.resample = resample
    torchaudio = ModuleType("torchaudio")
    torchaudio.functional = functional
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(sys.modules, "torchaudio.functional", functional)
    return resample_calls


def _reset_consumer(monkeypatch):
    monkeypatch.setattr(clap, "_BATCHING_STATE", threading.local())
    return clap._get_batching_state()


async def _stop_consumer(state):
    state.queue.put_nowait((None, None, None, None, None))
    await state.consumer_task


@pytest.mark.parametrize("active_request", [False, True])
@pytest.mark.asyncio
async def test_graceful_stop_settles_requests_queued_after_sentinel(monkeypatch, active_request):
    state = _reset_consumer(monkeypatch)
    loop = asyncio.get_running_loop()
    active_future = loop.create_future()
    trailing_future = loop.create_future()
    request = ("right", {}, "model", "cpu", active_future)

    monkeypatch.setattr(clap, "_score_batch", lambda requests: [(1.0, 48_000)] * len(requests))
    if active_request:
        state.queue.put_nowait(request)
    state.queue.put_nowait((None, None, None, None, None))
    state.queue.put_nowait(("up", {}, "model", "cpu", trailing_future))

    await clap._consumer_loop(state)

    if active_request:
        assert await active_future == (1.0, 48_000)
    with pytest.raises(RuntimeError, match="stopped before completing inference"):
        await trailing_future
    assert state.queue.empty()


def _score(prompt, waveform, sample_rate=48_000, **kwargs):
    return clap.compute_score(
        data_source="test",
        solution_image=None,
        ground_truth=prompt,
        extra_info={"audio": torch.tensor(waveform), "audio_sample_rate": sample_rate},
        device="cpu",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_score_queue_is_bounded(monkeypatch):
    state = _reset_consumer(monkeypatch)
    assert state.queue.maxsize == clap._MAX_BATCH_SIZE


@pytest.mark.parametrize("termination", ["finished", "cancelled"])
@pytest.mark.asyncio
async def test_ensure_consumer_restarts_stopped_task(monkeypatch, termination):
    state = _reset_consumer(monkeypatch)
    await clap._ensure_consumer(state)
    first_task = state.consumer_task

    if termination == "finished":
        await state.queue.put((None, None, None, None, None))
        await first_task
    else:
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    await clap._ensure_consumer(state)
    second_task = state.consumer_task

    assert second_task is not first_task
    assert not second_task.done()
    await _stop_consumer(state)


@pytest.mark.asyncio
async def test_active_inference_cancellation_settles_all_requests(monkeypatch):
    monkeypatch.setattr(clap, "_MAX_BATCH_SIZE", 2)
    state = _reset_consumer(monkeypatch)
    inference_started = threading.Event()
    release_inference = threading.Event()

    def blocking_score_batch(requests):
        inference_started.set()
        assert release_inference.wait(timeout=5)
        return [(1.0, 48_000)] * len(requests)

    monkeypatch.setattr(clap, "_score_batch", blocking_score_batch)
    callers = [asyncio.create_task(_score("right", [1.0, 0.0])) for _ in range(4)]
    assert await asyncio.to_thread(inference_started.wait, 5)
    for _ in range(100):
        if state.queue.qsize() == 2:
            break
        await asyncio.sleep(0.01)
    assert state.queue.qsize() == 2

    state.consumer_task.cancel()
    release_inference.set()
    results = await asyncio.wait_for(asyncio.gather(*callers, return_exceptions=True), timeout=5)
    await asyncio.gather(state.consumer_task, return_exceptions=True)

    assert all(isinstance(result, RuntimeError) for result in results)
    assert all("cancelled" in str(result) for result in results)


def test_batching_state_is_recreated_for_each_event_loop(monkeypatch):
    monkeypatch.setattr(clap, "_BATCHING_STATE", threading.local())
    monkeypatch.setattr(clap, "_score_batch", lambda requests: [(1.0, 48_000)] * len(requests))

    async def run_once():
        result = await _score("right", [1.0, 0.0])
        state = clap._get_batching_state()
        await _stop_consumer(state)
        return result, state

    first_result, first_state = asyncio.run(run_once())
    second_result, second_state = asyncio.run(run_once())

    assert first_result == second_result == {"score": 1.0, "source_sample_rate": 48_000}
    assert first_state is not second_state


@pytest.mark.asyncio
async def test_compute_score_batches_burst_requests_and_preserves_order(monkeypatch):
    _install_fake_torchaudio(monkeypatch)
    state = _reset_consumer(monkeypatch)
    processor = _FakeProcessor()
    monkeypatch.setattr(clap, "_load_clap", lambda model_name_or_path, device: (_FakeModel(), processor))

    results = await asyncio.gather(
        _score("right", [1.0, 0.0]),
        _score("up", [0.0, 1.0]),
        _score("right", [1.0, 1.0]),
        _score("up", [1.0, -1.0]),
    )
    await _stop_consumer(state)

    assert len(processor.batches) == 1
    assert processor.batches[0][0] == ["right", "up", "right", "up"]
    assert [result["score"] for result in results] == pytest.approx([1.0, 1.0, 2**-0.5, -(2**-0.5)])
    assert [result["source_sample_rate"] for result in results] == [48_000] * 4


@pytest.mark.asyncio
async def test_consumer_caps_each_micro_batch(monkeypatch):
    _install_fake_torchaudio(monkeypatch)
    state = _reset_consumer(monkeypatch)
    processor = _FakeProcessor()
    monkeypatch.setattr(clap, "_load_clap", lambda model_name_or_path, device: (_FakeModel(), processor))
    monkeypatch.setattr(clap, "_MAX_BATCH_SIZE", 2)

    results = await asyncio.gather(*(_score("right", [1.0, float(index)]) for index in range(5)))
    await _stop_consumer(state)

    assert [len(prompts) for prompts, _ in processor.batches] == [2, 2, 1]
    assert len(results) == 5


@pytest.mark.asyncio
async def test_consumer_groups_requests_by_model_and_device(monkeypatch):
    _install_fake_torchaudio(monkeypatch)
    state = _reset_consumer(monkeypatch)
    processors = {"model-a": _FakeProcessor(), "model-b": _FakeProcessor()}
    load_calls = []

    def load_clap(model_name_or_path, device):
        load_calls.append((model_name_or_path, device))
        return _FakeModel(), processors[model_name_or_path]

    monkeypatch.setattr(clap, "_load_clap", load_clap)
    results = await asyncio.gather(
        _score("right", [1.0, 0.0], model_name_or_path="model-a"),
        _score("up", [0.0, 1.0], model_name_or_path="model-b"),
        _score("up", [0.0, 1.0], model_name_or_path="model-a"),
        _score("right", [1.0, 0.0], model_name_or_path="model-b"),
    )
    await _stop_consumer(state)

    assert load_calls == [("model-a", "cpu"), ("model-b", "cpu")]
    assert [len(processors[name].batches[0][0]) for name in ("model-a", "model-b")] == [2, 2]
    assert [result["score"] for result in results] == pytest.approx([1.0] * 4)


@pytest.mark.asyncio
async def test_consumer_isolates_invalid_audio_and_resamples_valid_requests(monkeypatch):
    resample_calls = _install_fake_torchaudio(monkeypatch)
    state = _reset_consumer(monkeypatch)
    processor = _FakeProcessor()
    monkeypatch.setattr(clap, "_load_clap", lambda model_name_or_path, device: (_FakeModel(), processor))

    invalid = clap.compute_score(
        data_source="test",
        solution_image=None,
        ground_truth="right",
        extra_info={"audio_sample_rate": 48_000},
        device="cpu",
    )
    results = await asyncio.gather(
        _score("right", [1.0, 0.0], sample_rate=24_000),
        invalid,
        _score("up", [0.0, 1.0]),
        return_exceptions=True,
    )
    await _stop_consumer(state)

    assert results[0] == {"score": pytest.approx(1.0), "source_sample_rate": 24_000}
    assert isinstance(results[1], KeyError)
    assert results[2] == {"score": pytest.approx(1.0), "source_sample_rate": 48_000}
    assert resample_calls == [(24_000, 48_000)]
    assert len(processor.batches) == 1
    assert len(processor.batches[0][0]) == 2
