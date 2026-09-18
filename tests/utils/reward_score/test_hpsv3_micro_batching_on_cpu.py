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
"""CPU tests for frame-bounded HPSv3 reward batching."""

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

import pytest
import torch


def _load_module():
    stubs = {}
    if importlib.util.find_spec("verl") is None:
        verl = ModuleType("verl")
        verl_utils = ModuleType("verl.utils")
        verl_device = ModuleType("verl.utils.device")
        verl_device.get_device_name = lambda: "cpu"
        stubs = {
            "verl": verl,
            "verl.utils": verl_utils,
            "verl.utils.device": verl_device,
        }
        sys.modules.update(stubs)

    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/hpsv3_reward.py"
    spec = importlib.util.spec_from_file_location("hpsv3_reward_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        for name in stubs:
            sys.modules.pop(name, None)
    return module


hpsv3_reward = _load_module()


class _FakeInferencer:
    def __init__(self):
        self.batches = []

    def reward(self, images, prompts):
        self.batches.append((list(images), list(prompts)))
        raw_scores = []
        for image in images:
            pixel = image.getpixel((0, 0))
            raw_scores.append(float(pixel[0] if isinstance(pixel, tuple) else pixel))
        return torch.tensor([[score, -score] for score in raw_scores])


def _image(value: int) -> torch.Tensor:
    return torch.full((3, 2, 2), value, dtype=torch.uint8)


def _video(values) -> torch.Tensor:
    return torch.stack([_image(value) for value in values])


def _reset_consumer(monkeypatch, inferencer):
    monkeypatch.delenv("custom_reward_model_path", raising=False)
    monkeypatch.setattr(hpsv3_reward, "_BATCHING_STATE", threading.local())
    monkeypatch.setattr(hpsv3_reward, "_get_inferencer", lambda checkpoint_path, device: inferencer)
    return hpsv3_reward._get_batching_state()


async def _stop_consumer(state):
    state.queue.put_nowait(None)
    await state.consumer_task


def _request(future, value=1):
    return hpsv3_reward._ScoreRequest(
        prompt="prompt",
        solution_image=_image(value),
        extra_info={},
        checkpoint_path="fake-checkpoint",
        device="cpu",
        max_batch_size=4,
        reward_scale=0.1,
        future=future,
    )


def _score(
    solution_image,
    *,
    prompt="prompt",
    model_name="fake-checkpoint",
    device="cpu",
    max_batch_size=4,
    reward_scale=0.1,
    extra_info=None,
):
    return hpsv3_reward.compute_score_hpsv3(
        data_source="test",
        solution_image=solution_image,
        ground_truth=prompt,
        extra_info=extra_info or {},
        model_name=model_name,
        device=device,
        max_batch_size=max_batch_size,
        reward_scale=reward_scale,
    )


@pytest.mark.asyncio
async def test_score_queue_is_bounded(monkeypatch):
    state = _reset_consumer(monkeypatch, _FakeInferencer())
    assert state.queue.maxsize == hpsv3_reward._MAX_QUEUED_REQUESTS


def test_to_pil_preserves_uint8_pixel_values():
    image = hpsv3_reward._to_pil_hwc(_image(173))

    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (173, 173, 173)


@pytest.mark.asyncio
async def test_consumer_batches_burst_requests_and_preserves_order(monkeypatch):
    inferencer = _FakeInferencer()
    state = _reset_consumer(monkeypatch, inferencer)

    results = await asyncio.gather(*(_score(_image(index), prompt=f"prompt-{index}") for index in range(4)))
    await _stop_consumer(state)

    assert [len(images) for images, _ in inferencer.batches] == [4]
    assert inferencer.batches[0][1] == [f"prompt-{index}" for index in range(4)]
    assert [result["hpsv3_raw"] for result in results] == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert [result["score"] for result in results] == pytest.approx([0.0, 0.1, 0.2, 0.3])


@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.asyncio
async def test_single_video_is_split_by_total_frame_cap_and_averaged(monkeypatch, channels_last):
    inferencer = _FakeInferencer()
    state = _reset_consumer(monkeypatch, inferencer)
    video = _video([1, 2, 3, 4, 5])
    if channels_last:
        video = video.permute(0, 2, 3, 1)

    result = await _score(
        video,
        max_batch_size=2,
        reward_scale=0.5,
        extra_info={"frame_interval": 1},
    )
    await _stop_consumer(state)

    assert [len(images) for images, _ in inferencer.batches] == [2, 2, 1]
    assert result == {"score": pytest.approx(1.5), "hpsv3_raw": pytest.approx(3.0)}


@pytest.mark.parametrize("channels_last", [False, True])
def test_extract_frames_samples_the_time_axis(channels_last):
    video = _video([1, 2, 3, 4, 5])
    if channels_last:
        video = video.permute(0, 2, 3, 1)

    frames = hpsv3_reward._extract_frames(video, frame_interval=2)

    assert [frame.getpixel((0, 0))[0] for frame in frames] == [1, 3, 5]


def test_extract_frames_prefers_tchw_when_width_is_channel_sized():
    video = torch.stack([torch.full((3, 2, 3), value, dtype=torch.uint8) for value in [1, 2, 3]])

    frames = hpsv3_reward._extract_frames(video)

    assert [frame.size for frame in frames] == [(3, 2)] * 3
    assert [frame.getpixel((0, 0))[0] for frame in frames] == [1, 2, 3]


def test_extract_frames_preserves_legacy_cthw_direct_calls():
    video = _video([1, 2, 3, 4, 5]).permute(1, 0, 2, 3)

    frames = hpsv3_reward._extract_frames(video, frame_interval=2)

    assert [frame.getpixel((0, 0))[0] for frame in frames] == [1, 3, 5]


@pytest.mark.asyncio
async def test_invalid_request_is_isolated_from_valid_batch(monkeypatch):
    inferencer = _FakeInferencer()
    state = _reset_consumer(monkeypatch, inferencer)

    results = await asyncio.gather(_score(_image(7)), _score(object()), _score(_image(9)), return_exceptions=True)
    await _stop_consumer(state)

    assert results[0] == {"score": pytest.approx(0.7), "hpsv3_raw": pytest.approx(7.0)}
    assert isinstance(results[1], AttributeError)
    assert results[2] == {"score": pytest.approx(0.9), "hpsv3_raw": pytest.approx(9.0)}
    assert [len(images) for images, _ in inferencer.batches] == [2]


@pytest.mark.asyncio
async def test_consumer_groups_frames_by_checkpoint_device_and_batch_cap(monkeypatch):
    inferencers = {
        ("model-a", "cpu"): _FakeInferencer(),
        ("model-a", "cuda"): _FakeInferencer(),
        ("model-b", "cpu"): _FakeInferencer(),
    }
    state = _reset_consumer(monkeypatch, inferencers[("model-a", "cpu")])
    load_calls = []

    def get_inferencer(checkpoint_path, device):
        load_calls.append((checkpoint_path, device))
        return inferencers[(checkpoint_path, device)]

    monkeypatch.setattr(hpsv3_reward, "_get_inferencer", get_inferencer)
    results = await asyncio.gather(
        _score(_image(1), prompt="a-1", model_name="model-a", max_batch_size=2),
        _score(_image(2), prompt="a-2", model_name="model-a", max_batch_size=2),
        _score(_image(3), prompt="a-cap-1", model_name="model-a", max_batch_size=1),
        _score(_image(4), prompt="a-cuda", model_name="model-a", device="cuda", max_batch_size=2),
        _score(_image(5), prompt="b-cpu", model_name="model-b", max_batch_size=2),
    )
    await _stop_consumer(state)

    assert load_calls == [("model-a", "cpu"), ("model-a", "cpu"), ("model-a", "cuda"), ("model-b", "cpu")]
    assert [prompts for _, prompts in inferencers[("model-a", "cpu")].batches] == [
        ["a-1", "a-2"],
        ["a-cap-1"],
    ]
    assert inferencers[("model-a", "cuda")].batches[0][1] == ["a-cuda"]
    assert inferencers[("model-b", "cpu")].batches[0][1] == ["b-cpu"]
    assert [result["hpsv3_raw"] for result in results] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


@pytest.mark.parametrize("max_batch_size", [0, -1, 1.5, True])
@pytest.mark.asyncio
async def test_max_batch_size_must_be_a_positive_integer(max_batch_size):
    with pytest.raises(ValueError, match="max_batch_size must be a positive integer"):
        await _score(_image(1), max_batch_size=max_batch_size)


@pytest.mark.parametrize("termination", ["finished", "cancelled"])
@pytest.mark.asyncio
async def test_ensure_consumer_restarts_stopped_task(monkeypatch, termination):
    inferencer = _FakeInferencer()
    state = _reset_consumer(monkeypatch, inferencer)
    await hpsv3_reward._ensure_consumer(state)
    first_task = state.consumer_task

    if termination == "finished":
        await state.queue.put(None)
        await first_task
    else:
        first_task.cancel()
        await asyncio.gather(first_task, return_exceptions=True)

    await hpsv3_reward._ensure_consumer(state)
    second_task = state.consumer_task

    assert second_task is not first_task
    assert not second_task.done()
    await _stop_consumer(state)


@pytest.mark.parametrize("active_request", [False, True])
@pytest.mark.asyncio
async def test_graceful_stop_settles_requests_queued_after_sentinel(monkeypatch, active_request):
    state = _reset_consumer(monkeypatch, _FakeInferencer())
    loop = asyncio.get_running_loop()
    active_future = loop.create_future()
    trailing_future = loop.create_future()

    monkeypatch.setattr(
        hpsv3_reward,
        "_score_batch",
        lambda requests: [{"score": 0.1, "hpsv3_raw": 1.0}] * len(requests),
    )
    if active_request:
        state.queue.put_nowait(_request(active_future))
    state.queue.put_nowait(None)
    state.queue.put_nowait(_request(trailing_future))

    await hpsv3_reward._consumer_loop(state)

    if active_request:
        assert await active_future == {"score": 0.1, "hpsv3_raw": 1.0}
    with pytest.raises(RuntimeError, match="stopped before completing inference"):
        await trailing_future
    assert state.queue.empty()


@pytest.mark.asyncio
async def test_active_inference_cancellation_settles_all_requests(monkeypatch):
    monkeypatch.setattr(hpsv3_reward, "_MAX_QUEUED_REQUESTS", 2)
    inferencer = _FakeInferencer()
    state = _reset_consumer(monkeypatch, inferencer)
    inference_started = threading.Event()
    release_inference = threading.Event()

    def blocking_score_batch(requests):
        inference_started.set()
        assert release_inference.wait(timeout=5)
        return [{"score": 0.1, "hpsv3_raw": 1.0}] * len(requests)

    monkeypatch.setattr(hpsv3_reward, "_score_batch", blocking_score_batch)
    callers = [asyncio.create_task(_score(_image(index), max_batch_size=2)) for index in range(4)]
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
    monkeypatch.setattr(hpsv3_reward, "_BATCHING_STATE", threading.local())
    monkeypatch.setattr(
        hpsv3_reward,
        "_score_batch",
        lambda requests: [{"score": 0.1, "hpsv3_raw": 1.0}] * len(requests),
    )

    async def run_once():
        result = await _score(_image(1))
        state = hpsv3_reward._get_batching_state()
        await _stop_consumer(state)
        return result, state

    first_result, first_state = asyncio.run(run_once())
    second_result, second_state = asyncio.run(run_once())

    assert first_result == second_result == {"score": 0.1, "hpsv3_raw": 1.0}
    assert first_state is not second_state
