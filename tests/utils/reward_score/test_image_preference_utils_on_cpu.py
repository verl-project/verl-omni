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

"""CPU tests for the helpers shared by the local image preference scorers."""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from verl_omni.utils.reward_score.image_preference_utils import RequestBatcher, clip_feature_tensor, to_rgb_image


def test_clip_feature_tensor_unwraps_model_outputs() -> None:
    embeds = torch.ones(2, 4)

    assert clip_feature_tensor(embeds) is embeds
    assert clip_feature_tensor(SimpleNamespace(image_embeds=embeds)) is embeds
    assert clip_feature_tensor(SimpleNamespace(image_embeds=None, text_embeds=embeds)) is embeds
    with pytest.raises(TypeError, match="Unsupported CLIP feature"):
        clip_feature_tensor(SimpleNamespace())


@pytest.mark.parametrize("channels", [1, 3, 4])
def test_to_rgb_image_accepts_chw_tensors(channels) -> None:
    image = to_rgb_image(torch.full((channels, 2, 3), 7, dtype=torch.uint8))

    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (7, 7, 7)


def test_to_rgb_image_accepts_hwc_arrays_and_pil_images() -> None:
    assert to_rgb_image(np.zeros((2, 3, 3), dtype=np.uint8)).size == (3, 2)
    assert to_rgb_image(Image.new("RGBA", (2, 2))).mode == "RGB"
    with pytest.raises(TypeError, match="Unsupported image type"):
        to_rgb_image("not an image")


def test_request_batcher_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError, match="max_batch_size must be positive"):
        RequestBatcher(lambda payloads: payloads, max_batch_size=0)


@pytest.mark.asyncio
async def test_request_batcher_batches_bursts_and_isolates_item_errors() -> None:
    batches = []

    def process(payloads):
        batches.append(list(payloads))
        return [ValueError(f"bad {payload}") if payload == 2 else payload * 10 for payload in payloads]

    batcher = RequestBatcher(process, max_batch_size=3)
    results = await asyncio.gather(*(batcher.submit([index]) for index in range(5)), return_exceptions=True)
    await batcher.stop()

    assert results[:2] == [[0], [10]]
    assert isinstance(results[2], ValueError)
    assert results[3:] == [[30], [40]]
    assert batches == [[0, 1, 2], [3, 4]]
    assert batcher.task is None


@pytest.mark.asyncio
async def test_request_batcher_cancellation_cancels_in_flight_and_queued_requests() -> None:
    started, release = threading.Event(), threading.Event()

    def blocking_process(payloads):
        started.set()
        release.wait(timeout=5)
        return payloads

    executor = ThreadPoolExecutor(max_workers=1)
    batcher = RequestBatcher(blocking_process, max_batch_size=1, executor=executor)
    in_flight = asyncio.create_task(batcher.submit(["first"]))
    while not started.is_set():
        await asyncio.sleep(0.01)
    queued = asyncio.create_task(batcher.submit(["second"]))
    await asyncio.sleep(0)

    batcher.task.cancel()
    results = await asyncio.wait_for(asyncio.gather(in_flight, queued, return_exceptions=True), timeout=5)
    release.set()

    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert await batcher.submit(["restarted"]) == ["restarted"]
    await batcher.stop()
    executor.shutdown(wait=True)
