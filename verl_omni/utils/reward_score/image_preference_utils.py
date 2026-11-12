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

"""Plumbing shared by the local image preference scorers (PickScore, CLIPScore, preference mix)."""

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor
from typing import Any

import numpy as np
import torch
from PIL import Image

from verl_omni.utils.reward_score.reward_utils import image_tensor_to_pil, visual_reward_frames

_STOP = object()


def clip_feature_tensor(features) -> torch.Tensor:
    """Return the embedding tensor from a CLIP ``get_*_features`` result."""
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    raise TypeError(f"Unsupported CLIP feature return type: {type(features)!r}")


def to_rgb_image(image) -> Image.Image:
    """Convert a PIL image or a CHW/HWC array or tensor to an RGB PIL image."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image[..., 0]
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image).__name__}")
    return image.convert("RGB")


def decoded_reward_image(solution_image, extra_info: dict | None, *, scorer: str) -> Image.Image:
    """Select the one declared decoded image; latent responses are rejected."""
    frames = visual_reward_frames(solution_image, extra_info or {})
    if len(frames) != 1:
        raise ValueError(f"{scorer} requires a single decoded image")
    return image_tensor_to_pil(frames[0])


class RequestBatcher:
    """Collect concurrent single-item requests on one event loop into local inference batches.

    ``process_batch`` runs in ``executor`` and returns one result per payload; a
    result that is an exception fails only its own request. The consumer task is
    started on demand and restarted by the next request after it stops.
    """

    def __init__(
        self,
        process_batch: Callable[[list[Any]], list[Any]],
        *,
        max_batch_size: int,
        executor: Executor | None = None,
    ):
        if max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
        self._process_batch = process_batch
        self._max_batch_size = max_batch_size
        self._executor = executor
        self._queue = asyncio.Queue()
        self.task: asyncio.Task | None = None

    async def submit(self, payloads: list[Any]) -> list[Any]:
        """Queue ``payloads`` together and return their results in order."""
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self._consumer_loop())
        loop = asyncio.get_running_loop()
        futures = []
        for payload in payloads:
            future = loop.create_future()
            futures.append(future)
            self._queue.put_nowait((payload, future))
        return await asyncio.gather(*futures)

    async def stop(self) -> None:
        """Finish queued requests, including an in-flight batch, then stop the consumer."""
        task = self.task
        if task is not None and not task.done():
            self._queue.put_nowait(_STOP)
            await task
        self.task = None

    def _next_batch(self, first) -> tuple[list, bool]:
        batch = [first]
        while len(batch) < self._max_batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is _STOP:
                return batch, True
            batch.append(item)
        return batch, False

    def _cancel_queued(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is not _STOP:
                item[1].cancel()

    async def _consumer_loop(self) -> None:
        loop = asyncio.get_running_loop()
        batch = []
        try:
            while True:
                item = await self._queue.get()
                if item is _STOP:
                    return
                batch = [item]
                await asyncio.sleep(0)
                batch, should_stop = self._next_batch(item)
                try:
                    results = await loop.run_in_executor(
                        self._executor, self._process_batch, [payload for payload, _ in batch]
                    )
                except Exception as error:
                    results = [error] * len(batch)
                for (_, future), result in zip(batch, results, strict=True):
                    if future.done():
                        continue
                    if isinstance(result, BaseException):
                        future.set_exception(result)
                    else:
                        future.set_result(result)
                batch = []
                if should_stop:
                    return
        except asyncio.CancelledError:
            # Requests already taken or still queued would otherwise wait for a later restart.
            for _, future in batch:
                future.cancel()
            self._cancel_queued()
            raise
