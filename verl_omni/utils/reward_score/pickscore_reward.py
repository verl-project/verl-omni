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

import asyncio
import gc
import logging
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_PROCESSOR_PATH = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
_MODEL_PATH = "yuvalkirstain/PickScore_v1"
_MAX_BATCH_SIZE = 16

_inferencer = None
_score_queue = asyncio.Queue()
_consumer_task = None
_consumer_started = False
_consumer_lock = asyncio.Lock()


def _feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    raise TypeError(f"Unsupported CLIP feature return type: {type(features)!r}")


class _PickScoreInferencer:
    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype=torch.float32,
        model_path: str = _MODEL_PATH,
        processor_path: str = _PROCESSOR_PATH,
    ):
        if device is None:
            device = torch.device(get_device_name(), get_device_id())
        logger.info("Creating PickScore model from %s", model_path)
        self.device = torch.device(device)
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device)
        self.model = self.model.to(dtype=dtype)

    @torch.no_grad()
    def score(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:
        unique_prompts = list(dict.fromkeys(prompts))
        prompt_to_index = {prompt: index for index, prompt in enumerate(unique_prompts)}
        prompt_indices = [prompt_to_index[prompt] for prompt in prompts]

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        text_inputs = self.processor(
            text=unique_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_embs = _feature_tensor(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = text_embs[prompt_indices]

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        scores = scores / 26
        return scores


class PickScoreNativeScorer:
    """Lifecycle-friendly PickScore scorer for native deployments.

    ``RewardLoopWorker.compute_score_batch`` fans a batch out into concurrent
    single-item calls.  This scorer preserves the old PickScore batching
    behavior by collecting those calls locally and running one CLIP forward for
    up to ``_MAX_BATCH_SIZE`` items.  The queue belongs to this scorer instance
    (rather than module globals), so ``NativeRewardExecutor.sleep`` can stop it
    and release the model before actor update.
    """

    def __init__(self, model_path: str = _MODEL_PATH, device=None, dtype=torch.float32, processor_path=_PROCESSOR_PATH):
        self._inferencer = _PickScoreInferencer(
            device=device,
            dtype=dtype,
            model_path=model_path,
            processor_path=processor_path,
        )
        self._score_queue = asyncio.Queue()
        self._consumer_task = None
        self._consumer_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_consumer(self):
        if self._closed:
            raise RuntimeError("PickScore scorer is closed")
        if self._consumer_task is not None and not self._consumer_task.done():
            return
        async with self._consumer_lock:
            if self._closed:
                raise RuntimeError("PickScore scorer is closed")
            if self._consumer_task is None or self._consumer_task.done():
                self._consumer_task = asyncio.create_task(self._consumer_loop())

    def _score_requests(self, requests):
        prompts = [prompt for prompt, _, _ in requests]
        images = [image for _, image, _ in requests]
        return self._inferencer.score(prompts, images).tolist()

    async def _consumer_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            request = await self._score_queue.get()
            if request[0] is None:
                return

            requests = [request]
            should_stop = False
            # Let all compute_score() tasks created by compute_score_batch reach
            # the queue before taking the rest of this micro-batch.
            await asyncio.sleep(0)
            while len(requests) < _MAX_BATCH_SIZE:
                try:
                    request = self._score_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if request[0] is None:
                    should_stop = True
                    break
                requests.append(request)

            try:
                scores = await loop.run_in_executor(None, self._score_requests, requests)
                for (_, _, future), score in zip(requests, scores, strict=True):
                    if not future.done():
                        future.set_result(score)
            except BaseException as error:
                for *_, future in requests:
                    if not future.done():
                        future.set_exception(error)

            if should_stop:
                return

    async def score(self, prompts, images):
        prompts = list(prompts)
        images = list(images)
        if len(prompts) != len(images):
            raise ValueError("PickScore prompts and images must have the same length")
        await self._ensure_consumer()
        loop = asyncio.get_running_loop()
        futures = []
        for prompt, image in zip(prompts, images, strict=True):
            future = loop.create_future()
            futures.append(future)
            await self._score_queue.put((prompt, image, future))
        return await asyncio.gather(*futures)

    async def close(self):
        self._closed = True
        if self._consumer_task is not None and not self._consumer_task.done():
            await self._score_queue.put((None, None, None))
            await self._consumer_task
        self._consumer_task = None
        # Drop the whole inferencer before clearing the allocator: retaining a
        # local ``model`` variable here would keep its accelerator storage live.
        if hasattr(self, "_inferencer"):
            del self._inferencer
        gc.collect()
        accelerator = getattr(torch, get_device_name(), None)
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache) and getattr(accelerator, "is_available", lambda: False)():
            empty_cache()


def _to_pil_hwc(image) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return image


def _score_batch(requests) -> list[float | Exception]:
    """Convert and score a batch in a thread so the event loop is never blocked."""
    results = [None] * len(requests)
    prompts = []
    images = []
    valid_indices = []

    for index, (prompt, solution_image, _) in enumerate(requests):
        try:
            images.append(_to_pil_hwc(solution_image))
            prompts.append(prompt)
            valid_indices.append(index)
        except Exception as e:
            results[index] = e

    if valid_indices:
        try:
            scores = _inferencer.score(prompts, images).tolist()
            for index, score in zip(valid_indices, scores, strict=True):
                results[index] = score
        except Exception as e:
            for index in valid_indices:
                results[index] = e

    return results


async def _consumer_loop():
    loop = asyncio.get_running_loop()
    while True:
        request = await _score_queue.get()
        if request[0] is None:
            break

        requests = [request]
        should_stop = False
        await asyncio.sleep(0)
        while len(requests) < _MAX_BATCH_SIZE:
            try:
                request = _score_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if request[0] is None:
                should_stop = True
                break
            requests.append(request)

        results = await loop.run_in_executor(None, _score_batch, requests)
        for (_, _, future), result in zip(requests, results, strict=True):
            if future.done():
                continue
            if isinstance(result, Exception):
                logger.error("PickScore inference failed", exc_info=(type(result), result, result.__traceback__))
                future.set_exception(result)
            else:
                future.set_result(result)

        if should_stop:
            break


async def _ensure_consumer(device: str | None):
    global _inferencer, _consumer_started, _consumer_task
    if _consumer_started:
        return
    async with _consumer_lock:
        if not _consumer_started:
            # Model creation happens here so any error surfaces to the
            # first caller instead of silently killing the background task.
            _inferencer = _PickScoreInferencer(device=device)
            _consumer_started = True
            _consumer_task = asyncio.create_task(_consumer_loop())


async def compute_score_pickscore(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    **kwargs,
) -> dict:
    await _ensure_consumer(device)

    prompt = ground_truth if ground_truth else ""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await _score_queue.put((prompt, solution_image, future))
    raw_score = await future

    return {"score": raw_score, "pickscore_raw": raw_score}
