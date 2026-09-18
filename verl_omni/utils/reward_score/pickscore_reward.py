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

import aiohttp
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


def _pairwise_pickscore(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    logit_scale: float | torch.Tensor,
    score_divisor: float = 26.0,
) -> torch.Tensor:
    """Compute paired PickScore values from text and image embeddings."""
    if score_divisor == 0:
        raise ValueError("PickScore score_divisor must be non-zero")
    cosine = torch.nn.functional.cosine_similarity(text_embeddings, image_embeddings, dim=-1)
    return torch.as_tensor(logit_scale, device=cosine.device, dtype=cosine.dtype) * cosine / score_divisor


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
    def infer(self, prompts: list[str], images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Return the raw model outputs needed by PickScore."""
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
        text_embs = _feature_tensor(self.model.get_text_features(**text_inputs))
        text_embs = text_embs[prompt_indices]

        return {
            "text_embeddings": text_embs,
            "image_embeddings": image_embs,
            "logit_scale": self.model.logit_scale.exp(),
        }

    def score(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:
        """Score prompts and images for existing local callers."""
        output = self.infer(prompts, images)
        return _pairwise_pickscore(**output)


class PickScoreNativeModel:
    """PickScore inference model for native named reward models.

    Concurrent single-item requests are collected into local CLIP batches. The
    model returns embeddings; the configured reward function computes the score.
    """

    def __init__(
        self,
        model_path: str = _MODEL_PATH,
        processor_path: str = _PROCESSOR_PATH,
        device=None,
        dtype=torch.float32,
    ):
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
            raise RuntimeError("PickScore native model is closed")
        if self._consumer_task is not None and not self._consumer_task.done():
            return
        async with self._consumer_lock:
            if self._closed:
                raise RuntimeError("PickScore native model is closed")
            if self._consumer_task is None or self._consumer_task.done():
                self._consumer_task = asyncio.create_task(self._consumer_loop())

    def _infer_requests(self, requests):
        prompts = [prompt for prompt, _, _ in requests]
        images = [image for _, image, _ in requests]
        output = self._inferencer.infer(prompts, images)
        return [
            {
                "text_embeddings": output["text_embeddings"][index],
                "image_embeddings": output["image_embeddings"][index],
                "logit_scale": output["logit_scale"],
            }
            for index in range(len(requests))
        ]

    async def _consumer_loop(self):
        loop = asyncio.get_running_loop()
        while True:
            request = await self._score_queue.get()
            if request[0] is None:
                return

            requests = [request]
            should_stop = False
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
                outputs = await loop.run_in_executor(None, self._infer_requests, requests)
                for (_, _, future), output in zip(requests, outputs, strict=True):
                    if not future.done():
                        future.set_result(output)
            except BaseException as error:
                for *_, future in requests:
                    if not future.done():
                        future.set_exception(error)

            if should_stop:
                return

    async def infer(self, prompts, images):
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


async def compute_score_pickscore_native(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    reward_model,
    score_divisor: float = 26.0,
) -> dict:
    """Compute PickScore from outputs produced by a native named reward model."""
    del data_source, extra_info
    prompt = ground_truth or ""
    image = _to_pil_hwc(solution_image)
    outputs = await reward_model.infer(prompts=[prompt], images=[image])
    output = outputs[0]
    raw_score = _pairwise_pickscore(
        output["text_embeddings"].unsqueeze(0),
        output["image_embeddings"].unsqueeze(0),
        output["logit_scale"],
        score_divisor,
    ).item()
    return {"score": raw_score, "pickscore_raw": raw_score}


async def compute_score_pickscore_engine(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    model_name: str,
    logit_scale: float,
    score_divisor: float = 26.0,
) -> dict:
    """Score an image through an engine-backed CLIP embedding model.

    The named model provides only the managed router and model name. This
    reward function owns the PickScore embedding request format and formula.
    ``logit_scale`` is explicit because vLLM CLIP pooling does not restore the
    PickScore checkpoint's parameter.
    """
    del data_source, extra_info
    if not reward_router_address:
        raise ValueError("PickScore engine reward requires reward_router_address")
    if not model_name:
        raise ValueError("PickScore engine reward requires model_name")

    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    prompt = ground_truth or ""
    image = _to_pil_hwc(solution_image)
    loop = asyncio.get_running_loop()
    image_url = await loop.run_in_executor(None, pil_image_to_base64, image)
    text_payload = {"model": model_name, "input": prompt, "encoding_format": "float"}
    image_payload = {
        "model": model_name,
        "input": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}],
        "encoding_format": "float",
    }
    url = f"http://{reward_router_address}/v1/embeddings"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        async with session.post(url, json=text_payload) as response:
            response.raise_for_status()
            text_result = await response.json()
        async with session.post(url, json=image_payload) as response:
            response.raise_for_status()
            image_result = await response.json()

    text_embedding = torch.tensor(text_result["data"][0]["embedding"], dtype=torch.float32)
    image_embedding = torch.tensor(image_result["data"][0]["embedding"], dtype=torch.float32)
    raw_score = _pairwise_pickscore(
        text_embedding.unsqueeze(0), image_embedding.unsqueeze(0), logit_scale, score_divisor
    ).item()
    return {"score": raw_score, "pickscore_raw": raw_score}
