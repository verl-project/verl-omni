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
import logging
import os

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

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
    def __init__(self, device: str = "cuda", dtype=torch.float32):
        logger.info("Creating PickScore model from %s", _MODEL_PATH)
        self.device = device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(_PROCESSOR_PATH)
        self.model = CLIPModel.from_pretrained(_MODEL_PATH).eval().to(device)
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


def _to_pil_hwc(image) -> Image.Image:
    if isinstance(image, torch.Tensor):
        image = image.float().cpu().numpy()
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = image.transpose(1, 2, 0)
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
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


async def _ensure_consumer(device: str):
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
    device: str = "cuda",
    **kwargs,
) -> dict:
    await _ensure_consumer(device)

    prompt = ground_truth if ground_truth else ""
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await _score_queue.put((prompt, solution_image, future))
    raw_score = await future

    return {"score": raw_score, "pickscore_raw": raw_score}
