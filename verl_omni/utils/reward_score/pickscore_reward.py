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
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        text_inputs = self.processor(
            text=prompts,
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


def _score_one(prompt: str, solution_image) -> float:
    """Called from thread pool.  All heavy work (PIL conversion + CLIP inference)
    happens here so the event loop is never blocked."""
    pil_image = _to_pil_hwc(solution_image)
    scores = _inferencer.score([prompt], [pil_image])
    return scores[0].item()


async def _consumer_loop():
    loop = asyncio.get_running_loop()
    while True:
        prompt, solution_image, future = await _score_queue.get()
        if prompt is None:
            break
        try:
            raw_score = await loop.run_in_executor(None, _score_one, prompt, solution_image)
            future.set_result(raw_score)
        except Exception as e:
            logger.exception("PickScore inference failed")
            future.set_exception(e)


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
