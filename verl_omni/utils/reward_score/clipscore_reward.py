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

"""CLIPScore reward using the original CLIPScore cosine convention.

The default score is ``2.5 * max(cosine(image, text), 0)`` from Hessel et al.
The model defaults to OpenAI CLIP ViT-L/14 and concurrent one-image requests
are combined into local inference batches.
"""

import asyncio
import logging
import os
import threading

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_MODEL_PATH = "openai/clip-vit-large-patch14"
_MAX_BATCH_SIZE = 16

_inferencer = None
_inferencer_key = None
_inferencer_lock = threading.Lock()
# Queues, locks and tasks belong to one event loop; the reward manager may call
# ``asyncio.run`` more than once per process, so they are rebuilt per loop.
_score_queue = None
_consumer_task = None
_consumer_lock = None
_state_loop = None


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


def _pairwise_clipscore(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    scale_factor: float = 2.5,
) -> torch.Tensor:
    """Compute paired CLIPScore values from text and image embeddings."""
    # In bf16 the cosine rounds to ~4e-3, a third of the within-group reward spread.
    cosine = torch.nn.functional.cosine_similarity(text_embeddings.float(), image_embeddings.float(), dim=-1)
    return float(scale_factor) * cosine.clamp_min(0)


def _to_rgb_image(image) -> Image.Image:
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


def _decoded_reward_image(solution_image, extra_info: dict | None) -> Image.Image:
    """Select the one declared decoded image; latent responses are rejected."""
    from verl_omni.utils.reward_score.reward_utils import image_tensor_to_pil, visual_reward_frames

    frames = visual_reward_frames(solution_image, extra_info or {})
    if len(frames) != 1:
        raise ValueError("Image preference rewards require a single decoded image")
    return image_tensor_to_pil(frames[0])


class _CLIPScoreInferencer:
    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype=torch.float32,
        model_path: str = _MODEL_PATH,
    ):
        if device is None:
            device = torch.device(get_device_name(), get_device_id())
        logger.info("Creating CLIPScore model from %s", model_path)
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device, dtype=dtype)

    @torch.no_grad()
    def infer(self, prompts: list[str], images: list[Image.Image]) -> dict[str, torch.Tensor]:
        unique_prompts = list(dict.fromkeys(prompts))
        prompt_to_index = {prompt: index for index, prompt in enumerate(unique_prompts)}
        prompt_indices = [prompt_to_index[prompt] for prompt in prompts]

        image_inputs = self.processor(images=images, return_tensors="pt")
        image_inputs = {key: value.to(self.device) for key, value in image_inputs.items()}
        text_inputs = self.processor(
            text=unique_prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {key: value.to(self.device) for key, value in text_inputs.items()}

        image_embeddings = _feature_tensor(self.model.get_image_features(**image_inputs))
        text_embeddings = _feature_tensor(self.model.get_text_features(**text_inputs))[prompt_indices]
        return {"text_embeddings": text_embeddings, "image_embeddings": image_embeddings}

    def score(self, prompts: list[str], images: list[Image.Image], scale_factor: float = 2.5) -> torch.Tensor:
        return _pairwise_clipscore(**self.infer(prompts, images), scale_factor=scale_factor)


def _score_batch(requests) -> list[float | Exception]:
    results = [None] * len(requests)
    for index, (prompt, solution_image, _, _) in enumerate(requests):
        try:
            requests[index] = (prompt, _to_rgb_image(solution_image), requests[index][2], requests[index][3])
        except Exception as error:
            results[index] = error

    groups = {}
    for index, (prompt, image, _, scale_factor) in enumerate(requests):
        if results[index] is None:
            groups.setdefault(scale_factor, []).append((index, prompt, image))

    for scale_factor, group in groups.items():
        indices, prompts, images = zip(*group, strict=True)
        try:
            scores = _inferencer.score(list(prompts), list(images), scale_factor=scale_factor).tolist()
            for index, score in zip(indices, scores, strict=True):
                results[index] = score
        except Exception as error:
            for index in indices:
                results[index] = error
    return results


async def _consumer_loop():
    loop = asyncio.get_running_loop()
    while True:
        request = await _score_queue.get()
        if request[0] is None:
            return

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

        try:
            results = await loop.run_in_executor(None, _score_batch, requests)
        except asyncio.CancelledError:
            for _, _, future, _ in requests:
                future.cancel()
            raise
        except Exception as error:
            results = [error] * len(requests)
        for (_, _, future, _), result in zip(requests, results, strict=True):
            if future.done():
                continue
            if isinstance(result, Exception):
                future.set_exception(result)
            else:
                future.set_result(result)
        if should_stop:
            return


def _bind_loop_state() -> None:
    global _score_queue, _consumer_task, _consumer_lock, _state_loop
    loop = asyncio.get_running_loop()
    if _state_loop is not loop:
        _score_queue = asyncio.Queue()
        _consumer_lock = asyncio.Lock()
        _consumer_task = None
        _state_loop = loop


def _load_inferencer(device: str | None, model_path: str) -> None:
    global _inferencer, _inferencer_key
    key = (device, model_path)
    with _inferencer_lock:
        if _inferencer is None:
            _inferencer = _CLIPScoreInferencer(device=device, model_path=model_path)
            _inferencer_key = key
        elif _inferencer_key != key:
            raise RuntimeError(f"CLIPScore is already loaded with {_inferencer_key}, cannot switch to {key}")


async def _ensure_consumer(device: str | None, model_path: str):
    global _consumer_task
    _bind_loop_state()
    if _consumer_task is not None and not _consumer_task.done():
        _load_inferencer(device, model_path)
        return
    async with _consumer_lock:
        _load_inferencer(device, model_path)
        if _consumer_task is None or _consumer_task.done():
            _consumer_task = asyncio.create_task(_consumer_loop())


async def compute_score_clipscore(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    model_path: str = _MODEL_PATH,
    scale_factor: float = 2.5,
    **kwargs,
) -> dict[str, float]:
    """Score one generated image against its prompt."""
    del data_source, kwargs
    image = _decoded_reward_image(solution_image, extra_info)
    await _ensure_consumer(device, model_path)

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await _score_queue.put((ground_truth or "", image, future, float(scale_factor)))
    score = await future
    return {"score": score, "clipscore_raw": score}
