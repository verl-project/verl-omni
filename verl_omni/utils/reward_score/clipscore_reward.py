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

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from verl.utils.device import get_device_id, get_device_name

from verl_omni.utils.reward_score.image_preference_utils import (
    RequestBatcher,
    clip_feature_tensor,
    decoded_reward_image,
    to_rgb_image,
)

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

CLIPSCORE_MODEL_PATH = "openai/clip-vit-large-patch14"
_MAX_BATCH_SIZE = 16

_inferencer = None
_inferencer_key = None
_inferencer_lock = threading.Lock()
# A batcher's queue belongs to one event loop; the reward manager may call
# ``asyncio.run`` more than once per process, so it is rebuilt per loop.
_batcher = None
_batcher_loop = None


def pairwise_clipscore(
    text_embeddings: torch.Tensor,
    image_embeddings: torch.Tensor,
    scale_factor: float = 2.5,
) -> torch.Tensor:
    """Compute paired CLIPScore values from text and image embeddings."""
    # In bf16 the cosine rounds to ~4e-3, a third of the within-group reward spread.
    cosine = torch.nn.functional.cosine_similarity(text_embeddings.float(), image_embeddings.float(), dim=-1)
    return float(scale_factor) * cosine.clamp_min(0)


class CLIPScoreInferencer:
    """CLIP encoder returning paired text and image embeddings for CLIPScore."""

    def __init__(
        self,
        device: str | torch.device | None = None,
        dtype=torch.float32,
        model_path: str = CLIPSCORE_MODEL_PATH,
    ):
        if device is None:
            device = torch.device(get_device_name(), get_device_id())
        logger.info("Creating CLIPScore model from %s", model_path)
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(self.device, dtype=dtype)

    @torch.no_grad()
    def infer(self, prompts: list[str], images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """Encode a paired prompt/image batch while deduplicating repeated prompts."""
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

        image_embeddings = clip_feature_tensor(self.model.get_image_features(**image_inputs))
        text_embeddings = clip_feature_tensor(self.model.get_text_features(**text_inputs))[prompt_indices]
        return {"text_embeddings": text_embeddings, "image_embeddings": image_embeddings}

    def score(self, prompts: list[str], images: list[Image.Image], scale_factor: float = 2.5) -> torch.Tensor:
        """Return paired CLIPScore values."""
        return pairwise_clipscore(**self.infer(prompts, images), scale_factor=scale_factor)


def _score_batch(requests) -> list[float | Exception]:
    results = [None] * len(requests)
    groups = {}
    for index, (prompt, solution_image, scale_factor) in enumerate(requests):
        try:
            image = to_rgb_image(solution_image)
        except Exception as error:
            results[index] = error
            continue
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


def _load_inferencer(device: str | None, model_path: str) -> None:
    global _inferencer, _inferencer_key
    key = (device, model_path)
    with _inferencer_lock:
        if _inferencer is None:
            _inferencer = CLIPScoreInferencer(device=device, model_path=model_path)
            _inferencer_key = key
        elif _inferencer_key != key:
            raise RuntimeError(f"CLIPScore is already loaded with {_inferencer_key}, cannot switch to {key}")


def _loop_batcher() -> RequestBatcher:
    global _batcher, _batcher_loop
    loop = asyncio.get_running_loop()
    if _batcher_loop is not loop:
        _batcher = RequestBatcher(_score_batch, max_batch_size=_MAX_BATCH_SIZE)
        _batcher_loop = loop
    return _batcher


async def compute_score_clipscore(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    device: str | None = None,
    model_path: str = CLIPSCORE_MODEL_PATH,
    scale_factor: float = 2.5,
    **kwargs,
) -> dict[str, float]:
    """Score one generated image against its prompt."""
    del data_source, kwargs
    image = decoded_reward_image(solution_image, extra_info, scorer="CLIPScore")
    _load_inferencer(device, model_path)
    (score,) = await _loop_batcher().submit([(ground_truth or "", image, float(scale_factor))])
    return {"score": score, "clipscore_raw": score}
