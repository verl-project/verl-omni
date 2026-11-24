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

"""Batched HPSv2.1 + PickScore + CLIPScore reward for image preference RL.

The native model keeps all three encoders on one managed reward GPU and returns
their inference outputs. ``compute_score_preference_mix`` owns the score
formulas and 1:1:1 weighted aggregation.
"""

import asyncio
import gc
from concurrent.futures import ThreadPoolExecutor

import torch

from verl_omni.utils.reward_score.clipscore_reward import CLIPSCORE_MODEL_PATH, CLIPScoreInferencer, pairwise_clipscore
from verl_omni.utils.reward_score.hpsv2_reward import _HPSv2Scorer
from verl_omni.utils.reward_score.image_preference_utils import RequestBatcher, decoded_reward_image, to_rgb_image
from verl_omni.utils.reward_score.pickscore_reward import _MODEL_PATH as PICKSCORE_MODEL_PATH
from verl_omni.utils.reward_score.pickscore_reward import _PROCESSOR_PATH as PICKSCORE_PROCESSOR_PATH
from verl_omni.utils.reward_score.pickscore_reward import _pairwise_pickscore, _PickScoreInferencer


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    resolved = getattr(torch, dtype, None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unsupported preference reward dtype: {dtype!r}")
    return resolved


class PreferenceMixNativeModel:
    """One managed native deployment containing all three preference models."""

    def __init__(
        self,
        hpsv2_pretrained_path: str,
        hpsv2_checkpoint_path: str,
        device=None,
        dtype: str | torch.dtype = "bfloat16",
        pickscore_model_path: str = PICKSCORE_MODEL_PATH,
        pickscore_processor_path: str = PICKSCORE_PROCESSOR_PATH,
        clipscore_model_path: str = CLIPSCORE_MODEL_PATH,
        max_batch_size: int = 16,
    ):
        # Owned so close() can wait for a batch whose awaiting consumer was cancelled.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preference-mix")
        self._batcher = RequestBatcher(self._infer_requests, max_batch_size=max_batch_size, executor=self._executor)
        self.device = torch.device(device or "cpu")
        model_dtype = _resolve_dtype(dtype)
        self.hpsv2 = _HPSv2Scorer(hpsv2_pretrained_path, hpsv2_checkpoint_path, self.device)
        self.pickscore = _PickScoreInferencer(
            device=self.device,
            dtype=model_dtype,
            model_path=pickscore_model_path,
            processor_path=pickscore_processor_path,
        )
        self.clipscore = CLIPScoreInferencer(
            device=self.device,
            dtype=model_dtype,
            model_path=clipscore_model_path,
        )
        self._closed = False

    def _infer_requests(self, requests) -> list[dict[str, torch.Tensor]]:
        prompts = [prompt for prompt, _ in requests]
        images = [to_rgb_image(image) for _, image in requests]
        hpsv2 = self.hpsv2.infer(images, prompts)
        pickscore = self.pickscore.infer(prompts, images)
        clipscore = self.clipscore.infer(prompts, images)

        return [
            {
                "hpsv2_text_embedding": hpsv2["text_embeddings"][index],
                "hpsv2_image_embedding": hpsv2["image_embeddings"][index],
                "pickscore_text_embedding": pickscore["text_embeddings"][index],
                "pickscore_image_embedding": pickscore["image_embeddings"][index],
                "pickscore_logit_scale": pickscore["logit_scale"],
                "clipscore_text_embedding": clipscore["text_embeddings"][index],
                "clipscore_image_embedding": clipscore["image_embeddings"][index],
            }
            for index in range(len(requests))
        ]

    async def infer(self, prompts, images):
        prompts = list(prompts)
        images = list(images)
        if len(prompts) != len(images):
            raise ValueError("Preference reward prompts and images must have the same length")
        if self._closed:
            raise RuntimeError("Preference reward model is closed")
        return await self._batcher.submit(list(zip(prompts, images, strict=True)))

    async def close(self) -> None:
        self._closed = True
        await self._batcher.stop()
        await asyncio.to_thread(self._executor.shutdown, wait=True)
        for name in ("hpsv2", "pickscore", "clipscore"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        accelerator = getattr(torch, self.device.type, None)
        empty_cache = getattr(accelerator, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


async def compute_score_preference_mix(
    data_source: str,
    solution_image,
    ground_truth: str,
    extra_info: dict,
    reward_model,
    hpsv2_weight: float = 1.0,
    pickscore_weight: float = 1.0,
    clipscore_weight: float = 1.0,
    pickscore_divisor: float = 26.0,
    clipscore_scale: float = 2.5,
) -> dict[str, float]:
    """Compute and combine the three preference scores for one generated image."""
    del data_source
    image = decoded_reward_image(solution_image, extra_info, scorer="Preference-mix reward")
    outputs = await reward_model.infer(prompts=[ground_truth or ""], images=[image])
    output = outputs[0]

    hpsv2 = torch.nn.functional.cosine_similarity(
        output["hpsv2_text_embedding"].unsqueeze(0).float(),
        output["hpsv2_image_embedding"].unsqueeze(0).float(),
        dim=-1,
    ).item()
    pickscore = _pairwise_pickscore(
        output["pickscore_text_embedding"].unsqueeze(0),
        output["pickscore_image_embedding"].unsqueeze(0),
        output["pickscore_logit_scale"],
        score_divisor=pickscore_divisor,
    ).item()
    clipscore = pairwise_clipscore(
        output["clipscore_text_embedding"].unsqueeze(0),
        output["clipscore_image_embedding"].unsqueeze(0),
        scale_factor=clipscore_scale,
    ).item()
    score = hpsv2_weight * hpsv2 + pickscore_weight * pickscore + clipscore_weight * clipscore
    return {
        "score": score,
        "hpsv2_raw": hpsv2,
        "pickscore_raw": pickscore,
        "clipscore_raw": clipscore,
    }
