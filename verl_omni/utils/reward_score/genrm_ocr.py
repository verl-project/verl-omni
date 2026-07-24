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

"""OCR scoring backed by a generative reward model (GRM).

The :func:`compute_score_ocr` function sends a generated image to a vision
language model served behind an OpenAI-compatible router and uses the model's
transcription, compared to a ground truth, to produce a score in ``[0, 1]``.
"""

import json
import os
import re
from typing import Optional

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from PIL import Image
from transformers import PreTrainedTokenizer

DEFAULT_GRM_PROMPT = (
    "Please output only the text content from the image without any additional descriptions or formatting."
)
DEFAULT_GRM_MODEL_PATH = "~/models/tiny-random/qwen3-vl"
DEFAULT_SAMPLING_PARAMS = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}


async def _chat_complete(router_address: str, chat_complete_request: dict) -> ChatCompletion:
    """POST a chat completion request to the GRM router and parse the response."""
    url = f"http://{router_address}/v1/chat/completions"
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=chat_complete_request) as resp:
            output = await resp.text()
    return ChatCompletion(**json.loads(output))


def _to_pil(image) -> Image.Image:
    """Normalize a tensor / array / PIL image to a uint8 RGB PIL image."""
    if isinstance(image, torch.Tensor):
        image = image.float().permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "must be in HWC format"
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)
    return image


def _levenshtein_score(text: str, ground_truth: str) -> float:
    """Compute a normalized OCR score in ``[0, 1]`` from raw GRM text."""
    import Levenshtein

    gt = re.sub(r"\s+", "", ground_truth).lower()
    text = re.sub(r"\s+", "", text).lower()
    if gt in text:
        dist = 0
    else:
        dist = Levenshtein.distance(text, gt)
    # If GRM hallucinates many extra characters, only apply a one-character penalty.
    dist = min(dist, len(gt))
    if len(gt) > 0:
        return 1 - dist / len(gt)
    # Empty ground truth: only an empty transcription is a perfect match.
    return 1.0 if len(text) == 0 else 0.0


def _sampling_params() -> dict:
    params = dict(DEFAULT_SAMPLING_PARAMS)
    env_overrides = {
        "temperature": ("GENRM_OCR_TEMPERATURE", float),
        "top_p": ("GENRM_OCR_TOP_P", float),
        "max_tokens": ("GENRM_OCR_MAX_TOKENS", int),
        "seed": ("GENRM_OCR_SEED", int),
    }
    for key, (env_name, parser) in env_overrides.items():
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        params[key] = parser(raw_value)
    return params


async def compute_score_ocr(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
):
    """Compute an image OCR score via a generative reward model (GRM).

    The image is sent to a GRM via an OpenAI-compatible router; the returned
    transcription is compared to ``ground_truth`` using Levenshtein distance to
    yield a score in ``[0, 1]`` (1 = perfect match).

    Args:
        data_source: Source dataset identifier. Unused, kept for interface
            consistency.
        solution_image: The solution image or video to be evaluated.
        ground_truth: The ground truth text for comparison.
        extra_info: Additional information; ``frame_interval`` controls video
            frame subsampling.
        reward_router_address: ``host:port`` of the GRM router.
        reward_model_tokenizer: Tokenizer for the reward model. Unused, kept
            for interface consistency.
        model_name: Name or path of the GRM. Defaults to
            ``DEFAULT_GRM_MODEL_PATH``.

    Returns:
        dict: ``{"score": float, "genrm_response": str}``.
    """
    from verl.utils.ray_utils import get_event_loop

    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    # Normalize any input format to [N, C, H, W] (frames × channels × height × width).
    # Detected formats:
    #   3D:        [C, H, W]             — single image (QwenImage FlowGRPO)
    #   4D:        [C, F, H, W]          — channels-first video (raw VAE)
    #              [F, H, W, C]          — channels-last video (engine postprocess)
    #   5D:        [B, C, F, H, W]       — batched channels-first
    #              [B, F, H, W, C]       — batched channels-last (Wan22 DanceGRPO)
    is_channels_last = solution_image.shape[-1] in (1, 3)

    if solution_image.ndim == 3:
        frame_interval = extra_info.get("frame_interval", 1)
        # [C, H, W] → [1, C, H, W]
        solution_image = solution_image.unsqueeze(0)

    elif solution_image.ndim == 4:
        frame_interval = extra_info.get("frame_interval", 4)
        if is_channels_last:
            # [F, H, W, C] → [C, F, H, W]
            solution_image = solution_image.permute(3, 0, 1, 2)
        # Now [C, F, H, W]: subsample frames, then frame-dim first
        solution_image = solution_image[:, ::frame_interval]  # [C, F', H, W]
        solution_image = solution_image.permute(1, 0, 2, 3)  # [F', C, H, W]

    elif solution_image.ndim == 5:
        frame_interval = extra_info.get(
            "frame_interval",
        )
        if is_channels_last:
            # [B, F, H, W, C] → [B, C, F, H, W]
            solution_image = solution_image.permute(0, 4, 1, 2, 3)
        # Now [B, C, F, H, W]: subsample frames, flatten batch + frames
        solution_image = solution_image[:, :, ::frame_interval]  # [B, C, F', H, W]
        solution_image = solution_image.permute(0, 2, 1, 3, 4)  # [B, F', C, H, W]
        solution_image = solution_image.reshape(-1, *solution_image.shape[2:])  # [B*F', C, H, W]

    model_name = model_name or os.path.expanduser(DEFAULT_GRM_MODEL_PATH)
    loop = get_event_loop()

    grm_response = ""
    scores = []
    for image in solution_image:
        pil_image = _to_pil(image)
        image_base64 = await loop.run_in_executor(None, pil_image_to_base64, pil_image)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": DEFAULT_GRM_PROMPT},
                ],
            },
        ]
        # TODO: make sampling params configurable
        chat_complete_request = {
            "messages": messages,
            "model": model_name,
            **_sampling_params(),
        }
        result = await _chat_complete(
            router_address=reward_router_address,
            chat_complete_request=chat_complete_request,
        )
        grm_response = result.choices[0].message.content
        scores.append(_levenshtein_score(grm_response, ground_truth))

    score = sum(scores) / len(scores)
    return {"score": score, "genrm_response": grm_response}
