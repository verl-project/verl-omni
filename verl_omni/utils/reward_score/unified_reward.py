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

"""Human-preference scoring backed by UnifiedReward 2.0."""

import asyncio
import json
import re
from typing import Optional

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from PIL import Image
from transformers import PreTrainedTokenizer

DEFAULT_UNIFIED_REWARD_MODEL_PATH = "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b"
DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
UNIFIED_REWARD_SCORE_PATTERN = re.compile(
    r"(Alignment|Coherence|Style)\s+Score\s*(?:\(1-5\))?\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


async def _chat_complete(
    router_address: str,
    chat_complete_request: dict,
    session: aiohttp.ClientSession | None = None,
) -> ChatCompletion:
    """POST a chat completion request to the reward router and parse the response."""
    url = f"http://{router_address}/v1/chat/completions"
    if session is None:
        timeout = aiohttp.ClientTimeout(total=None)
        async with aiohttp.ClientSession(timeout=timeout) as owned_session:
            async with owned_session.post(url, json=chat_complete_request) as resp:
                output = await resp.text()
    else:
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


def _prepare_solution_frames(solution_image: np.ndarray | torch.Tensor, frame_interval: int):
    """Normalize an image/video tensor or array into an iterable of frames."""
    if solution_image.ndim == 3:  # image
        if isinstance(solution_image, torch.Tensor):
            return solution_image.unsqueeze(0)
        return np.expand_dims(solution_image, axis=0)
    if solution_image.ndim == 4:  # video
        return solution_image[::frame_interval]
    raise ValueError(f"Expected image/video with 3 or 4 dimensions, got shape {solution_image.shape}")


def _build_unified_reward_prompt(caption: str) -> str:
    """Build the official point-score prompt for UnifiedReward 2.0."""
    return (
        "You are presented with a generated image and its associated text caption. "
        "Your task is to analyze the image across multiple dimensions in relation to the caption. "
        "Specifically:\n"
        "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
        "- Alignment Score: How well the image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent the image is "
        "(absence of visual glitches, object distortions, etc.).\n"
        "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
        "Output your evaluation using the format below:\n\n"
        "Alignment Score (1-5): X\n"
        "Coherence Score (1-5): Y\n"
        "Style Score (1-5): Z\n\n"
        "Your task is provided as follows:\n"
        f"Text Caption: [{caption}]"
    )


def _parse_unified_reward_scores(model_output: str) -> dict[str, float]:
    """Parse UnifiedReward point scores from raw model output."""
    scores: dict[str, float] = {}
    for match in UNIFIED_REWARD_SCORE_PATTERN.finditer(model_output):
        name = match.group(1).lower()
        scores[name] = float(match.group(2))

    if scores:
        return scores

    # Fallback for slightly malformed responses that still contain the three values.
    numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", model_output)
    if len(numbers) >= 3:
        return {
            "alignment": float(numbers[0]),
            "coherence": float(numbers[1]),
            "style": float(numbers[2]),
        }
    return {}


def _aggregate_unified_reward_scores(scores: dict[str, float]) -> tuple[float, float]:
    """Return ``(normalized_score, raw_score)`` from 1-5 UnifiedReward axes."""
    if not scores:
        return 0.0, 0.0
    raw_score = sum(scores.values()) / len(scores)
    clipped_raw_score = min(max(raw_score, 1.0), 5.0)
    normalized_score = (clipped_raw_score - 1.0) / 4.0
    return normalized_score, raw_score


async def _score_single_image(
    image,
    caption: str,
    router_address: str,
    model_name: str,
    loop,
    session: aiohttp.ClientSession,
) -> tuple[float, float, str]:
    """Score one image frame against ``caption`` via UnifiedReward."""
    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    pil_image = _to_pil(image)
    image_base64 = await loop.run_in_executor(None, pil_image_to_base64, pil_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": _build_unified_reward_prompt(caption)},
            ],
        },
    ]
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        **DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS,
    }
    result = await _chat_complete(
        router_address=router_address,
        chat_complete_request=chat_complete_request,
        session=session,
    )
    unified_reward_response = result.choices[0].message.content or ""
    axis_scores = _parse_unified_reward_scores(unified_reward_response)
    normalized_score, raw_score = _aggregate_unified_reward_scores(axis_scores)
    return normalized_score, raw_score, unified_reward_response


async def compute_score_unified_reward(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
):
    """Compute a human-preference score via UnifiedReward 2.0.

    The reward model scores the generated image against its text caption on
    Alignment, Coherence, and Style axes. The returned ``score`` is the mean of
    those axes normalized from the model's 1-5 scale to ``[0, 1]``.
    """
    from verl.utils.ray_utils import get_event_loop

    del data_source, reward_model_tokenizer

    extra_info = extra_info or {}
    caption = extra_info.get("prompt") or extra_info.get("raw_prompt") or ground_truth or ""
    frame_interval = extra_info.get("frame_interval", 1)
    solution_image = _prepare_solution_frames(solution_image, frame_interval)

    model_name = model_name or DEFAULT_UNIFIED_REWARD_MODEL_PATH
    loop = get_event_loop()

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        frame_results = await asyncio.gather(
            *[
                _score_single_image(
                    image=image,
                    caption=caption,
                    router_address=reward_router_address,
                    model_name=model_name,
                    loop=loop,
                    session=session,
                )
                for image in solution_image
            ]
        )

    normalized_scores = [result[0] for result in frame_results]
    raw_scores = [result[1] for result in frame_results]
    unified_reward_response = frame_results[-1][2]

    score = sum(normalized_scores) / len(normalized_scores)
    raw_score = sum(raw_scores) / len(raw_scores)
    return {"score": score, "raw_score": raw_score, "response": unified_reward_response}
