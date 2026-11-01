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

"""VLM-as-a-judge reward scorers.

This module is the shared home for the generative reward-model scorers that ask a
vision language model to judge multimodal output: OCR transcription scoring and
UnifiedReward 2.0 preference scoring.

The historical modules :mod:`verl_omni.utils.reward_score.genrm_ocr` and
:mod:`verl_omni.utils.reward_score.unified_reward` remain importable wrappers for
existing configs and third-party imports.
"""

import asyncio
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
DEFAULT_UNIFIED_REWARD_MODEL_PATH = "CodeGoat24/UnifiedReward-2.0-qwen3vl-2b"
DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512}
UNIFIED_REWARD_SCORE_PATTERN = re.compile(
    r"(Alignment|Coherence|Style)\s+Score\s*(?:\(1-5\))?\s*:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


async def _post_chat_completion(
    router_address: str,
    chat_complete_request: dict,
    session: aiohttp.ClientSession | None = None,
) -> ChatCompletion:
    """POST a chat completion request to an OpenAI-compatible router."""
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


async def _chat_complete(router_address: str, chat_complete_request: dict) -> ChatCompletion:
    """POST a chat completion request with a per-request session."""
    return await _post_chat_completion(router_address, chat_complete_request)


async def _chat_complete_with_session(
    router_address: str,
    chat_complete_request: dict,
    session: aiohttp.ClientSession | None = None,
) -> ChatCompletion:
    """POST a chat completion request, optionally reusing a shared session."""
    return await _post_chat_completion(router_address, chat_complete_request, session=session)


def _to_pil(image) -> Image.Image:
    """Convert a uint8 tensor / array / PIL image to an RGB PIL image."""
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "must be in HWC format"
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


async def _compute_score_ocr_impl(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
    sampling_params: dict = None,
    *,
    user_prompt: Optional[str] = None,
    transport,
):
    """Shared OCR scoring implementation with an injectable chat transport."""
    from verl.utils.ray_utils import get_event_loop

    from verl_omni.utils.reward_score.reward_utils import (
        pil_image_to_base64,
        resolve_prompt_builder,
        visual_reward_frames,
    )

    solution_image = visual_reward_frames(solution_image, extra_info, extra_info.get("frame_interval", 4))

    model_name = model_name or os.path.expanduser(DEFAULT_GRM_MODEL_PATH)
    if user_prompt is None:
        user_prompt = DEFAULT_GRM_PROMPT
    elif user_prompt.startswith("pkg://"):
        user_prompt = resolve_prompt_builder(user_prompt)()
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
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]
        params = _sampling_params()
        if sampling_params is not None:
            params.update(sampling_params)
            if "seed" not in sampling_params:
                params.pop("seed", None)
        chat_complete_request = {
            "messages": messages,
            "model": model_name,
            **params,
        }
        result = await transport(
            router_address=reward_router_address,
            chat_complete_request=chat_complete_request,
        )
        grm_response = result.choices[0].message.content
        scores.append(_levenshtein_score(grm_response, ground_truth))

    score = sum(scores) / len(scores)
    return {"score": score, "genrm_response": grm_response}


async def compute_score_ocr(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
    sampling_params: dict = None,
    *,
    user_prompt: Optional[str] = None,
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
        sampling_params: Override the defaults from :func:`_sampling_params`.
            An explicit ``seed`` opts in to reproducible sampling; without one
            any env-inherited seed is dropped.
        user_prompt: User prompt for the GRM. Defaults to
            ``DEFAULT_GRM_PROMPT``. A ``pkg://`` spec names a zero-argument
            prompt builder.

    Returns:
        dict: ``{"score": float, "genrm_response": str}``.
    """
    return await _compute_score_ocr_impl(
        data_source,
        solution_image,
        ground_truth,
        extra_info,
        reward_router_address,
        reward_model_tokenizer,
        model_name,
        sampling_params,
        user_prompt=user_prompt,
        transport=_chat_complete,
    )


def _prepare_solution_frames(solution_image, frame_interval: int, *, extra_info: dict):
    """Select canonical frames from the declared decoded visual artifact."""
    from verl_omni.utils.reward_score.reward_utils import visual_reward_frames

    return visual_reward_frames(solution_image, extra_info, frame_interval)


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
    required_axes = {"alignment", "coherence", "style"}
    if set(scores) != required_axes:
        return {}
    if any(not np.isfinite(value) or not 1.0 <= value <= 5.0 for value in scores.values()):
        return {}
    return scores


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
    prompt: str,
    router_address: str,
    model_name: str,
    loop,
    session: aiohttp.ClientSession,
    *,
    transport,
) -> tuple[float, float, str]:
    """Score one image frame against ``prompt`` via UnifiedReward."""
    from verl_omni.utils.reward_score.reward_utils import pil_image_to_base64

    pil_image = _to_pil(image)
    image_base64 = await loop.run_in_executor(None, pil_image_to_base64, pil_image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_base64}},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        **DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS,
    }
    result = await transport(
        router_address=router_address,
        chat_complete_request=chat_complete_request,
        session=session,
    )
    unified_reward_response = result.choices[0].message.content or ""
    axis_scores = _parse_unified_reward_scores(unified_reward_response)
    if not axis_scores:
        return None
    normalized_score, raw_score = _aggregate_unified_reward_scores(axis_scores)
    return normalized_score, raw_score, unified_reward_response


async def _compute_score_unified_reward_impl(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
    *,
    user_prompt: Optional[str] = None,
    transport,
):
    """Shared UnifiedReward scoring implementation with an injectable transport."""
    from verl.utils.ray_utils import get_event_loop

    from verl_omni.utils.reward_score.reward_utils import resolve_prompt_builder

    del data_source, reward_model_tokenizer

    extra_info = extra_info or {}
    caption = extra_info.get("prompt") or extra_info.get("raw_prompt") or ground_truth or ""
    frame_interval = extra_info.get("frame_interval", 1)
    solution_image = _prepare_solution_frames(solution_image, frame_interval, extra_info=extra_info)

    if user_prompt is None:
        prompt = _build_unified_reward_prompt(caption)
    elif user_prompt.startswith("pkg://"):
        prompt = resolve_prompt_builder(user_prompt)(caption)
    else:
        prompt = user_prompt.replace("{caption}", caption)

    model_name = model_name or DEFAULT_UNIFIED_REWARD_MODEL_PATH
    loop = get_event_loop()

    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        frame_results = await asyncio.gather(
            *[
                _score_single_image(
                    image=image,
                    prompt=prompt,
                    router_address=reward_router_address,
                    model_name=model_name,
                    loop=loop,
                    session=session,
                    transport=transport,
                )
                for image in solution_image
            ]
        )

    valid_results = [result for result in frame_results if result is not None]
    malformed_frames = len(frame_results) - len(valid_results)
    reward_extra_info = {}
    if malformed_frames:
        reward_extra_info["malformed_frames"] = malformed_frames
        extra_info["unified_reward/malformed_frames"] = malformed_frames
    if not valid_results:
        raise ValueError(
            "UnifiedReward returned malformed output for every frame; expected all three labeled axes "
            "(Alignment, Coherence, Style) with scores in [1, 5]."
        )

    normalized_scores = [result[0] for result in valid_results]
    raw_scores = [result[1] for result in valid_results]
    unified_reward_response = valid_results[-1][2]

    score = sum(normalized_scores) / len(normalized_scores)
    raw_score = sum(raw_scores) / len(raw_scores)
    return {
        "score": score,
        "raw_score": raw_score,
        "response": unified_reward_response,
        "reward_extra_info": reward_extra_info,
    }


async def compute_score_unified_reward(
    data_source: str,
    solution_image: np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
    model_name: Optional[str] = None,
    *,
    user_prompt: Optional[str] = None,
):
    """Compute a human-preference score via UnifiedReward 2.0.

    The reward model scores the generated image against its text caption on
    Alignment, Coherence, and Style axes. The returned ``score`` is the mean of
    those axes normalized from the model's 1-5 scale to ``[0, 1]``.

    ``user_prompt`` overrides the default prompt. A ``pkg://`` spec names a
    builder called with ``caption``; literal text replaces ``{caption}``.
    """
    return await _compute_score_unified_reward_impl(
        data_source,
        solution_image,
        ground_truth,
        extra_info,
        reward_router_address,
        reward_model_tokenizer,
        model_name,
        user_prompt=user_prompt,
        transport=_chat_complete_with_session,
    )


__all__ = [
    "DEFAULT_GRM_MODEL_PATH",
    "DEFAULT_GRM_PROMPT",
    "DEFAULT_SAMPLING_PARAMS",
    "DEFAULT_UNIFIED_REWARD_MODEL_PATH",
    "DEFAULT_UNIFIED_REWARD_SAMPLING_PARAMS",
    "UNIFIED_REWARD_SCORE_PATTERN",
    "compute_score_ocr",
    "compute_score_unified_reward",
]
