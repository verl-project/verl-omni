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

"""Safetensors HTTP client for diffusion-native latent rewards."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import aiohttp
import torch
from safetensors.torch import save as save_tensors

PROTOCOL_VERSION = "1"
PROTOCOL_HEADER = "x-drm-protocol-version"


def _batched_tensor(value: Any, name: str, expected_rank: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor, got {type(value).__name__}")
    value = value.detach()
    if value.ndim == expected_rank - 1:
        value = value.unsqueeze(0)
    elif value.ndim != expected_rank:
        raise ValueError(
            f"{name} must have rank {expected_rank - 1} or {expected_rank}, got shape {tuple(value.shape)}"
        )
    if value.shape[0] != 1:
        raise ValueError(f"RewardLoopWorker expects one sample, but {name} has batch size {value.shape[0]}")
    return value.cpu().contiguous()


def _serialize_request(
    solution_image: torch.Tensor,
    extra_info: dict,
    noise_level: float,
    noise_seed: int | None,
) -> bytes:
    latents = extra_info.get("latents_clean")
    if latents is None and isinstance(solution_image, torch.Tensor) and solution_image.shape[-3] == 16:
        latents = solution_image

    tensors = {
        "latents": _batched_tensor(latents, "latents_clean", expected_rank=4),
        "prompt_embeds": _batched_tensor(extra_info.get("prompt_embeds"), "prompt_embeds", expected_rank=3),
        "pooled_prompt_embeds": _batched_tensor(
            extra_info.get("pooled_prompt_embeds"),
            "pooled_prompt_embeds",
            expected_rank=2,
        ),
        "u": torch.tensor([noise_level], dtype=torch.float32),
    }
    if noise_seed is not None:
        tensors["seeds"] = torch.tensor([noise_seed], dtype=torch.int64)
    return save_tensors(tensors)


async def _session() -> aiohttp.ClientSession:
    session = getattr(compute_score, "_session", None)
    if session is None or session.closed:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        compute_score._session = session
    return session


async def _request_score(server_url: str, payload: bytes, timeout: float) -> float:
    session = await _session()
    headers = {
        "content-type": "application/octet-stream",
        PROTOCOL_HEADER: PROTOCOL_VERSION,
    }
    request_timeout = aiohttp.ClientTimeout(total=timeout)
    async with session.post(server_url, data=payload, headers=headers, timeout=request_timeout) as response:
        if response.status != 200:
            detail = await response.text()
            raise RuntimeError(f"DRM server returned HTTP {response.status}: {detail}")
        try:
            result = await response.json()
        except (aiohttp.ContentTypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError("DRM server returned an invalid JSON response") from exc

    if not isinstance(result, dict):
        raise RuntimeError(f"DRM server response must be a JSON object, got {type(result).__name__}")
    if result.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            f"DRM protocol mismatch: expected {PROTOCOL_VERSION!r}, got {result.get('protocol_version')!r}"
        )
    raw_scores = result.get("raw_scores")
    if not isinstance(raw_scores, list) or len(raw_scores) != 1:
        raise RuntimeError(f"DRM server must return one raw score, got {raw_scores!r}")
    try:
        raw_score = float(raw_scores[0])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"DRM server returned an invalid raw score: {raw_scores[0]!r}") from exc
    if not math.isfinite(raw_score):
        raise RuntimeError(f"DRM server returned a non-finite score: {raw_score}")
    return raw_score


async def compute_score(
    solution_image: torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    server_url: str,
    noise_level: float = 0.4,
    noise_seed: int | None = 0,
    score_scale: float = 1.0,
    score_bias: float = 0.0,
    timeout: float = 120.0,
    max_retries: int = 2,
    retry_backoff: float = 0.5,
    **kwargs,
) -> dict:
    """Score one SD3.5 clean latent with an external DRM server.

    The rollout must use ``output_type=latent`` or ``output_type=both`` so
    ``extra_info`` contains ``latents_clean``. Prompt embeddings are reused
    from the rollout and therefore remain exactly aligned with generation.
    """
    del ground_truth, kwargs
    if not 0 <= noise_level <= 1:
        raise ValueError(f"noise_level must be in [0, 1], got {noise_level}")
    if timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")
    if retry_backoff < 0:
        raise ValueError(f"retry_backoff must be non-negative, got {retry_backoff}")

    # Tensor copies and safetensors packing are synchronous, so keep them off
    # the reward worker's event loop.
    payload = await asyncio.to_thread(_serialize_request, solution_image, extra_info, noise_level, noise_seed)
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            raw_score = await _request_score(server_url, payload, timeout)
            score = raw_score * score_scale + score_bias
            return {"score": score, "drm_raw_score": raw_score}
        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as exc:
            last_error = exc
            if attempt < max_retries:
                await asyncio.sleep(retry_backoff * (2**attempt))

    message = f"DRM scoring failed after {max_retries + 1} attempts: {last_error}"
    raise RuntimeError(message) from last_error
