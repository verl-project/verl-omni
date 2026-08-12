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

"""Generic HTTP reward client for external scorer services.

Sends generated images to an external HTTP scorer service using pickle protocol
and returns the score. Compatible with all scorer services under
rewards_services/api_services/ that accept the standard payload format::

    POST with pickle-serialized {"images": List[bytes], "prompts": List[str], "metadata": dict}
    Response: pickle-serialized {"scores": List[float]}
"""

import asyncio
import io
import pickle

import aiohttp
import numpy as np
import torch
from PIL import Image


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to a uint8 RGB PIL image."""
    if image.ndim == 4:
        image = image[0]
    image = image.float().permute(1, 2, 0).cpu().numpy()
    image = (image * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _serialize_image(pil_image: Image.Image) -> bytes:
    """Serialize a PIL image to JPEG bytes."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    return buf.getvalue()


def _prepare_image_bytes(image: torch.Tensor) -> bytes:
    """Convert image tensor to JPEG bytes (CPU-heavy, run in thread pool)."""
    pil_image = _tensor_to_pil(image)
    return _serialize_image(pil_image)


def _error_detail(response_bytes: bytes) -> str:
    """Extract a readable error message from a pickled or plain-text response."""
    try:
        response_data = pickle.loads(response_bytes)
    except (pickle.UnpicklingError, EOFError):
        return response_bytes.decode("utf-8", errors="replace")
    if isinstance(response_data, dict) and "error" in response_data:
        return str(response_data["error"])
    return response_bytes.decode("utf-8", errors="replace")


async def compute_score(
    solution_image: torch.Tensor,
    ground_truth: str,
    server_url: str,
    max_retries: int = 2,
    retry_backoff: float = 0.5,
    **kwargs,
) -> dict:
    """Compute reward by calling an external HTTP scorer service.

    Args:
        solution_image: Generated image tensor (C, H, W) or (N, C, H, W).
        ground_truth: Prompt string passed directly to the scorer service.
        server_url: Full URL of the scorer service (e.g., "http://localhost:19082").
        max_retries: Number of retries after the initial HTTP attempt.
        retry_backoff: Initial delay in seconds between retries. The delay doubles after each failure.

    Returns:
        dict with "score" key.

    Raises:
        RuntimeError: If the scorer returns an HTTP error, reports an error, or returns no scores.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}")
    if retry_backoff < 0:
        raise ValueError(f"retry_backoff must be non-negative, got {retry_backoff}")

    loop = asyncio.get_running_loop()
    image_bytes = await loop.run_in_executor(None, _prepare_image_bytes, solution_image)

    payload = pickle.dumps(
        {
            "images": [image_bytes],
            "prompts": [ground_truth],
            "metadata": {},
        }
    )

    if not hasattr(compute_score, "_session") or compute_score._session.closed:
        timeout = aiohttp.ClientTimeout(total=120)
        compute_score._session = aiohttp.ClientSession(timeout=timeout)

    session = compute_score._session
    attempts = max_retries + 1
    last_error = None
    for attempt in range(attempts):
        try:
            async with session.post(server_url, data=payload) as resp:
                response_bytes = await resp.read()
                if resp.status != 200:
                    error = RuntimeError(f"Scorer server returned HTTP {resp.status}: {_error_detail(response_bytes)}")
                    retryable = resp.status in {408, 429} or 500 <= resp.status < 600
                    if not retryable:
                        raise error
                    last_error = error
                else:
                    response_data = pickle.loads(response_bytes)
                    break
        except (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ) as exc:
            last_error = exc
        if attempt < max_retries:
            await asyncio.sleep(retry_backoff * (2**attempt))
    else:
        raise RuntimeError(f"HTTP scoring failed after {attempts} attempts: {last_error}") from last_error

    if "error" in response_data:
        raise RuntimeError(f"Scorer server error: {response_data['error']}")

    scores = response_data.get("scores")
    if not scores:
        raise RuntimeError("Scorer server returned no scores")

    score = float(scores[0])
    return {"score": score}
