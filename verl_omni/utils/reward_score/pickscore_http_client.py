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

"""Load-balanced HTTP client for PickScore scorer servers.

Sends generated images to one or more PickScore HTTP scorer servers using the
standard pickle protocol (same as ``http_scorer_client``), with round-robin
load balancing across multiple server URLs.

Protocol (compatible with ``examples/reward_servers/pickscore_http_server.py``)::

    POST with pickle-serialized {"images": List[bytes], "prompts": List[str], "metadata": dict}
    Response: pickle-serialized {"scores": List[float]}
"""

import asyncio
import io
import itertools
import logging
import pickle
from typing import Sequence

import aiohttp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_counter = itertools.count()


def _parse_server_urls(server_urls: str | Sequence[str]) -> list[str]:
    """Parse a comma-separated string or sequence into a list of URLs."""
    if isinstance(server_urls, str):
        urls = [item.strip().strip("'\"") for item in server_urls.split(",")]
    else:
        urls = [str(item).strip().strip("'\"") for item in server_urls]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError("pickscore_http_client requires at least one server URL")
    return urls


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to a uint8 RGB PIL image."""
    if image.ndim == 4:
        image = image[0]
    image = image.float().permute(1, 2, 0).cpu().numpy()
    image = (image * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _serialize_image(image: torch.Tensor | np.ndarray | Image.Image | dict) -> bytes:
    """Serialize an image (tensor / ndarray / PIL / parquet dict) to JPEG bytes."""
    if isinstance(image, dict):
        if image.get("bytes") is not None:
            return image["bytes"]
        if image.get("path") is not None:
            image = Image.open(image["path"])

    if isinstance(image, torch.Tensor):
        image = _tensor_to_pil(image)
    elif isinstance(image, np.ndarray):
        if image.ndim == 4:
            image = image[0]
        if image.dtype != np.uint8:
            image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")

    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


def _prepare_image_bytes(image: torch.Tensor | np.ndarray | Image.Image | dict) -> bytes:
    """Convert image to JPEG bytes (CPU-heavy, run in thread pool)."""
    return _serialize_image(image)


async def compute_score(
    solution_image: torch.Tensor | np.ndarray | Image.Image | dict,
    ground_truth: str,
    server_urls: str | Sequence[str],
    **kwargs,
) -> dict[str, float]:
    """Compute reward by round-robin dispatching to PickScore HTTP servers.

    Args:
        solution_image: Generated image in CHW / NCHW tensor, HWC / NHWC ndarray,
            PIL image, or parquet image dict format.
        ground_truth: Text prompt or edit instruction.
        server_urls: Single URL, comma-separated string, or sequence of URLs.
            Requests are dispatched round-robin across all URLs for load balancing.

    Returns:
        dict with "score" key.
    """
    urls = _parse_server_urls(server_urls)
    server_url = urls[next(_counter) % len(urls)]

    loop = asyncio.get_event_loop()
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
    async with session.post(server_url, data=payload) as resp:
        if resp.status != 200:
            error_text = await resp.text()
            logger.error("PickScore server %s returned %s: %s", server_url, resp.status, error_text)
            return {"score": 0.0}
        response_data = pickle.loads(await resp.read())

    if "error" in response_data:
        logger.error("PickScore server %s error: %s", server_url, response_data["error"])
        return {"score": 0.0}

    scores = response_data.get("scores", [])
    score = float(scores[0]) if scores else 0.0
    return {"score": score}
