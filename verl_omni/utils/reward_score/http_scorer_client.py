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
rewards_services/api_services/ that accept the standard payload format:
    POST with pickle-serialized {"images": List[bytes], "prompts": List[str], "metadata": dict}
    Response: pickle-serialized {"scores": List[float]}
"""

import io
import logging
import pickle

import aiohttp
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


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


async def compute_score(
    solution_image: torch.Tensor,
    ground_truth: str,
    server_url: str,
    **kwargs,
) -> dict:
    """Compute reward by calling an external HTTP scorer service.

    Args:
        solution_image: Generated image tensor (C, H, W) or (N, C, H, W).
        ground_truth: Prompt string passed directly to the scorer service.
        server_url: Full URL of the scorer service (e.g., "http://localhost:19082").

    Returns:
        dict with "score" key.
    """
    pil_image = _tensor_to_pil(solution_image)
    image_bytes = _serialize_image(pil_image)

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
            logger.error(f"Scorer server returned {resp.status}: {error_text}")
            return {"score": 0.0}
        response_data = pickle.loads(await resp.read())

    if "error" in response_data:
        logger.error(f"Scorer server error: {response_data['error']}")
        return {"score": 0.0}

    scores = response_data["scores"]
    score = float(scores[0]) if scores else 0.0
    return {"score": score}
