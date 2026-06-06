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

"""PickScore reward function for verl-omni.

PickScore is a text-to-image human-preference scoring model built on top of
CLIP-ViT-H-14. It computes a cosine similarity between text and image embeddings,
scaled by a learned logit-scale parameter.

Reference: https://github.com/yuvalkirstain/PickScore
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---- lazy-loaded singleton ----

_processor: Any = None
_model: Any = None
_device: str = "cuda"

PROCESSOR_NAME = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_NAME = "yuvalkirstain/PickScore_v1"


def _ensure_model(device: Optional[str] = None) -> None:
    """Lazily load the PickScore model and CLIP processor."""
    global _processor, _model

    if _model is not None:
        return

    from transformers import AutoModel, AutoProcessor

    if device is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev = device

    logger.info("Loading PickScore processor: %s", PROCESSOR_NAME)
    _processor = AutoProcessor.from_pretrained(PROCESSOR_NAME)

    logger.info("Loading PickScore model: %s", MODEL_NAME)
    _model = AutoModel.from_pretrained(MODEL_NAME).eval().to(dev)


# ---- tensor helpers ----


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a CHW float tensor in [0, 1] to an RGB PIL image."""
    if image.ndim == 4:
        image = image[0]  # (N, C, H, W) -> (C, H, W)
    image = image.float().permute(1, 2, 0).cpu().numpy()
    image = (image * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _to_pil(image: Any) -> Image.Image:
    """Normalize various image representations to a PIL Image."""
    if isinstance(image, dict):
        import io as _io

        if image.get("bytes") is not None:
            image = Image.open(_io.BytesIO(image["bytes"]))
        elif image.get("path") is not None:
            image = Image.open(image["path"])
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[-1] == 3:
            image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    if isinstance(image, torch.Tensor):
        image = _tensor_to_pil(image)
    if not isinstance(image, Image.Image):
        raise TypeError(f"Unsupported image type: {type(image)}")
    return image.convert("RGB")


# ---- public API ----


def compute_score(
    data_source: str,
    solution_image: torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    device: Optional[str] = None,
    **kwargs,
) -> dict[str, float]:
    """Compute PickScore between a text prompt and generated image.

    Args:
        data_source: Dataset identifier (unused, kept for interface compatibility).
        solution_image: Generated image tensor (C, H, W) or (N, C, H, W) in [0, 1].
        ground_truth: Text prompt used to generate the image.
        extra_info: Additional metadata (unused here, kept for interface compatibility).
        device: Torch device string (``"cuda"``, ``"cpu"``). Defaults to ``"cuda"`` if
            available, otherwise ``"cpu"``.

    Returns:
        ``{"score": float}`` where score is the cosine similarity in ``[0, 1]``
        between the prompt and image embeddings.
    """
    _ensure_model(device=device)

    # Convert the output image tensor to PIL
    pil_image = _to_pil(solution_image)

    # Preprocess
    image_inputs = _processor(
        images=[pil_image],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(_device)

    text_inputs = _processor(
        text=[ground_truth],
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        image_embs = _model.get_image_features(**image_inputs).pooler_output
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = _model.get_text_features(**text_inputs).pooler_output
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # Cosine similarity between text and image embeddings.
        # PickScore's ``logit_scale`` is designed for softmax-based
        # contrastive comparison across multiple images; using it on a
        # single pair would saturate sigmoid to ~1.0 and kill reward
        # variance.  We use the raw cosine similarity rescaled to [0, 1]
        # instead, which is interpretable and preserves the ranking
        # needed for GRPO advantage computation.
        cos_sim = (text_embs @ image_embs.T)[0, 0]  # in [-1, 1]
        score = (cos_sim + 1.0) / 2.0

    return {"score": float(score.cpu())}
