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

"""HPSv2.1 reward scorer backed by ``open_clip_torch``.

The score follows the HPSv2 convention used by MindSpeed-MM: the diagonal of
``image_features @ text_features.T``. Advantage normalization remains in the
trainer and no additional logit or reward scaling is applied here.

Required environment variables:

- ``HPSV2_PRETRAINED_PATH``: OpenCLIP ViT-H-14 pretrained checkpoint.
- ``CUSTOM_REWARD_MODEL_PATH``: HPS-v2.1 reward checkpoint.

Optional environment variable:

- ``REWARD_DEVICE``: inference device, default ``cpu``. ``HPSV2_DEVICE`` is
  accepted as a backward-compatible fallback.
"""

import logging
import os
import threading
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_load_lock = threading.Lock()
_scorer = None
_scorer_key: tuple[str, str, str] | None = None


def _required_file(env_name: str) -> str:
    path = os.getenv(env_name)
    if not path:
        raise ValueError(f"{env_name} must point to an HPSv2.1 checkpoint file")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{env_name} is not a file: {path}")
    return path


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    configured = os.getenv("REWARD_DEVICE") or os.getenv("HPSV2_DEVICE") or "cpu"
    return torch.device(configured)


class _HPSv2Scorer:
    """One OpenCLIP scorer instance owned by one RewardLoopWorker process."""

    def __init__(self, pretrained_path: str, reward_checkpoint_path: str, device: torch.device):
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError("HPSv2 reward requires `pip install open_clip_torch`") from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=pretrained_path,
            precision="amp",
            device="cpu",
            jit=False,
            output_dict=True,
        )
        checkpoint = torch.load(reward_checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(device).eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        self.device = device
        logger.info("HPSv2.1 scorer loaded on %s", device)

    def score(self, image: Image.Image, prompt: str) -> float:
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_input = self.tokenizer([prompt]).to(self.device)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type):
            output = self.model(image_input, text_input)
            raw_score = torch.diagonal(output["image_features"] @ output["text_features"].T)
        return float(raw_score.item())


def _get_hpsv2_scorer(device: str | torch.device | None = None) -> _HPSv2Scorer:
    """Load exactly one configured scorer in each RewardLoopWorker process."""
    global _scorer, _scorer_key

    pretrained_path = _required_file("HPSV2_PRETRAINED_PATH")
    reward_checkpoint_path = _required_file("CUSTOM_REWARD_MODEL_PATH")
    resolved_device = _resolve_device(device)
    scorer_key = (pretrained_path, reward_checkpoint_path, str(resolved_device))

    if _scorer is not None:
        if _scorer_key != scorer_key:
            raise RuntimeError(f"HPSv2 scorer is already loaded with {_scorer_key}, cannot switch to {scorer_key}")
        return _scorer

    with _load_lock:
        if _scorer is None:
            if resolved_device.type != "cpu":
                logger.warning(
                    "HPSv2 is using %s from a RewardLoopWorker that does not reserve Ray accelerator resources.",
                    resolved_device,
                )
            loaded_scorer = _HPSv2Scorer(pretrained_path, reward_checkpoint_path, resolved_device)
            _scorer_key = scorer_key
            _scorer = loaded_scorer
        elif _scorer_key != scorer_key:
            raise RuntimeError(f"HPSv2 scorer is already loaded with {_scorer_key}, cannot switch to {scorer_key}")
        return _scorer


def _to_rgb_image(value: Image.Image | np.ndarray | torch.Tensor) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(value).__name__}")

    array = value
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError(f"HPSv2 scores one image per call, got batch shape {tuple(array.shape)}")
        array = array[0]
    if array.ndim == 3 and array.shape[0] in (1, 3, 4):
        array = np.moveaxis(array, 0, -1)
    if array.ndim not in (2, 3):
        raise ValueError(f"Expected an HxW, HxWxC, or CxHxW image, got shape {tuple(array.shape)}")
    if array.ndim == 3 and array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3, or 4 image channels, got shape {tuple(array.shape)}")
    if not np.isfinite(array).all():
        raise ValueError("Image contains non-finite values")

    array = array.astype(np.float32, copy=False)
    minimum = float(array.min())
    maximum = float(array.max())
    if minimum < 0 or maximum > 255:
        raise ValueError(f"Image values must be in [0, 1] or [0, 255], got [{minimum}, {maximum}]")
    if maximum <= 1:
        array = array * 255
    array = np.rint(array).astype(np.uint8)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array).convert("RGB")


def compute_score_hpsv2(
    solution_image: Image.Image | np.ndarray | torch.Tensor,
    ground_truth: Any = None,
    **_: Any,
) -> dict[str, float]:
    """Score one generated image using the VisualRewardManager contract."""
    scorer = _get_hpsv2_scorer()
    prompt = "" if ground_truth is None else str(ground_truth)
    score = scorer.score(_to_rgb_image(solution_image), prompt)
    return {"score": score, "hpsv2_raw": score}
