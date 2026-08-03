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

"""Image/video conversion helpers shared by reward scoring and trainer output paths."""

import base64
from io import BytesIO

import torch
from PIL import Image


def video_tensor_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert an RGB ``[T, C, H, W]`` tensor in ``[0, 1]`` to PIL frames.

    PIL (not NumPy) frames avoid ``export_to_video`` rescaling already-uint8 input
    by 255, which would invert colors modulo 256.
    """
    if video.ndim != 4 or video.shape[1] != 3:
        raise ValueError(f"Expected an RGB video tensor with shape [T, 3, H, W], got {tuple(video.shape)}")

    video = video.detach().permute(0, 2, 3, 1).to(dtype=torch.float32)
    video = torch.nan_to_num(video, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0, 1)
    frames = video.mul_(255).round_().to(dtype=torch.uint8, device="cpu").contiguous().numpy()
    return [Image.fromarray(frame) for frame in frames]


def pil_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded data URI string.

    Args:
        image: The PIL Image to convert.

    Returns:
        A base64-encoded PNG data URI string (e.g. ``data:image/png;base64,...``).
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_image = f"data:image/png;base64,{encoded_image_text}"
    return base64_image
