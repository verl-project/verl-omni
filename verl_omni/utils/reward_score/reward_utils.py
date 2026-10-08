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


def audio_info_from_batch(extra_info: dict | None, batch, *, scorer: str) -> dict:
    """Use a single sample's decoded audio/rate in preference to extra_info."""
    info = dict(extra_info or {})
    if batch is None:
        return info
    if len(batch) != 1:
        raise ValueError(f"{scorer} scoring requires exactly one sample.")
    item = batch[0]
    for key in ("audio", "audio_sample_rate"):
        if key in item.batch:
            info[key] = item.batch[key]
        elif key in item.non_tensor_batch:
            info[key] = item.non_tensor_batch[key]
    return info


def get_audio(extra_info: dict, *, default_sample_rate: int = 48_000) -> tuple[torch.Tensor, int]:
    """Normalize decoded audio to CPU mono samples and return its source rate."""
    audio = extra_info.get("audio")
    if audio is None:
        raise KeyError("Audio reward requires decoded audio in extra_info['audio'].")
    audio = torch.as_tensor(audio).detach().float().cpu()
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    elif audio.ndim != 1:
        raise ValueError(f"Expected audio shape (T,) or (C,T), got {tuple(audio.shape)}.")
    sample_rate = extra_info.get("audio_sample_rate", default_sample_rate)
    if isinstance(sample_rate, torch.Tensor):
        sample_rate = sample_rate.item()
    if sample_rate is None:
        raise KeyError("Audio reward requires extra_info['audio_sample_rate'].")
    return audio, int(sample_rate)


def resample_audio(waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
    """Resample one mono waveform, leaving matching rates untouched."""
    if source_rate == target_rate:
        return waveform
    import torchaudio.functional as audio_functional

    return audio_functional.resample(waveform.unsqueeze(0), orig_freq=source_rate, new_freq=target_rate).squeeze(0)


def load_torch_state_dict(path: str):
    """Load tensor-only weights, including legacy torch.save files without mmap."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used with files saved with" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def normalize_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """Normalize an RGB uint8 video to the ``[T, 3, H, W]`` layout.

    Accepted layouts are time-first channels-first ``[T, 3, H, W]``,
    channels-first temporal ``[3, T, H, W]``, and channels-last
    ``[T, H, W, 3]``. Ambiguous shapes such as ``[3, 3, H, W]`` retain the
    canonical time-first interpretation.
    """
    if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8:
        dtype = video.dtype if isinstance(video, torch.Tensor) else type(video)
        raise ValueError(f"Expected a uint8 video tensor, got {dtype}")
    if video.ndim != 4:
        raise ValueError(f"Expected an RGB video tensor with shape [T, 3, H, W], got {tuple(video.shape)}")

    if video.shape[1] == 3:
        normalized = video
    elif video.shape[0] == 3:
        normalized = video.permute(1, 0, 2, 3)
    elif video.shape[-1] == 3:
        normalized = video.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected an RGB video tensor with shape [T, 3, H, W], got {tuple(video.shape)}")

    if normalized.shape[0] == 0 or normalized.shape[2] == 0 or normalized.shape[3] == 0:
        raise ValueError(f"Expected a video with non-empty time and spatial dimensions, got {tuple(video.shape)}")
    return normalized


def video_tensor_to_pil_frames(video: torch.Tensor) -> list[Image.Image]:
    """Convert a normalized RGB uint8 video tensor to PIL frames.

    PIL (not NumPy) frames avoid ``export_to_video`` rescaling already-uint8 input
    by 255, which would invert colors modulo 256.
    """
    video = normalize_video_tensor(video)
    frames = video.detach().permute(0, 2, 3, 1).to(device="cpu").contiguous().numpy()
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
