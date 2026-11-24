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
import math
from collections.abc import Callable
from functools import lru_cache
from io import BytesIO
from numbers import Real

import torch
from PIL import Image


@lru_cache
def resolve_prompt_builder(spec: str) -> Callable:
    """Resolve a ``pkg://<module>.<function>`` prompt builder spec."""
    if not isinstance(spec, str) or not spec.startswith("pkg://"):
        raise ValueError(f"Prompt builder spec must start with 'pkg://', got {spec!r}")
    module_name, separator, object_name = spec.removeprefix("pkg://").rpartition(".")
    if not separator or not module_name or not object_name:
        raise ValueError(f"Prompt builder spec must be 'pkg://<module>.<function>', got {spec!r}")
    try:
        from verl.utils.import_utils import load_extern_object

        builder = load_extern_object(f"pkg://{module_name}", object_name)
    except Exception as error:
        raise ValueError(f"Failed to load prompt builder from {spec!r}: {error}") from error
    if not callable(builder):
        raise ValueError(f"Prompt builder from {spec!r} is not callable")
    return builder


def audio_info_from_batch(extra_info: dict | None, batch, *, scorer: str) -> dict:
    """Use a single sample's decoded audio/rate in preference to extra_info."""
    info = dict(extra_info or {})
    if batch is None:
        return info
    if len(batch) != 1:
        raise ValueError(f"{scorer} scoring requires exactly one sample.")
    item = batch[0]
    for key in ("audio", "audio_sample_rate", "audio_layout"):
        if key in item.batch:
            info[key] = item.batch[key]
        elif key in item.non_tensor_batch:
            info[key] = item.non_tensor_batch[key]
    return info


def get_audio(extra_info: dict, *, audio_layout: str | None = None) -> tuple[torch.Tensor, int]:
    """Normalize decoded audio to CPU mono samples and return its source rate."""
    if "media_artifacts" in extra_info:
        from verl_omni.pipelines.rollout_media import ArtifactContractError, select_artifact

        artifact = select_artifact(
            extra_info["media_artifacts"], name="audio", modality="audio", representation="decoded"
        )
        if artifact.spec.layout != "CT" or audio_layout not in (None, "CT"):
            raise ArtifactContractError(f"Audio rewards require canonical CT audio, got {artifact.spec.layout}")
        waveform = artifact.data.detach().float().cpu().mean(dim=0)
        if not torch.isfinite(waveform).all():
            raise ArtifactContractError("Audio rewards require finite decoded audio samples")
        return waveform, artifact.spec.sample_rate
    audio = extra_info.get("audio")
    if audio is None:
        raise KeyError("Audio reward requires decoded audio in extra_info['audio'].")
    audio = torch.as_tensor(audio).detach().float().cpu()
    if audio.numel() == 0 or not torch.isfinite(audio).all():
        raise ValueError("Audio reward requires non-empty audio containing only finite samples.")
    declared_layout = extra_info.get("audio_layout")
    if audio_layout is not None and declared_layout is not None and audio_layout != declared_layout:
        raise ValueError("Conflicting audio_layout in scorer config and extra_info.")
    layout = audio_layout if audio_layout is not None else declared_layout
    if layout not in (None, "T", "CT"):
        raise ValueError("Audio layout must be 'T' or 'CT'; channels-last audio is not supported.")
    while audio.ndim > 2 and audio.shape[0] == 1:
        audio = audio[0]
    if audio.ndim == 2:
        if layout != "CT":
            raise ValueError("Two-dimensional audio requires explicit audio_layout='CT' for shape (C, T).")
        audio = audio.mean(dim=0)
    elif audio.ndim != 1:
        raise ValueError(f"Expected audio shape (T,) or (C,T), got {tuple(audio.shape)}.")
    sample_rate = extra_info.get("audio_sample_rate")
    if isinstance(sample_rate, torch.Tensor):
        if sample_rate.numel() != 1:
            raise ValueError("Audio sample rate must be a scalar positive integer.")
        sample_rate = sample_rate.item()
    if sample_rate is None:
        raise KeyError("Audio reward requires extra_info['audio_sample_rate'].")
    if (
        isinstance(sample_rate, bool)
        or not isinstance(sample_rate, Real)
        or not math.isfinite(sample_rate)
        or sample_rate <= 0
        or sample_rate != int(sample_rate)
    ):
        raise ValueError("Audio sample rate must be a scalar positive integer.")
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
    """Validate canonical RGB uint8 TCHW; layout conversion belongs to the adapter."""
    if not isinstance(video, torch.Tensor) or video.dtype != torch.uint8:
        dtype = video.dtype if isinstance(video, torch.Tensor) else type(video)
        raise ValueError(f"Expected a uint8 video tensor, got {dtype}")
    if video.ndim != 4:
        raise ValueError(f"Expected an RGB video tensor with shape [T, 3, H, W], got {tuple(video.shape)}")

    if video.shape[1] != 3 or any(size <= 0 for size in video.shape):
        raise ValueError(f"Expected a nonempty RGB video with shape [T, 3, H, W], got {tuple(video.shape)}")
    return video


def visual_reward_frames(solution_image, extra_info: dict, frame_interval: int = 1) -> torch.Tensor:
    """Select decoded visual media and return NCHW frames using declared modality, never rank."""
    from verl_omni.pipelines.rollout_media import PREVIEW_ARTIFACT, ArtifactContractError, MediaArtifact, MediaSpec

    if isinstance(frame_interval, bool) or not isinstance(frame_interval, int) or frame_interval < 1:
        raise ValueError("frame_interval must be a positive integer")
    artifacts = extra_info.get("media_artifacts")
    if artifacts is not None:
        name = extra_info.get(PREVIEW_ARTIFACT)
        if name not in artifacts:
            raise ArtifactContractError(
                f"Decoded preview artifact={name!r} is absent; cannot score latent responses as pixels"
            )
        artifact = artifacts[name]
    else:
        kind = extra_info.get("media_kind")
        if kind not in ("image", "video"):
            raise ArtifactContractError("Visual rewards require named artifacts or an explicit media_kind")
        artifact = MediaArtifact(
            MediaSpec(kind, "decoded", "CHW" if kind == "image" else "TCHW", fps=extra_info.get("fps")),
            solution_image,
        )
    try:
        artifact.validate(context="visual reward", name="preview")
    except (TypeError, ValueError) as error:
        raise ArtifactContractError(str(error)) from error
    expected_layout = {"image": "CHW", "video": "TCHW"}.get(artifact.spec.modality)
    if artifact.spec.representation != "decoded" or artifact.spec.layout != expected_layout:
        raise ArtifactContractError(f"Visual rewards require canonical decoded media, got {artifact.spec}")
    if artifact.spec.modality == "image":
        return artifact.data.unsqueeze(0)
    return artifact.data[::frame_interval]


def image_tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert an explicitly canonical uint8 CHW image to RGB PIL."""
    from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec

    MediaArtifact(MediaSpec("image", "decoded", "CHW"), image).validate(context="PIL conversion", name="image")
    array = image.detach().permute(1, 2, 0).cpu().numpy()
    if image.shape[0] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array).convert("RGB")


def video_reward_tensor(solution_image, extra_info: dict, fps: float) -> torch.Tensor:
    """Select a named decoded video, checking its FPS against scorer configuration."""
    if "media_artifacts" not in extra_info:
        return solution_image
    from verl_omni.pipelines.rollout_media import ArtifactContractError, select_artifact

    artifact = select_artifact(
        extra_info["media_artifacts"],
        name=extra_info.get("preview_artifact"),
        modality="video",
        representation="decoded",
    )
    if artifact.spec.fps != fps:
        raise ArtifactContractError(f"Video reward FPS={fps} conflicts with preview FPS={artifact.spec.fps}")
    return visual_reward_frames(solution_image, extra_info)


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
