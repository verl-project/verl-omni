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
    from verl_omni.pipelines.rollout_artifacts import PREVIEW_ARTIFACT, ArtifactContractError, MediaArtifact
    from verl_omni.pipelines.rollout_media import MediaSpec

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
    from verl_omni.pipelines.rollout_artifacts import MediaArtifact
    from verl_omni.pipelines.rollout_media import MediaSpec

    MediaArtifact(MediaSpec("image", "decoded", "CHW"), image).validate(context="PIL conversion", name="image")
    array = image.detach().permute(1, 2, 0).cpu().numpy()
    if image.shape[0] == 1:
        array = array[:, :, 0]
    return Image.fromarray(array).convert("RGB")


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
