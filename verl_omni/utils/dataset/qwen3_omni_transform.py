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

"""Qwen3-Omni sample transform for offline MLLM DPO without a VeOmni dependency.

The logic mirrors VeOmni's ``process_sample_qwen_omni`` data transform and the
minimal multimodal media helpers it relies on.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IGNORE_INDEX = -100
IMAGE_INPUT_INDEX = -200
VIDEO_INPUT_INDEX = -300
AUDIO_INPUT_INDEX = -400

QWEN_OMNI_SYSTEM_MESSAGE = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)

__all__ = ["process_qwen3_omni_sample"]


def _align_to_factor_remainder_floor(n: float, factor: int, remainder: int) -> int:
    adjusted = n - remainder
    return int(adjusted // factor) * factor + remainder


def _align_to_factor_remainder_ceil(n: float, factor: int, remainder: int) -> int:
    adjusted = n - remainder
    return math.ceil(adjusted / factor) * factor + remainder


def _smart_resize_image(
    image: Image.Image,
    *,
    scale_factor: int | None = None,
    image_min_pixels: int | None = None,
    image_max_pixels: int | None = None,
    max_ratio: int | None = None,
    **kwargs,
) -> Image.Image:
    del kwargs
    width, height = image.size
    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if image_max_pixels is not None and h_bar * w_bar > image_max_pixels:
        beta = math.sqrt((height * width) / image_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)
    if image_min_pixels is not None and h_bar * w_bar < image_min_pixels:
        beta = math.sqrt(image_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)
    return image.resize((w_bar, h_bar))


def _fetch_images(images: Sequence[str], **kwargs) -> list[Image.Image]:
    max_image_nums = kwargs.get("max_image_nums", len(images))
    loaded = [Image.open(path).convert("RGB") for path in images[:max_image_nums]]
    return [_smart_resize_image(image, **kwargs) for image in loaded]


def _calculate_frame_indices(
    total_frames: int,
    video_fps: float,
    *,
    fps: float = 2.0,
    frame_factor: int | None = None,
    frame_factor_remainder: int = 0,
    min_frames: int | None = None,
    max_frames: int | None = None,
    **kwargs,
) -> tuple[list[int], int]:
    del kwargs
    r = frame_factor_remainder
    if frame_factor is not None:
        if frame_factor <= 0:
            raise ValueError(f"frame_factor must be a positive integer, got {frame_factor}")
        if not 0 <= r < frame_factor:
            raise ValueError(f"frame_factor_remainder must be in [0, {frame_factor}), got {r}")

    nframes = total_frames / video_fps * fps
    if min_frames is not None:
        if frame_factor is not None:
            min_frames = _align_to_factor_remainder_ceil(min_frames, frame_factor, r)
        nframes = max(min_frames, nframes)
    if max_frames is not None:
        if frame_factor is not None:
            max_frames = _align_to_factor_remainder_floor(max_frames, frame_factor, r)
        nframes = min(max_frames, nframes)
    if frame_factor is not None:
        nframes = _align_to_factor_remainder_floor(nframes, frame_factor, r)
        min_valid = r if r > 0 else frame_factor
        nframes = max(nframes, min_valid)

    nframes = int(max(1, nframes))
    pad_count = max(0, nframes - total_frames)
    sample_count = min(nframes, total_frames)
    if sample_count > 0:
        indices = np.linspace(0, total_frames - 1, sample_count).round().astype(int).tolist()
    else:
        indices = []
    return indices, pad_count


def _smart_resize_video(
    video: torch.Tensor,
    *,
    scale_factor: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    max_ratio: int | None = None,
    **kwargs,
) -> torch.Tensor:
    del kwargs
    if video.ndim != 4:
        raise ValueError(f"video must be 4-dim, but got {video.ndim}")
    _, _, height, width = video.shape
    if max_ratio is not None:
        ratio = max(width, height) / min(width, height)
        if ratio > max_ratio:
            raise ValueError(f"absolute aspect ratio must be smaller than {max_ratio}, got {ratio}")

    if scale_factor is not None:
        h_bar = max(scale_factor, round(height / scale_factor) * scale_factor)
        w_bar = max(scale_factor, round(width / scale_factor) * scale_factor)
    else:
        h_bar = height
        w_bar = width

    if video_max_pixels is not None and h_bar * w_bar > video_max_pixels:
        beta = math.sqrt((height * width) / video_max_pixels)
        if scale_factor is not None:
            h_bar = math.floor(height / beta / scale_factor) * scale_factor
            w_bar = math.floor(width / beta / scale_factor) * scale_factor
        else:
            h_bar = math.floor(height / beta)
            w_bar = math.floor(width / beta)
    if video_min_pixels is not None and h_bar * w_bar < video_min_pixels:
        beta = math.sqrt(video_min_pixels / (height * width))
        if scale_factor is not None:
            h_bar = math.ceil(height * beta / scale_factor) * scale_factor
            w_bar = math.ceil(width * beta / scale_factor) * scale_factor
        else:
            h_bar = math.ceil(height * beta)
            w_bar = math.ceil(width * beta)

    return F.interpolate(video, size=(h_bar, w_bar), mode="bicubic", align_corners=False).float()


def _smart_video_nframes(
    video: torch.Tensor,
    video_fps: float,
    *,
    fps: float = 2.0,
    frame_factor: int | None = None,
    min_frames: int | None = None,
    max_frames: int | None = None,
    **kwargs,
) -> torch.Tensor:
    indices, pad_count = _calculate_frame_indices(
        total_frames=video.shape[0],
        video_fps=video_fps,
        fps=fps,
        frame_factor=frame_factor,
        min_frames=min_frames,
        max_frames=max_frames,
        **kwargs,
    )
    video = video[indices]
    if pad_count > 0:
        last_frame = video[-1:].expand(pad_count, -1, -1, -1)
        video = torch.cat([video, last_frame], dim=0)
    return video


def _pil_images_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    tensors = []
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensors.append(torch.from_numpy(np.array(img)).permute(2, 0, 1))
    return torch.stack(tensors)


def _decode_video_path(video_path: str) -> tuple[torch.Tensor, float]:
    try:
        import av
    except ImportError as exc:
        raise ImportError("PyAV (`pip install av`) is required to decode video files for Qwen3-Omni DPO.") from exc

    container = av.open(video_path)
    if not container.streams.video:
        container.close()
        raise ValueError(f"Video {video_path} contains no video streams.")
    stream = container.streams.video[0]
    video_fps = float(stream.average_rate) if stream.average_rate else 2.0
    frames = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
    container.close()
    if not frames:
        raise ValueError(f"Video {video_path} contains no decodable frames.")
    return _pil_images_to_tensor(frames), video_fps


def _fetch_videos(videos: Sequence[str], **kwargs) -> tuple[list[torch.Tensor], list[Any]]:
    video_inputs: list[torch.Tensor] = []
    audio_inputs: list[Any] = []
    for video in videos:
        if not isinstance(video, str):
            raise NotImplementedError("Only local video file paths are supported.")
        tensor, video_fps = _decode_video_path(video)
        tensor = _smart_video_nframes(_smart_resize_video(tensor, **kwargs), video_fps, **kwargs)
        video_inputs.append(tensor)
        audio_inputs.append(None)
    return video_inputs, audio_inputs


def _load_audio_from_path(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required to load audio files for Qwen3-Omni DPO. Install with `pip install librosa`."
        ) from exc
    return librosa.load(audio_path, sr=sample_rate)[0]


def _fetch_audios(audios: Sequence[str], **kwargs) -> list[np.ndarray]:
    sample_rate = kwargs.get("sample_rate", 16000)
    return [_load_audio_from_path(audio, sample_rate=sample_rate) for audio in audios]


def _get_omni_token_ids(processor) -> tuple[int, int, int]:
    tokenizer = getattr(processor, "tokenizer", processor)
    vocab = tokenizer.get_vocab()
    image_token_id = vocab.get("<|image_pad|>", vocab.get("<|IMAGE|>"))
    video_token_id = vocab.get("<|video_pad|>", vocab.get("<|VIDEO|>"))
    audio_token_id = vocab.get("<|audio_pad|>", vocab.get("<|AUDIO|>"))
    if image_token_id is None:
        raise ValueError("Cannot find image token (<|image_pad|> or <|IMAGE|>) in tokenizer vocab.")
    if video_token_id is None:
        raise ValueError("Cannot find video token (<|video_pad|> or <|VIDEO|>) in tokenizer vocab.")
    if audio_token_id is None:
        raise ValueError("Cannot find audio token (<|audio_pad|> or <|AUDIO|>) in tokenizer vocab.")
    return image_token_id, video_token_id, audio_token_id


def _mark_assistant_content(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    marked_messages = copy.deepcopy(messages)
    markers: list[tuple[str, str]] = []
    for index, message in enumerate(marked_messages):
        if message.get("role") != "assistant":
            continue
        start_marker = f"__verl_omni_assistant_start_{index}__"
        end_marker = f"__verl_omni_assistant_end_{index}__"
        markers.append((start_marker, end_marker))
        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = f"{start_marker}{content}{end_marker}"
        else:
            message["content"] = [
                {"type": "text", "text": start_marker},
                *content,
                {"type": "text", "text": end_marker},
            ]
    return marked_messages, markers


def _assistant_char_spans_from_template(
    input_conversations: list[dict[str, Any]],
    processor,
    rendered_text: str,
) -> list[tuple[int, int]]:
    """Locate assistant content spans from the structured chat template."""
    marked_conversations, markers = _mark_assistant_content(input_conversations)
    if not markers:
        return []

    marked_text = processor.apply_chat_template(marked_conversations, tokenize=False)
    spans: list[tuple[int, int]] = []
    stripped_text = marked_text
    for start_marker, end_marker in markers:
        start = stripped_text.find(start_marker)
        if start < 0:
            raise ValueError("Cannot locate assistant start marker in rendered Qwen3-Omni chat template.")
        stripped_text = stripped_text[:start] + stripped_text[start + len(start_marker) :]
        end = stripped_text.find(end_marker, start)
        if end < 0:
            raise ValueError("Cannot locate assistant end marker in rendered Qwen3-Omni chat template.")
        spans.append((start, end))
        stripped_text = stripped_text[:end] + stripped_text[end + len(end_marker) :]

    if stripped_text != rendered_text:
        raise ValueError("Marked Qwen3-Omni chat template rendering does not match the unmarked rendering.")
    return spans


def _assistant_token_mask_from_template(
    input_conversations: list[dict[str, Any]],
    processor,
    rendered_text: str,
    input_ids: torch.Tensor,
    media_token_ids: tuple[int, int, int],
) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    char_spans = _assistant_char_spans_from_template(input_conversations, processor, rendered_text)
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    if not char_spans:
        return loss_mask

    tokenized = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = tokenized["offset_mapping"]
    token_ids = tokenized["input_ids"]
    token_to_processor_pos = _align_template_tokens_to_processor_tokens(token_ids, input_ids, set(media_token_ids))

    span_i = 0
    for token_i, (start, end) in enumerate(offsets):
        if start == end:
            continue
        while span_i < len(char_spans) and end > char_spans[span_i][1]:
            span_i += 1
        if span_i >= len(char_spans):
            break
        span_start, span_end = char_spans[span_i]
        if start < span_end and end > span_start:
            loss_mask[token_to_processor_pos[token_i]] = True
    return loss_mask


def _align_template_tokens_to_processor_tokens(
    token_ids: Sequence[int],
    processor_input_ids: torch.Tensor,
    media_token_ids: set[int],
) -> list[int]:
    """Map chat-template token positions to processor-expanded input positions."""
    token_to_processor_pos: list[int] = []
    processor_ids = processor_input_ids.tolist()
    processor_i = 0
    for token_i, token_id in enumerate(token_ids):
        if processor_i >= len(processor_ids):
            raise ValueError(
                "Tokenizer chat-template tokens are longer than processor input_ids while building assistant labels."
            )
        if processor_ids[processor_i] != token_id:
            raise ValueError(
                "Cannot align tokenizer chat-template tokens with processor input_ids at "
                f"token index {token_i}: tokenizer id {token_id}, processor id {processor_ids[processor_i]}."
            )
        token_to_processor_pos.append(processor_i)
        processor_i += 1
        if token_id in media_token_ids:
            while processor_i < len(processor_ids) and processor_ids[processor_i] == token_id:
                processor_i += 1

    if processor_i != len(processor_ids):
        raise ValueError(
            "Processor input_ids contain trailing tokens that are not present in the rendered chat template "
            f"({len(processor_ids) - processor_i} extra token(s))."
        )
    return token_to_processor_pos


def process_qwen3_omni_sample(
    sample: dict[str, Any],
    processor,
    position_id_func: Callable,
    **kwargs,
) -> list[dict[str, Any]]:
    """Transform one offline preference sample into Qwen3-Omni model inputs."""
    image_token_id, video_token_id, audio_token_id = _get_omni_token_ids(processor)
    conversations = (
        sample["conversations"] if ("conversations" in sample and len(sample["conversations"]) > 0) else sample
    )

    input_conversations = [
        {
            "role": "system",
            "content": [{"type": "text", "text": QWEN_OMNI_SYSTEM_MESSAGE}],
        },
    ]
    for conversation in conversations:
        contents = []
        for message in conversation[1:]:
            contents.append({"type": message[0], message[0]: message[1]})
        input_conversations.append({"role": conversation[0], "content": contents})
    text = processor.apply_chat_template(input_conversations, tokenize=False)

    images = _fetch_images(sample.get("images", []), **kwargs) if sample.get("images") else []
    videos, video_audios = _fetch_videos(sample.get("videos", []), **kwargs) if sample.get("videos") else ([], [])
    audio_audios = _fetch_audios(sample.get("audios", []), **kwargs) if sample.get("audios") else []

    video_audios_iter = iter(video_audios)
    audio_audios_iter = iter(audio_audios)
    audios = []
    for item in input_conversations:
        for content in item["content"]:
            if content["type"] == "video":
                audios.append(next(video_audios_iter))
            elif content["type"] == "audio":
                audios.append(next(audio_audios_iter))

    model_inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
    )
    model_inputs = model_inputs.data
    feature_attention_mask = model_inputs.get("feature_attention_mask", None)

    if feature_attention_mask is not None:
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        model_inputs["audio_feature_lengths"] = audio_feature_lengths
    else:
        audio_feature_lengths = None

    input_ids = model_inputs["input_ids"].squeeze(0)
    raw_input_ids = input_ids.clone()
    image_mask = input_ids == image_token_id
    video_mask = input_ids == video_token_id
    audio_mask = input_ids == audio_token_id
    input_ids[image_mask] = IMAGE_INPUT_INDEX
    input_ids[video_mask] = VIDEO_INPUT_INDEX
    input_ids[audio_mask] = AUDIO_INPUT_INDEX

    position_id_returns = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=model_inputs.get("image_grid_thw", None),
        video_grid_thw=model_inputs.get("video_grid_thw", None),
        attention_mask=model_inputs["attention_mask"],
        audio_seqlens=audio_feature_lengths,
        second_per_grids=model_inputs.pop("video_second_per_grid", None),
    )
    position_id_returns["position_ids"] = position_id_returns["position_ids"].clone()
    model_inputs["position_ids"] = position_id_returns["position_ids"]

    model_inputs["image_mask"] = image_mask
    model_inputs["video_mask"] = video_mask
    model_inputs["audio_mask"] = audio_mask
    input_ids[image_mask | video_mask | audio_mask] = 0
    model_inputs["input_ids"] = input_ids
    model_inputs["attention_mask"] = model_inputs["attention_mask"].squeeze(0)

    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    assistant_loss_mask = _assistant_token_mask_from_template(
        input_conversations,
        processor,
        text,
        raw_input_ids,
        (image_token_id, video_token_id, audio_token_id),
    )
    labels[assistant_loss_mask] = input_ids[assistant_loss_mask]
    model_inputs["labels"] = labels
    return [model_inputs]
