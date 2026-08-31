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
"""MiniCPM sample transform for offline MLLM DPO.

The transform consumes the same intermediate conversation format as
``OfflineMLLMDPODataset`` and renders MiniCPM-style turn-based messages with
``<image>``, ``<video>``, and ``<audio>`` placeholders.  It intentionally keeps
the dependency surface small: MiniCPM's official ``AutoProcessor`` /
``AutoTokenizer`` path is used when available, while ``minicpmo.utils`` remains
optional for training-only preprocessing.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import torch

from verl_omni.utils.dataset.qwen3_omni_transform import (
    AUDIO_INPUT_INDEX,
    IGNORE_INDEX,
    IMAGE_INPUT_INDEX,
    VIDEO_INPUT_INDEX,
    _fetch_audios,
    _fetch_images,
    _fetch_videos,
)

__all__ = ["process_minicpm_sample"]

_MODALITY_PLACEHOLDERS = {
    "image": "<image>",
    "video": "<video>",
    "audio": "<audio>",
}


def _tokenizer_from_processor(processor):
    return getattr(processor, "tokenizer", processor)


def _convert_token_to_id(tokenizer, token: str) -> int | None:
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
    except Exception:
        return None
    if token_id is None:
        return None
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    if token_id == unk_token_id:
        return None
    try:
        return int(token_id)
    except (TypeError, ValueError):
        return None


def _get_media_token_ids(processor) -> dict[str, int]:
    tokenizer = _tokenizer_from_processor(processor)
    candidates = {
        "image": ("<image>", "<|image_pad|>", "<|IMAGE|>"),
        "video": ("<video>", "<|video_pad|>", "<|VIDEO|>"),
        "audio": ("<audio>", "<|audio_pad|>", "<|AUDIO|>"),
    }
    token_ids: dict[str, int] = {}
    for modality, tokens in candidates.items():
        for token in tokens:
            token_id = _convert_token_to_id(tokenizer, token)
            if token_id is not None:
                token_ids[modality] = token_id
                break
    return token_ids


def _append_content_text(parts: list[str], item: Any) -> None:
    if isinstance(item, dict):
        if item.get("type") == "text":
            parts.append(str(item.get("text", "")))
        elif "text" in item:
            parts.append(str(item["text"]))
        else:
            parts.append(str(item))
        return
    parts.append(str(item))


def _conversation_to_message(conversation: Sequence[Any]) -> dict[str, str]:
    role = str(conversation[0] or "user")
    parts: list[str] = []
    for item in conversation[1:]:
        if isinstance(item, (list | tuple)) and item:
            item_type = str(item[0])
            placeholder = _MODALITY_PLACEHOLDERS.get(item_type)
            if placeholder is not None:
                parts.append(placeholder)
                if len(item) > 1 and item[1]:
                    parts.append(str(item[1]))
                continue
            if item_type == "text" and len(item) > 1:
                parts.append(str(item[1]))
                continue
        _append_content_text(parts, item)
    return {"role": role, "content": "".join(parts)}


def _build_minicpm_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    conversations = sample["conversations"] if ("conversations" in sample and sample["conversations"]) else sample
    return [_conversation_to_message(conversation) for conversation in conversations]


def _mark_final_assistant_content(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, str] | None]:
    marked_messages = copy.deepcopy(messages)
    final_assistant_index = None
    for index in range(len(marked_messages) - 1, -1, -1):
        if marked_messages[index].get("role") == "assistant":
            final_assistant_index = index
            break
    if final_assistant_index is None:
        return marked_messages, None

    start_marker = "__verl_omni_minicpm_final_assistant_start__"
    end_marker = "__verl_omni_minicpm_final_assistant_end__"
    content = marked_messages[final_assistant_index].get("content", "")
    marked_messages[final_assistant_index]["content"] = f"{start_marker}{content}{end_marker}"
    return marked_messages, (start_marker, end_marker)


def _render_chat(messages: list[dict[str, Any]], processor, **kwargs) -> str:
    tokenizer = _tokenizer_from_processor(processor)
    template_kwargs = {
        key: kwargs[key]
        for key in (
            "use_image_id",
            "max_slice_nums",
            "slice_mode",
            "downsample_mode",
        )
        if key in kwargs
    }
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, **template_kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False)


def _processor_data(output: Any) -> dict[str, Any]:
    if hasattr(output, "data"):
        output = output.data
    if not isinstance(output, dict):
        raise TypeError(f"MiniCPM processor must return a dict-like object, got {type(output).__name__}.")
    return dict(output)


def _without_empty_media(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in kwargs.items() if value is not None and not (isinstance(value, list) and not value)
    }


def _call_processor(processor, *, text: str, images: list[Any], videos: list[Any], audios: list[Any]) -> dict[str, Any]:
    attempts = [
        {"text": text, "images": images, "videos": videos, "audios": audios, "return_tensors": "pt", "padding": True},
        {"text": [text], "images": images, "videos": videos, "audios": audios, "return_tensors": "pt", "padding": True},
        {"text": text, "image": images, "video": videos, "audio": audios, "return_tensors": "pt", "padding": True},
    ]
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return _processor_data(processor(**_without_empty_media(kwargs)))
        except TypeError as exc:
            last_error = exc
            continue
    tokenizer = _tokenizer_from_processor(processor)
    try:
        return _processor_data(tokenizer(text, return_tensors="pt", padding=True))
    except TypeError as exc:
        raise TypeError("MiniCPM processor rejected the supported training input signatures.") from last_error or exc


def _squeeze_batch_dim(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == 1:
        return value.squeeze(0).contiguous()
    return value


def _align_template_tokens_to_processor_tokens(
    token_ids: Sequence[int],
    processor_input_ids: torch.Tensor,
    media_token_ids: set[int],
) -> list[int]:
    token_to_processor_pos: list[int] = []
    processor_ids = processor_input_ids.tolist()
    processor_i = 0
    for token_id in token_ids:
        if processor_i >= len(processor_ids):
            break
        if processor_ids[processor_i] != token_id:
            token_to_processor_pos.append(processor_i)
            processor_i += 1
            continue
        token_to_processor_pos.append(processor_i)
        processor_i += 1
        if token_id in media_token_ids:
            while processor_i < len(processor_ids) and processor_ids[processor_i] == token_id:
                processor_i += 1
    return token_to_processor_pos


def _final_assistant_token_mask(
    messages: list[dict[str, Any]],
    processor,
    rendered_text: str,
    input_ids: torch.Tensor,
    media_token_ids: set[int],
    **kwargs,
) -> torch.Tensor:
    tokenizer = _tokenizer_from_processor(processor)
    marked_messages, markers = _mark_final_assistant_content(messages)
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    if markers is None:
        return loss_mask

    marked_text = _render_chat(marked_messages, processor, **kwargs)
    start_marker, end_marker = markers
    start = marked_text.find(start_marker)
    if start < 0:
        raise ValueError("Cannot locate MiniCPM assistant start marker in rendered chat template.")
    stripped_text = marked_text[:start] + marked_text[start + len(start_marker) :]
    end = stripped_text.find(end_marker, start)
    if end < 0:
        raise ValueError("Cannot locate MiniCPM assistant end marker in rendered chat template.")
    stripped_text = stripped_text[:end] + stripped_text[end + len(end_marker) :]
    if stripped_text != rendered_text:
        raise ValueError("Marked MiniCPM chat template rendering does not match the unmarked rendering.")

    tokenized = tokenizer(rendered_text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = tokenized.get("offset_mapping")
    token_ids = tokenized.get("input_ids")
    if offsets is None or token_ids is None:
        raise ValueError("MiniCPM tokenizer must provide offset_mapping to build final-assistant DPO labels.")

    token_to_processor_pos = _align_template_tokens_to_processor_tokens(token_ids, input_ids, media_token_ids)
    for token_i, (token_start, token_end) in enumerate(offsets[: len(token_to_processor_pos)]):
        if token_start == token_end:
            continue
        if token_start < end and token_end > start:
            loss_mask[token_to_processor_pos[token_i]] = True
    return loss_mask


def _default_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)


def process_minicpm_sample(sample: dict[str, Any], processor, position_id_func=None, **kwargs) -> list[dict[str, Any]]:
    """Transform one offline preference sample into MiniCPM model inputs."""

    messages = _build_minicpm_messages(sample)
    rendered_text = _render_chat(messages, processor, **kwargs)
    images = _fetch_images(sample.get("images", []), **kwargs) if sample.get("images") else []
    videos, _video_audios = _fetch_videos(sample.get("videos", []), **kwargs) if sample.get("videos") else ([], [])
    audios = _fetch_audios(sample.get("audios", []), **kwargs) if sample.get("audios") else []

    model_inputs = _call_processor(processor, text=rendered_text, images=images, videos=videos, audios=audios)
    model_inputs = {key: _squeeze_batch_dim(value) for key, value in model_inputs.items()}

    if "input_ids" not in model_inputs:
        raise ValueError("MiniCPM processor output must contain input_ids.")
    input_ids = model_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        model_inputs["input_ids"] = input_ids
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(input_ids)

    media_token_ids_by_modality = _get_media_token_ids(processor)
    media_token_ids = set(media_token_ids_by_modality.values())
    for modality, token_id in media_token_ids_by_modality.items():
        mask = input_ids == token_id
        model_inputs[f"{modality}_mask"] = mask
        if modality == "image":
            model_inputs["input_ids"] = torch.where(
                mask, torch.full_like(input_ids, IMAGE_INPUT_INDEX), model_inputs["input_ids"]
            )
        elif modality == "video":
            model_inputs["input_ids"] = torch.where(
                mask, torch.full_like(input_ids, VIDEO_INPUT_INDEX), model_inputs["input_ids"]
            )
        elif modality == "audio":
            model_inputs["input_ids"] = torch.where(
                mask, torch.full_like(input_ids, AUDIO_INPUT_INDEX), model_inputs["input_ids"]
            )

    raw_input_ids = input_ids.clone()

    if "position_ids" not in model_inputs:
        if position_id_func is not None:
            position_returns = position_id_func(
                input_ids=model_inputs["input_ids"].unsqueeze(0),
                attention_mask=model_inputs["attention_mask"].unsqueeze(0),
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
            )
            if isinstance(position_returns, dict):
                model_inputs["position_ids"] = _squeeze_batch_dim(position_returns["position_ids"])
            else:
                model_inputs["position_ids"] = _squeeze_batch_dim(position_returns[0])
        else:
            model_inputs["position_ids"] = _default_position_ids(input_ids)

    labels = torch.full_like(raw_input_ids, fill_value=IGNORE_INDEX)
    assistant_mask = _final_assistant_token_mask(
        messages,
        processor,
        rendered_text,
        raw_input_ids,
        media_token_ids,
        **kwargs,
    )
    labels[assistant_mask] = raw_input_ids[assistant_mask]
    model_inputs["input_ids"] = raw_input_ids
    model_inputs["labels"] = labels
    return [model_inputs]
