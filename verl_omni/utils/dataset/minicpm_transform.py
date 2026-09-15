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
``OfflineMLLMDPODataset``.  Parquet rows should keep compact semantic markers
(``<image>``, ``<video>``, ``<audio>``) in prompt text; this module rewrites
them to MiniCPM processor slots (``<image>./</image>``, ``<audio>./</audio>``)
after chat-template rendering and before calling ``MiniCPMOProcessor``.

Training should use two batch kinds only:

* **image-only** batches from ``image/*.parquet`` rows.
* **audio-only** batches from ``audio/*.parquet`` rows with standalone ``audios`` paths.

Do not use ``video/*.parquet`` for MiniCPM DPO: the remote processor has no video
input path, and the previous video+audio workaround only fed decoded audio anyway.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from verl_omni.utils.dataset.qwen3_omni_transform import (
    IGNORE_INDEX,
    _fetch_audios,
    _fetch_images,
)

__all__ = ["process_minicpm_sample"]

# Remote ``processing_minicpmo.MiniCPMOProcessor`` scans for these inline slots in text.
_MINICPM_IMAGE_SLOT = "<image>./</image>"
_MINICPM_AUDIO_SLOT = "<audio>./</audio>"

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
    del kwargs
    tokenizer = _tokenizer_from_processor(processor)
    return tokenizer.apply_chat_template(messages, tokenize=False)


def _replace_bare_media_tokens(text: str, modality: str, slot: str) -> str:
    token = f"<{modality}>"
    if slot in text:
        return text
    pattern = re.compile(rf"{re.escape(token)}(?!\./</{modality}>)")
    return pattern.sub(slot, text)


def _inject_processor_media_slots(
    text: str,
    *,
    image_count: int,
    audio_count: int,
    video_count: int = 0,
    use_audio_in_video: bool = False,
) -> str:
    """Rewrite bare ``<image>`` / ``<video>`` / ``<audio>`` markers into MiniCPM processor slots."""

    if use_audio_in_video and video_count > 0:
        text = _replace_bare_media_tokens(text, "video", _MINICPM_AUDIO_SLOT)
    text = _replace_bare_media_tokens(text, "image", _MINICPM_IMAGE_SLOT)
    text = _replace_bare_media_tokens(text, "audio", _MINICPM_AUDIO_SLOT)

    if image_count and text.count(_MINICPM_IMAGE_SLOT) != image_count:
        raise ValueError(
            f"MiniCPM processor expects {image_count} {_MINICPM_IMAGE_SLOT!r} slot(s) in text, "
            f"found {text.count(_MINICPM_IMAGE_SLOT)} after chat-template rendering."
        )
    if audio_count and text.count(_MINICPM_AUDIO_SLOT) != audio_count:
        raise ValueError(
            f"MiniCPM processor expects {audio_count} {_MINICPM_AUDIO_SLOT!r} slot(s) in text, "
            f"found {text.count(_MINICPM_AUDIO_SLOT)} after chat-template rendering."
        )
    return text


def _render_processor_text(
    messages: list[dict[str, Any]],
    processor,
    *,
    image_count: int,
    audio_count: int,
    video_count: int,
    use_audio_in_video: bool,
    **kwargs,
) -> str:
    rendered = _render_chat(messages, processor, **kwargs)
    return _inject_processor_media_slots(
        rendered,
        image_count=image_count,
        audio_count=audio_count,
        video_count=video_count,
        use_audio_in_video=use_audio_in_video,
    )


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


def _call_processor(
    processor,
    *,
    text: str,
    images: list[Any],
    videos: list[Any],
    audios: list[Any],
    **kwargs,
) -> dict[str, Any]:
    processor_kwargs = {
        key: kwargs[key]
        for key in ("max_slice_nums", "use_image_id", "max_length", "sampling_rate")
        if key in kwargs and kwargs[key] is not None
    }
    if "sampling_rate" not in processor_kwargs and kwargs.get("sample_rate") is not None:
        processor_kwargs["sampling_rate"] = kwargs["sample_rate"]
    attempts = [
        {
            "text": text,
            "images": images,
            "videos": videos,
            "audios": audios,
            "return_tensors": "pt",
            "padding": True,
            **processor_kwargs,
        },
        {
            "text": [text],
            "images": images,
            "videos": videos,
            "audios": audios,
            "return_tensors": "pt",
            "padding": True,
            **processor_kwargs,
        },
        {
            "text": text,
            "image": images,
            "video": videos,
            "audio": audios,
            "return_tensors": "pt",
            "padding": True,
            **processor_kwargs,
        },
    ]
    last_error: Exception | None = None
    for attempt_kwargs in attempts:
        try:
            return _processor_data(processor(**_without_empty_media(attempt_kwargs)))
        except TypeError as exc:
            last_error = exc
            continue
    # Empty media lists are stripped before the processor call, so a text-only
    # sample already reaches the processor as ``text`` / ``padding`` only.
    if images or videos or audios:
        raise TypeError(
            "MiniCPM processor rejected the supported training input signatures "
            f"while media was present (images={len(images)}, audios={len(audios)}, "
            f"videos={len(videos)}). Refusing a text-only tokenizer fallback."
        ) from last_error
    tokenizer = _tokenizer_from_processor(processor)
    try:
        return _processor_data(tokenizer(text, return_tensors="pt", padding=True))
    except TypeError as exc:
        raise TypeError(
            "MiniCPM processor rejected the supported training input signatures, "
            "and the tokenizer also failed for a text-only sample."
        ) from last_error or exc


def _squeeze_batch_dim(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[0] == 1:
        return value.squeeze(0).contiguous()
    return value


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _is_empty_audio_features(value: Any) -> bool:
    """True when there are no mel frames, including collated empty placeholders.

    MiniCPMO uses ``len(data['audio_features']) > 0`` to decide whether a batch
    has audio. A collated image-only batch is often ``[[], [], ...]``, which has
    length equal to the text batch size even though no clip exists.
    """
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, tuple)):
        return len(value) == 0 or all(_is_empty_audio_features(item) for item in value)
    return False


def _one_sample_audio_feature_lens(sample: Any, device: torch.device) -> torch.Tensor:
    if sample is None or (isinstance(sample, (list, tuple)) and not sample):
        return torch.zeros(0, dtype=torch.long, device=device)
    if isinstance(sample, torch.Tensor):
        return sample.to(device=device, dtype=torch.long).reshape(-1).contiguous()
    if isinstance(sample, np.ndarray):
        return torch.as_tensor(sample, dtype=torch.long, device=device).reshape(-1).contiguous()
    if isinstance(sample, (list, tuple)):
        if len(sample) == 1 and isinstance(sample[0], (list, tuple, torch.Tensor, np.ndarray)):
            return _one_sample_audio_feature_lens(sample[0], device)
        if sample and not isinstance(sample[0], (int, float, np.integer, np.floating)):
            return torch.cat([_one_sample_audio_feature_lens(item, device) for item in sample], dim=0)
        return torch.as_tensor(sample, dtype=torch.long, device=device).reshape(-1).contiguous()
    return torch.as_tensor(sample, dtype=torch.long, device=device).reshape(-1).contiguous()


def _batch_audio_feature_lens(value: Any, device: torch.device) -> list[torch.Tensor]:
    """List of 1D tensors so ``torch.hstack(audio_feature_lens_raw)`` succeeds."""
    if _is_empty_audio_features(value):
        return []
    if isinstance(value, torch.Tensor) and value.ndim <= 1:
        return [_one_sample_audio_feature_lens(value, device)]
    if isinstance(value, (list, tuple)):
        return [_one_sample_audio_feature_lens(sample, device) for sample in value]
    return [_one_sample_audio_feature_lens(value, device)]


def _normalize_audio_features(value: Any) -> torch.Tensor | list:
    """Empty batches become ``[]``; real clips become ``(n_clips, 80, frames)``."""
    if _is_empty_audio_features(value):
        return []
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            return value.unsqueeze(0).contiguous()
        return value.contiguous()
    if isinstance(value, np.ndarray):
        return _normalize_audio_features(torch.as_tensor(value))
    if isinstance(value, (list, tuple)):
        clips: list[torch.Tensor] = []
        for item in value:
            if _is_empty_audio_features(item):
                continue
            packed = _normalize_audio_features(item)
            if isinstance(packed, list):
                continue
            if packed.ndim == 2:
                clips.append(packed)
            else:
                clips.extend(clip.contiguous() for clip in packed)
        if not clips:
            return []
        max_frames = max(int(clip.shape[-1]) for clip in clips)
        padded = []
        for clip in clips:
            if clip.ndim != 2:
                raise ValueError(f"MiniCPM audio clip must be (n_mels, frames), got shape {tuple(clip.shape)}.")
            pad = max_frames - int(clip.shape[-1])
            if pad:
                clip = torch.nn.functional.pad(clip, (0, pad))
            padded.append(clip)
        return torch.stack(padded, dim=0)
    return _normalize_audio_features(_as_tensor(value))


def _sample_pixel_slices(pixel_values: Any) -> list[torch.Tensor]:
    """Per-sample slices for MiniCPMO.get_vision_embedding.

    The processor returns a batch list ``[[slice, slice, ...]]``.  Remote code
    then does ``i.flatten(end_dim=1)`` on each slice, so each ``i`` must be a
    tensor, not another list.
    """
    if pixel_values is None:
        return []
    if isinstance(pixel_values, torch.Tensor):
        if pixel_values.numel() == 0:
            return []
        if pixel_values.ndim >= 3:
            return [slice_tensor.contiguous() for slice_tensor in pixel_values]
        return [pixel_values.contiguous()]
    if isinstance(pixel_values, np.ndarray):
        return _sample_pixel_slices(torch.as_tensor(pixel_values))
    if isinstance(pixel_values, (list, tuple)):
        if not pixel_values:
            return []
        first = pixel_values[0]
        if isinstance(first, (list, tuple)):
            if len(pixel_values) == 1:
                return _sample_pixel_slices(first)
            slices: list[torch.Tensor] = []
            for item in pixel_values:
                slices.extend(_sample_pixel_slices(item))
            return slices
        return [_as_tensor(item).contiguous() for item in pixel_values]
    return [_as_tensor(pixel_values).contiguous()]


def _sample_tgt_sizes(tgt_sizes: Any, *, n_slices: int, device: torch.device) -> torch.Tensor:
    if tgt_sizes is None or (isinstance(tgt_sizes, (list | tuple)) and not tgt_sizes):
        return torch.zeros(0, 2, dtype=torch.int32, device=device)
    if isinstance(tgt_sizes, (list, tuple)) and len(tgt_sizes) == 1 and not isinstance(tgt_sizes[0], (int, float)):
        inner = tgt_sizes[0]
        if isinstance(inner, (list | tuple | np.ndarray | torch.Tensor)):
            return _sample_tgt_sizes(inner, n_slices=n_slices, device=device)
    sizes = torch.as_tensor(tgt_sizes, dtype=torch.int32, device=device)
    if sizes.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.int32, device=device)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(0)
    return sizes.reshape(-1, 2).contiguous()


def _encode_like_processor(tokenizer, text: str, *, max_length: int | None = None) -> list[int]:
    """Mirror ``MiniCPMOProcessor._convert`` tokenization (drop ``<|listen|>``)."""

    listen_token_id = _convert_token_to_id(tokenizer, "<|listen|>")
    encode = getattr(tokenizer, "encode", None)
    if encode is not None:
        raw_ids = [int(token) for token in encode(text)]
    else:
        result = tokenizer(text, add_special_tokens=False)
        raw_ids = [int(token) for token in result["input_ids"]]

    token_ids: list[int] = []
    for token in raw_ids:
        if listen_token_id is not None and token == listen_token_id:
            continue
        token_ids.append(token)
    if max_length is not None:
        token_ids = token_ids[:max_length]
    return token_ids


def _normalise_image_sizes(image_sizes: Any) -> list[Any]:
    if isinstance(image_sizes, torch.Tensor):
        if image_sizes.numel() == 0:
            return []
        return image_sizes.reshape(-1, 2).tolist()
    if not image_sizes:
        return []
    if isinstance(image_sizes, list):
        if len(image_sizes) == 1 and isinstance(image_sizes[0], list):
            inner = image_sizes[0]
            if inner and isinstance(inner[0], (list | tuple)):
                return [tuple(item) for item in inner]
            return list(inner)
        return list(image_sizes)
    return [image_sizes]


def _expand_processor_slots(
    processor,
    text: str,
    *,
    image_sizes: Sequence[Any],
    audios: Sequence[np.ndarray],
    max_slice_nums: int | None = None,
    use_image_id: bool | None = None,
    stream_input: bool = False,
    chunk_length: int = 1,
) -> str:
    """Expand MiniCPM processor slots the same way as ``_convert_omni_to_inputs``."""

    if use_image_id is None:
        use_image_id = True

    split_pattern = f"({re.escape(_MINICPM_IMAGE_SLOT)}|{re.escape(_MINICPM_AUDIO_SLOT)})"
    text_chunks = re.split(split_pattern, text)
    image_sizes_list = _normalise_image_sizes(image_sizes)
    audios_list = list(audios)
    image_id = 0
    audio_id = 0
    for index, chunk in enumerate(text_chunks):
        if chunk == _MINICPM_IMAGE_SLOT:
            if image_id >= len(image_sizes_list):
                raise ValueError(
                    f"MiniCPM label alignment expects {len(image_sizes_list)} image slot(s), "
                    f"found at least {image_id + 1} in text."
                )
            image_processor = processor.image_processor
            text_chunks[index] = image_processor.get_slice_image_placeholder(
                image_sizes_list[image_id],
                image_id,
                max_slice_nums,
                use_image_id,
            )
            image_id += 1
        elif chunk == _MINICPM_AUDIO_SLOT:
            if audio_id >= len(audios_list):
                raise ValueError(
                    f"MiniCPM label alignment expects {len(audios_list)} audio slot(s), "
                    f"found at least {audio_id + 1} in text."
                )
            text_chunks[index] = processor.get_audio_placeholder(
                len(audios_list[audio_id]),
                chunk_input=stream_input,
                chunk_length=chunk_length,
            )
            audio_id += 1
    return "".join(text_chunks)


def _strip_leading_padding(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    if attention_mask is None or attention_mask.ndim != 1:
        return input_ids, 0
    valid = attention_mask.to(dtype=torch.bool)
    if valid.all():
        return input_ids, 0
    first_valid = int(valid.int().argmax().item()) if valid.any() else 0
    if first_valid == 0:
        return input_ids, 0
    return input_ids[first_valid:], first_valid


def _find_token_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> slice | None:
    if not needle or len(needle) > len(haystack):
        return None
    for start in range(len(haystack) - len(needle), -1, -1):
        if list(haystack[start : start + len(needle)]) == list(needle):
            return slice(start, start + len(needle))
    return None


def _final_assistant_token_mask(
    messages: list[dict[str, Any]],
    processor,
    rendered_text: str,
    input_ids: torch.Tensor,
    *,
    image_sizes: Sequence[Any],
    audios: Sequence[np.ndarray],
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    tokenizer = _tokenizer_from_processor(processor)
    marked_messages, markers = _mark_final_assistant_content(messages)
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    if markers is None:
        return loss_mask

    image_count = int(kwargs.get("_minicpm_image_count", 0))
    audio_count = int(kwargs.get("_minicpm_audio_count", 0))
    video_count = int(kwargs.get("_minicpm_video_count", 0))
    use_audio_in_video = bool(kwargs.get("use_audio_in_video"))
    max_slice_nums = kwargs.get("max_slice_nums")
    use_image_id = kwargs.get("use_image_id")
    max_length = kwargs.get("max_length")
    stream_input = bool(kwargs.get("stream_input", False))
    chunk_length = int(kwargs.get("chunk_length", 1))

    marked_text = _render_processor_text(
        marked_messages,
        processor,
        image_count=image_count,
        audio_count=audio_count,
        video_count=video_count,
        use_audio_in_video=use_audio_in_video,
        **kwargs,
    )
    expand_kwargs = {
        "image_sizes": image_sizes,
        "audios": audios,
        "max_slice_nums": max_slice_nums,
        "use_image_id": use_image_id,
        "stream_input": stream_input,
        "chunk_length": chunk_length,
    }
    expanded_rendered = _expand_processor_slots(processor, rendered_text, **expand_kwargs)
    expanded_marked = _expand_processor_slots(processor, marked_text, **expand_kwargs)

    start_marker, end_marker = markers
    start = expanded_marked.find(start_marker)
    if start < 0:
        raise ValueError("Cannot locate MiniCPM assistant start marker in rendered chat template.")
    end = expanded_marked.find(end_marker, start + len(start_marker))
    if end < 0:
        raise ValueError("Cannot locate MiniCPM assistant end marker in rendered chat template.")

    assistant_char_start = start
    assistant_char_end = end - len(start_marker)
    expected_answer = expanded_marked[start + len(start_marker) : end]
    if expanded_rendered[assistant_char_start:assistant_char_end] != expected_answer:
        raise ValueError("Marked MiniCPM chat template rendering does not match the unmarked rendering.")

    encoded_ids = _encode_like_processor(tokenizer, expanded_rendered, max_length=max_length)
    processor_ids_tensor, pad_offset = _strip_leading_padding(input_ids, attention_mask)
    processor_ids = [int(token) for token in processor_ids_tensor.tolist()]

    if encoded_ids != processor_ids:
        answer_ids = _encode_like_processor(tokenizer, expected_answer)
        answer_span = _find_token_subsequence(processor_ids, answer_ids)
        if answer_span is not None:
            for token_i in range(answer_span.start, answer_span.stop):
                loss_mask[pad_offset + token_i] = True
            return loss_mask
        raise ValueError(
            "MiniCPM expanded label encoding does not match processor input_ids; "
            f"lengths {len(encoded_ids)} vs {len(processor_ids)}."
        )

    tokenized = tokenizer(expanded_rendered, return_offsets_mapping=True)
    offsets = tokenized.get("offset_mapping")
    if offsets is None:
        raise ValueError("MiniCPM tokenizer must provide offset_mapping to build final-assistant DPO labels.")

    for token_i, (token_start, token_end) in enumerate(offsets[: len(processor_ids)]):
        if token_start == token_end:
            continue
        if token_start < assistant_char_end and token_end > assistant_char_start:
            loss_mask[pad_offset + token_i] = True
    return loss_mask


def _default_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.arange(input_ids.shape[-1], dtype=torch.long, device=input_ids.device)


def _empty_bound(device: torch.device) -> torch.Tensor:
    return torch.zeros(0, 2, dtype=torch.long, device=device)


def _sample_bound(value: Any, device: torch.device) -> torch.Tensor:
    if value is None or (isinstance(value, (list, tuple)) and not value):
        return _empty_bound(device)
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (torch.Tensor, np.ndarray, list)):
        return _sample_bound(value[0], device)
    bound = torch.as_tensor(value, dtype=torch.long, device=device)
    if bound.numel() == 0:
        return _empty_bound(device)
    if bound.ndim == 1:
        bound = bound.unsqueeze(0)
    return bound.reshape(-1, 2).contiguous()


def _fill_minicpm_forward_fields(model_inputs: dict[str, Any], input_ids: torch.Tensor) -> dict[str, Any]:
    """Guarantee keys MiniCPMO.forward reads from ``data``.

    Remote code indexes ``data["pixel_values"]``, ``data["tgt_sizes"]``,
    ``data["image_bound"]``, ``data["audio_bounds"]``, ``data["position_ids"]``,
    and ``len(data["input_ids"])``. Missing media keys must still be present as
    empty per-sample values so batching can keep MiniCPM's list layout.
    """
    if "position_ids" not in model_inputs or model_inputs["position_ids"] is None:
        model_inputs["position_ids"] = _default_position_ids(input_ids)
    position_ids = model_inputs["position_ids"]
    if not isinstance(position_ids, torch.Tensor):
        position_ids = torch.tensor(position_ids, dtype=torch.long, device=input_ids.device)
    if position_ids.ndim > 1 and position_ids.shape[0] == 1:
        position_ids = position_ids.squeeze(0)
    model_inputs["position_ids"] = position_ids.to(dtype=torch.long, device=input_ids.device)
    if model_inputs["position_ids"].shape[-1] != input_ids.shape[-1]:
        raise ValueError(
            "MiniCPM position_ids length "
            f"{model_inputs['position_ids'].shape[-1]} does not match input_ids length {input_ids.shape[-1]}."
        )

    model_inputs["pixel_values"] = _sample_pixel_slices(model_inputs.get("pixel_values"))
    model_inputs["tgt_sizes"] = _sample_tgt_sizes(
        model_inputs.get("tgt_sizes"),
        n_slices=len(model_inputs["pixel_values"]),
        device=input_ids.device,
    )
    model_inputs["image_bound"] = _sample_bound(model_inputs.get("image_bound"), input_ids.device)
    model_inputs["audio_bounds"] = _sample_bound(model_inputs.get("audio_bounds"), input_ids.device)
    audio_features = _normalize_audio_features(model_inputs.get("audio_features"))
    if audio_features == []:
        model_inputs.pop("audio_features", None)
        model_inputs.pop("audio_feature_lens", None)
    else:
        model_inputs["audio_features"] = audio_features
        model_inputs["audio_feature_lens"] = _batch_audio_feature_lens(
            model_inputs.get("audio_feature_lens"), input_ids.device
        )
    return model_inputs


def _fetch_minicpm_audios(sample: dict[str, Any], **kwargs) -> list[np.ndarray]:
    return _fetch_audios(sample.get("audios", []), **kwargs) if sample.get("audios") else []


def process_minicpm_sample(sample: dict[str, Any], processor, position_id_func=None, **kwargs) -> list[dict[str, Any]]:
    """Transform one offline preference sample into MiniCPM model inputs."""

    del position_id_func

    messages = _build_minicpm_messages(sample)
    if sample.get("videos"):
        raise ValueError("MiniCPM offline DPO does not support video rows. Use image-only or audio-only parquet.")
    images = _fetch_images(sample.get("images", []), **kwargs) if sample.get("images") else []
    audios = _fetch_minicpm_audios(sample, **kwargs)
    rendered_text = _render_processor_text(
        messages,
        processor,
        image_count=len(images),
        audio_count=len(audios),
        video_count=0,
        use_audio_in_video=False,
        **kwargs,
    )

    label_kwargs = {
        **kwargs,
        "_minicpm_image_count": len(images),
        "_minicpm_audio_count": len(audios),
        "_minicpm_video_count": 0,
    }
    model_inputs = _call_processor(processor, text=rendered_text, images=images, videos=[], audios=audios, **kwargs)
    model_inputs = {
        key: value if key in {"audio_features", "audio_feature_lens"} else _squeeze_batch_dim(value)
        for key, value in model_inputs.items()
    }

    if "input_ids" not in model_inputs:
        raise ValueError("MiniCPM processor output must contain input_ids.")
    input_ids = model_inputs["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        model_inputs["input_ids"] = input_ids
    if "attention_mask" not in model_inputs:
        model_inputs["attention_mask"] = torch.ones_like(input_ids)

    raw_input_ids = input_ids.clone()
    model_inputs = _fill_minicpm_forward_fields(model_inputs, raw_input_ids)

    labels = torch.full_like(raw_input_ids, fill_value=IGNORE_INDEX)
    assistant_mask = _final_assistant_token_mask(
        messages,
        processor,
        rendered_text,
        raw_input_ids,
        image_sizes=model_inputs.get("image_sizes", []),
        audios=audios,
        attention_mask=model_inputs.get("attention_mask"),
        **label_kwargs,
    )
    labels[assistant_mask] = raw_input_ids[assistant_mask]
    model_inputs["input_ids"] = raw_input_ids
    model_inputs["labels"] = labels
    return [model_inputs]
