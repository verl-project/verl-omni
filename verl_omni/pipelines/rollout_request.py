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
"""Typed, CPU-importable contract for one diffusion rollout request.

Before this module the rollout ``generate`` template threaded the prompt token
ids, prompt mask, negatives, per-encoder token ids and the image/video/audio
condition data through the server as a handful of loose keyword arguments, and
every strategy re-derived the same conventions on its own. :class:`OmniRolloutRequest`
gives both the AR and diffusion strategies a single typed object to consume so
the request shape is declared in one place.

This module keeps the *interpretation* of the request (how the fields map onto
the engine ``OmniCustomPrompt`` / vLLM prompt dict) inside each strategy; it does
not change the wire keys the pipelines already read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, get_args

from verl_omni.pipelines.rollout_media import Modality

#: Payload keys inspected, in priority order, when locating condition images in
#: a rollout request payload (verl-omni's ``custom_prompt`` dict). This mirrors
#: the historical fallback order and is centralized here so a future PR can
#: collapse the aliases without hunting through adapters.
_CONDITION_IMAGE_KEYS: tuple[str, ...] = (
    "images",
    "image",
    "multi_modal_data.image",
    "extra_args.multi_modal_data.image",
    "additional_information.condition_images",
)


@dataclass(frozen=True)
class MediaInput:
    """A single condition media stream attached to a rollout request.

    Attributes:
        modality: The media kind (``"image"``, ``"video"`` or ``"audio"``).
        data: The raw condition payload (e.g. a PIL image or tensor).
    """

    modality: Modality
    data: Any

    def __post_init__(self) -> None:
        if self.modality not in get_args(Modality):
            raise ValueError(f"Unsupported media modality: {self.modality!r}")
        if self.data is None:
            raise ValueError(f"MediaInput data for {self.modality!r} must not be None")


@dataclass(frozen=True)
class PromptBundle:
    """The prompt side of a diffusion rollout request.

    Attributes:
        token_ids: The primary prompt token ids.
        mask: Optional prompt attention mask.
        negative_token_ids: Optional classifier-free-guidance negative prompt ids.
        extra_token_ids: Optional per-extra-encoder token ids (e.g. CLIP + T5).
        negative_extra_token_ids: Optional per-extra-encoder negative token ids.
        mm_processor_kwargs: Optional multimodal-processor kwargs forwarded to the engine.
    """

    token_ids: list[int]
    mask: Any | None = None
    negative_token_ids: Optional[list[int]] = None
    extra_token_ids: Optional[Mapping[str, list[int]]] = None
    negative_extra_token_ids: Optional[Mapping[str, list[int]]] = None
    mm_processor_kwargs: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class OmniRolloutRequest:
    """Single typed contract for one diffusion/AR rollout request.

    Attributes:
        prompt: The prompt token ids and their negatives / per-encoder variants.
        media: Condition media streams in declaration order.
    """

    prompt: PromptBundle
    media: tuple[MediaInput, ...] = ()

    @classmethod
    def from_generate_kwargs(
        cls,
        *,
        prompt_ids: list[int],
        prompt_mask: Any | None = None,
        negative_prompt_ids: Optional[list[int]] = None,
        extra_prompt_ids: Optional[Mapping[str, list[int]]] = None,
        negative_extra_prompt_ids: Optional[Mapping[str, list[int]]] = None,
        mm_processor_kwargs: Optional[Mapping[str, Any]] = None,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
    ) -> OmniRolloutRequest:
        """Build a request from the server ``generate`` keyword arguments.

        A media stream is included only when its data is not ``None`` (an empty
        list is still a present-but-empty stream), matching the previous
        ``_build_multi_modal_data`` behavior.
        """
        media: list[MediaInput] = []
        if image_data is not None:
            media.append(MediaInput("image", image_data))
        if video_data is not None:
            media.append(MediaInput("video", video_data))
        if audio_data is not None:
            media.append(MediaInput("audio", audio_data))
        return cls(
            prompt=PromptBundle(
                token_ids=prompt_ids,
                mask=prompt_mask,
                negative_token_ids=negative_prompt_ids,
                extra_token_ids=extra_prompt_ids,
                negative_extra_token_ids=negative_extra_prompt_ids,
                mm_processor_kwargs=mm_processor_kwargs,
            ),
            media=tuple(media),
        )

    def multi_modal_data(self) -> dict[str, Any]:
        """Assemble the vLLM ``multi_modal_data`` dict from the media streams.

        Raises:
            ValueError: If the request declares the same modality more than once.
        """
        multi_modal_data: dict[str, Any] = {}
        for stream in self.media:
            if stream.modality in multi_modal_data:
                raise ValueError(f"Duplicate media modality in rollout request: {stream.modality!r}")
            multi_modal_data[stream.modality] = stream.data
        return multi_modal_data


def _optional_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"Condition-image container {key!r} must be a mapping, got {type(value).__name__}")
    return value


def _normalize_condition_images(images: Any) -> list[Any]:
    if isinstance(images, tuple):
        return list(images)
    if isinstance(images, list):
        return images
    return [images]


def _alias_values_match(left: Any, right: Any) -> bool:
    """Compare alias payloads without assuming tensor/array equality is scalar."""
    if left is right:
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_alias_values_match(left[key], right[key]) for key in left)
    if isinstance(left, list | tuple) and isinstance(right, list | tuple):
        return len(left) == len(right) and all(_alias_values_match(a, b) for a, b in zip(left, right, strict=True))
    try:
        result = left == right
        return result if isinstance(result, bool) else bool(result)
    except (TypeError, ValueError, RuntimeError):
        return False


def condition_images_from_payload(payload: Mapping[str, Any]) -> list[Any]:
    """Extract condition images from a rollout request payload.

    Accepts one source, or multiple aliases that contain equivalent values,
    and normalizes the result to a flat list. Conflicting aliases fail instead
    of silently selecting the first one. This is the single place that knows
    the legacy image-candidate aliases; image-conditioned diffusion adapters
    call it instead of re-implementing the fallback chain.

    Raises:
        TypeError: If a nested alias container is not a mapping.
        ValueError: If multiple aliases provide different condition images.
    """
    multi_modal_data = _optional_mapping(payload, "multi_modal_data")
    extra_args = _optional_mapping(payload, "extra_args")
    extra_multi_modal_data = _optional_mapping(extra_args, "multi_modal_data") if extra_args is not None else None
    additional_information = _optional_mapping(payload, "additional_information")

    values = (
        payload.get("images"),
        payload.get("image"),
        multi_modal_data.get("image") if multi_modal_data is not None else None,
        extra_multi_modal_data.get("image") if extra_multi_modal_data is not None else None,
        additional_information.get("condition_images") if additional_information is not None else None,
    )
    present = [
        (key, _normalize_condition_images(value))
        for key, value in zip(_CONDITION_IMAGE_KEYS, values, strict=True)
        if value is not None
    ]
    if not present:
        return []

    selected_key, selected_images = present[0]
    conflicting_keys = [key for key, images in present[1:] if not _alias_values_match(selected_images, images)]
    if conflicting_keys:
        fields = ", ".join((selected_key, *conflicting_keys))
        raise ValueError(
            f"Conflicting condition-image aliases were provided: {fields}. "
            "Pass condition images through one field or provide equivalent values."
        )
    return selected_images
