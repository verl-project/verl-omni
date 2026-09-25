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
"""Adapter-owned named diffusion output declarations; no positional media protocol."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

Modality = Literal["image", "video", "audio"]


def resolve_is_video(ndim: int, media_kind: str | None) -> bool:
    """Read the declared modality; rank is never a modality discriminator."""
    del ndim
    if media_kind is None:
        raise ValueError("Explicit media_kind required, got None")
    if media_kind not in ("image", "video", "audio"):
        raise ValueError(f"Unsupported media kind: {media_kind!r}")
    return media_kind == "video"


def resolve_batch_media_kind(media_kinds: Iterable[str | None]) -> str | None:
    """Require one declared modality per pipeline batch; ignore absent legacy metadata."""
    resolved = None
    for kind in media_kinds:
        if kind is None:
            continue
        resolve_is_video(0, kind)
        if resolved is not None and resolved != kind:
            raise ValueError(f"Conflicting media kinds in one rollout batch: {resolved!r} and {kind!r}")
        resolved = kind
    return resolved


def validate_visual_media_batch_rank(ndim: int, media_kind: str | None) -> None:
    """Validate declared image/video batches before legacy layout normalization."""
    if media_kind is None:
        return
    if media_kind == "image" and ndim != 4:
        raise ValueError(f"Declared media_kind='image' requires an NCHW batch, got rank {ndim}.")
    if media_kind == "video" and ndim != 5:
        raise ValueError(f"Declared media_kind='video' requires a rank-5 batch, got rank {ndim}.")
    if media_kind == "audio":
        raise ValueError("Cannot dump declared audio as a visual generation batch.")
    resolve_is_video(ndim, media_kind)


@dataclass(frozen=True)
class MediaSpec:
    """One media declaration; dtype belongs to the tensor, not config."""

    modality: Modality
    representation: Literal["decoded", "latent"]
    layout: str
    sample_rate: int | None = None
    fps: float | None = None


@dataclass(frozen=True)
class DiffusionIOSpec:
    """Available named artifacts, with canonical decoded and native latent axes.

    Each request selects its primary and optional preview explicitly. Runtime
    sample rate/FPS live on returned artifacts; declarations constrain only the
    adapter-owned artifact names, representations, and layouts.
    """

    artifacts: Mapping[str, MediaSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, Mapping) or not self.artifacts:
            raise ValueError("DiffusionIOSpec requires named artifacts")
        if any(not isinstance(name, str) or not isinstance(spec, MediaSpec) for name, spec in self.artifacts.items()):
            raise TypeError("DiffusionIOSpec requires artifact-name to MediaSpec declarations")
