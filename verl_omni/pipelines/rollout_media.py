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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

Modality = Literal["image", "video", "audio"]


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
    sample rate/FPS live on the returned artifacts; a declaration can constrain a
    fixed rate or leave it to a model's runtime decoder configuration.
    """

    artifacts: Mapping[str, MediaSpec]

    def __post_init__(self) -> None:
        if not isinstance(self.artifacts, Mapping) or not self.artifacts:
            raise ValueError("DiffusionIOSpec requires named artifacts")
        if any(not isinstance(name, str) or not isinstance(spec, MediaSpec) for name, spec in self.artifacts.items()):
            raise TypeError("DiffusionIOSpec requires artifact-name to MediaSpec declarations")
