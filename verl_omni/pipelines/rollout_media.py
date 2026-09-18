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
"""Typed, CPU-importable contracts for diffusion rollout media.

These types let a diffusion adapter *declare* what media its pipeline emits
(primary stream plus any auxiliary streams such as joint audio) instead of the
rollout strategy hard-coding model-specific conventions like "audio lives at
tuple position 1" or "the audio sample rate is 32000 Hz". The diffusion
strategy consults the adapter-owned :class:`DiffusionIOSpec` when it converts an
engine result into a rollout output, so adding a new combination of existing
modalities only touches the adapter, not the shared server/strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

#: Media kinds a diffusion pipeline can emit.
Modality = Literal["image", "video", "audio"]


@dataclass(frozen=True)
class MediaSpec:
    """Declaration of a single media stream produced by a diffusion pipeline.

    Attributes:
        modality: The media kind (``"image"``, ``"video"`` or ``"audio"``).
        sample_rate: Default audio sample rate in Hz. Audio streams only; used
            as a fallback when the adapter does not attach a runtime sample rate
            through the rollout metadata.
        fps: Default frames-per-second. Video streams only; ``None`` when the
            pipeline does not declare one.

    The float-latent vs. uint8-pixel distinction is intentionally *not* declared
    here: it is decided per request by the sampling ``output_type`` (``latent``
    keeps floats, otherwise pixels are quantized to uint8 in ``[0, 255]``).
    """

    modality: Modality
    sample_rate: Optional[int] = None
    fps: Optional[int] = None


@dataclass(frozen=True)
class DiffusionIOSpec:
    """Adapter-owned declaration of a diffusion pipeline's rollout outputs.

    Attributes:
        primary: The main media stream. It is carried on
            ``DiffusionOutput.diffusion_output`` and, when the pipeline emits a
            media tuple, occupies position 0.
        auxiliary: Additional media streams in tuple order, so ``auxiliary[i]``
            describes media-tuple position ``i + 1`` (e.g. a single ``audio``
            entry describes the joint-audio stream at position 1).
    """

    primary: MediaSpec
    auxiliary: tuple[MediaSpec, ...] = ()
