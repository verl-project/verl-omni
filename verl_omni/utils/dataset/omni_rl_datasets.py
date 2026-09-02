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
"""Audio-aware RL dataset utilities for omni-modal training."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset

# Whisper mel-frame stride at 16kHz; keep in sync with the feature extractor's
# hop_length so actor recompute and vllm-omni rollout frame audio identically.
DEFAULT_AUDIO_HOP_LENGTH = 160


def pad_audio_to_hop_multiple(audio: np.ndarray, hop_length: int = DEFAULT_AUDIO_HOP_LENGTH) -> np.ndarray:
    """Zero-pad audio to a multiple of hop_length (no-op when already aligned)."""
    pad_length = -audio.shape[-1] % hop_length
    if pad_length:
        return np.pad(audio, (0, pad_length))
    return audio


class QwenOmniRLHFDataset(RLHFDataset):
    """Adapt Qwen's multimodal media loader to verl's RL dataset interface.

    verl turns parquet media columns into structured messages. Qwen's
    ``process_mm_info`` then resolves image/audio/video paths into the media
    objects expected by the Qwen3-Omni processor and vLLM-Omni rollout. Video
    audio extraction is controlled by ``data.mm_processor_kwargs.use_audio_in_video``
    and remains disabled by default.
    """

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        from qwen_omni_utils import process_mm_info

        mm_processor_kwargs = config.get("mm_processor_kwargs", {}) if config is not None else {}
        use_audio_in_video = bool(mm_processor_kwargs.get("use_audio_in_video", False))

        # Qwen returns (audios, images, videos); verl expects
        # (images, videos, audios). AVQA keeps the default because it supplies
        # a standalone audio track; video tasks can opt into the video audio.
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=r".*__audioread_load.*",
                )
                audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        except Exception as error:
            raise RuntimeError("Failed to process multimodal sample") from error

        if audios is not None:
            audios = [pad_audio_to_hop_multiple(a) for a in audios]
        return images, videos, audios
