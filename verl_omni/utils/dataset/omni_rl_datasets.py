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

from typing import Any

import numpy as np
from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset

# Whisper mel-frame stride at 16kHz; keep in sync with the feature extractor's
# hop_length so actor recompute and vllm-omni rollout frame audio identically.
DEFAULT_AUDIO_HOP_LENGTH = 160

# MiniCPM-o's Whisper feature extractor runs at 16kHz mono float32.
DEFAULT_SAMPLING_RATE = 16000


def pad_audio_to_hop_multiple(audio: np.ndarray, hop_length: int = DEFAULT_AUDIO_HOP_LENGTH) -> np.ndarray:
    """Zero-pad audio to a multiple of hop_length (no-op when already aligned)."""
    pad_length = -audio.shape[-1] % hop_length
    if pad_length:
        return np.pad(audio, (0, pad_length))
    return audio


class OmniAudioRLHFDataset(RLHFDataset):
    """Shared audio-aware RL dataset: resolve media, then hop-pad audio.

    Subclasses implement ``_resolve_media_from_messages`` returning verl's
    ``(images, videos, audios)`` order.
    """

    @classmethod
    def _resolve_media_from_messages(
        cls,
        messages: list[dict],
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        """Return ``(images, videos, audios)`` loaded from message content blocks."""
        raise NotImplementedError("OmniAudioRLHFDataset subclasses must implement _resolve_media_from_messages")

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        images, videos, audios = cls._resolve_media_from_messages(messages, config)
        # vllm-omni pads audio to a hop multiple before feature extraction while
        # the HF side drops the tail frame; pad first so both sides expand the
        # prompt to the same audio token count.
        if audios is not None:
            audios = [pad_audio_to_hop_multiple(a) for a in audios]
        return images, videos, audios


class QwenOmniRLHFDataset(OmniAudioRLHFDataset):
    """Adapt Qwen's multimodal media loader to verl's RL dataset interface.

    verl turns parquet media columns into structured messages. Qwen's
    ``process_mm_info`` then resolves image/audio/video paths into the media
    objects expected by the Qwen3-Omni processor and vLLM-Omni rollout.
    """

    @classmethod
    def _resolve_media_from_messages(
        cls,
        messages: list[dict],
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        from qwen_omni_utils import process_mm_info

        # qwen_omni_utils returns (audios, images, videos). AVQA uses a
        # standalone audio track rather than extracting audio from a video.
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        return images, videos, audios


def _load_minicpm_audio(path: str, sampling_rate: int) -> np.ndarray:
    """Load one audio clip as float32 mono at ``sampling_rate``."""
    import soundfile as sf

    wav, source_rate = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if source_rate != sampling_rate:
        import torch
        import torchaudio.functional as audio_f

        wav = audio_f.resample(torch.from_numpy(wav), orig_freq=source_rate, new_freq=sampling_rate).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


class MiniCPMORLHFDataset(OmniAudioRLHFDataset):
    """MiniCPM-o media loader for verl's RL dataset interface.

    Walks the same structured content blocks verl builds from ``<image>`` /
    ``<audio>`` parquet markers (no Qwen dependency) and loads RGB images plus
    16kHz mono waveforms — the formats MiniCPMO's processor and vLLM-Omni's
    MiniCPM input mapper accept.
    """

    @classmethod
    def _sampling_rate_from_config(cls, config: DictConfig | dict | None) -> int:
        mm_kwargs = dict(config or {}).get("mm_processor_kwargs") or {}
        if mm_kwargs.get("sampling_rate") is not None:
            return int(mm_kwargs["sampling_rate"])
        return DEFAULT_SAMPLING_RATE

    @classmethod
    def _resolve_media_from_messages(
        cls,
        messages: list[dict],
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        from PIL import Image

        sampling_rate = cls._sampling_rate_from_config(config)
        images: list[Any] = []
        audios: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "image":
                    image_ref = block.get("image") or block.get("image_url")
                    if isinstance(image_ref, dict):
                        image_ref = image_ref.get("url")
                    if image_ref is None:
                        raise ValueError(f"MiniCPM image block has no path: {block!r}")
                    with Image.open(image_ref) as image:  # convert() returns a new image; close the handle
                        images.append(image.convert("RGB"))
                elif block_type == "audio":
                    audio_ref = block.get("audio") or block.get("audio_url")
                    if audio_ref is None:
                        raise ValueError(f"MiniCPM audio block has no path: {block!r}")
                    audios.append(_load_minicpm_audio(audio_ref, sampling_rate))
                elif block_type == "video":
                    raise ValueError(
                        "MiniCPMORLHFDataset does not support video rows; AVQA training is image+audio only."
                    )
        return images, None, audios
