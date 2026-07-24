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

import os
from typing import Any
from urllib.parse import unquote, urlparse

from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset


class OmniRLHFDataset(RLHFDataset):
    """Decode audio paths into waveforms before invoking HF processors.

    verl turns parquet media columns into structured messages. Vision helpers
    resolve image/video paths, while standalone audio paths otherwise remain
    strings. Qwen3-Omni's feature extractor expects waveform arrays, so this
    class resolves audio at the dataset processing seam.
    """

    @staticmethod
    def _normalize_audio_path(audio: Any) -> Any:
        if isinstance(audio, os.PathLike):
            return os.fspath(audio)
        if isinstance(audio, str) and audio.startswith("file://"):
            parsed = urlparse(audio)
            if parsed.netloc in ("", "localhost"):
                return unquote(parsed.path)
        return audio

    @staticmethod
    def _audio_sampling_rate(config: DictConfig | dict | None) -> int:
        config = config or {}
        processor_kwargs = config.get("mm_processor_kwargs", {}) or {}
        return int(processor_kwargs.get("sampling_rate", 16000))

    @classmethod
    def _load_audio_for_processor(cls, audio: Any, sampling_rate: int) -> Any:
        if isinstance(audio, dict):
            if "array" in audio:
                return audio["array"]
            for key in ("audio", "audio_url", "path"):
                if key in audio:
                    return cls._load_audio_for_processor(audio[key], sampling_rate)
            return audio

        audio = cls._normalize_audio_path(audio)
        if isinstance(audio, str):
            if not audio.startswith(("http://", "https://")) and not os.path.isfile(audio):
                raise FileNotFoundError(
                    f"Audio path does not exist on this worker: {audio}. "
                    "Ensure the AVQA media directory is mounted at the same path on every node."
                )
            from transformers.audio_utils import load_audio

            return load_audio(audio, sampling_rate=sampling_rate)
        return audio

    @classmethod
    def _extract_audio_info(
        cls,
        messages: list[dict],
        sampling_rate: int = 16000,
    ) -> list[Any] | None:
        audios: list[Any] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                if "audio" in item:
                    audio = item["audio"]
                elif "audio_url" in item:
                    audio = item["audio_url"]
                else:
                    audio = {key: value for key, value in item.items() if key != "type"}
                audios.append(cls._load_audio_for_processor(audio, sampling_rate))
        return audios or None

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        has_visual = any(
            isinstance(message.get("content"), list)
            and any(isinstance(item, dict) and item.get("type") in {"image", "video"} for item in message["content"])
            for message in messages
        )
        if has_visual:
            from qwen_vl_utils import process_vision_info

            images, videos = process_vision_info(
                messages,
                image_patch_size=image_patch_size,
                return_video_metadata=True,
            )
        else:
            images, videos = None, None
        audios = cls._extract_audio_info(messages, cls._audio_sampling_rate(config))
        return images, videos, audios
