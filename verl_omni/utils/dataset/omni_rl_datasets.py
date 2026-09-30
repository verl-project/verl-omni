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

__all__ = [
    "MiniCPMORLHFDataset",
    "QwenOmniRLHFDataset",
    "pad_audio_to_hop_multiple",
]


class QwenOmniRLHFDataset(RLHFDataset):
    """Adapt Qwen's multimodal media loader to verl's RL dataset interface.

    verl turns parquet media columns into structured messages. Qwen's
    ``process_mm_info`` then resolves image/audio/video paths into the media
    objects expected by the Qwen3-Omni processor and vLLM-Omni rollout.
    """

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        from qwen_omni_utils import process_mm_info

        # Qwen returns (audios, images, videos); verl expects
        # (images, videos, audios). AVQA uses a standalone audio track rather
        # than extracting audio from a video.
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        # vllm-omni pads audio to a hop multiple before feature extraction while
        # the HF side drops the tail frame; pad first so both sides expand the
        # prompt to the same audio token count.
        if audios is not None:
            audios = [pad_audio_to_hop_multiple(a) for a in audios]
        return images, videos, audios


class MiniCPMORLHFDataset(RLHFDataset):
    """MiniCPM-o media loader for verl's RL dataset interface.

    Walks the same structured content blocks verl builds from ``<image>`` /
    ``<audio>`` parquet markers (no Qwen dependency) and loads RGB images plus
    16kHz mono waveforms — the formats MiniCPMO's processor and vLLM-Omni's
    MiniCPM input mapper accept.
    """

    @classmethod
    def _process_multi_modal_info(
        cls,
        messages: list[dict],
        image_patch_size: int,
        config: DictConfig | dict | None,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        mm_kwargs = dict(config or {}).get("mm_processor_kwargs") or {}
        sampling_rate = mm_kwargs.get("sampling_rate")
        # 16000 Hz is MiniCPM-o's Whisper rate; decoding at another rate would
        # desync the waveform from the mel frames the processor expects.
        sampling_rate = 16000 if sampling_rate is None else int(sampling_rate)
        images, audios = _load_minicpm_media(messages, sampling_rate)
        # vllm-omni pads audio to a hop multiple before feature extraction while
        # the HF side drops the tail frame; pad first so both sides expand the
        # prompt to the same audio token count.
        return images, None, [pad_audio_to_hop_multiple(a) for a in audios]


def pad_audio_to_hop_multiple(audio: np.ndarray, hop_length: int = 160) -> np.ndarray:
    """Zero-pad audio to a multiple of ``hop_length`` (no-op when already aligned).

    160 is Whisper's mel-frame stride at 16kHz; keep it in sync with the feature
    extractor's hop_length so actor recompute and vllm-omni rollout frame audio
    identically.
    """
    pad_length = -audio.shape[-1] % hop_length
    if pad_length:
        return np.pad(audio, (0, pad_length))
    return audio


def _media_ref(block: dict, kind: str) -> str:
    """Path/URL inside one media content block; fails closed when it carries none."""
    ref = block.get(kind) or block.get(f"{kind}_url")
    if isinstance(ref, dict):  # verl's image blocks may nest the path under "url"
        ref = ref.get("url")
    if ref is None:
        raise ValueError(f"MiniCPM {kind} block has no path: {block!r}")
    return ref


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


def _load_minicpm_media(messages: list[dict], sampling_rate: int) -> tuple[list[Any], list[Any]]:
    """Load one sample's images and waveforms from verl's content blocks."""
    from PIL import Image

    images: list[Any] = []
    audios: list[Any] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "image":
                with Image.open(_media_ref(block, "image")) as image:  # convert() returns a new image; close the handle
                    images.append(image.convert("RGB"))
            elif kind == "audio":
                audios.append(_load_minicpm_audio(_media_ref(block, "audio"), sampling_rate))
            elif kind == "video":
                raise ValueError("MiniCPMORLHFDataset does not support video rows; AVQA training is image+audio only.")
    return images, audios
