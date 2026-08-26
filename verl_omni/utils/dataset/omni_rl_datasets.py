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

import copy
import json
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from verl.utils.dataset.rl_dataset import RLHFDataset

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
_MEDIA_CACHE_SIZE = max(0, int(os.getenv("OMNIVIDEO_INPUT_CACHE_SIZE", "8")))
_MEDIA_KEYS = {"audio", "audio_url", "image", "image_url", "video", "video_url"}


def _ffmpeg_executable() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    from imageio_ffmpeg import get_ffmpeg_exe

    bundled_ffmpeg = get_ffmpeg_exe()
    if not Path(bundled_ffmpeg).is_file():
        raise RuntimeError(f"FFmpeg executable is unavailable (resolved path: {bundled_ffmpeg!r})")
    return bundled_ffmpeg


def _media_decode_timeout() -> float:
    return max(1.0, float(os.getenv("OMNIVIDEO_INPUT_DECODE_TIMEOUT", "60")))


def _probe_video_duration(ffmpeg: str, source_path: str, timeout: float) -> float:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-i", source_path],
        check=False,
        capture_output=True,
        text=True,
        timeout=min(timeout, 15.0),
    )
    match = _DURATION_RE.search(result.stderr or "")
    if match is None:
        raise ValueError(f"Unable to determine video duration for {source_path}")
    hours, minutes, seconds = match.groups()
    duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    if duration <= 0:
        raise ValueError(f"Invalid video duration {duration} for {source_path}")
    return duration


def _decode_audio(ffmpeg: str, source_path: str, start: float, duration: float, timeout: float):
    import numpy as np

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        source_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if not audio.size:
        raise ValueError(f"FFmpeg decoded no audio from {source_path}")
    return audio


def _materialize_video_item(item: dict, temp_dir: str, index: int) -> tuple[str, Any]:
    source = item.get("video", item.get("video_url"))
    if not isinstance(source, str):
        raise TypeError(f"Expected a video path, got {type(source).__name__}")
    source_path = source[7:] if source.startswith("file://") else source
    ffmpeg = _ffmpeg_executable()
    timeout = _media_decode_timeout()
    full_duration = _probe_video_duration(ffmpeg, source_path, timeout)
    start = max(0.0, float(item.get("video_start", 0.0)))
    end = min(full_duration, float(item.get("video_end", full_duration)))
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Invalid video range {start}-{end} for {source_path}")

    min_frames = max(2, int(item.get("min_frames", 4)))
    max_frames = max(min_frames, int(item.get("max_frames", 64)))
    requested_fps = max(0.01, float(item.get("fps", 2.0)))
    target_frames = min(max(round(duration * requested_fps), min_frames), max_frames)
    target_frames -= target_frames % 2
    target_frames = max(2, target_frames)
    sampling_fps = target_frames / duration

    output_path = Path(temp_dir) / f"video_{index:04d}.mp4"
    video_command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-filter_threads",
        "1",
        "-threads",
        "1",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        source_path,
        "-an",
        "-sn",
        "-dn",
        "-vf",
        f"fps={sampling_fps:.8f},scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-frames:v",
        str(target_frames),
        "-c:v",
        "mpeg4",
        "-q:v",
        "3",
        "-y",
        str(output_path),
    ]
    subprocess.run(video_command, check=True, capture_output=True, timeout=timeout)
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise ValueError(f"FFmpeg decoded no video from {source_path}")

    audio = _decode_audio(ffmpeg, source_path, start, duration, timeout)
    return str(output_path.resolve()), audio


def _process_audio_video_with_ffmpeg(messages: list[dict], image_patch_size: int):
    from qwen_omni_utils import process_mm_info

    transformed = copy.deepcopy(messages)
    audios = []
    with tempfile.TemporaryDirectory(prefix="omnivideo_input_") as temp_dir:
        video_index = 0
        for message in transformed:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "video":
                    continue
                normalized_video, audio = _materialize_video_item(item, temp_dir, video_index)
                video_index += 1
                item["video"] = normalized_video
                item.pop("video_url", None)
                item.pop("video_start", None)
                item.pop("video_end", None)
                audios.append(audio)

        _, images, videos = process_mm_info(
            transformed,
            use_audio_in_video=False,
            image_patch_size=image_patch_size,
        )
    return audios or None, images, videos


def _serialize_media_messages(messages: list[dict]) -> str:
    media_messages = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        media_content = [item for item in content if isinstance(item, dict) and (_MEDIA_KEYS & item.keys())]
        if media_content:
            media_messages.append({"role": message.get("role", "user"), "content": media_content})
    return json.dumps(media_messages, sort_keys=True, separators=(",", ":"), default=str)


@lru_cache(maxsize=_MEDIA_CACHE_SIZE)
def _process_audio_video_cached(serialized_messages: str, image_patch_size: int):
    return _process_audio_video_with_ffmpeg(json.loads(serialized_messages), image_patch_size)


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
        config: DictConfig,
    ) -> tuple[list[Any] | None, list[Any] | None, list[Any] | None]:
        from qwen_omni_utils import process_mm_info

        # Qwen returns (audios, images, videos); verl expects
        # (images, videos, audios). AVQA uses a standalone audio track, while
        # datasets such as OmniVideo-R1 can read the audio stream directly
        # from each video to avoid duplicating media on disk.
        use_audio_in_video = bool(config.get("use_audio_in_video", False))
        if use_audio_in_video:
            audios, images, videos = _process_audio_video_cached(_serialize_media_messages(messages), image_patch_size)
        else:
            audios, images, videos = process_mm_info(
                messages,
                use_audio_in_video=False,
                image_patch_size=image_patch_size,
            )
        return images, videos, audios
