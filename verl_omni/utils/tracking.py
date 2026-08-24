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

"""Experiment-tracking helpers layered on verl.utils.tracking."""

import os
import subprocess
import tempfile
import wave
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl_omni.utils.reward_score.reward_utils import video_tensor_to_pil_frames


def batch_items(values: Any, batch_size: int, name: str) -> list[Any]:
    """Normalize an optional scalar or batched value to one item per sample."""
    if values is None:
        return [None] * batch_size
    if isinstance(values, torch.Tensor | np.ndarray):
        if values.ndim == 0:
            return [values] * batch_size
        if values.shape[0] == batch_size:
            return list(values)
        if batch_size == 1:
            return [values]
        raise ValueError(f"{name} batch size {values.shape[0]} does not match output batch size {batch_size}.")
    if isinstance(values, Sequence) and not isinstance(values, str | bytes):
        if len(values) != batch_size:
            raise ValueError(f"{name} batch size {len(values)} does not match output batch size {batch_size}.")
        return list(values)
    return [values] * batch_size


def _write_wav(audio: Any, sample_rate: Any, path: Path) -> None:
    waveform = torch.as_tensor(audio).detach().cpu().float()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected audio shape [T] or [C, T], got {tuple(waveform.shape)}.")
    if waveform.shape[0] > 8 and waveform.shape[1] <= 8:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] > 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    sample_rate = int(torch.as_tensor(sample_rate).item())
    if sample_rate <= 0:
        raise ValueError(f"Audio sample rate must be positive, got {sample_rate}.")
    pcm = (torch.nan_to_num(waveform).clamp(-1, 1).transpose(0, 1).numpy() * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _export_video(
    output: torch.Tensor,
    output_path: str,
    *,
    fps: int,
    audio: Any = None,
    audio_sample_rate: Any = None,
    video_exporter: Callable[..., str] | None = None,
    ffmpeg_exe: str | None = None,
) -> None:
    if video_exporter is None:
        from diffusers.utils import export_to_video

        video_exporter = export_to_video

    frames = video_tensor_to_pil_frames(output)
    if audio is None:
        video_exporter(frames, output_path, fps=fps)
        return
    if audio_sample_rate is None:
        raise ValueError("audio_sample_rate is required when logging a video with audio.")
    if ffmpeg_exe is None:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_exe = get_ffmpeg_exe()

    output_path = Path(output_path)
    silent_path = output_path.with_suffix(".silent.mp4")
    audio_path = output_path.with_suffix(".wav")
    try:
        video_exporter(frames, str(silent_path), fps=fps)
        _write_wav(audio, audio_sample_rate, audio_path)
        subprocess.run(
            [
                ffmpeg_exe,
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(silent_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-shortest",
                str(output_path),
            ],
            check=True,
        )
    finally:
        silent_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


def wrap_val_samples_for_wandb(samples, fps=24, output_dir=None):
    """Wrap validation samples and prepare top-level ``wandb`` video media.

    Video outputs ``[T, C, H, W]`` are encoded to mp4 and passed to
    ``wandb.Video`` by path. Provide ``output_dir`` to keep the media available
    for asynchronous upload; otherwise a temp dir is returned for cleanup.
    Optional tuple elements four and five carry audio and its sample rate. The
    table stores a stable media key because offline ``wandb`` tables do not
    reliably persist nested videos. Other outputs become ``wandb.Image``.
    """
    import wandb

    video_dir = output_dir
    video_tmp_dir = None
    wrapped = []
    media_to_log = {}
    for sample in samples:
        inp, out, score = sample[:3]
        audio = sample[3] if len(sample) > 3 else None
        audio_sample_rate = sample[4] if len(sample) > 4 else None
        if hasattr(out, "ndim") and out.ndim == 5:
            # Batched video [B, T, C, H, W]; log the first sample.
            out = out[0]
        if hasattr(out, "ndim") and out.ndim == 4:
            if video_dir is None:
                video_tmp_dir = tempfile.mkdtemp(prefix="val_video_")
                video_dir = video_tmp_dir
            else:
                os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(video_dir, f"{len(wrapped)}.mp4")
            _export_video(out, video_path, fps=fps, audio=audio, audio_sample_rate=audio_sample_rate)
            media_key = f"val/videos/sample_{len(wrapped) + 1}"
            media_to_log[media_key] = wandb.Video(video_path, format="mp4")
            media = media_key
        else:
            if not isinstance(out, torch.Tensor) or out.dtype != torch.uint8:
                raise ValueError(f"Expected a uint8 image tensor, got {getattr(out, 'dtype', type(out))}.")
            media = wandb.Image(out, file_type="jpg", normalize=False)
        wrapped.append((inp, media, score))
    return wrapped, video_tmp_dir, media_to_log


def log_wandb_media(media: dict[str, Any], step: int) -> None:
    """Buffer top-level ``wandb`` media for the validation table log at ``step``."""
    if not media:
        return

    import wandb

    if wandb.run is not None:
        wandb.log(media, step=step, commit=False)
