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

import logging
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, resolve_is_video, validate_audio
from verl_omni.utils.reward_score.reward_utils import normalize_video_tensor

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT_S = 600


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


def validate_audio_batch(audios: Any, rates: Any, count: int, *, context: str) -> tuple[list[Any], list[Any]]:
    """Normalize audio/sample-rate pairs and validate every CT waveform before retention.

    Args:
        audios: Optional CT waveforms, as a batch or per-sample sequence.
        rates: Optional scalar or per-sample sample rates.
        count: Full sample count before export limits are applied.
        context: Description included in waveform validation errors.

    Returns:
        Validated per-sample waveforms and sample rates.
    """
    audios = batch_items(audios, count, "audio")
    rates = batch_items(rates, count, "audio_sample_rate")
    for audio, rate in zip(audios, rates, strict=True):
        validate_audio(audio, rate, context=context)
    return audios, rates


def _temp_path_beside(path: Path, suffix: str) -> Path:
    fd, temp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=suffix)
    os.close(fd)
    return Path(temp_path)


def _write_wav(audio: Any, sample_rate: Any, path: Path) -> None:
    waveform = torch.as_tensor(audio).detach().cpu().float()
    while waveform.ndim > 2 and waveform.shape[0] == 1:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected audio shape [T] or [C, T], got {tuple(waveform.shape)}.")
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


def _video_tensor_to_rgb24(video: torch.Tensor | MediaArtifact) -> tuple[np.ndarray, int, int]:
    """Return a contiguous RGB24 array from a supported RGB video layout."""
    if isinstance(video, MediaArtifact):
        if (video.spec.modality, video.spec.representation) != ("video", "decoded"):
            raise ValueError(f"Video export requires a decoded video artifact, got {video.spec}")
        video = video.normalized(context="video export", name="preview").data
        if video.shape[1] == 1:
            video = video.expand(-1, 3, -1, -1)
        elif video.shape[1] == 4:
            video = video[:, :3]  # RGB24 has no alpha channel.
    else:
        video = normalize_video_tensor(video)
    frames = video.detach().permute(0, 2, 3, 1).to(device="cpu").contiguous().numpy()
    return frames, int(frames.shape[2]), int(frames.shape[1])


def _export_video(
    output: torch.Tensor,
    output_path: str,
    *,
    fps: int,
    audio: Any = None,
    audio_sample_rate: Any = None,
    ffmpeg_exe: str | None = None,
) -> None:
    """Encode RGB frames and optional audio in one ffmpeg invocation.

    The RGB frames are spooled to a transient file, like the audio track, instead
    of being written to ffmpeg's stdin. Exports run on a background thread, and a
    process forked meanwhile (e.g. a DataLoader worker) would inherit any pipe to
    ffmpeg and keep it open, so the export would never finish. A file also avoids
    imageio-ffmpeg's ``stdin.write`` path, which can silently truncate a frame.
    """
    if ffmpeg_exe is None:
        from imageio_ffmpeg import get_ffmpeg_exe

        ffmpeg_exe = get_ffmpeg_exe()
    # CPython only uses posix_spawn for an executable given by path.
    ffmpeg_exe = os.path.abspath(shutil.which(ffmpeg_exe) or ffmpeg_exe)

    if isinstance(output, MediaArtifact) and output.spec.fps is not None:
        fps = output.spec.fps
    else:
        fps = int(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    frames, width, height = _video_tensor_to_rgb24(output)
    output_path = Path(output_path)
    frames_path = _temp_path_beside(output_path, ".rgb")
    audio_path = None
    command = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        str(frames_path),
    ]
    try:
        frames.tofile(frames_path)
        if audio is None:
            command.append("-an")
        else:
            if audio_sample_rate is None:
                raise ValueError("audio_sample_rate is required when logging a video with audio.")
            audio_path = _temp_path_beside(output_path, ".wav")
            _write_wav(audio, audio_sample_rate, audio_path)
            command.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"])

        command.extend(
            [
                # yuv420p/H.264 needs even dimensions. Padding only affects odd-sized inputs.
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]
        )
        if audio is not None:
            # Keep the full video even when generated audio is shorter, and do not
            # extend the artifact when audio is longer.
            command.extend(["-c:a", "aac", "-t", str(frames.shape[0] / fps)])
        command.append(str(output_path))
        try:
            # No pipes, and close_fds=False lets CPython use posix_spawn, which creates no
            # exec-status pipe either, so a concurrent fork inherits nothing from this call.
            subprocess.run(command, stdin=subprocess.DEVNULL, close_fds=False, check=True, timeout=_FFMPEG_TIMEOUT_S)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    finally:
        frames_path.unlink(missing_ok=True)
        if audio_path is not None:
            audio_path.unlink(missing_ok=True)


def wrap_val_samples_for_wandb(samples, fps=24, output_dir=None, media_kinds=None):
    """Wrap validation samples and prepare top-level ``wandb`` media.

    Declared ``media_kinds`` or artifact specs decide whether each output is an
    image or video. Video outputs are normalized to ``[T, C, H, W]`` and encoded
    to mp4 and passed to ``wandb.Video`` by path. Provide ``output_dir`` to keep
    the media available for asynchronous upload; otherwise a temp dir is
    returned for cleanup. Optional tuple elements four and five carry audio and
    its sample rate. Media conversion failures are represented in the table and
    logged instead of propagating into the training loop.
    """
    import wandb

    samples = list(samples)
    media_kinds = batch_items(media_kinds, len(samples), "media_kind")
    video_dir = output_dir
    video_tmp_dir = None
    wrapped = []
    media_to_log = {}
    for sample, media_kind in zip(samples, media_kinds, strict=True):
        inp, out, score = sample[:3]
        audio = sample[3] if len(sample) > 3 else None
        audio_sample_rate = sample[4] if len(sample) > 4 else None
        if isinstance(out, MediaArtifact):
            out.validate(context="W&B", name="preview")
            if out.spec.representation != "decoded" or out.spec.modality not in ("image", "video"):
                raise ValueError(f"W&B requires a decoded visual artifact, got {out.spec}")
            media_kind = out.spec.modality
            if media_kind == "image":
                out = out.normalized(context="W&B", name="preview").data
        output_ndim = getattr(out, "ndim", -1)
        is_video = resolve_is_video(media_kind)
        if is_video and output_ndim == 5:
            # Batched video [B, T, C, H, W]; TCHW is validated below.
            out = out[0]
        if not isinstance(out, MediaArtifact):
            MediaArtifact(
                MediaSpec(media_kind, "decoded", "TCHW" if is_video else "CHW", fps=fps if is_video else None),
                out,
            ).validate(context="W&B", name="preview")
        if is_video:
            try:
                validate_audio(audio, audio_sample_rate, context="W&B")
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
            except Exception as error:
                logger.warning("Could not log validation sample %d video: %s", len(wrapped), error)
                media = f"[validation media unavailable: {type(error).__name__}: {error}]"
        else:
            try:
                if not isinstance(out, torch.Tensor) or out.dtype != torch.uint8:
                    raise ValueError(f"Expected a uint8 image tensor, got {getattr(out, 'dtype', type(out))}.")
                media = wandb.Image(out, file_type="jpg")
            except Exception as error:
                logger.warning("Could not log validation sample %d image: %s", len(wrapped), error)
                media = f"[validation media unavailable: {type(error).__name__}: {error}]"
        wrapped.append((inp, media, score))
    return wrapped, video_tmp_dir, media_to_log


def log_wandb_media(media: dict[str, Any], step: int) -> None:
    """Buffer top-level ``wandb`` media for the validation table log at ``step``."""
    if not media:
        return

    import wandb

    if wandb.run is not None:
        wandb.log(media, step=step, commit=False)
