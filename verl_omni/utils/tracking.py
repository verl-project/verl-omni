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
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, resolve_is_video, validate_audio
from verl_omni.utils.reward_score.reward_utils import normalize_video_tensor

logger = logging.getLogger(__name__)

_FFMPEG_TIMEOUT_S = 600
# GIF is far larger than H.264 at the same size, so SwanLab previews are capped.
_SWANLAB_GIF_MAX_SIDE = 512


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


def _audio_to_pcm16(audio: Any, sample_rate: Any) -> tuple[np.ndarray, int]:
    """Return ``[T, C]`` 16-bit PCM (at most two channels) and the integer sample rate."""
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
    return pcm, sample_rate


def _write_wav(audio: Any, sample_rate: Any, path: Path) -> None:
    pcm, sample_rate = _audio_to_pcm16(audio, sample_rate)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(pcm.shape[1])
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _resolve_video_fps(video: Any, fps: Any) -> int | float:
    """Return a video artifact's declared fps, or the positive integer fallback."""
    if isinstance(video, MediaArtifact) and video.spec.fps is not None:
        fps = video.spec.fps
    else:
        fps = int(fps)
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    return fps


def _video_tensor_to_rgb24(video: torch.Tensor | MediaArtifact) -> tuple[Iterator[np.ndarray], int, int]:
    """Validate a supported RGB video layout and return lazy RGB24 frames with the width and height.

    Each ``[H, W, 3]`` frame is converted when it is consumed, so an export never
    holds a second full-resolution copy of the clip.
    """
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
    video = video.detach()
    frames = (frame.permute(1, 2, 0).to(device="cpu").contiguous().numpy() for frame in video)
    return frames, int(video.shape[3]), int(video.shape[2])


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

    fps = _resolve_video_fps(output, fps)

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
        frame_count = 0
        with open(frames_path, "wb") as frames_file:
            for frame in frames:
                frame.tofile(frames_file)
                frame_count += 1
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
            command.extend(["-c:a", "aac", "-t", str(frame_count / fps)])
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


def _validated_val_output(out: Any, media_kind: str | None, *, fps: int, context: str) -> tuple[Any, bool]:
    """Validate one decoded validation output and return it with whether it is a video."""
    if isinstance(out, MediaArtifact):
        out.validate(context=context, name="preview")
        if out.spec.representation != "decoded" or out.spec.modality not in ("image", "video"):
            raise ValueError(f"{context} requires a decoded visual artifact, got {out.spec}")
        media_kind = out.spec.modality
        if media_kind == "image":
            out = out.normalized(context=context, name="preview").data
    output_ndim = getattr(out, "ndim", -1)
    is_video = resolve_is_video(media_kind)
    if is_video and output_ndim == 5:
        # Batched video [B, T, C, H, W]; TCHW is validated below.
        out = out[0]
    if not isinstance(out, MediaArtifact):
        MediaArtifact(
            MediaSpec(media_kind, "decoded", "TCHW" if is_video else "CHW", fps=fps if is_video else None),
            out,
        ).validate(context=context, name="preview")
    return out, is_video


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
        out, is_video = _validated_val_output(out, media_kind, fps=fps, context="W&B")
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


def _swanlab_image(output: Any, swanlab: Any) -> Any:
    """Build a ``swanlab.Image`` from a uint8 ``[C, H, W]`` tensor."""
    frame = output.detach().permute(1, 2, 0).cpu().numpy()
    # JPEG has no alpha channel, and swanlab.Image takes grayscale as [H, W].
    frame = frame[..., 0] if frame.shape[-1] == 1 else frame[..., :3]
    return swanlab.Image(frame, file_type="jpg")


def _swanlab_gif(output: Any, swanlab: Any, *, fps: int) -> Any:
    """Build a looping-GIF ``swanlab.Video``, the only video format SwanLab accepts."""
    fps = _resolve_video_fps(output, fps)
    frames, _, _ = _video_tensor_to_rgb24(output)
    images = []
    for frame in frames:
        # Downscale before keeping the frame, so only capped previews wait for the encode.
        image = Image.fromarray(frame)
        image.thumbnail((_SWANLAB_GIF_MAX_SIDE, _SWANLAB_GIF_MAX_SIDE))
        images.append(image)
    # swanlab.Video reads the file while it is constructed, so the temp file can go right after.
    with tempfile.TemporaryDirectory(prefix="val_gif_") as gif_dir:
        gif_path = os.path.join(gif_dir, "sample.gif")
        images[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / fps),
            loop=0,
        )
        return swanlab.Video(gif_path)


def wrap_val_samples_for_swanlab(
    samples: Iterable[Sequence[Any]], fps: int = 24, media_kinds: Any = None
) -> tuple[list[tuple[Any, str, Any]], dict[str, Any]]:
    """Prepare validation samples for SwanLab's text table and top-level media.

    SwanLab tables only hold text, so each row names the media key that was
    logged for that sample. GIF, the only video format ``swanlab.Video`` takes,
    has no audio track, so optional tuple elements four and five (audio and its
    sample rate) go to their own key. As on W&B, only video samples carry audio.
    Media conversion failures are represented in the table and logged instead of
    propagating into the training loop.
    """
    import swanlab

    samples = list(samples)
    media_kinds = batch_items(media_kinds, len(samples), "media_kind")
    rows = []
    media_to_log = {}
    for sample, media_kind in zip(samples, media_kinds, strict=True):
        inp, out, score = sample[:3]
        audio = sample[3] if len(sample) > 3 else None
        audio_sample_rate = sample[4] if len(sample) > 4 else None
        sample_number = len(rows) + 1
        out, is_video = _validated_val_output(out, media_kind, fps=fps, context="SwanLab")
        try:
            if is_video:
                media = _swanlab_gif(out, swanlab, fps=fps)
                cell = f"val/videos/sample_{sample_number}"
            else:
                media = _swanlab_image(out, swanlab)
                cell = f"val/images/sample_{sample_number}"
            media_to_log[cell] = media
        except Exception as error:
            logger.warning("Could not log validation sample %d media to SwanLab: %s", len(rows), error)
            cell = f"[validation media unavailable: {type(error).__name__}: {error}]"
        if is_video and audio is not None:
            try:
                validate_audio(audio, audio_sample_rate, context="SwanLab")
                pcm, rate = _audio_to_pcm16(audio, audio_sample_rate)
                media_to_log[f"val/audios/sample_{sample_number}"] = swanlab.Audio(pcm.T, sample_rate=rate)
            except Exception as error:
                logger.warning("Could not log validation sample %d audio to SwanLab: %s", len(rows), error)
        rows.append((inp, cell, score))
    return rows, media_to_log


def log_swanlab_media(media: dict[str, Any], step: int) -> None:
    """Log top-level SwanLab media for the validation table at ``step``."""
    if not media:
        return

    import swanlab

    swanlab.log(media, step=step)


def log_val_samples(
    generations_logger: Any,
    loggers: Sequence[str],
    samples: Sequence[Sequence[Any]],
    step: int,
    *,
    fps: int = 24,
    media_root: str | None = None,
) -> None:
    """Log validation samples and their media to every configured logger.

    Media and table logging failures are logged instead of propagating into the
    training loop.

    Args:
        generations_logger: Table logger with a ``log(loggers, samples, step)`` method.
        loggers: Configured logger names.
        samples: Retained ``(input, output, score, audio, audio_sample_rate, media_kind)`` rows.
        step: Global step of the samples.
        fps: Fallback frame rate for video outputs.
        media_root: Directory that keeps W&B videos for its asynchronous upload.
            They go to a temporary directory when it is unset.
    """
    media_kinds = [sample[5] for sample in samples]

    # SwanLab's table takes text and its media types differ, so it gets its own rows.
    table_loggers = loggers
    swanlab_rows = None
    swanlab_media = {}
    if "swanlab" in loggers:
        table_loggers = [name for name in loggers if name != "swanlab"]
        swanlab_rows, swanlab_media = wrap_val_samples_for_swanlab(samples, fps=fps, media_kinds=media_kinds)

    # Keep wandb videos in persistent storage.
    video_tmp_dir = None
    wandb_media = {}
    if "wandb" in table_loggers:
        wandb_video_dir = os.path.join(media_root, "wandb_val_media", f"global_step_{step}") if media_root else None
        rows, video_tmp_dir, wandb_media = wrap_val_samples_for_wandb(
            samples, fps=fps, output_dir=wandb_video_dir, media_kinds=media_kinds
        )
    else:
        rows = [
            (input_, output.data if isinstance(output, MediaArtifact) else output, score)
            for input_, output, score, *_ in samples
        ]

    try:
        try:
            log_wandb_media(wandb_media, step)
        except Exception as error:
            logger.warning("Skipping validation media log at step %s: %s", step, error)
        try:
            log_swanlab_media(swanlab_media, step)
        except Exception as error:
            logger.warning("Skipping SwanLab validation media log at step %s: %s", step, error)
        try:
            generations_logger.log(table_loggers, rows, step)
        except Exception as error:
            logger.warning("Skipping validation table log at step %s: %s", step, error)
        if swanlab_rows is not None:
            try:
                generations_logger.log(["swanlab"], swanlab_rows, step)
            except Exception as error:
                logger.warning("Skipping SwanLab validation table log at step %s: %s", step, error)
    finally:
        if video_tmp_dir is not None:
            shutil.rmtree(video_tmp_dir, ignore_errors=True)
