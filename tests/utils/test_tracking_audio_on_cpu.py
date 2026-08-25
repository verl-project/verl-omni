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

"""CPU tests for audio-video experiment tracking."""

import importlib.util
import sys
import types
import wave
from pathlib import Path

import torch


def _load_tracking_module(monkeypatch):
    module_path = Path(__file__).parents[2] / "verl_omni/utils/tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_export_video_encodes_rgb_and_audio_in_one_ffmpeg_invocation(monkeypatch, tmp_path):
    tracking = _load_tracking_module(monkeypatch)
    commands = []
    output = torch.arange(5 * 3 * 8 * 10, dtype=torch.uint8).reshape(5, 3, 8, 10)

    def fake_ffmpeg(command, *, input, check):
        assert check is True
        commands.append(command)
        expected_rgb = output.permute(0, 2, 3, 1).contiguous().numpy().tobytes()
        assert input == expected_rgb
        second_input = command.index("-i", command.index("-i") + 1)
        with wave.open(command[second_input + 1], "rb") as wav_file:
            assert wav_file.getframerate() == 48_000
            assert wav_file.getnchannels() == 1
        Path(command[-1]).write_bytes(b"video-with-audio")

    monkeypatch.setattr(tracking.subprocess, "run", fake_ffmpeg)
    output_path = tmp_path / "sample.mp4"
    tracking._export_video(
        output,
        str(output_path),
        fps=24,
        audio=torch.zeros(1, 800),
        audio_sample_rate=48_000,
        ffmpeg_exe="/fake/ffmpeg",
    )

    assert output_path.read_bytes() == b"video-with-audio"
    command = commands[0]
    assert command[command.index("-f") : command.index("-f") + 8] == [
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        "10x8",
        "-pix_fmt",
        "rgb24",
    ]
    codec_index = command.index("-c:v")
    assert command[codec_index : codec_index + 4] == ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    assert command[command.index("-c:a") : command.index("-c:a") + 2] == ["-c:a", "aac"]
    assert "+faststart" in command


def test_wandb_wrapper_forwards_audio_to_video_export(monkeypatch):
    tracking = _load_tracking_module(monkeypatch)
    captured = []
    wandb = types.ModuleType("wandb")
    wandb.Video = lambda path, **kwargs: (path, kwargs)
    wandb.Image = lambda output, **kwargs: (output, kwargs)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    def fake_export(output, path, **kwargs):
        captured.append((output, path, kwargs))

    monkeypatch.setattr(tracking, "_export_video", fake_export)
    clip = torch.zeros(5, 3, 8, 10)
    audio = torch.zeros(1, 800)
    wrapped, temp_dir, media_to_log = tracking.wrap_val_samples_for_wandb(
        [("prompt", clip, 0.5, audio, 48_000)],
        fps=24,
    )

    try:
        assert len(captured) == 1
        assert captured[0][2] == {"fps": 24, "audio": audio, "audio_sample_rate": 48_000}
        assert wrapped == [("prompt", "val/videos/sample_1", 0.5)]
        assert media_to_log == {"val/videos/sample_1": (captured[0][1], {"format": "mp4"})}
    finally:
        Path(temp_dir).rmdir()


def test_wandb_media_is_logged_as_a_top_level_payload(monkeypatch):
    tracking = _load_tracking_module(monkeypatch)
    calls = []
    wandb = types.ModuleType("wandb")
    wandb.run = object()
    wandb.log = lambda payload, step, commit: calls.append((payload, step, commit))
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    media = {"val/videos/sample_1": object()}
    tracking.log_wandb_media(media, step=7)

    assert calls == [(media, 7, False)]
