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

import pytest
import torch


def _load_tracking_module(monkeypatch):
    root = Path(__file__).parents[2]
    reward_utils_path = root / "verl_omni/utils/reward_score/reward_utils.py"
    reward_utils_spec = importlib.util.spec_from_file_location("reward_utils_under_test", reward_utils_path)
    reward_utils = importlib.util.module_from_spec(reward_utils_spec)
    assert reward_utils_spec.loader is not None
    reward_utils_spec.loader.exec_module(reward_utils)

    # Avoid importing the package root, whose optional rollout dependencies are
    # intentionally absent from this focused utility-test environment.
    package_names = ("verl_omni", "verl_omni.utils", "verl_omni.utils.reward_score")
    for name in package_names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, "verl_omni.utils.reward_score.reward_utils", reward_utils)

    module_path = root / "verl_omni/utils/tracking.py"
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
        assert isinstance(input, memoryview)
        assert input.tobytes() == expected_rgb
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
    assert command[command.index("-vf") : command.index("-vf") + 2] == [
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
    ]
    codec_index = command.index("-c:v")
    assert command[codec_index : codec_index + 4] == ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    assert command[command.index("-c:a") : command.index("-c:a") + 4] == ["-c:a", "aac", "-t", "0.20833333333333334"]
    assert "-shortest" not in command
    assert "+faststart" in command


@pytest.mark.parametrize("layout", ["tchw", "cthw", "thwc"])
def test_video_tensor_to_rgb24_normalizes_supported_layouts(monkeypatch, layout):
    tracking = _load_tracking_module(monkeypatch)
    canonical = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(2, 3, 4, 5)
    video = {
        "tchw": canonical,
        "cthw": canonical.permute(1, 0, 2, 3),
        "thwc": canonical.permute(0, 2, 3, 1),
    }[layout]

    frames, width, height = tracking._video_tensor_to_rgb24(video)

    assert (width, height) == (5, 4)
    assert frames.tobytes() == canonical.permute(0, 2, 3, 1).contiguous().numpy().tobytes()


def test_export_video_pads_odd_dimensions_and_preserves_video_when_audio_is_short(monkeypatch, tmp_path):
    imageio = pytest.importorskip("imageio")
    tracking = _load_tracking_module(monkeypatch)
    output_path = tmp_path / "odd.mp4"

    tracking._export_video(
        torch.zeros(8, 3, 5, 7, dtype=torch.uint8),
        str(output_path),
        fps=8,
        audio=torch.zeros(1, 12_000),
        audio_sample_rate=48_000,
    )

    reader = imageio.get_reader(output_path)
    try:
        frames = [reader.get_data(index) for index in range(8)]
        with pytest.raises(IndexError):
            reader.get_data(8)
    finally:
        reader.close()
    assert len(frames) == 8
    assert frames[0].shape == (6, 8, 3)


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


def test_wandb_wrapper_skips_media_failure_and_continues(monkeypatch, tmp_path):
    tracking = _load_tracking_module(monkeypatch)
    wandb = types.ModuleType("wandb")
    wandb.Video = lambda path, **kwargs: (path, kwargs)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    def fake_export(output, path, **kwargs):
        if path.endswith("0.mp4"):
            raise ValueError("bad media")
        Path(path).write_bytes(b"video")

    monkeypatch.setattr(tracking, "_export_video", fake_export)
    clip = torch.zeros(5, 3, 8, 10, dtype=torch.uint8)
    wrapped, temp_dir, media_to_log = tracking.wrap_val_samples_for_wandb(
        [("first", clip, 0.5), ("second", clip, 0.6)],
        output_dir=str(tmp_path),
    )

    assert wrapped[0][1] == "[validation media unavailable: ValueError: bad media]"
    assert wrapped[1] == ("second", "val/videos/sample_2", 0.6)
    assert list(media_to_log) == ["val/videos/sample_2"]
    assert temp_dir is None


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
