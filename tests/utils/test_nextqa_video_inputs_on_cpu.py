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

"""Exercise media boundaries with mocked decoders, without a rollout engine."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset_module(monkeypatch):
    # Only the external base class is isolated. Load the real shared padding
    # implementation and the complete new dataset module from disk.
    base = ModuleType("verl.utils.dataset.rl_dataset")
    base.RLHFDataset = type("RLHFDataset", (), {})
    monkeypatch.setitem(sys.modules, "verl.utils.dataset.rl_dataset", base)
    shared = load_file("omni_dataset_test", "verl_omni/utils/dataset/omni_rl_datasets.py")
    monkeypatch.setitem(sys.modules, "verl_omni.utils.dataset.omni_rl_datasets", shared)
    return load_file("nextqa_dataset_test", "verl_omni/utils/dataset/nextqa_rl_dataset.py")


def test_soundtrack_is_explicit_and_input_is_not_modified(dataset_module):
    video = {"type": "video", "video": "/clip.mp4", "video_start": 2, "video_end": 5, "fps": 1}
    messages = [{"role": "user", "content": [video, {"type": "text", "text": "Why?"}]}]
    result = dataset_module.with_video_soundtracks(messages)
    assert result[0]["content"][1] == {"type": "audio", "audio": "/clip.mp4", "audio_start": 2, "audio_end": 5}
    assert [item["type"] for item in result[0]["content"]] == ["video", "audio", "text"]
    assert len(messages[0]["content"]) == 2
    assert result[0]["content"][0] is not video


def test_decode_preserves_metadata_and_pads_audio(dataset_module, monkeypatch):
    frames = object()
    audio = np.ones(801, dtype=np.float32)
    metadata = {"fps": 30, "total_num_frames": 1800, "frames_indices": np.arange(32)}
    calls = []

    def decode(messages, **kwargs):
        calls.append(kwargs)
        return [audio], None, [(frames, metadata)]

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=decode))
    images, videos, audios = dataset_module.NextQARLHFDataset._process_multi_modal_info([], 16, {})
    assert images is None
    assert videos[0][0] is frames
    assert videos[0][1]["frames_indices"] == list(range(32))
    assert videos[0][1]["total_num_frames"] == 1800
    assert videos[0][1]["duration"] == 60
    assert audios[0].shape == (960,)
    np.testing.assert_array_equal(audios[0][:801], audio)
    np.testing.assert_array_equal(audios[0][801:], 0)
    assert calls == [{"use_audio_in_video": False, "image_patch_size": 16, "return_video_metadata": True}]


def test_mp4_soundtrack_reaches_qwen_as_waveform(dataset_module, monkeypatch):
    messages = dataset_module.with_video_soundtracks(
        [{"content": [{"type": "video", "video": "/clips/a b.mp4", "video_start": 2, "video_end": 3}]}]
    )
    waveform = np.ones(16000, dtype="<f4")

    def run(command, **kwargs):
        assert command[command.index("-i") + 1] == "/clips/a b.mp4"
        assert command[command.index("-ss") + 1] == "2.0"
        assert command[command.index("-t") + 1] == "1.0"
        return SimpleNamespace(stdout=waveform.tobytes())

    def decode(decoded, **kwargs):
        video, audio = decoded[0]["content"]
        assert video == messages[0]["content"][0]
        assert set(audio) == {"type", "audio"}  # no second crop in Qwen
        np.testing.assert_array_equal(audio["audio"], waveform)
        return [audio["audio"]], None, None

    monkeypatch.setattr(dataset_module.subprocess, "run", run)
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=decode))
    _, _, audios = dataset_module.NextQARLHFDataset._process_multi_modal_info(messages, 16, {})
    np.testing.assert_array_equal(audios[0], waveform)
    assert messages[0]["content"][1]["audio"] == "/clips/a b.mp4"
    assert messages[0]["content"][1]["audio_start"] == 2


@pytest.mark.parametrize("failure", ["missing", "codec", "timeout", "empty", "nonfinite"])
def test_soundtrack_decode_failure_is_actionable(dataset_module, monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == "missing":
            raise FileNotFoundError()
        if failure == "codec":
            raise subprocess.CalledProcessError(1, args[0], stderr=b"no audio stream")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], 120)
        return SimpleNamespace(stdout=b"" if failure == "empty" else np.array([np.nan], dtype="<f4").tobytes())

    monkeypatch.setattr(dataset_module.subprocess, "run", run)
    messages = dataset_module.with_video_soundtracks([{"content": [{"type": "video", "video": "clip.mp4"}]}])
    with pytest.raises((RuntimeError, ValueError), match="ffmpeg|clip.mp4") as caught:
        dataset_module.decode_video_soundtracks(messages)
    expected_causes = {
        "missing": FileNotFoundError,
        "codec": subprocess.CalledProcessError,
        "timeout": subprocess.TimeoutExpired,
    }
    if failure in expected_causes:
        assert isinstance(caught.value.__cause__, expected_causes[failure])


def test_real_aac_mp4_soundtrack_and_clip(dataset_module, monkeypatch, tmp_path):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    path = tmp_path / "clip with spaces.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=s=32x32:r=8:d=2",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=2",
            "-c:v",
            "mpeg4",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    real_run = subprocess.run

    def run(command, **kwargs):
        return real_run([ffmpeg, *command[1:]], **kwargs)

    monkeypatch.setattr(dataset_module.subprocess, "run", run)
    messages = dataset_module.with_video_soundtracks(
        [{"content": [{"type": "video", "video": str(path), "video_start": 0.5, "video_end": 1.5}]}]
    )
    audio = dataset_module.decode_video_soundtracks(messages)[0]["content"][1]
    assert audio["audio"].shape == (16000,)
    assert audio["audio"].dtype == np.float32
    assert np.isfinite(audio["audio"]).all()
    assert np.max(np.abs(audio["audio"])) > 0.01
    assert set(audio) == {"type", "audio"}


@pytest.mark.parametrize("options", [{"use_audio_in_video": True}, {"sampling_rate": 48000}])
def test_incompatible_audio_configuration_fails_before_decode(dataset_module, monkeypatch, options):
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=None))
    with pytest.raises(ValueError):
        dataset_module.NextQARLHFDataset._process_multi_modal_info([], 14, {"mm_processor_kwargs": options})


@pytest.mark.parametrize("start,end", [(-1, None), (float("nan"), None), (0, float("inf")), (2, 2), (2, 1)])
def test_invalid_soundtrack_interval_fails_before_ffmpeg(dataset_module, monkeypatch, start, end):
    def unexpected_decode(*args, **kwargs):
        pytest.fail("Invalid intervals must be rejected before launching FFmpeg")

    monkeypatch.setattr(dataset_module.subprocess, "run", unexpected_decode)
    messages = dataset_module.with_video_soundtracks(
        [{"content": [{"type": "video", "video": "clip.mp4", "video_start": start, "video_end": end}]}]
    )
    with pytest.raises(ValueError, match="Invalid soundtrack interval"):
        dataset_module.decode_video_soundtracks(messages)


def test_qwen_decode_error_preserves_original_exception(dataset_module, monkeypatch):
    original = ValueError("invalid video metadata")

    def decode(*args, **kwargs):
        raise original

    monkeypatch.setitem(sys.modules, "qwen_omni_utils", SimpleNamespace(process_mm_info=decode))
    with pytest.raises(ValueError) as caught:
        dataset_module.NextQARLHFDataset._process_multi_modal_info([], 16, {})
    assert caught.value is original


def test_text_and_standalone_audio_are_not_treated_as_soundtracks(dataset_module, monkeypatch):
    messages = [
        {"role": "system", "content": "Answer the question."},
        {"content": [{"type": "audio", "audio": "standalone.wav"}, {"type": "text", "text": "Why?"}]},
    ]

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Standalone audio must be left for the Qwen audio loader")

    monkeypatch.setattr(dataset_module.subprocess, "run", unexpected_decode)
    result = dataset_module.decode_video_soundtracks(dataset_module.with_video_soundtracks(messages))
    assert result == messages
    assert result is not messages
