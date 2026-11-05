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

"""CPU tests for the SwanLab helpers in verl_omni.utils.tracking."""

import io
import wave

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers")

from tests.utils.swanlab_stub import use_swanlab  # noqa: E402
from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec  # noqa: E402
from verl_omni.utils.tracking import log_swanlab_media, wrap_val_samples_for_swanlab  # noqa: E402

WARM = (191, 89, 38)


@pytest.fixture(autouse=True)
def swanlab(monkeypatch):
    return use_swanlab(monkeypatch)


def _warm_image(channels=3, h=32, w=48):
    image = torch.zeros(channels, h, w, dtype=torch.uint8)
    for channel, value in enumerate(WARM[:channels]):
        image[channel] = value
    return image


def _warm_clip(t=4, h=32, w=48):
    clip = _warm_image(h=h, w=w).unsqueeze(0).repeat(t, 1, 1, 1)
    # PIL merges identical GIF frames, so each frame gets its own blue level.
    clip[:, 2] += (torch.arange(t, dtype=torch.uint8) * 4).view(t, 1, 1)
    return clip


def _decoded_image(media):
    return Image.open(io.BytesIO(media.buffer.getvalue()))


def test_image_sample_becomes_a_swanlab_image_named_in_its_row():
    rows, media = wrap_val_samples_for_swanlab([("prompt", _warm_image(), 0.5)], media_kinds=["image"])

    assert rows == [("prompt", "val/images/sample_1", 0.5)]
    assert list(media) == ["val/images/sample_1"]
    decoded = np.asarray(_decoded_image(media["val/images/sample_1"]).convert("RGB"))
    assert decoded.shape == (32, 48, 3)
    r, g, b = decoded.reshape(-1, 3).mean(axis=0)
    assert r > g > b, f"warm image inverted: R={r:.1f} G={g:.1f} B={b:.1f}"


@pytest.mark.parametrize("channels", [1, 4])
def test_grayscale_and_alpha_images_are_accepted(channels):
    rows, media = wrap_val_samples_for_swanlab([("prompt", _warm_image(channels), 0.5)], media_kinds=["image"])

    assert rows[0][1] == "val/images/sample_1"
    assert _decoded_image(media["val/images/sample_1"]).size == (48, 32)


def test_decoded_image_artifact_uses_its_declared_modality():
    preview = MediaArtifact(MediaSpec("image", "decoded", "CHW"), _warm_image())

    rows, media = wrap_val_samples_for_swanlab([("prompt", preview, 0.5)], media_kinds=[None])

    assert rows[0][1] == "val/images/sample_1"
    assert "val/images/sample_1" in media


def test_video_sample_becomes_a_decodable_gif(swanlab):
    rows, media = wrap_val_samples_for_swanlab([("prompt", _warm_clip(t=5), 1.0)], fps=10, media_kinds=["video"])

    assert rows == [("prompt", "val/videos/sample_1", 1.0)]
    video = media["val/videos/sample_1"]
    assert isinstance(video, swanlab.Video)
    gif = Image.open(io.BytesIO(video.buffer.getvalue()))
    assert gif.format == "GIF"
    assert gif.size == (48, 32)
    assert gif.n_frames == 5
    assert gif.info["duration"] == 100
    r, g, b = np.asarray(gif.convert("RGB")).reshape(-1, 3).mean(axis=0)
    assert r > g > b, f"warm frame inverted: R={r:.1f} G={g:.1f} B={b:.1f}"


def test_batched_video_and_artifact_fps():
    batched = _warm_clip(t=3).unsqueeze(0)
    rows, media = wrap_val_samples_for_swanlab([("one", batched, 1.0)], fps=10, media_kinds=["video"])
    assert rows[0][1] == "val/videos/sample_1"
    assert Image.open(io.BytesIO(media["val/videos/sample_1"].buffer.getvalue())).n_frames == 3

    preview = MediaArtifact(MediaSpec("video", "decoded", "TCHW", fps=4), _warm_clip(t=3))
    _, media = wrap_val_samples_for_swanlab([("two", preview, 1.0)], fps=10, media_kinds=[None])
    assert Image.open(io.BytesIO(media["val/videos/sample_1"].buffer.getvalue())).info["duration"] == 250


def test_large_video_is_downscaled_keeping_its_aspect_ratio():
    clip = _warm_clip(t=2, h=720, w=1280)

    _, media = wrap_val_samples_for_swanlab([("prompt", clip, 1.0)], media_kinds=["video"])

    assert Image.open(io.BytesIO(media["val/videos/sample_1"].buffer.getvalue())).size == (512, 288)


def test_large_video_frames_are_downscaled_before_the_next_one_converts(monkeypatch):
    images = []
    full_size_images = []
    fromarray = Image.fromarray

    def tracking_fromarray(frame, *args, **kwargs):
        full_size_images.append(sum(image.size != (512, 32) for image in images))
        images.append(fromarray(frame, *args, **kwargs))
        return images[-1]

    monkeypatch.setattr(Image, "fromarray", tracking_fromarray)

    wrap_val_samples_for_swanlab([("prompt", _warm_clip(t=8, h=64, w=1024), 1.0)], media_kinds=["video"])

    assert len(images) == 8
    assert max(full_size_images) == 0


def test_image_sample_audio_is_not_logged_like_the_wandb_path():
    sample = ("prompt", _warm_image(), 1.0, torch.zeros(1, 800), 8000)

    rows, media = wrap_val_samples_for_swanlab([sample], media_kinds=["image"])

    assert rows[0][1] == "val/images/sample_1"
    assert set(media) == {"val/images/sample_1"}


def test_audio_goes_to_its_own_key_next_to_the_video():
    waveform = torch.linspace(-0.5, 0.5, 800).unsqueeze(0)
    sample = ("prompt", _warm_clip(), 1.0, waveform, 8000)

    rows, media = wrap_val_samples_for_swanlab([sample], media_kinds=["video"])

    assert rows[0][1] == "val/videos/sample_1"
    assert set(media) == {"val/videos/sample_1", "val/audios/sample_1"}
    with wave.open(io.BytesIO(media["val/audios/sample_1"].buffer.getvalue())) as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 8000
        assert wav_file.getnframes() == 800


def test_multichannel_audio_is_downmixed_for_swanlab():
    sample = ("prompt", _warm_clip(), 1.0, torch.zeros(6, 800), torch.tensor(8000))

    _, media = wrap_val_samples_for_swanlab([sample], media_kinds=["video"])

    with wave.open(io.BytesIO(media["val/audios/sample_1"].buffer.getvalue())) as wav_file:
        assert wav_file.getnchannels() == 1


def test_bad_audio_keeps_the_video():
    sample = ("prompt", _warm_clip(), 1.0, torch.zeros(1, 800), -1)

    rows, media = wrap_val_samples_for_swanlab([sample], media_kinds=["video"])

    assert rows[0][1] == "val/videos/sample_1"
    assert set(media) == {"val/videos/sample_1"}


def test_one_failed_video_is_skipped_and_the_others_continue(monkeypatch, swanlab):
    real_video = swanlab.Video
    calls = []

    def flaky_video(path):
        calls.append(path)
        if len(calls) == 1:
            raise TypeError("unsupported")
        return real_video(path)

    monkeypatch.setattr(swanlab, "Video", flaky_video)

    samples = [(f"prompt {i}", _warm_clip(), float(i)) for i in range(2)]
    rows, media = wrap_val_samples_for_swanlab(samples, media_kinds=["video"] * 2)

    assert rows[0][1].startswith("[validation media unavailable: TypeError: unsupported")
    assert rows[1][1] == "val/videos/sample_2"
    assert list(media) == ["val/videos/sample_2"]


def test_failed_image_conversion_is_reported_in_its_row(monkeypatch, swanlab):
    def broken_image(*args, **kwargs):
        raise OSError("cannot encode")

    monkeypatch.setattr(swanlab, "Image", broken_image)

    rows, media = wrap_val_samples_for_swanlab([("prompt", _warm_image(), 0.5)], media_kinds=["image"])

    assert rows[0][1] == "[validation media unavailable: OSError: cannot encode]"
    assert media == {}


@pytest.mark.parametrize(
    ("output", "media_kind", "match"),
    [
        (_warm_image(), None, "media_kind"),
        (_warm_image().float(), "image", "uint8"),
    ],
)
def test_contract_errors_raise_like_the_wandb_path(output, media_kind, match):
    with pytest.raises(ValueError, match=match):
        wrap_val_samples_for_swanlab([("prompt", output, 0.5)], media_kinds=[media_kind])


def test_log_swanlab_media_passes_the_step_and_skips_empty_media(monkeypatch, swanlab):
    logged = []
    monkeypatch.setattr(swanlab, "log", lambda data, step=None: logged.append((data, step)))

    log_swanlab_media({}, step=3)
    assert logged == []

    log_swanlab_media({"val/images/sample_1": "image"}, step=3)
    assert logged == [({"val/images/sample_1": "image"}, 3)]
