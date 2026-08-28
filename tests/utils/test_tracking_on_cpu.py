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

"""CPU tests for verl_omni.utils.tracking.wrap_val_samples_for_wandb."""

import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("diffusers")
wandb = pytest.importorskip("wandb")
imageio = pytest.importorskip("imageio")

from verl_omni.utils.tracking import _video_tensor_to_rgb24, wrap_val_samples_for_wandb  # noqa: E402


def _warm_clip(t=8, h=32, w=32):
    """A solid warm clip ``[T, C, H, W]`` with R > G > B, so an inverted encode is detectable."""
    clip = torch.zeros(t, 3, h, w, dtype=torch.uint8)
    clip[:, 0] = 191  # R
    clip[:, 1] = 89  # G
    clip[:, 2] = 38  # B
    return clip


def test_channels_first_video_is_normalized_to_rgb24_frames():
    video = _warm_clip(t=5, h=8, w=12).permute(1, 0, 2, 3)

    frames, width, height = _video_tensor_to_rgb24(video)

    assert frames.shape == (5, 8, 12, 3)
    assert (width, height) == (12, 8)
    np.testing.assert_array_equal(frames[0, 0, 0], np.array([191, 89, 38], dtype=np.uint8))


def test_video_samples_become_wandb_video_with_a_real_mp4_in_output_dir(monkeypatch, tmp_path):
    captured = []

    class _FakeVideo:
        def __init__(self, path, *args, **kwargs):
            assert os.path.isfile(path), f"wandb.Video got a non-existent path: {path}"
            reader = imageio.get_reader(path)  # ffmpeg backend -> proves it decodes
            frame = reader.get_data(4)
            reader.close()
            captured.append(
                SimpleNamespace(
                    path=path, kwargs=dict(kwargs), rgb=tuple(float(frame[..., c].mean()) for c in range(3))
                )
            )
            self.data_or_path = path

    monkeypatch.setattr(wandb, "Video", _FakeVideo)

    output_dir = tmp_path / "wandb_val_media" / "global_step_3"
    samples = [(f"prompt {i}", _warm_clip(), float(i)) for i in range(2)]
    wrapped, video_tmp_dir, media_to_log = wrap_val_samples_for_wandb(samples, fps=8, output_dir=str(output_dir))

    assert video_tmp_dir is None
    assert len(wrapped) == 2
    assert len(captured) == 2
    for index, ((inp, media_key, score), c) in enumerate(zip(wrapped, captured, strict=True), start=1):
        assert media_key == f"val/videos/sample_{index}"
        assert media_to_log[media_key].data_or_path == c.path
        assert c.path.endswith(".mp4") and c.kwargs.get("format") == "mp4"
        assert os.path.dirname(c.path) == str(output_dir)
        assert os.path.isfile(c.path)
        r, g, b = c.rgb
        assert r > g > b and r > 128 and b < 128, f"warm frame inverted: R={r:.1f} G={g:.1f} B={b:.1f}"
    assert os.path.isdir(output_dir)


def test_video_samples_without_output_dir_return_cleanup_temp_dir(monkeypatch):
    captured = []

    class _FakeVideo:
        def __init__(self, path, *args, **kwargs):
            assert os.path.isfile(path), f"wandb.Video got a non-existent path: {path}"
            captured.append(SimpleNamespace(path=path, kwargs=dict(kwargs)))
            self.data_or_path = path

    monkeypatch.setattr(wandb, "Video", _FakeVideo)

    samples = [("prompt", _warm_clip(), 1.0)]
    wrapped, video_tmp_dir, media_to_log = wrap_val_samples_for_wandb(samples, fps=8)

    try:
        assert video_tmp_dir is not None
        assert os.path.basename(video_tmp_dir).startswith("val_video_")
        assert os.path.isdir(video_tmp_dir)
        assert len(wrapped) == 1
        assert len(captured) == 1
        assert media_to_log["val/videos/sample_1"].data_or_path == captured[0].path
        assert captured[0].path.endswith(".mp4")
        assert captured[0].kwargs.get("format") == "mp4"
        assert os.path.dirname(captured[0].path) == video_tmp_dir
        assert os.path.isfile(captured[0].path)
    finally:
        shutil.rmtree(video_tmp_dir, ignore_errors=True)


def test_image_samples_become_wandb_image_and_no_temp_dir(monkeypatch):
    captured = []

    def _fake_image(data, *args, **kwargs):
        captured.append((data, kwargs))
        return SimpleNamespace(data=data)

    monkeypatch.setattr(wandb, "Image", _fake_image)

    samples = [("prompt", torch.randint(256, (3, 16, 16), dtype=torch.uint8), 1.0)]
    wrapped, video_tmp_dir, media_to_log = wrap_val_samples_for_wandb(samples)

    assert video_tmp_dir is None
    assert media_to_log == {}
    assert len(wrapped) == 1 and wrapped[0][0] == "prompt"
    assert captured[0][0].dtype == torch.uint8
    assert captured[0][1] == {"file_type": "jpg"}


def test_image_samples_reject_non_uint8_input(monkeypatch):
    monkeypatch.setattr(wandb, "Image", lambda *args, **kwargs: pytest.fail("wandb.Image should not be called"))

    with pytest.raises(ValueError, match=r"Expected a uint8 image tensor, got torch\.float32\."):
        wrap_val_samples_for_wandb([("prompt", torch.rand(3, 16, 16), 1.0)])


def test_uint8_image_is_logged_by_real_wandb_offline_run(monkeypatch, tmp_path):
    """Exercise W&B image encoding and logging without an account or network access."""
    for name in ("cache", "config", "data"):
        directory = tmp_path / f"wandb-{name}"
        directory.mkdir()
        monkeypatch.setenv(f"WANDB_{name.upper()}_DIR", str(directory))
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "false")
    monkeypatch.setenv("WANDB_SILENT", "true")

    image = torch.empty((3, 32, 48), dtype=torch.uint8)
    expected_colors = np.array([[16, 32, 64], [96, 128, 160], [192, 224, 240]], dtype=np.float32)
    for index, color in enumerate(expected_colors):
        image[:, :, index * 16 : (index + 1) * 16] = torch.from_numpy(color.astype(np.uint8)).view(3, 1, 1)

    run = wandb.init(
        project="verl-omni-offline-verification",
        mode="offline",
        dir=str(tmp_path),
        settings=wandb.Settings(silent=True),
    )
    assert run.offline
    run_dir = Path(run.dir)
    try:
        wrapped, video_tmp_dir, media_to_log = wrap_val_samples_for_wandb([("synthetic", image, 1.0)])
        assert video_tmp_dir is None
        assert media_to_log == {}
        wandb.log({"verification/image": wrapped[0][1]}, step=1)
    finally:
        run.finish()

    logged_images = list((run_dir / "media" / "images").rglob("*.jpg"))
    assert len(logged_images) == 1

    decoded = np.asarray(Image.open(logged_images[0]).convert("RGB"), dtype=np.float32)
    decoded_colors = np.stack(
        [decoded[:, index * 16 + 4 : (index + 1) * 16 - 4].mean(axis=(0, 1)) for index in range(3)]
    )
    np.testing.assert_allclose(decoded_colors, expected_colors, atol=3)
