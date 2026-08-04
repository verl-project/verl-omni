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
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("diffusers")
wandb = pytest.importorskip("wandb")
imageio = pytest.importorskip("imageio")

from verl_omni.utils.tracking import wrap_val_samples_for_wandb  # noqa: E402


def _warm_clip(t=8, h=32, w=32):
    """A solid warm clip ``[T, C, H, W]`` with R > G > B, so an inverted encode is detectable."""
    clip = torch.zeros(t, 3, h, w)
    clip[:, 0] = 0.75  # R
    clip[:, 1] = 0.35  # G
    clip[:, 2] = 0.15  # B
    return clip


def test_video_samples_become_wandb_video_with_a_real_mp4(monkeypatch):
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

    monkeypatch.setattr(wandb, "Video", _FakeVideo)

    samples = [(f"prompt {i}", _warm_clip(), float(i)) for i in range(2)]
    wrapped, video_tmp_dir = wrap_val_samples_for_wandb(samples, fps=8)

    try:
        assert len(wrapped) == 2
        assert len(captured) == 2
        for (inp, media, score), c in zip(wrapped, captured, strict=True):
            assert isinstance(media, _FakeVideo)
            assert c.path.endswith(".mp4") and c.kwargs.get("format") == "mp4"
            r, g, b = c.rgb
            assert r > g > b and r > 128 and b < 128, f"warm frame inverted: R={r:.1f} G={g:.1f} B={b:.1f}"
        # One shared temp dir the caller is expected to remove.
        assert os.path.basename(video_tmp_dir).startswith("val_video_")
        assert os.path.isdir(video_tmp_dir)
    finally:
        shutil.rmtree(video_tmp_dir, ignore_errors=True)


def test_image_samples_become_wandb_image_and_no_temp_dir(monkeypatch):
    monkeypatch.setattr(wandb, "Image", lambda data, *a, **k: SimpleNamespace(data=data))

    samples = [("prompt", torch.rand(3, 16, 16), 1.0)]
    wrapped, video_tmp_dir = wrap_val_samples_for_wandb(samples)

    assert video_tmp_dir is None
    assert len(wrapped) == 1 and wrapped[0][0] == "prompt"
