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
"""CPU tests for the Qwen3-Omni video decoder's frame-rate handling."""

import sys
from fractions import Fraction
from types import SimpleNamespace

import pytest
from PIL import Image

from verl_omni.utils.dataset import qwen3_omni_transform


class _FakeContainer:
    def __init__(self, *, average_rate, guessed_rate):
        stream = SimpleNamespace(average_rate=average_rate, guessed_rate=guessed_rate)
        self.streams = SimpleNamespace(video=[stream])
        self.closed = False

    def decode(self, video):
        for _ in range(3):
            yield SimpleNamespace(to_image=lambda: Image.new("RGB", (4, 2)))

    def close(self):
        self.closed = True


def _decode(monkeypatch, container):
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda path: container))
    return qwen3_omni_transform._decode_video_path("clip.mp4")


def test_decode_uses_the_average_frame_rate(monkeypatch):
    frames, video_fps = _decode(monkeypatch, _FakeContainer(average_rate=Fraction(30, 1), guessed_rate=Fraction(25, 1)))

    assert video_fps == 30.0
    assert tuple(frames.shape) == (3, 3, 2, 4)


def test_decode_falls_back_to_the_guessed_frame_rate(monkeypatch):
    _, video_fps = _decode(monkeypatch, _FakeContainer(average_rate=None, guessed_rate=Fraction(24000, 1001)))

    assert video_fps == pytest.approx(23.976, abs=1e-3)


def test_decode_rejects_a_video_without_a_frame_rate(monkeypatch):
    container = _FakeContainer(average_rate=None, guessed_rate=None)

    with pytest.raises(ValueError, match=r"clip\.mp4 reports no frame rate"):
        _decode(monkeypatch, container)

    assert container.closed
