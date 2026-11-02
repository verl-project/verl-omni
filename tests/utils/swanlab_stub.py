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

"""Stand-in for the optional ``swanlab`` package, keeping the rules verl-omni relies on."""

import io
import sys
import wave
from types import ModuleType

import numpy as np
from PIL import Image as PILImage


class Image:
    def __init__(self, data: np.ndarray, file_type: str = "png"):
        self.buffer = io.BytesIO()
        PILImage.fromarray(data).save(self.buffer, format="JPEG" if file_type == "jpg" else file_type.upper())


class Video:
    def __init__(self, path: str):
        if not path.endswith(".gif"):
            raise TypeError(f"Unsupported file extension in {path!r}. Supported: .gif")
        with open(path, "rb") as gif_file:
            self.buffer = io.BytesIO(gif_file.read())


class Audio:
    def __init__(self, data: np.ndarray, sample_rate: int):
        if data.ndim != 2 or data.shape[0] not in (1, 2):
            raise TypeError(f"Expected (num_channels, num_frames) with 1 or 2 channels, got {data.shape}")
        self.buffer = io.BytesIO()
        with wave.open(self.buffer, "wb") as wav_file:
            wav_file.setnchannels(data.shape[0])
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(data.T.tobytes())


def log(data: dict, step: int | None = None) -> None:
    raise RuntimeError("You must call swanlab.init() before swanlab.log()")


def use_swanlab(monkeypatch) -> ModuleType:
    """Return the real package when the extra is installed, otherwise install this stand-in for the test."""
    try:
        import swanlab
    except ImportError:
        swanlab = sys.modules[__name__]
        monkeypatch.setitem(sys.modules, "swanlab", swanlab)
    return swanlab
