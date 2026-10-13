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

"""CPU tests for image/video conversion helpers in reward_score.reward_utils."""

import base64
import importlib.util
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

_SRC = Path(__file__).resolve().parents[3] / "verl_omni" / "utils" / "reward_score" / "reward_utils.py"
_SPEC = importlib.util.spec_from_file_location("reward_utils_under_test", _SRC)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
normalize_video_tensor = _MODULE.normalize_video_tensor
video_tensor_to_pil_frames = _MODULE.video_tensor_to_pil_frames
pil_image_to_base64 = _MODULE.pil_image_to_base64


@pytest.mark.parametrize("layout", ["tchw", "cthw", "thwc"])
def test_normalize_video_tensor_accepts_supported_rgb_layouts(layout):
    canonical = torch.arange(2 * 3 * 4 * 5, dtype=torch.uint8).reshape(2, 3, 4, 5)
    video = {
        "tchw": canonical,
        "cthw": canonical.permute(1, 0, 2, 3),
        "thwc": canonical.permute(0, 2, 3, 1),
    }[layout]

    torch.testing.assert_close(normalize_video_tensor(video), canonical)


def test_video_tensor_to_pil_frames_preserves_uint8_pixels():
    video = torch.tensor([[[[0, 1, 255]], [[64, 128, 191]], [[255, 32, 0]]]], dtype=torch.uint8)

    frames = video_tensor_to_pil_frames(video)

    assert len(frames) == 1
    assert frames[0].mode == "RGB"
    pixels = np.asarray(frames[0])
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[[0, 64, 255], [1, 128, 32], [255, 191, 0]]]


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
def test_video_tensor_to_pil_frames_rejects_non_uint8(dtype):
    with pytest.raises(ValueError, match=f"Expected a uint8 video tensor, got {dtype}"):
        video_tensor_to_pil_frames(torch.zeros((1, 3, 4, 5), dtype=dtype))


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 4, 4, 5), (2, 3, 4, 5, 1)])
def test_video_tensor_to_pil_frames_rejects_non_tchw_rgb(shape):
    with pytest.raises(ValueError, match="Expected an RGB video tensor"):
        video_tensor_to_pil_frames(torch.zeros(shape, dtype=torch.uint8))


def test_pil_image_to_base64_roundtrips_to_the_same_image():
    image = Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))

    uri = pil_image_to_base64(image)

    assert uri.startswith("data:image/png;base64,")
    decoded = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.format == "PNG"
    assert np.array_equal(np.asarray(decoded), np.asarray(image))


@pytest.mark.parametrize("rate", [24000, 24000.0, torch.tensor(24000)])
def test_audio_requires_explicit_rate_and_normalizes_declared_channels(rate):
    audio, sample_rate = _MODULE.get_audio(
        {
            "audio": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            "audio_sample_rate": rate,
            "audio_layout": "CT",
        }
    )
    torch.testing.assert_close(audio, torch.full((3,), 0.5))
    assert sample_rate == 24000


def test_audio_rejects_missing_sample_rate():
    with pytest.raises(KeyError, match="audio_sample_rate"):
        _MODULE.get_audio({"audio": torch.ones(8)})


@pytest.mark.parametrize("rate", [None, 0, -1, 24000.5, float("nan"), float("inf"), True, "24000"])
def test_audio_rejects_invalid_rates(rate):
    with pytest.raises((KeyError, ValueError), match="sample.rate|sample_rate"):
        _MODULE.get_audio({"audio": torch.ones(8), "audio_sample_rate": rate})


@pytest.mark.parametrize(
    "audio", [torch.empty(0), torch.empty(0, 8), torch.tensor([float("nan")]), torch.tensor([float("inf")])]
)
def test_audio_rejects_empty_or_nonfinite_input(audio):
    with pytest.raises(ValueError, match="non-empty audio containing only finite"):
        _MODULE.get_audio({"audio": audio, "audio_sample_rate": 24000})


@pytest.mark.parametrize("shape", [(8, 2), (2, 8), (1, 2, 8)])
def test_audio_does_not_guess_two_dimensional_layout(shape):
    with pytest.raises(ValueError, match="explicit audio_layout='CT'"):
        _MODULE.get_audio({"audio": torch.ones(shape), "audio_sample_rate": 24000})


def test_audio_rejects_channels_last_and_conflicting_layouts():
    info = {"audio": torch.ones(8, 2), "audio_sample_rate": 24000, "audio_layout": "TC"}
    with pytest.raises(ValueError, match="channels-last audio is not supported"):
        _MODULE.get_audio(info)
    with pytest.raises(ValueError, match="Conflicting audio_layout"):
        _MODULE.get_audio(info, audio_layout="CT")
