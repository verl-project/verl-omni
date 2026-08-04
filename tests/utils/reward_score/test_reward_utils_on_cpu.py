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
video_tensor_to_pil_frames = _MODULE.video_tensor_to_pil_frames
pil_image_to_base64 = _MODULE.pil_image_to_base64


def test_video_tensor_to_pil_frames_clips_before_uint8_quantization():
    video = torch.tensor(
        [
            [
                [[-0.1, 0.0, 1.0, 1.1]],
                [[float("nan"), 0.5, float("inf"), -float("inf")]],
                [[0.25, 0.75, 0.999, 0.001]],
            ]
        ]
    )

    frames = video_tensor_to_pil_frames(video)

    assert len(frames) == 1
    assert frames[0].mode == "RGB"
    pixels = np.asarray(frames[0])
    assert pixels.dtype == np.uint8
    assert pixels.tolist() == [[[0, 0, 64], [0, 128, 191], [255, 255, 255], [255, 0, 0]]]


@pytest.mark.parametrize("shape", [(3, 4, 5), (2, 4, 4, 5), (2, 3, 4, 5, 1)])
def test_video_tensor_to_pil_frames_rejects_non_tchw_rgb(shape):
    with pytest.raises(ValueError, match="Expected an RGB video tensor"):
        video_tensor_to_pil_frames(torch.zeros(shape))


def test_pil_image_to_base64_roundtrips_to_the_same_image():
    image = Image.fromarray(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))

    uri = pil_image_to_base64(image)

    assert uri.startswith("data:image/png;base64,")
    decoded = Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.format == "PNG"
    assert np.array_equal(np.asarray(decoded), np.asarray(image))
