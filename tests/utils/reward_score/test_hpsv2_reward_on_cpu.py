# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch
from PIL import Image

from verl_omni.utils.reward_score.hpsv2_reward import _to_rgb_image


def test_to_rgb_image_converts_tensor_and_pil_inputs() -> None:
    tensor = torch.tensor(
        [
            [[0.0, 1.0], [0.5, 0.25]],
            [[1.0, 0.0], [0.5, 0.25]],
            [[0.0, 0.5], [1.0, 0.25]],
        ]
    )
    converted_tensor = _to_rgb_image(tensor)
    converted_pil = _to_rgb_image(Image.new("RGBA", (2, 1), (1, 2, 3, 4)))

    assert converted_tensor.mode == "RGB"
    assert converted_tensor.size == (2, 2)
    assert converted_tensor.getpixel((0, 0)) == (0, 255, 0)
    assert converted_pil.mode == "RGB"
    assert converted_pil.getpixel((0, 0)) == (1, 2, 3)


def test_to_rgb_image_accepts_singleton_batch_and_grayscale() -> None:
    image = _to_rgb_image(np.array([[[0.0, 1.0], [0.5, 0.25]]], dtype=np.float32))

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert np.asarray(image).tolist() == [
        [[0, 0, 0], [255, 255, 255]],
        [[128, 128, 128], [64, 64, 64]],
    ]


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (object(), "Unsupported image type"),
        (np.zeros((2, 1, 1, 3)), "scores one image per call"),
        (np.zeros((2, 2, 2, 2, 2)), "Expected an HxW"),
        (np.zeros((2, 2, 2)), "Expected 1, 3, or 4 image channels"),
        (np.array([[np.nan]], dtype=np.float32), "non-finite"),
        (np.array([[-1.0]], dtype=np.float32), r"must be in \[0, 1\] or \[0, 255\]"),
        (np.array([[256.0]], dtype=np.float32), r"must be in \[0, 1\] or \[0, 255\]"),
    ],
)
def test_to_rgb_image_rejects_invalid_inputs(value, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _to_rgb_image(value)
