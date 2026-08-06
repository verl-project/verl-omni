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
"""CPU tests for JPEG compressibility reward scoring."""

import numpy as np
import torch

from verl_omni.utils.reward_score.jpeg_compressibility import (
    compute_score,
    jpeg_compressibility,
    jpeg_incompressibility,
)


def test_jpeg_incompressibility_accepts_tensor_batch():
    images = torch.zeros(2, 3, 8, 8, dtype=torch.float32)

    scores, meta = jpeg_incompressibility()(images, prompts=None)

    assert scores.shape == (2,)
    assert scores.dtype == np.float64
    assert (scores > 0).all()
    assert meta == {}


def test_jpeg_compressibility_is_negative_scaled_incompressibility():
    images = torch.ones(1, 3, 8, 8, dtype=torch.float32)

    incompressible_scores, _ = jpeg_incompressibility()(images, prompts=None)
    compressible_scores, meta = jpeg_compressibility()(images, prompts=None)

    np.testing.assert_allclose(compressible_scores, -incompressible_scores / 500)
    assert meta == {}


def test_compute_score_accepts_single_image_tensor():
    image = torch.zeros(3, 8, 8, dtype=torch.float32)

    score = compute_score(image)

    assert isinstance(score, float)
    assert score < 0
