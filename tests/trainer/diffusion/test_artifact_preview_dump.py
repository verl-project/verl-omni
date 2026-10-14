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
"""GPU regressions for decoded previews paired with native latent responses."""

import pytest
import torch
from PIL import Image
from test_artifact_preview_dump_on_cpu import _dump_image_previews

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec


@pytest.mark.parametrize("version", [0, 1])
@pytest.mark.parametrize("latent_shape", [(2, 16), (2, 16, 2, 2)])
def test_cuda_latent_primary_exports_image_previews(tmp_path, version, latent_shape):
    assert torch.cuda.is_available()
    latents = torch.randn(latent_shape, dtype=torch.bfloat16, device="cuda")
    before = latents.clone()
    previews = [
        MediaArtifact(
            MediaSpec("image", "decoded", "CHW"), torch.full((3, 4, 5), 37, dtype=torch.uint8, device="cuda")
        ),
        MediaArtifact(
            MediaSpec("image", "decoded", "HWC"), torch.full((2, 3, 3), 211, dtype=torch.uint8, device="cuda")
        ),
    ]
    rows = _dump_image_previews(tmp_path, version, latents, previews)
    assert len(rows) == 2
    for row, size, value in zip(rows, [(5, 4), (3, 2)], [37, 211], strict=True):
        with Image.open(row["output"]) as image:
            assert image.size == size
            assert image.getpixel((0, 0)) == (value, value, value)
    assert latents.is_cuda and all(preview.data.is_cuda for preview in previews)
    torch.testing.assert_close(latents, before)
