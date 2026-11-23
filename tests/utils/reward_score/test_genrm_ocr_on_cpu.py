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

"""OCR 接收 RGBA 时仅对白底评分预览做转换，不修改原始产物。"""

import asyncio
import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from verl_omni.utils.reward_score import genrm_ocr
from verl_omni.utils.reward_score.reward_utils import image_tensor_to_pil


@pytest.mark.parametrize(
    ("channels", "alpha", "expected"),
    [
        (3, None, (0, 0, 0)),
        (4, 0, (255, 255, 255)),
        (4, 128, (127, 127, 127)),
        (4, 255, (0, 0, 0)),
    ],
)
def test_ocr_scores_visible_pixels_and_preserves_source(monkeypatch, channels, alpha, expected):
    image = torch.zeros(channels, 2, 3, dtype=torch.uint8)
    if alpha is not None:
        image[3].fill_(alpha)
    original = image.clone()
    requests = []

    async def complete(router_address, chat_complete_request):
        requests.append(chat_complete_request)
        uri = chat_complete_request["messages"][1]["content"][0]["image_url"]["url"]
        with Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as preview:
            assert preview.mode == "RGB"
            assert preview.size == (3, 2)
            assert set(preview.getdata()) == {expected}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

    monkeypatch.setattr(genrm_ocr, "_chat_complete", complete)
    result = asyncio.run(
        genrm_ocr.compute_score_ocr(
            data_source="flow_grpo/ocr",
            solution_image=image,
            ground_truth="hello",
            extra_info={"media_kind": "image"},
            reward_router_address="unused",
        )
    )
    assert result["score"] == 1.0 and len(requests) == 1
    torch.testing.assert_close(image, original)


def test_pil_conversion_keeps_existing_default_alpha_policy():
    image = torch.zeros(4, 1, 1, dtype=torch.uint8)
    image[0] = 10
    assert image_tensor_to_pil(image).getpixel((0, 0)) == (10, 0, 0)
    assert image_tensor_to_pil(image, alpha_background=(255, 255, 255)).getpixel((0, 0)) == (255, 255, 255)
