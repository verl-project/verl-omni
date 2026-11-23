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

"""编辑 scorer 的参考图顺序、透明度、严格解析与真实 manager 参数合约。"""

import asyncio
import base64
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from verl import DataProto

from verl_omni.pipelines.rollout_media import ArtifactContractError
from verl_omni.reward_loop.reward_manager.visual import VisualRewardManager
from verl_omni.utils.reward_score import image_edit_reward


def reference(color):
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGBA", (3, 2), color).save(buffer, format="PNG")
    payload = buffer.getvalue()
    return {"bytes": payload}


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_manager_forwards_reference_images_in_order(monkeypatch, count):
    from PIL import Image

    requests = []
    references = [reference((index * 40, 0, 0, 255)) for index in range(count)]
    generated = torch.zeros(4, 2, 3, dtype=torch.uint8)
    original = generated.clone()

    async def complete(router_address, chat_complete_request):
        assert router_address == "judge.invalid:8000"
        requests.append(chat_complete_request)
        content = chat_complete_request["messages"][1]["content"]
        images = [item["image_url"]["url"] for item in content if item["type"] == "image_url"]
        assert len(images) == count + 1
        for index, uri in enumerate(images):
            with Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
                expected = (index * 40, 0, 0) if index < count else (255, 255, 255)
                assert image.mode == "RGB" and image.getpixel((0, 0)) == expected
        assert json.loads(content[-1]["text"].split(": ", 1)[1]) == "Change only the background to blue."
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"edit_success":0.25,"preservation":0.81}'),
                )
            ]
        )

    monkeypatch.setattr(image_edit_reward, "_chat_complete", complete)
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "verl_omni/trainer/config"), version_base=None):
        config = compose(
            config_name="diffusion_trainer",
            overrides=[
                "reward.reward_model.model_path=/models/Qwen3-VL-8B-Instruct",
                "reward.reward_model.rollout.response_length=512",
            ],
        )
    manager = VisualRewardManager(config, None, image_edit_reward.compute_score_image_edit, "judge.invalid:8000")
    data = DataProto.from_dict(
        tensors={"responses": generated.unsqueeze(0)},
        non_tensors={
            "data_source": np.array(["controlled_edit"], dtype=object),
            "reward_model": np.array([{"ground_truth": "Change only the background to blue."}], dtype=object),
            "extra_info": np.array([{"reference_images": references, "media_kind": "image"}], dtype=object),
            "media_kind": np.array(["image"], dtype=object),
        },
    )
    result = asyncio.run(manager.run_single(data))
    assert result["reward_score"] == pytest.approx(0.45)
    assert result["reward_extra_info"]["preservation"] == 0.81
    assert result["reward_extra_info"]["reference_count"] == count
    assert requests[0]["model"] == "/models/Qwen3-VL-8B-Instruct"
    assert requests[0]["temperature"] == 0.0
    torch.testing.assert_close(generated, original)


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "[]",
        "{}",
        '{"edit_success":0.5}',
        '{"edit_success":true,"preservation":1}',
        '{"edit_success":1.1,"preservation":0.5}',
        '{"edit_success":-0.1,"preservation":0.5}',
        '{"edit_success":NaN,"preservation":0.5}',
        '{"edit_success":"0.5","preservation":0.5}',
        '{"edit_success":1,"preservation":1,"other":0}',
    ],
)
def test_invalid_scores_fail_instead_of_clamping(response):
    with pytest.raises(ValueError):
        image_edit_reward.parse_edit_scores(response)


@pytest.mark.parametrize("count", [0, 5])
def test_unsupported_reference_count_fails_before_service(count):
    with pytest.raises(ArtifactContractError, match="1–4"):
        asyncio.run(
            image_edit_reward.compute_score_image_edit(
                "controlled_edit",
                torch.zeros(3, 2, 2, dtype=torch.uint8),
                "Change the background.",
                {"media_kind": "image", "reference_images": [reference((0, 0, 0, 255))] * count},
                "judge.invalid:8000",
                model_name="/models/Qwen3-VL-8B-Instruct",
            )
        )


def test_reference_decoding_and_transparency():
    from PIL import Image

    transparent = reference((0, 0, 0, 0))
    uri = image_edit_reward._reference_uri(transparent)
    with Image.open(BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
        assert image.getpixel((0, 0)) == (255, 255, 255)
    with pytest.raises(ArtifactContractError, match="Cannot decode"):
        image_edit_reward._reference_uri({"bytes": b"not an image"})


def test_video_is_not_treated_as_a_single_image():
    with pytest.raises(ArtifactContractError, match="not video"):
        asyncio.run(
            image_edit_reward.compute_score_image_edit(
                "controlled_edit",
                torch.zeros(1, 3, 2, 2, dtype=torch.uint8),
                "Change the background.",
                {"media_kind": "video", "fps": 1, "reference_images": [reference((0, 0, 0, 255))]},
                "judge.invalid:8000",
                model_name="/models/Qwen3-VL-8B-Instruct",
            )
        )
