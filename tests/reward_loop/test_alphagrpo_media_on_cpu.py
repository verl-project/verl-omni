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

"""Named image selectors and generated thinking text share the reward boundary."""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.pipelines.rollout_media import MediaArtifact, MediaSpec, artifact_fields
from verl_omni.reward_loop.reward_manager.media import _reward_extra_info


@pytest.mark.parametrize("transport", ["top", "tool"])
def test_generated_thinking_preserves_named_media(transport):
    latent = torch.zeros(4, 16)
    image = torch.full((3, 8, 8), 31, dtype=torch.uint8)
    fields = artifact_fields(
        {
            "image_latent": MediaArtifact(MediaSpec("image", "latent", "LC"), latent),
            "image_preview": MediaArtifact(MediaSpec("image", "decoded", "CHW"), image),
        },
        primary="image_latent",
        preview="image_preview",
    )
    fields["thinking_text"] = "<think>generated</think>"
    non = {"extra_info": {"thinking_text": "stale", "semantic_questions": ["Cat?"]}}
    if transport == "tool":
        non["tool_extra_fields"] = fields
        batch = {}
    else:
        batch = {key: value for key, value in fields.items() if isinstance(value, torch.Tensor)}
        non.update({key: value for key, value in fields.items() if not isinstance(value, torch.Tensor)})
    info = _reward_extra_info(SimpleNamespace(batch=batch, non_tensor_batch=non))
    assert info["thinking_text"] == "<think>generated</think>"
    assert info["primary_artifact"] == "image_latent"
    assert info["preview_artifact"] == "image_preview"
    assert info["semantic_questions"] == ["Cat?"]
    torch.testing.assert_close(info["media_artifacts"]["image_preview"].data, image)
    torch.testing.assert_close(info["media_artifacts"]["image_latent"].data, latent)
