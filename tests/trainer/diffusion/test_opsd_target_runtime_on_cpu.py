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
"""CPU tests for the differentiable-reward feature unwrapping in the OPSD runtime.

Necessity: the GPU smoke run failed because transformers>=5.13 returns
``BaseModelOutputWithPooling`` from ``CLIPModel.get_{text,image}_features`` while the
differentiable path assumed a plain tensor (``AttributeError: ... has no attribute
'norm'``). The loss stayed finite only because the reward never ran; this test pins the
version-tolerant unwrapping so the regression cannot come back silently.
"""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.trainer.diffusion.opsd_target_runtime import _unwrap_clip_features


def test_tensor_input_passes_through():
    features = torch.randn(2, 1024)
    assert _unwrap_clip_features(features) is features


@pytest.mark.parametrize("attribute", ["image_embeds", "text_embeds", "pooler_output"])
def test_model_output_unwraps_projected_embedding(attribute):
    embedding = torch.randn(2, 1024)
    output = SimpleNamespace(**{attribute: embedding})
    unwrapped = _unwrap_clip_features(output)
    assert unwrapped is embedding


def test_model_output_prefers_projected_embeddings_over_pooler():
    projected = torch.randn(2, 1024)
    pooler = torch.randn(2, 768)
    output = SimpleNamespace(image_embeds=projected, pooler_output=pooler)
    assert _unwrap_clip_features(output) is projected


def test_unknown_return_type_raises():
    with pytest.raises(TypeError, match="Unsupported CLIP feature return type"):
        _unwrap_clip_features(SimpleNamespace(latent=torch.randn(2, 4)))
