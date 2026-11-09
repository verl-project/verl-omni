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

"""Real tiny-CLIP forward and native scorer contracts, without accelerator access."""

import copy

import pytest
import torch
from transformers import CLIPConfig, CLIPModel

from verl_omni.reward_loop.native_fsdp import CLIPScoreForward, NativeFSDPExecutor
from verl_omni.utils.reward_score import clipscore_reward
from verl_omni.utils.reward_score.clipscore_reward import (
    CLIPScoreNativeModel,
    compute_score_clipscore_native,
    pairwise_clipscore,
)
from verl_omni.utils.reward_score.image_preference_utils import clip_feature_tensor
from verl_omni.workers.config.reward import parse_reward_model_config


@pytest.fixture
def tiny_clip():
    torch.manual_seed(73)
    return CLIPModel(
        CLIPConfig(
            text_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "max_position_embeddings": 8,
                "vocab_size": 16,
                "eos_token_id": 2,
            },
            vision_config={
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 8,
                "patch_size": 4,
            },
            projection_dim=8,
        )
    ).eval()


def test_clipscore_forward_matches_real_unsharded_features(tiny_clip):
    images = {"pixel_values": torch.randn(3, 3, 8, 8)}
    text = {"input_ids": torch.tensor([[1, 4, 2], [1, 5, 2], [1, 4, 2]])}
    with torch.no_grad():
        expected = {
            "image_embeddings": clip_feature_tensor(tiny_clip.get_image_features(**images)),
            "text_embeddings": clip_feature_tensor(tiny_clip.get_text_features(**text)),
        }
        actual = CLIPScoreForward(copy.deepcopy(tiny_clip))(images, text)
    assert set(actual) == {"image_embeddings", "text_embeddings"}
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(pairwise_clipscore(**actual), pairwise_clipscore(**expected), rtol=0, atol=0)


def test_openai_transposed_projection_keeps_values_and_becomes_shardable(tiny_clip):
    original = tiny_clip.visual_projection.weight.detach().clone()
    tiny_clip.visual_projection.weight = torch.nn.Parameter(original.t().contiguous().t())
    assert not tiny_clip.visual_projection.weight.is_contiguous()
    adapter = CLIPScoreForward(tiny_clip)
    assert all(parameter.is_contiguous() for parameter in adapter.parameters())
    torch.testing.assert_close(adapter.clip.visual_projection.weight, original, rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in adapter.parameters())


def test_clipscore_adapter_is_explicit_and_reuses_existing_placement_contract():
    model = parse_reward_model_config(
        "clipscore",
        {
            "backend": "native",
            "model_path": "explicit-checkpoint",
            "placement": {"replica_groups": [[0, 1], [2, 3]]},
            "executor": {
                "model": "verl_omni.utils.reward_score.clipscore_reward:CLIPScoreNativeModel",
                "kwargs": {"processor_path": "explicit-processor"},
            },
        },
    )
    assert model.placement.bundle_indices == [0, 1, 2, 3]
    assert model.fsdp.timeout == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("cosine", [1.0, -1.0])
async def test_native_scorer_reuses_clamp_and_explicit_scale(cosine):
    class Handle:
        async def infer(self, prompts, images):
            assert prompts == ["a prompt"]
            assert len(images) == 1
            return [{"text_embeddings": torch.tensor([1.0, 0.0]), "image_embeddings": torch.tensor([cosine, 0.0])}]

    result = await compute_score_clipscore_native(
        "test",
        torch.zeros(3, 8, 8, dtype=torch.uint8),
        "a prompt",
        {"media_kind": "image"},
        Handle(),
        scale_factor=3.0,
    )
    expected = 3.0 if cosine > 0 else 0.0
    assert result == {"score": expected, "clipscore_raw": expected}


@pytest.mark.asyncio
async def test_fsdp_executor_accepts_embedding_only_adapter():
    class Replica:
        def infer(self, prompts, images):
            return {"text_embeddings": torch.arange(6).reshape(3, 2), "image_embeddings": torch.ones(3, 2)}

    result = await NativeFSDPExecutor(Replica()).infer(["a", "b", "c"], [None] * 3)
    assert len(result) == 3
    assert all(set(item) == {"text_embeddings", "image_embeddings"} for item in result)
    torch.testing.assert_close(result[2]["text_embeddings"], torch.tensor([4, 5]))


def test_native_clipscore_reuses_inferencer_and_preserves_rows(monkeypatch, tiny_clip):
    class Processor:
        def __call__(self, images=None, text=None, **kwargs):
            if images is not None:
                return {"pixel_values": torch.stack(images)}
            return {"input_ids": torch.tensor([[1, 4 if item == "same" else 5, 2] for item in text])}

    monkeypatch.setattr(clipscore_reward.CLIPModel, "from_pretrained", lambda *args, **kwargs: tiny_clip)
    monkeypatch.setattr(clipscore_reward.CLIPProcessor, "from_pretrained", lambda *args, **kwargs: Processor())
    model = CLIPScoreNativeModel(model_path="tiny", device="cpu")
    images = [torch.randn(3, 8, 8) for _ in range(3)]
    rows = model.infer(["same", "other", "same"], images)
    reference = model._inferencer.infer(["same", "other", "same"], images)
    for index, row in enumerate(rows):
        for key in reference:
            torch.testing.assert_close(row[key], reference[key][index])
    model.close()
    with pytest.raises(RuntimeError, match="closed"):
        model.infer(["same"], images[:1])
