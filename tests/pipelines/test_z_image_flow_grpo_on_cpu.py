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
"""CPU tests for the Z-Image FlowGRPO common helpers and registry binding."""

import pytest
import torch

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.z_image_flow_grpo import ZImage
from verl_omni.pipelines.z_image_flow_grpo.common import (
    Z_IMAGE_VAE_SCALE_FACTOR,
    apply_z_image_cfg,
    latents_to_transformer_input,
    split_padded_embeds_to_list,
    stack_transformer_output,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_config(architecture: str):
    from verl_omni.workers.config.diffusion.model import DiffusionModelConfig

    cfg = object.__new__(DiffusionModelConfig)
    object.__setattr__(cfg, "architecture", architecture)
    object.__setattr__(cfg, "external_lib", None)
    return cfg


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestZImageRegistry:
    def test_zimage_registered(self):
        cfg = _make_model_config("ZImagePipeline")
        assert DiffusionModelBase.get_class(cfg) is ZImage


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


class TestCommonHelpers:
    def test_vae_scale_factor_default(self):
        assert Z_IMAGE_VAE_SCALE_FACTOR == 8

    def test_split_padded_embeds_to_list_returns_per_sample_valid_prefix(self):
        # B=2, L=4, D=3
        embeds = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        out = split_padded_embeds_to_list(embeds, mask)
        assert isinstance(out, list)
        assert len(out) == 2
        assert out[0].shape == (3, 3)
        assert out[1].shape == (2, 3)
        assert torch.equal(out[0], embeds[0, :3])
        assert torch.equal(out[1], embeds[1, :2])

    def test_latents_to_transformer_input_shapes(self):
        latents = torch.zeros(2, 16, 8, 8)
        out = latents_to_transformer_input(latents)
        assert isinstance(out, list)
        assert len(out) == 2
        for t in out:
            assert t.shape == (16, 1, 8, 8)

    def test_stack_transformer_output_negates_and_squeezes(self):
        per_sample = [torch.ones(16, 1, 8, 8), torch.full((16, 1, 8, 8), 2.0)]
        out = stack_transformer_output(per_sample)
        assert out.shape == (2, 16, 8, 8)
        # Negation convention: +1 -> -1, +2 -> -2.
        assert torch.allclose(out[0], -torch.ones(16, 8, 8))
        assert torch.allclose(out[1], -torch.full((16, 8, 8), 2.0))

    def test_apply_z_image_cfg_matches_formula(self):
        pos = torch.tensor([[[[1.0, 2.0]]]])  # (1,1,1,2)
        neg = torch.tensor([[[[0.5, 1.0]]]])
        scale = 4.0

        out = apply_z_image_cfg(pos, neg, scale, cfg_normalization=False)
        expected = pos + scale * (pos - neg)
        assert torch.allclose(out, expected)

    def test_apply_z_image_cfg_renormalizes_when_clip_active(self):
        torch.manual_seed(0)
        pos = torch.randn(2, 4, 3, 3)
        neg = torch.zeros_like(pos)
        out = apply_z_image_cfg(pos, neg, cfg_scale=10.0, cfg_normalization=True)

        pos_norm = pos.flatten(1).norm(dim=1)
        out_norm = out.flatten(1).norm(dim=1)
        # With cfg_normalization=True the post-CFG norm cannot exceed the
        # positive-prediction norm.
        assert torch.all(out_norm <= pos_norm + 1e-5)


# ---------------------------------------------------------------------------
# Optional: vllm-omni registration is gated on import availability. We only
# assert it when the dependency is installed.
# ---------------------------------------------------------------------------


def test_vllm_omni_pipeline_registered_when_available():
    try:
        from verl_omni.pipelines.model_base import VllmOmniPipelineBase
        from verl_omni.pipelines.z_image_flow_grpo import ZImagePipelineWithLogProb
    except ImportError:
        pytest.skip("vllm-omni not installed")

    if ZImagePipelineWithLogProb is None:
        pytest.skip("vllm-omni Z-Image pipeline not available")

    assert VllmOmniPipelineBase.get_class("ZImagePipeline") is ZImagePipelineWithLogProb
