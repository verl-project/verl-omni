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
"""CPU-only unit tests for ``WanFlops``."""

import math

import pytest

from verl_omni.utils.mfu import DiffusionFlopsCounter, WanFlops
from verl_omni.utils.mfu.diffusion_flops_counter import _REGISTRY

# Mirrors a real Wan2.1/2.2 14B transformer/config.json.
WAN_CONFIG: dict = {
    "_class_name": "WanTransformer3DModel",
    "patch_size": (1, 2, 2),
    "num_attention_heads": 40,
    "attention_head_dim": 128,
    "in_channels": 16,
    "out_channels": 16,
    "text_dim": 4096,
    "ffn_dim": 13824,
    "num_layers": 40,
    "added_kv_proj_dim": None,
}


def _counter(cfg: dict | None = None) -> DiffusionFlopsCounter:
    return DiffusionFlopsCounter("WanPipeline", cfg or WAN_CONFIG)


def _reference_wan_flops(
    config: dict,
    latent_seqlens: list[int],
    prompt_seqlens: list[int],
    delta_time: float,
    *,
    num_timesteps: int,
    num_forward_passes: int,
) -> float:
    """Independent reference re-derived from ``WanTransformerBlock`` source."""
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    num_layers = int(config["num_layers"])
    ffn_dim = int(config["ffn_dim"])
    in_channels = int(config["in_channels"])
    out_channels = int(config.get("out_channels") or in_channels)
    text_dim = int(config.get("text_dim", 0))
    patch_size = config.get("patch_size", (1, 2, 2))
    patch_prod = 1
    for p in patch_size:
        patch_prod *= int(p)

    dim = num_heads * head_dim
    added_kv = int(config.get("added_kv_proj_dim") or dim)
    batch_size = max(len(latent_seqlens), len(prompt_seqlens))
    img_tot = sum(latent_seqlens)
    txt_tot = sum(prompt_seqlens)

    flops_fwd = 0.0
    for _layer in range(num_layers):
        # attn1 (self, QKV+out on latents): 4*dim^2.
        flops_fwd += 2 * (4 * dim * dim) * img_tot
        # attn2 (cross): Q + out on latents (2*dim^2), K/V from text (2*dim*added_kv).
        flops_fwd += 2 * (2 * dim * dim) * img_tot
        flops_fwd += 2 * (2 * dim * added_kv) * txt_tot
        # ffn: dim -> ffn_dim -> dim.
        flops_fwd += 2 * (2 * dim * ffn_dim) * img_tot

    flops_fwd += 2 * (in_channels * patch_prod * dim) * img_tot  # patch_embedding
    flops_fwd += 2 * (patch_prod * out_channels * dim) * img_tot  # proj_out
    flops_fwd += 2 * (text_dim * dim + dim * dim) * txt_tot  # text_embedder (2-layer)
    flops_fwd += 2 * (6 * dim * dim) * batch_size  # condition_embedder.time_proj

    for latent_s, prompt_s in zip(latent_seqlens, prompt_seqlens, strict=False):
        flops_fwd += num_layers * 2 * 2 * (latent_s**2) * head_dim * num_heads  # self-attn
        flops_fwd += num_layers * 2 * 2 * (latent_s * prompt_s) * head_dim * num_heads  # cross-attn

    flops_fwd_bwd = 3 * flops_fwd
    flops_all_steps = flops_fwd_bwd * num_timesteps * num_forward_passes
    return flops_all_steps / delta_time / 1e12


class TestWanFlopsRegistry:
    def test_registered(self):
        assert _REGISTRY["WanPipeline"] is WanFlops


class TestWanFlopsLatentSeqlens:
    def test_divides_by_patch_volume(self):
        from tests.utils.test_diffusion_flops_counter_on_cpu import _Tensor

        # (B=1, C=16, T=21, H=60, W=104) raw latent, patch=(1,2,2) -> 21*30*52.
        data = {"image_latents": _Tensor((1, 16, 21, 60, 104))}
        seqs = WanFlops(WAN_CONFIG).get_latent_seqlens(data)
        assert seqs == [21 * 30 * 52]


class TestWanFlopsScaling:
    def _kwargs(self, **overrides):
        defaults = dict(
            latent_seqlens=[512, 512],
            prompt_seqlens=[64, 96],
            delta_time=2.0,
            num_timesteps=10,
            num_forward_passes=2,
        )
        defaults.update(overrides)
        return defaults

    def test_linear_in_num_timesteps(self):
        counter = _counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(num_timesteps=10))
        est_b, _ = counter.estimate_flops(**self._kwargs(num_timesteps=30))
        assert math.isclose(est_b / est_a, 3.0, rel_tol=1e-9)

    def test_superlinear_in_latent_seqlen(self):
        # Wan's huge ffn_dim (13824) makes the dense (linear) term dominate
        # at moderate seqlens, unlike Qwen/SD3 where attention dominates
        # quickly; the quadratic self-attn term only overtakes it at scale.
        # A 16x input growth should still yield a strictly-superlinear
        # (>16x) output growth once the self-attn term is large enough.
        counter = _counter()
        kw = dict(prompt_seqlens=[64], delta_time=1.0, num_timesteps=1, num_forward_passes=1)
        small, _ = counter.estimate_flops(latent_seqlens=[4096], **kw)
        large, _ = counter.estimate_flops(latent_seqlens=[65536], **kw)
        assert large / small > 16.0

    def test_matches_hand_rolled_reference(self):
        counter = _counter()
        kwargs = self._kwargs()
        est, _ = counter.estimate_flops(**kwargs)
        ref = _reference_wan_flops(WAN_CONFIG, **kwargs)
        assert math.isclose(est, ref, rel_tol=1e-9), (est, ref)

    def test_matches_reference_across_shapes_and_added_kv(self):
        cfg = dict(WAN_CONFIG, added_kv_proj_dim=2048)
        counter = _counter(cfg)
        scenarios = [
            dict(latent_seqlens=[256], prompt_seqlens=[64], delta_time=0.5, num_timesteps=1, num_forward_passes=1),
            dict(
                latent_seqlens=[1024, 4096],
                prompt_seqlens=[128, 512],
                delta_time=8.0,
                num_timesteps=50,
                num_forward_passes=2,
            ),
        ]
        for kwargs in scenarios:
            est, _ = counter.estimate_flops(**kwargs)
            ref = _reference_wan_flops(cfg, **kwargs)
            assert math.isclose(est, ref, rel_tol=1e-9), (kwargs, est, ref)

    def test_mismatched_seqlen_lengths_raises(self):
        counter = _counter()
        with pytest.raises(ValueError, match="same length"):
            counter.estimate_flops(
                latent_seqlens=[1024, 1024],
                prompt_seqlens=[256],
                delta_time=1.0,
                num_timesteps=1,
                num_forward_passes=1,
            )


class TestWanFlopsParamCount:
    """Ground-truth check against an instantiated ``WanTransformer3DModel``."""

    @pytest.fixture(scope="class")
    def tiny_wan(self):
        from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

        return WanTransformer3DModel(
            patch_size=(1, 2, 2),
            num_attention_heads=4,
            attention_head_dim=16,
            in_channels=8,
            out_channels=8,
            text_dim=32,
            freq_dim=32,
            ffn_dim=128,
            num_layers=2,
            added_kv_proj_dim=None,
        )

    def test_per_layer_weight_count_matches_module_numel(self, tiny_wan):
        dim = tiny_wan.config.num_attention_heads * tiny_wan.config.attention_head_dim
        block = tiny_wan.blocks[0]

        self_attn_weights = sum(
            m.weight.numel() for m in (block.attn1.to_q, block.attn1.to_k, block.attn1.to_v, block.attn1.to_out[0])
        )
        cross_q_out_weights = sum(m.weight.numel() for m in (block.attn2.to_q, block.attn2.to_out[0]))
        cross_kv_weights = sum(m.weight.numel() for m in (block.attn2.to_k, block.attn2.to_v))
        ffn_weights = sum(p.numel() for p in block.ffn.parameters() if p.dim() == 2)

        assert self_attn_weights == 4 * dim * dim
        assert cross_q_out_weights == 2 * dim * dim
        assert cross_kv_weights == 2 * dim * dim  # added_kv_proj_dim=None -> falls back to dim
        assert ffn_weights == 2 * dim * tiny_wan.config.ffn_dim
