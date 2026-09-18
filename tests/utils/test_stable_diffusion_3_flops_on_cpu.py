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
"""CPU-only unit tests for ``StableDiffusion3Flops``."""

import math
import warnings

import pytest

from verl_omni.utils.mfu import DiffusionFlopsCounter, StableDiffusion3Flops
from verl_omni.utils.mfu.diffusion_flops_counter import _REGISTRY

# Mirrors a real SD3-medium transformer/config.json (no dual_attention_layers).
SD3_CONFIG: dict = {
    "_class_name": "SD3Transformer2DModel",
    "sample_size": 128,
    "patch_size": 2,
    "in_channels": 16,
    "num_layers": 24,
    "attention_head_dim": 64,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "caption_projection_dim": 1536,
    "pooled_projection_dim": 2048,
    "out_channels": 16,
    "dual_attention_layers": (),
}


def _counter() -> DiffusionFlopsCounter:
    return DiffusionFlopsCounter("StableDiffusion3Pipeline", SD3_CONFIG)


def _reference_sd3_flops(
    config: dict,
    latent_seqlens: list[int],
    prompt_seqlens: list[int],
    delta_time: float,
    *,
    num_timesteps: int,
    num_forward_passes: int,
) -> float:
    """Independent reference re-derived from ``JointTransformerBlock`` source."""
    num_heads = int(config["num_attention_heads"])
    head_dim = int(config["attention_head_dim"])
    num_layers = int(config["num_layers"])
    in_channels = int(config["in_channels"])
    joint_attention_dim = int(config["joint_attention_dim"])
    caption_projection_dim = int(config["caption_projection_dim"])
    patch_size = int(config.get("patch_size", 2))
    out_channels = int(config.get("out_channels") or in_channels)

    dim = num_heads * head_dim
    batch_size = max(len(latent_seqlens), len(prompt_seqlens))
    img_tot = sum(latent_seqlens)
    txt_tot = sum(prompt_seqlens)

    flops_fwd = 0.0
    for layer in range(num_layers):
        is_last = layer == num_layers - 1
        # Image stream: attn (QKV+out, 4*dim^2) + ff (dim->4dim->dim, 8*dim^2).
        flops_fwd += 2 * (12 * dim * dim) * img_tot
        # Text stream: same as image, except the last (context_pre_only)
        # layer drops to_add_out and ff_context, keeping only add_q/k/v_proj.
        txt_n = 12 * dim * dim if not is_last else 3 * dim * dim
        flops_fwd += 2 * txt_n * txt_tot
        # AdaLN modulation, batch-scaled (one temb per sample).
        img_mod_n = 6 * dim * dim
        txt_mod_n = 6 * dim * dim if not is_last else 2 * dim * dim
        flops_fwd += 2 * (img_mod_n + txt_mod_n) * batch_size

        # Joint full attention over the concatenated (img, txt) sequence.
        for img_s, txt_s in zip(latent_seqlens, prompt_seqlens, strict=False):
            joint_s = img_s + txt_s
            flops_fwd += 2 * 2 * (joint_s**2) * head_dim * num_heads

    flops_fwd += 2 * (in_channels * patch_size * patch_size * dim) * img_tot  # pos_embed
    flops_fwd += 2 * (joint_attention_dim * caption_projection_dim) * txt_tot  # context_embedder
    flops_fwd += 2 * (patch_size * patch_size * out_channels * dim) * img_tot  # proj_out

    flops_fwd_bwd = 3 * flops_fwd
    flops_all_steps = flops_fwd_bwd * num_timesteps * num_forward_passes
    return flops_all_steps / delta_time / 1e12


class TestStableDiffusion3FlopsRegistry:
    def test_registered(self):
        assert _REGISTRY["StableDiffusion3Pipeline"] is StableDiffusion3Flops
        assert _REGISTRY["StableDiffusion3PipelineWithLogProb"] is StableDiffusion3Flops


class TestStableDiffusion3FlopsLatentSeqlens:
    def test_divides_by_patch_area(self):
        # 512x512 raw latent, patch_size=2 -> 64x64 = 4096 tokens.
        from tests.utils.test_diffusion_flops_counter_on_cpu import _Tensor

        data = {"image_latents": _Tensor((2, 16, 64, 64))}
        seqs = StableDiffusion3Flops(SD3_CONFIG).get_latent_seqlens(data)
        assert seqs == [32 * 32] * 2


class TestStableDiffusion3FlopsScaling:
    def _kwargs(self, **overrides):
        defaults = dict(
            latent_seqlens=[1024, 1024],
            prompt_seqlens=[256, 192],
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

    def test_linear_in_num_forward_passes(self):
        counter = _counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(num_forward_passes=1))
        est_b, _ = counter.estimate_flops(**self._kwargs(num_forward_passes=2))
        assert math.isclose(est_b / est_a, 2.0, rel_tol=1e-9)

    def test_matches_hand_rolled_reference(self):
        counter = _counter()
        kwargs = self._kwargs()
        est, _ = counter.estimate_flops(**kwargs)
        ref = _reference_sd3_flops(SD3_CONFIG, **kwargs)
        assert math.isclose(est, ref, rel_tol=1e-9), (est, ref)

    def test_matches_reference_across_shapes(self):
        counter = _counter()
        scenarios = [
            dict(latent_seqlens=[256], prompt_seqlens=[64], delta_time=0.5, num_timesteps=1, num_forward_passes=1),
            dict(
                latent_seqlens=[1024, 4096],
                prompt_seqlens=[128, 512],
                delta_time=8.0,
                num_timesteps=50,
                num_forward_passes=2,
            ),
            dict(latent_seqlens=[1], prompt_seqlens=[1], delta_time=1.0, num_timesteps=1, num_forward_passes=1),
        ]
        for kwargs in scenarios:
            est, _ = counter.estimate_flops(**kwargs)
            ref = _reference_sd3_flops(SD3_CONFIG, **kwargs)
            assert math.isclose(est, ref, rel_tol=1e-9), (kwargs, est, ref)

    def test_attention_is_quadratic_in_joint_seqlen(self):
        counter = _counter()
        est_small, _ = counter.estimate_flops(
            latent_seqlens=[512], prompt_seqlens=[256], delta_time=1.0, num_timesteps=1, num_forward_passes=1
        )
        est_large, _ = counter.estimate_flops(
            latent_seqlens=[1024], prompt_seqlens=[512], delta_time=1.0, num_timesteps=1, num_forward_passes=1
        )
        ratio = est_large / est_small
        assert 2.0 < ratio < 4.0, ratio

    def test_dual_attention_layers_warns(self):
        cfg = dict(SD3_CONFIG, dual_attention_layers=(0, 1, 2))
        counter = DiffusionFlopsCounter("StableDiffusion3Pipeline", cfg)
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            counter.estimate_flops(
                latent_seqlens=[256], prompt_seqlens=[64], delta_time=1.0, num_timesteps=1, num_forward_passes=1
            )
        assert any("dual_attention_layers" in str(w.message) for w in warned)


class TestStableDiffusion3FlopsParamCount:
    """Ground-truth check against an instantiated ``SD3Transformer2DModel``."""

    @pytest.fixture(scope="class")
    def tiny_sd3(self):
        from diffusers import SD3Transformer2DModel

        return SD3Transformer2DModel(
            sample_size=32,
            patch_size=2,
            in_channels=8,
            num_layers=3,
            attention_head_dim=16,
            num_attention_heads=4,
            joint_attention_dim=48,
            caption_projection_dim=64,
            pooled_projection_dim=32,
            out_channels=8,
            dual_attention_layers=(),
        )

    def test_per_layer_weight_count_matches_module_numel(self, tiny_sd3):
        dim = tiny_sd3.config.num_attention_heads * tiny_sd3.config.attention_head_dim

        non_last_block = tiny_sd3.transformer_blocks[0]
        last_block = tiny_sd3.transformer_blocks[-1]

        def stream_weights(block, img: bool) -> int:
            attn = block.attn
            if img:
                linears = [attn.to_q, attn.to_k, attn.to_v, attn.to_out[0]]
                ff = block.ff
            else:
                linears = [attn.add_q_proj, attn.add_k_proj, attn.add_v_proj]
                if attn.to_add_out is not None:
                    linears.append(attn.to_add_out)
                ff = block.ff_context
            total = sum(m.weight.numel() for m in linears)
            if ff is not None:
                total += sum(p.numel() for p in ff.parameters() if p.dim() == 2)
            return total

        assert stream_weights(non_last_block, img=True) == 12 * dim * dim
        assert stream_weights(non_last_block, img=False) == 12 * dim * dim
        assert stream_weights(last_block, img=True) == 12 * dim * dim
        assert stream_weights(last_block, img=False) == 3 * dim * dim
        assert last_block.context_pre_only is True
        assert non_last_block.context_pre_only is False
