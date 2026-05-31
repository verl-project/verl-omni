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
"""CPU-only unit tests for ``DiffusionFlopsCounter``.

Verifies the architecture-registry dispatch, the Qwen-Image FLOPs formula's
linearity / quadratic scaling, **absolute correctness against a hand-rolled
reference**, and that the formula's implied ``N_params`` matches the actual
``model.numel()`` of a tiny ``QwenImageTransformer2DModel`` instance built
through ``diffusers``. Unknown architectures degrade gracefully to zero
estimated FLOPs (matching the upstream LLM counter).
"""

import math
import warnings

import pytest

from verl_omni.utils.diffusion_flops_counter import (
    _REGISTRY,
    DiffusionFlopsCounter,
    estimate_qwen_image_flops,
    image_seqlens_from_latent_shape,
    register_diffusion_flops_estimator,
    resolve_cfg_passes,
)


def _reference_qwen_image_flops(
    config: dict,
    image_seqlens: list[int],
    prompt_seqlens: list[int],
    delta_time: float,
    *,
    num_timesteps: int,
    cfg_passes: int,
) -> float:
    """Independent, deliberately verbose reference re-implementation.

    Re-derived from the diffusers ``QwenImageTransformerBlock`` source rather
    than from ``estimate_qwen_image_flops``, so that a regression in the
    production formula is caught even if both were edited at once.
    """
    num_attention_heads = int(config["num_attention_heads"])
    attention_head_dim = int(config["attention_head_dim"])
    num_layers = int(config["num_layers"])
    in_channels = int(config["in_channels"])
    joint_attention_dim = int(config["joint_attention_dim"])
    patch_size = int(config.get("patch_size", 2))
    out_channels = int(config.get("out_channels") or in_channels)

    dim = num_attention_heads * attention_head_dim
    batch_size = max(len(image_seqlens), len(prompt_seqlens))
    img_tot = sum(image_seqlens)
    txt_tot = sum(prompt_seqlens)

    # Forward FLOPs per call. Factor 2 below is FLOPs per MAC
    # (one multiply + one add); the 3x backward expansion happens at the
    # end. Dense terms scale per-token; attention scales per joint seq^2;
    # modulation scales per-sample (one timestep embedding per sample).
    flops_fwd = 0.0
    for layer in range(num_layers):
        del layer
        # Image stream per-block linears: to_q/k/v + to_out[0] (4*dim^2)
        # plus img_mlp dim->4*dim->dim (2*4*dim^2) = 12*dim^2 per layer.
        per_img_token = 2 * dim * dim * (3 + 1 + 8)
        flops_fwd += per_img_token * img_tot
        # Text stream is symmetric (add_q/k/v_proj, to_add_out, txt_mlp).
        per_txt_token = 2 * dim * dim * (3 + 1 + 8)
        flops_fwd += per_txt_token * txt_tot
        # img_mod + txt_mod: dim -> 6*dim, applied to one temb per sample.
        flops_fwd += 2 * (2 * 6 * dim * dim) * batch_size

        # Joint full attention per sample: Q@K^T + softmax(weights)@V over
        # the combined (img_s + txt_s) sequence; 2 matmuls of (s, d_h) x
        # (d_h, s) at 2*s^2*d_h FLOPs each, summed over heads.
        for img_s, txt_s in zip(image_seqlens, prompt_seqlens, strict=False):
            joint_s = img_s + txt_s
            flops_fwd += 2 * 2 * (joint_s**2) * attention_head_dim * num_attention_heads

    # Embedding-side linears applied once per token (not per layer).
    flops_fwd += 2 * (in_channels * dim) * img_tot  # img_in
    flops_fwd += 2 * (joint_attention_dim * dim) * txt_tot  # txt_in
    flops_fwd += 2 * (patch_size * patch_size * out_channels * dim) * img_tot  # proj_out

    # Backward computes both dL/dx and dL/dw, each at forward cost, so
    # fwd+bwd = 3 * fwd (verl convention).
    flops_fwd_bwd = 3 * flops_fwd

    flops_all_steps = flops_fwd_bwd * num_timesteps * cfg_passes
    return flops_all_steps / delta_time / 1e12


# Real Qwen-Image transformer config (mirrors
# ~/models/Qwen/Qwen-Image/transformer/config.json so the test does not depend
# on local model files being present).
QWEN_IMAGE_CONFIG: dict = {
    "_class_name": "QwenImageTransformer2DModel",
    "attention_head_dim": 128,
    "guidance_embeds": False,
    "in_channels": 64,
    "joint_attention_dim": 3584,
    "num_attention_heads": 24,
    "num_layers": 60,
    "out_channels": 16,
    "patch_size": 2,
}


def _qwen_counter() -> DiffusionFlopsCounter:
    return DiffusionFlopsCounter("QwenImagePipeline", QWEN_IMAGE_CONFIG)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestDiffusionFlopsRegistry:
    def test_qwen_image_is_registered(self):
        assert "QwenImagePipeline" in _REGISTRY
        assert _REGISTRY["QwenImagePipeline"] is estimate_qwen_image_flops

    def test_custom_architecture_dispatch(self):
        sentinel = 1234.5

        @register_diffusion_flops_estimator("_TestArch_CPU")
        def _stub(config, image_seqlens, prompt_seqlens, delta_time, *, num_timesteps, cfg_passes):
            del config, image_seqlens, prompt_seqlens, delta_time, num_timesteps, cfg_passes
            return sentinel

        counter = DiffusionFlopsCounter("_TestArch_CPU", {})
        est, prom = counter.estimate_flops(
            image_seqlens=[1], prompt_seqlens=[1], delta_time=1.0, num_timesteps=1, cfg_passes=1
        )
        assert est == sentinel
        assert prom > 0

    def test_unknown_architecture_warns_once_and_returns_zero(self):
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            counter = DiffusionFlopsCounter("__DoesNotExist__", {})
            assert any("no FLOPs estimator registered" in str(w.message) for w in warned)
        est, _ = counter.estimate_flops(
            image_seqlens=[1], prompt_seqlens=[1], delta_time=1.0, num_timesteps=1, cfg_passes=1
        )
        assert est == 0.0


# ---------------------------------------------------------------------------
# Numerical scaling
# ---------------------------------------------------------------------------


class TestQwenImageFlopsScaling:
    def _kwargs(self, **overrides):
        defaults = dict(
            image_seqlens=[1024, 1024],
            prompt_seqlens=[256, 192],
            delta_time=2.0,
            num_timesteps=10,
            cfg_passes=2,
        )
        defaults.update(overrides)
        return defaults

    def test_linear_in_num_timesteps(self):
        counter = _qwen_counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(num_timesteps=10))
        est_b, _ = counter.estimate_flops(**self._kwargs(num_timesteps=30))
        assert math.isclose(est_b / est_a, 3.0, rel_tol=1e-9)

    def test_linear_in_cfg_passes(self):
        counter = _qwen_counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(cfg_passes=1))
        est_b, _ = counter.estimate_flops(**self._kwargs(cfg_passes=2))
        assert math.isclose(est_b / est_a, 2.0, rel_tol=1e-9)

    def test_inverse_in_delta_time(self):
        counter = _qwen_counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(delta_time=2.0))
        est_b, _ = counter.estimate_flops(**self._kwargs(delta_time=4.0))
        # FLOPs/s scales inverse with wall-clock time.
        assert math.isclose(est_a / est_b, 2.0, rel_tol=1e-9)

    def test_zero_for_non_positive_inputs(self):
        counter = _qwen_counter()
        est_zero_time, _ = counter.estimate_flops(**self._kwargs(delta_time=0.0))
        est_zero_steps, _ = counter.estimate_flops(**self._kwargs(num_timesteps=0))
        est_zero_cfg, _ = counter.estimate_flops(**self._kwargs(cfg_passes=0))
        assert est_zero_time == 0.0
        assert est_zero_steps == 0.0
        assert est_zero_cfg == 0.0

    def test_attention_is_quadratic_in_joint_seqlen(self):
        """Attention FLOPs scale as :math:`(img_s + txt_s)^2` per sample.

        Doubling the joint seqlen at fixed batch size should ~quadruple
        attention FLOPs while only doubling dense FLOPs. The resulting total
        is between 2x and 4x.
        """
        counter = _qwen_counter()
        est_small, _ = counter.estimate_flops(
            image_seqlens=[512], prompt_seqlens=[256], delta_time=1.0, num_timesteps=1, cfg_passes=1
        )
        est_large, _ = counter.estimate_flops(
            image_seqlens=[1024], prompt_seqlens=[512], delta_time=1.0, num_timesteps=1, cfg_passes=1
        )
        ratio = est_large / est_small
        assert 2.0 < ratio < 4.0, ratio

    def test_matches_hand_rolled_reference(self):
        """Absolute correctness: production estimator agrees with a verbose,
        independent reference re-derived from the diffusers source."""
        counter = _qwen_counter()
        kwargs = self._kwargs()
        est, _ = counter.estimate_flops(**kwargs)
        ref = _reference_qwen_image_flops(QWEN_IMAGE_CONFIG, **kwargs)
        assert math.isclose(est, ref, rel_tol=1e-9), (est, ref)

    def test_matches_reference_across_shapes(self):
        """Same as ``test_matches_hand_rolled_reference`` but for a range of
        batch shapes, including unequal img/txt seqlens, single sample, and
        unusual ``num_timesteps`` / ``cfg_passes``."""
        counter = _qwen_counter()
        scenarios = [
            dict(image_seqlens=[256], prompt_seqlens=[64], delta_time=0.5, num_timesteps=1, cfg_passes=1),
            dict(image_seqlens=[1024, 4096], prompt_seqlens=[128, 512], delta_time=8.0, num_timesteps=50, cfg_passes=2),
            dict(image_seqlens=[4096] * 8, prompt_seqlens=[256] * 8, delta_time=12.0, num_timesteps=10, cfg_passes=1),
        ]
        for kwargs in scenarios:
            est, _ = counter.estimate_flops(**kwargs)
            ref = _reference_qwen_image_flops(QWEN_IMAGE_CONFIG, **kwargs)
            assert math.isclose(est, ref, rel_tol=1e-9), (kwargs, est, ref)

    def test_realistic_size_band_consistent_with_reference(self):
        """Loose sanity band that catches order-of-magnitude regressions."""
        counter = _qwen_counter()
        # 4 samples * 1024 image tokens * 256 text tokens * 1 step * 1 cfg.
        kwargs = dict(
            image_seqlens=[1024] * 4,
            prompt_seqlens=[256] * 4,
            delta_time=1.0,
            num_timesteps=1,
            cfg_passes=1,
        )
        est, _ = counter.estimate_flops(**kwargs)
        ref = _reference_qwen_image_flops(QWEN_IMAGE_CONFIG, **kwargs)
        # Hand sanity check: each token only flows through ONE of the two
        # ~10.2B-param streams. Image stream: 6 * 10.2B * 4096 = 2.5e14; text
        # stream: 6 * 10.2B * 1024 = 6.3e13; total dense ≈ 3.1e14 FLOPs at
        # delta_time=1s. Add ~15 TFLOPS of attention. Empirically ~224 TFLOPS.
        assert 180.0 < est < 320.0, est
        assert math.isclose(est, ref, rel_tol=1e-9)


class TestQwenImageFlopsParamCount:
    """Ground-truth correctness: the formula's implied per-token parameter
    count should match the actual ``model.numel()`` of an instantiated
    ``QwenImageTransformer2DModel`` (modulo small per-sample mod params and
    norms / biases the convention ignores)."""

    @pytest.fixture(scope="class")
    def tiny_qwen_image(self):
        from diffusers import QwenImageTransformer2DModel

        # The on-disk Qwen-Image config has axes_dims_rope=(16, 56, 56) and
        # attention_head_dim=128 (sum=128). Keep the same invariant for the
        # tiny model so RoPE construction succeeds.
        return QwenImageTransformer2DModel(
            num_attention_heads=4,
            attention_head_dim=16,
            num_layers=3,
            in_channels=32,
            out_channels=8,
            patch_size=2,
            joint_attention_dim=48,
            axes_dims_rope=(4, 6, 6),
            guidance_embeds=False,
        )

    def _config_dict(self, model) -> dict:
        # ``register_to_config`` puts the constructor args on ``model.config``;
        # those are the same fields ``estimate_qwen_image_flops`` reads.
        return dict(model.config)

    def test_per_stream_weight_count_matches_module_numel(self, tiny_qwen_image):
        """The per-stream, per-layer token-scaling param count
        (``12 * dim**2``) must equal the sum of *weight* params (no biases)
        across the eight per-token linears of one ``QwenImageTransformerBlock``
        on a single stream.
        """
        block = tiny_qwen_image.transformer_blocks[0]
        cfg = self._config_dict(tiny_qwen_image)
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]

        # Image stream weights only.
        img_stream_weights = sum(
            m.weight.numel() for m in (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.attn.to_out[0])
        ) + sum(p.numel() for p in block.img_mlp.parameters() if p.dim() == 2)

        # Text stream weights only.
        txt_stream_weights = sum(
            m.weight.numel()
            for m in (block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj, block.attn.to_add_out)
        ) + sum(p.numel() for p in block.txt_mlp.parameters() if p.dim() == 2)

        formula_per_stream = 12 * dim * dim
        assert img_stream_weights == formula_per_stream, (img_stream_weights, formula_per_stream)
        assert txt_stream_weights == formula_per_stream, (txt_stream_weights, formula_per_stream)

    def test_per_token_dense_flops_matches_module_numel(self, tiny_qwen_image):
        """Subtract attention / modulation / embedding contributions from
        ``estimate_qwen_image_flops`` and confirm the remaining "block dense"
        FLOPs equal ``6 * (img_stream_weights * img_tot + txt_stream_weights *
        txt_tot)`` computed directly from ``model.numel()``.
        """
        cfg = self._config_dict(tiny_qwen_image)
        dim = cfg["num_attention_heads"] * cfg["attention_head_dim"]
        num_layers = cfg["num_layers"]
        block = tiny_qwen_image.transformer_blocks[0]

        # Real weight counts (no biases) per stream from the instantiated module.
        img_stream_weights = sum(
            m.weight.numel() for m in (block.attn.to_q, block.attn.to_k, block.attn.to_v, block.attn.to_out[0])
        ) + sum(p.numel() for p in block.img_mlp.parameters() if p.dim() == 2)
        txt_stream_weights = sum(
            m.weight.numel()
            for m in (block.attn.add_q_proj, block.attn.add_k_proj, block.attn.add_v_proj, block.attn.to_add_out)
        ) + sum(p.numel() for p in block.txt_mlp.parameters() if p.dim() == 2)

        img_tot = 7
        txt_tot = 5
        counter = DiffusionFlopsCounter("QwenImagePipeline", cfg)
        est_tflops, _ = counter.estimate_flops(
            image_seqlens=[img_tot],
            prompt_seqlens=[txt_tot],
            delta_time=1.0,
            num_timesteps=1,
            cfg_passes=1,
        )
        est_flops = est_tflops * 1e12

        # Strip attention, modulation, and embedding terms.
        seqlen_square_sum = (img_tot + txt_tot) ** 2
        attn_flops = 12 * num_layers * cfg["num_attention_heads"] * cfg["attention_head_dim"] * seqlen_square_sum
        mod_flops = 6 * num_layers * (12 * dim * dim) * 1  # batch_size = 1
        emb_flops = 6 * (
            cfg["in_channels"] * dim * img_tot
            + cfg["joint_attention_dim"] * dim * txt_tot
            + cfg.get("patch_size", 2) ** 2 * (cfg["out_channels"] or cfg["in_channels"]) * dim * img_tot
        )

        block_dense_flops = est_flops - attn_flops - mod_flops - emb_flops
        # block_dense_flops should equal
        #   6 * num_layers * (img_stream_weights * img_tot + txt_stream_weights * txt_tot)
        expected = 6 * num_layers * (img_stream_weights * img_tot + txt_stream_weights * txt_tot)
        assert math.isclose(block_dense_flops, expected, rel_tol=1e-9), (block_dense_flops, expected)


class TestDPGlobalConsistency:
    """Mirror the wiring invariant in ``TrainingWorker._postprocess_output``:
    the counter is fed global (allgathered) seqlens and the resulting MFU is
    divided by ``world_size`` to recover per-GPU achieved compute.

    A single rank reporting per-rank lists and dividing by ``world_size``
    would double-count if the convention were the LLM-style global form.
    These tests pin the per-rank == global / world_size invariant so the
    upstream pattern keeps holding for diffusion.
    """

    def test_global_then_div_world_size_equals_per_rank_no_div(self):
        counter = _qwen_counter()
        per_rank_img = [1024] * 16
        per_rank_txt = [256] * 16
        world_size = 4
        global_img = per_rank_img * world_size
        global_txt = per_rank_txt * world_size

        global_est, prom = counter.estimate_flops(
            image_seqlens=global_img,
            prompt_seqlens=global_txt,
            delta_time=1.0,
            num_timesteps=1,
            cfg_passes=1,
        )
        per_rank_est, _ = counter.estimate_flops(
            image_seqlens=per_rank_img,
            prompt_seqlens=per_rank_txt,
            delta_time=1.0,
            num_timesteps=1,
            cfg_passes=1,
        )
        # Global / world_size should equal per-rank only when the formula is
        # linear in token count (it is, for the dense terms). Attention is
        # also linear when each sample's joint seqlen is identical.
        assert math.isclose(global_est / world_size, per_rank_est, rel_tol=1e-9), (
            global_est / world_size,
            per_rank_est,
        )

    def test_mfu_below_one_at_realistic_step_time(self):
        """At a realistic ``delta_time`` for a 4-GPU L20 Qwen-Image step,
        MFU stays in [0, 1] after dividing by ``world_size``."""
        counter = _qwen_counter()
        world_size = 4
        # ppo_micro_batch_size_per_gpu=16, image 1024x1024 (=1024 latent
        # tokens after 2x patching at 64x64), text seqlen 256, 10 denoising
        # steps, true_cfg_scale=4.0 (=> 2 cfg passes).
        per_rank_img = [1024] * 16
        per_rank_txt = [256] * 16
        global_img = per_rank_img * world_size
        global_txt = per_rank_txt * world_size

        # delta_time = 260 s matches the observed end-to-end actor train
        # block on 4xL20 with LoRA + param/optim offload (rollout/reward
        # excluded).
        est, prom = counter.estimate_flops(
            image_seqlens=global_img,
            prompt_seqlens=global_txt,
            delta_time=260.0,
            num_timesteps=10,
            cfg_passes=2,
        )
        # Mimic TrainingWorker._postprocess_output's final step.
        mfu = est / prom / world_size
        # On L20 (119.5 TFLOPS BF16 peak) with offloaded params + LoRA-only
        # bwd this should be in the 40-80% band; we use a generous bound to
        # catch order-of-magnitude wiring regressions, not steady-state
        # performance fluctuations.
        assert 0.0 < mfu < 1.0, mfu


class TestImageSeqlensFromLatentShape:
    """Generality contract for the latent-shape -> seqlen extractor used by
    ``TrainingWorker._collect_diffusion_flops_meta``.

    Covers image DiTs (3-D / 4-D), video DiTs (5-D), and the FlowGRPO
    time-stacked ``all_latents`` variants of each.
    """

    def test_3d_image_latents_BLC(self):
        assert image_seqlens_from_latent_shape((4, 1024, 64)) == 1024

    def test_4d_image_latents_BCHW(self):
        # 1024x1024 VAE latents at 8x downsample -> 128x128 = 16384 tokens.
        assert image_seqlens_from_latent_shape((4, 16, 128, 128)) == 128 * 128

    def test_5d_video_latents_BCTHW(self):
        # Wan / Hunyuan / LTX / CogVideoX: (B, C, T, H, W).
        assert image_seqlens_from_latent_shape((1, 16, 21, 60, 104)) == 21 * 60 * 104

    def test_4d_all_latents_BTLC(self):
        # FlowGRPO-style image rollouts: (B, T_steps, L, C).
        assert image_seqlens_from_latent_shape((2, 10, 1024, 64), all_latents_layout=True) == 1024

    def test_5d_all_latents_BTCHW(self):
        # FlowGRPO-style image rollouts in (B, T_steps, C, H, W) layout.
        assert image_seqlens_from_latent_shape((2, 10, 16, 128, 128), all_latents_layout=True) == 128 * 128

    def test_6d_all_latents_video(self):
        # FlowGRPO-style video rollouts: (B, T_steps, C, T_lat, H, W).
        assert image_seqlens_from_latent_shape((1, 10, 16, 21, 60, 104), all_latents_layout=True) == 21 * 60 * 104

    def test_none_and_garbage_shapes_return_zero(self):
        assert image_seqlens_from_latent_shape(None) == 0
        assert image_seqlens_from_latent_shape(()) == 0
        assert image_seqlens_from_latent_shape((4,)) == 0
        assert image_seqlens_from_latent_shape((4, 16, 32, 32, 32, 32, 32)) == 0


class TestResolveCfgPasses:
    """Generality contract for CFG-pass detection. The counter API treats
    ``cfg_passes`` as a positive int; this helper resolves it from the
    pipeline + transformer configs without architecture-specific knowledge."""

    def test_no_cfg_returns_one(self):
        # Unconditional / class-conditioned model, no guidance fields.
        assert resolve_cfg_passes({}, {}) == 1
        assert resolve_cfg_passes(None, None) == 1

    def test_true_cfg_scale_qwen_image_style(self):
        # Qwen-Image: pipeline.true_cfg_scale=4.0 -> 2 forward passes.
        assert resolve_cfg_passes({"true_cfg_scale": 4.0}, {}) == 2
        assert resolve_cfg_passes({"true_cfg_scale": 1.0}, {}) == 1
        assert resolve_cfg_passes({"true_cfg_scale": None}, {}) == 1

    def test_guidance_scale_wan_sd3_style(self):
        # Standard CFG: pipeline.guidance_scale=5.0 -> 2 passes (Wan, SD3).
        assert resolve_cfg_passes({"guidance_scale": 5.0}, {}) == 2
        # 1.0 means CFG disabled at inference time.
        assert resolve_cfg_passes({"guidance_scale": 1.0}, {}) == 1

    def test_guidance_distilled_flux_style(self):
        # Flux: transformer_config.guidance_embeds=True means the guidance
        # scalar is consumed *inside* the model; only one forward runs even
        # when guidance_scale > 1.0.
        assert resolve_cfg_passes({"guidance_scale": 3.5}, {"guidance_embeds": True}) == 1
        # guidance_embeds=False is the explicit non-distilled case.
        assert resolve_cfg_passes({"guidance_scale": 3.5}, {"guidance_embeds": False}) == 2

    def test_explicit_override_wins(self):
        # Pipeline can force the value (e.g. custom rollout that batches
        # cond+uncond into one tensor).
        assert resolve_cfg_passes({"cfg_passes": 1, "guidance_scale": 7.5}, {}) == 1
        assert resolve_cfg_passes({"cfg_passes": 2, "true_cfg_scale": 1.0}, {}) == 2
        # Below-1 override is clamped (we never run fewer than one pass).
        assert resolve_cfg_passes({"cfg_passes": 0}, {}) == 1

    def test_unknown_garbage_values_dont_crash(self):
        # Helper survives non-numeric junk for forward-compat with weird
        # pipeline configs.
        assert resolve_cfg_passes({"guidance_scale": "off"}, {}) == 1
        assert resolve_cfg_passes({"true_cfg_scale": object()}, {}) == 1


class TestNonCfgQwenImageFlops:
    """Pin the formula behaviour for the non-CFG case so future estimators
    in the registry can rely on ``cfg_passes=1`` being a no-op multiplier."""

    def test_cfg_passes_one_halves_two(self):
        counter = _qwen_counter()
        kwargs = dict(
            image_seqlens=[1024] * 4,
            prompt_seqlens=[256] * 4,
            delta_time=2.0,
            num_timesteps=5,
        )
        est_one, _ = counter.estimate_flops(cfg_passes=1, **kwargs)
        est_two, _ = counter.estimate_flops(cfg_passes=2, **kwargs)
        assert math.isclose(est_two, 2 * est_one, rel_tol=1e-9)

    def test_unconditional_zero_text_runs(self):
        # Class-conditioned / unconditional: prompt_seqlens = [0]*B should
        # still produce a sensible non-zero FLOPs estimate driven by the
        # image stream alone (attn quadratic term collapses to img_seq^2).
        counter = _qwen_counter()
        est, _ = counter.estimate_flops(
            image_seqlens=[1024] * 4,
            prompt_seqlens=[0] * 4,
            delta_time=1.0,
            num_timesteps=1,
            cfg_passes=1,
        )
        assert est > 0


class TestDiffusionFlopsCounterApi:
    def test_promised_flops_returned_in_tflops(self):
        counter = _qwen_counter()
        _, prom = counter.estimate_flops(
            image_seqlens=[1024], prompt_seqlens=[256], delta_time=1.0, num_timesteps=1, cfg_passes=1
        )
        # ``get_device_flops`` returns TFLOPS by default; CPU baseline is
        # 448 GFLOPS = 0.448 TFLOPS. We accept anything > 0 to keep the test
        # device-agnostic.
        assert prom > 0

    def test_mismatched_seqlen_lengths_raises(self):
        counter = _qwen_counter()
        with pytest.raises(ValueError, match="same length"):
            counter.estimate_flops(
                image_seqlens=[1024, 1024],
                prompt_seqlens=[256],
                delta_time=1.0,
                num_timesteps=1,
                cfg_passes=1,
            )
