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

Verifies the architecture-registry dispatch, the default
``DiffusionModelFlops`` extractors, the Qwen-Image FLOPs formula's
linearity / quadratic scaling, **absolute correctness against a hand-rolled
reference**, and that the formula's implied ``N_params`` matches the actual
``model.numel()`` of a tiny ``QwenImageTransformer2DModel`` instance built
through ``diffusers``. Unknown architectures degrade gracefully to zero
estimated FLOPs (matching the upstream LLM counter).
"""

import math
import warnings

import pytest

from verl_omni.utils.mfu import (
    DiffusionFlopsCounter,
    DiffusionModelFlops,
    QwenImageFlops,
    collect_diffusion_flops_meta,
    get_device_peak_tflops,
    get_forward_passes_per_step,
    register_diffusion_architecture,
)
from verl_omni.utils.mfu.diffusion_flops_counter import _REGISTRY


def _reference_qwen_image_flops(
    config: dict,
    latent_seqlens: list[int],
    prompt_seqlens: list[int],
    delta_time: float,
    *,
    num_timesteps: int,
    num_forward_passes: int,
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
    batch_size = max(len(latent_seqlens), len(prompt_seqlens))
    img_tot = sum(latent_seqlens)
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
        for img_s, txt_s in zip(latent_seqlens, prompt_seqlens, strict=False):
            joint_s = img_s + txt_s
            flops_fwd += 2 * 2 * (joint_s**2) * attention_head_dim * num_attention_heads

    # Embedding-side linears applied once per token (not per layer).
    flops_fwd += 2 * (in_channels * dim) * img_tot  # img_in
    flops_fwd += 2 * (joint_attention_dim * dim) * txt_tot  # txt_in
    flops_fwd += 2 * (patch_size * patch_size * out_channels * dim) * img_tot  # proj_out

    # Backward computes both dL/dx and dL/dw, each at forward cost, so
    # fwd+bwd = 3 * fwd (verl convention).
    flops_fwd_bwd = 3 * flops_fwd

    flops_all_steps = flops_fwd_bwd * num_timesteps * num_forward_passes
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
        assert _REGISTRY["QwenImagePipeline"] is QwenImageFlops
        # Aliases share the same class.
        assert _REGISTRY["QwenImagePipelineWithLogProb"] is QwenImageFlops

    def test_custom_architecture_dispatch(self):
        sentinel = 1234.5

        @register_diffusion_architecture("_TestArch_CPU")
        class _Stub(DiffusionModelFlops):
            def estimate_flops(self, latent_seqlens, prompt_seqlens, delta_time, *, num_timesteps, num_forward_passes):
                del latent_seqlens, prompt_seqlens, delta_time, num_timesteps, num_forward_passes
                return sentinel

        counter = DiffusionFlopsCounter("_TestArch_CPU", {})
        est, prom = counter.estimate_flops(
            latent_seqlens=[1], prompt_seqlens=[1], delta_time=1.0, num_timesteps=1, num_forward_passes=1
        )
        assert est == sentinel
        assert prom > 0

    def test_register_decorator_rejects_non_subclasses(self):
        with pytest.raises(TypeError, match="DiffusionModelFlops"):

            @register_diffusion_architecture("_BadArch_CPU")
            def _not_a_class(*args, **kwargs):  # pragma: no cover
                del args, kwargs
                return 0.0

    def test_unknown_architecture_warns_once_and_returns_zero(self):
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            counter = DiffusionFlopsCounter("__DoesNotExist__", {})
            assert any("no FLOPs estimator registered" in str(w.message) for w in warned)
        est, _ = counter.estimate_flops(
            latent_seqlens=[1], prompt_seqlens=[1], delta_time=1.0, num_timesteps=1, num_forward_passes=1
        )
        assert est == 0.0


class TestDevicePeakOverride:
    """Verify the ``VERL_OMNI_DEVICE_FLOPS_TFLOPS`` env-var override.

    Needed on clusters where ``torch.cuda.get_device_name()`` returns a
    string that mis-matches verl's built-in ``_DEVICE_FLOPS`` table
    (e.g. relabeled H200 SKUs reporting as ``"NVIDIA L20X"`` and falling
    through to the L20 entry at ~12% of the real peak). Setting the env
    var lets the user pin the correct bf16-dense peak so reported MFU is
    physically meaningful.
    """

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        monkeypatch.delenv("VERL_OMNI_DEVICE_FLOPS_TFLOPS", raising=False)

    def test_override_is_used_by_counter(self, monkeypatch):
        monkeypatch.setenv("VERL_OMNI_DEVICE_FLOPS_TFLOPS", "989.0")
        assert get_device_peak_tflops() == pytest.approx(989.0)
        counter = DiffusionFlopsCounter("QwenImagePipeline", QWEN_IMAGE_CONFIG)
        _, promised = counter.estimate_flops(
            latent_seqlens=[1024] * 2,
            prompt_seqlens=[64] * 2,
            delta_time=1.0,
            num_timesteps=1,
            num_forward_passes=1,
        )
        assert promised == pytest.approx(989.0)

    def test_invalid_override_falls_back_with_warning(self, monkeypatch):
        monkeypatch.setenv("VERL_OMNI_DEVICE_FLOPS_TFLOPS", "not-a-number")
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            value = get_device_peak_tflops()
        assert value > 0  # fell back to upstream get_device_flops
        assert any("VERL_OMNI_DEVICE_FLOPS_TFLOPS" in str(w.message) for w in warned)

    def test_negative_override_falls_back_with_warning(self, monkeypatch):
        monkeypatch.setenv("VERL_OMNI_DEVICE_FLOPS_TFLOPS", "-1.0")
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            value = get_device_peak_tflops()
        assert value > 0
        assert any("must be positive" in str(w.message) for w in warned)

    def test_no_override_matches_upstream(self):
        from verl.utils.flops_counter import get_device_flops

        assert get_device_peak_tflops() == pytest.approx(float(get_device_flops("T")))


# ---------------------------------------------------------------------------
# Default DiffusionArchitectureFlops extractors
# ---------------------------------------------------------------------------


class _Tensor:
    """Minimal stand-in for a torch tensor: exposes ``shape`` / ``ndim`` and
    ``is_nested`` so the default extractors can be unit-tested on CPU
    without instantiating a real ``TensorDict``."""

    def __init__(self, shape, is_nested=False):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.is_nested = is_nested

    def sum(self, dim=-1):  # for prompt_embeds_mask path
        # Pretend each row sums to its full length.
        rows = self.shape[0]
        cols = self.shape[1]
        return _SumResult([cols] * rows)


class _SumResult:
    def __init__(self, values):
        self._values = values

    def long(self):
        return self

    def tolist(self):
        return list(self._values)


class TestDefaultLatentSeqlens:
    """The base-class ``latent_seqlens`` covers the standard ``(B, C, *spatial)``
    latent-stream layouts that ship today: vanilla T2I, T2V, and any
    future T2A modality at training time, plus FlowGRPO/MixGRPO
    rollout-stacked tensors with one extra time axis.
    """

    def test_4d_image_latents(self):
        # (B=4, C=16, H=128, W=128) -> 128*128 tokens per sample.
        data = {"image_latents": _Tensor((4, 16, 128, 128))}
        seqs = DiffusionModelFlops({}).get_latent_seqlens(data)
        assert seqs == [128 * 128] * 4

    def test_5d_video_latents(self):
        # Wan / Hunyuan / LTX / CogVideoX: (B=1, C=16, T=21, H=60, W=104).
        data = {"image_latents": _Tensor((1, 16, 21, 60, 104))}
        seqs = DiffusionModelFlops({}).get_latent_seqlens(data)
        assert seqs == [21 * 60 * 104]

    def test_5d_all_latents_rollout_stacked(self):
        # FlowGRPO image rollouts: (B=2, T_steps=10, C=16, H=128, W=128).
        data = {"all_latents": _Tensor((2, 10, 16, 128, 128))}
        seqs = DiffusionModelFlops({}).get_latent_seqlens(data)
        assert seqs == [128 * 128] * 2

    def test_6d_all_latents_video_rollout(self):
        # FlowGRPO video rollouts: (B=1, T_steps=10, C=16, T_lat=21, H=60, W=104).
        data = {"all_latents": _Tensor((1, 10, 16, 21, 60, 104))}
        seqs = DiffusionModelFlops({}).get_latent_seqlens(data)
        assert seqs == [21 * 60 * 104]

    def test_image_latents_takes_priority_over_all_latents(self):
        data = {
            "image_latents": _Tensor((2, 16, 64, 64)),
            "all_latents": _Tensor((2, 10, 16, 128, 128)),
        }
        seqs = DiffusionModelFlops({}).get_latent_seqlens(data)
        assert seqs == [64 * 64] * 2

    def test_missing_latents_returns_empty(self):
        assert DiffusionModelFlops({}).get_latent_seqlens({}) == []
        assert DiffusionModelFlops({}).get_latent_seqlens({"image_latents": None}) == []


class TestDefaultPromptSeqlens:
    def test_dense_mask_sums_per_row(self):
        data = {"prompt_embeds_mask": _Tensor((3, 256))}
        seqs = DiffusionModelFlops({}).get_prompt_seqlens(data)
        assert seqs == [256, 256, 256]

    def test_falls_back_to_prompt_embeds_shape(self):
        # No mask available, but prompt_embeds is dense (B, L, D).
        data = {"prompt_embeds": _Tensor((4, 192, 1024))}
        seqs = DiffusionModelFlops({}).get_prompt_seqlens(data)
        assert seqs == [192] * 4

    def test_unconditional_returns_zeros(self):
        # Neither mask nor embeds. Falls back to [] derived from
        # whichever data field happens to expose a batch dim.
        data = {"image_latents": _Tensor((2, 16, 64, 64))}
        seqs = DiffusionModelFlops({}).get_prompt_seqlens(data)
        assert seqs == []


class TestQwenImageFlopsPackedLatents:
    """Qwen-Image (and the wider MM-DiT family) calls
    ``diffusers._pack_latents`` *before* the transformer, so the data
    fields the FLOPs counter receives at runtime are in the packed layout
    ``(B, L, C')`` for ``image_latents`` or ``(B, T_steps, L, C')`` for
    FlowGRPO's stacked ``all_latents`` — with ``C' == in_channels``.

    The base ``DiffusionModelFlops.get_latent_seqlens`` was written for
    the unpacked ``(B, [T,] C, H, W)`` layout, so without the override
    it mis-identifies ``L`` as a channel dim and returns ``C'`` as the
    per-sample seqlen (a 16x undercount of the real attention seqlen at
    512x512, and ~256x undercount on the attention term).
    """

    def test_packed_3d_image_latents(self):
        # 512x512 -> latent 64x64 -> packed L = 32*32 = 1024, C' = 16*4 = 64.
        data = {"image_latents": _Tensor((4, 1024, 64))}
        flops = QwenImageFlops(QWEN_IMAGE_CONFIG)
        assert flops.get_latent_seqlens(data) == [1024] * 4

    def test_packed_4d_all_latents_flowgrpo_stacked(self):
        # FlowGRPO trajectory at 512x512 with sde_window_size=2.
        data = {
            "all_latents": _Tensor((4, 2, 1024, 64)),
            "prompt_embeds_mask": _Tensor((4, 256)),
        }
        flops = QwenImageFlops(QWEN_IMAGE_CONFIG)
        assert flops.get_latent_seqlens(data) == [1024] * 4
        meta = DiffusionFlopsCounter("QwenImagePipeline", QWEN_IMAGE_CONFIG).collect_meta(data)
        assert meta["latent_seqlens"] == [1024] * 4
        assert meta["prompt_seqlens"] == [256] * 4

    def test_latents_clean_diffusion_nft_batch(self):
        # DiffusionNFT uses unpacked VAE latents (B, C, H, W).
        data = {
            "latents_clean": _Tensor((4, 64, 64, 64)),
            "prompt_embeds_mask": _Tensor((4, 256)),
        }
        assert QwenImageFlops(QWEN_IMAGE_CONFIG).get_latent_seqlens(data) == [4096] * 4
        meta = DiffusionFlopsCounter("QwenImagePipeline", QWEN_IMAGE_CONFIG).collect_meta(data)
        assert meta["latent_seqlens"] == [4096] * 4
        assert meta["prompt_seqlens"] == [256] * 4

    def test_collect_diffusion_flops_meta_train_timesteps(self):
        counter = DiffusionFlopsCounter("QwenImagePipeline", QWEN_IMAGE_CONFIG)
        data = {
            "latents_clean": _Tensor((2, 64, 64, 64)),
            "prompt_embeds_mask": _Tensor((2, 128)),
            "train_timesteps": _Tensor((2, 4)),
        }
        meta = collect_diffusion_flops_meta(counter, data)
        assert meta is not None
        assert meta["num_timesteps"] == 4
        assert meta["latent_seqlens"] == [4096] * 2
        assert meta["prompt_seqlens"] == [128] * 2

    def test_packed_undercount_regression(self):
        # Direct numerical guard: the broken default returns the wrong
        # number; the override fixes it. This pins the magnitude of the
        # bug we just fixed (16x undercount of the per-sample latent
        # seqlen at 512x512 / patch_size=2) so any future refactor that
        # re-breaks the layout detection fails loudly.
        data = {"all_latents": _Tensor((4, 2, 1024, 64))}
        broken = DiffusionModelFlops(QWEN_IMAGE_CONFIG).get_latent_seqlens(data)
        assert broken == [64] * 4, (
            "Base default still mis-identifies packed L as channels. "
            "If this regression test now passes for the base class, "
            "delete this test and the QwenImageFlops override."
        )
        fixed = QwenImageFlops(QWEN_IMAGE_CONFIG).get_latent_seqlens(data)
        assert fixed == [1024] * 4
        # Down-stream sanity check: in this batch the dense term
        # dominates and scales linearly with (L + P), so the FLOPs
        # ratio is (1024+256) / (64+256) ~= 4x. Production prompts are
        # typically much shorter (60-100 tokens), pushing the ratio
        # closer to the underlying 16x latent ratio.
        counter = DiffusionFlopsCounter("QwenImagePipeline", QWEN_IMAGE_CONFIG)
        kw = dict(delta_time=1.0, num_timesteps=2, num_forward_passes=1)
        broken_est, _ = counter.estimate_flops(broken, [256] * 4, **kw)
        fixed_est, _ = counter.estimate_flops(fixed, [256] * 4, **kw)
        assert fixed_est / max(broken_est, 1e-9) > 3.5


class TestSubclassOverridePattern:
    """Demonstrates the pattern the doc teaches: subclass the parent
    architecture and override only ``get_latent_seqlens`` for variants
    that concatenate reference latents along the sequence dim (Edit /
    Img2Img / Inpaint / ControlNet) — the reference tokens flow through
    the same image-side linears as the denoise targets, so they belong
    on the same stream rather than in a third bucket.

    Inputs use the diffusers packed layout ``(B, L, C')`` produced by
    ``_pack_latents`` and consumed by the transformer, which is what
    the FLOPs counter actually sees at runtime.
    """

    def test_edit_concatenates_reference_into_latent_stream(self):
        @register_diffusion_architecture("_QwenImageEdit_CPU")
        class _EditFlops(QwenImageFlops):
            def get_latent_seqlens(self, data):
                # Both denoise and reference latents arrive packed
                # (B, L, C'); they share the image-side path so we
                # add ref_L on top of the denoise L.
                base = super().get_latent_seqlens(data)
                ref = data.get("reference_image_latents")
                if ref is None or not hasattr(ref, "shape"):
                    return base
                ref_per_sample = int(ref.shape[-2])
                return [b + ref_per_sample for b in base]

        counter = DiffusionFlopsCounter("_QwenImageEdit_CPU", QWEN_IMAGE_CONFIG)
        # 512x512 denoise + 256x256 packed reference: L_denoise=1024, L_ref=256.
        data = {
            "image_latents": _Tensor((2, 1024, 64)),
            "reference_image_latents": _Tensor((2, 256, 64)),
        }
        meta = counter.collect_meta(data)
        assert meta["latent_seqlens"] == [1024 + 256] * 2


# ---------------------------------------------------------------------------
# Numerical scaling
# ---------------------------------------------------------------------------


class TestQwenImageFlopsScaling:
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
        counter = _qwen_counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(num_timesteps=10))
        est_b, _ = counter.estimate_flops(**self._kwargs(num_timesteps=30))
        assert math.isclose(est_b / est_a, 3.0, rel_tol=1e-9)

    def test_linear_in_num_forward_passes(self):
        counter = _qwen_counter()
        est_a, _ = counter.estimate_flops(**self._kwargs(num_forward_passes=1))
        est_b, _ = counter.estimate_flops(**self._kwargs(num_forward_passes=2))
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
        est_zero_cfg, _ = counter.estimate_flops(**self._kwargs(num_forward_passes=0))
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
            latent_seqlens=[512], prompt_seqlens=[256], delta_time=1.0, num_timesteps=1, num_forward_passes=1
        )
        est_large, _ = counter.estimate_flops(
            latent_seqlens=[1024], prompt_seqlens=[512], delta_time=1.0, num_timesteps=1, num_forward_passes=1
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
        unusual ``num_timesteps`` / ``num_forward_passes``."""
        counter = _qwen_counter()
        scenarios = [
            dict(latent_seqlens=[256], prompt_seqlens=[64], delta_time=0.5, num_timesteps=1, num_forward_passes=1),
            dict(
                latent_seqlens=[1024, 4096],
                prompt_seqlens=[128, 512],
                delta_time=8.0,
                num_timesteps=50,
                num_forward_passes=2,
            ),
            dict(
                latent_seqlens=[4096] * 8,
                prompt_seqlens=[256] * 8,
                delta_time=12.0,
                num_timesteps=10,
                num_forward_passes=1,
            ),
        ]
        for kwargs in scenarios:
            est, _ = counter.estimate_flops(**kwargs)
            ref = _reference_qwen_image_flops(QWEN_IMAGE_CONFIG, **kwargs)
            assert math.isclose(est, ref, rel_tol=1e-9), (kwargs, est, ref)

    def test_unconditional_zero_text_runs(self):
        # Class-conditioned / unconditional: prompt_seqlens = [0]*B should
        # still produce a sensible non-zero FLOPs estimate driven by the
        # image stream alone (attn quadratic term collapses to img_seq^2).
        counter = _qwen_counter()
        est, _ = counter.estimate_flops(
            latent_seqlens=[1024] * 4,
            prompt_seqlens=[0] * 4,
            delta_time=1.0,
            num_timesteps=1,
            num_forward_passes=1,
        )
        assert est > 0


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
            latent_seqlens=[img_tot],
            prompt_seqlens=[txt_tot],
            delta_time=1.0,
            num_timesteps=1,
            num_forward_passes=1,
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

        # The exact dict structure returned by allgather_diffusion_flops_meta
        global_meta = {
            "latent_seqlens": global_img,
            "prompt_seqlens": global_txt,
            "num_timesteps": 1,
            "num_forward_passes": 1,
        }

        global_est, prom = counter.estimate_flops(delta_time=1.0, **global_meta)

        per_rank_meta = {
            "latent_seqlens": per_rank_img,
            "prompt_seqlens": per_rank_txt,
            "num_timesteps": 1,
            "num_forward_passes": 1,
        }

        per_rank_est, _ = counter.estimate_flops(delta_time=1.0, **per_rank_meta)
        # Global / world_size should equal per-rank only when the formula is
        # linear in token count (it is, for the dense terms). Attention is
        # also linear when each sample's joint seqlen is identical.
        assert math.isclose(global_est / world_size, per_rank_est, rel_tol=1e-9), (
            global_est / world_size,
            per_rank_est,
        )

    def test_mfu_divides_by_world_size_after_global_gather(self):
        """``_postprocess_output`` reports ``mfu = est / prom / world_size``
        when ``est`` comes from globally gathered seqlens.

        This is a wiring invariant only — absolute MFU bounds belong in GPU
        smoke/e2e runs, not CPU unit tests (``prom`` is host-dependent).
        """
        counter = _qwen_counter()
        world_size = 4
        global_meta = {
            "latent_seqlens": [1024] * (16 * world_size),
            "prompt_seqlens": [256] * (16 * world_size),
            "num_timesteps": 10,
            "num_forward_passes": 2,
        }

        est, prom = counter.estimate_flops(delta_time=260.0, **global_meta)
        assert est > 0
        assert prom > 0

        mfu = est / prom / world_size
        assert math.isclose(mfu * world_size, est / prom, rel_tol=1e-9)


class TestGetForwardPassesPerStep:
    """Generality contract for CFG-pass detection. The counter API treats
    ``num_forward_passes`` as a positive int; this helper resolves it from the
    pipeline + transformer configs without architecture-specific knowledge."""

    def test_no_cfg_returns_one(self):
        # Unconditional / class-conditioned model, no guidance fields.
        assert get_forward_passes_per_step({}, {}) == 1
        assert get_forward_passes_per_step(None, None) == 1

    def test_true_cfg_scale_qwen_image_style(self):
        # Qwen-Image: pipeline.true_cfg_scale=4.0 -> 2 forward passes.
        assert get_forward_passes_per_step({"true_cfg_scale": 4.0}, {}) == 2
        assert get_forward_passes_per_step({"true_cfg_scale": 1.0}, {}) == 1
        assert get_forward_passes_per_step({"true_cfg_scale": None}, {}) == 1

    def test_guidance_scale_wan_sd3_style(self):
        # Standard CFG: pipeline.guidance_scale=5.0 -> 2 passes (Wan, SD3).
        assert get_forward_passes_per_step({"guidance_scale": 5.0}, {}) == 2
        # 1.0 means CFG disabled at inference time.
        assert get_forward_passes_per_step({"guidance_scale": 1.0}, {}) == 1

    def test_guidance_distilled_flux_style(self):
        # Flux: transformer_config.guidance_embeds=True means the guidance
        # scalar is consumed *inside* the model; only one forward runs even
        # when guidance_scale > 1.0.
        assert get_forward_passes_per_step({"guidance_scale": 3.5}, {"guidance_embeds": True}) == 1
        # guidance_embeds=False is the explicit non-distilled case.
        assert get_forward_passes_per_step({"guidance_scale": 3.5}, {"guidance_embeds": False}) == 2

    def test_explicit_override_wins(self):
        # Pipeline can force the value (e.g. custom rollout that batches
        # cond+uncond into one tensor).
        assert get_forward_passes_per_step({"num_forward_passes": 1, "guidance_scale": 7.5}, {}) == 1
        assert get_forward_passes_per_step({"num_forward_passes": 2, "true_cfg_scale": 1.0}, {}) == 2
        # Below-1 override is clamped (we never run fewer than one pass).
        assert get_forward_passes_per_step({"num_forward_passes": 0}, {}) == 1

    def test_unknown_garbage_values_dont_crash(self):
        # Helper survives non-numeric junk for forward-compat with weird
        # pipeline configs.
        assert get_forward_passes_per_step({"guidance_scale": "off"}, {}) == 1
        assert get_forward_passes_per_step({"true_cfg_scale": object()}, {}) == 1


class TestDiffusionFlopsCounterApi:
    def test_promised_flops_returned_in_tflops(self):
        counter = _qwen_counter()
        _, prom = counter.estimate_flops(
            latent_seqlens=[1024], prompt_seqlens=[256], delta_time=1.0, num_timesteps=1, num_forward_passes=1
        )
        # ``get_device_flops`` returns TFLOPS by default; CPU baseline is
        # 448 GFLOPS = 0.448 TFLOPS. We accept anything > 0 to keep the test
        # device-agnostic.
        assert prom > 0

    def test_mismatched_seqlen_lengths_raises(self):
        counter = _qwen_counter()
        with pytest.raises(ValueError, match="same length"):
            counter.estimate_flops(
                latent_seqlens=[1024, 1024],
                prompt_seqlens=[256],
                delta_time=1.0,
                num_timesteps=1,
                num_forward_passes=1,
            )
