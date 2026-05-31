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

"""FLOPs / MFU counter for diffusion DiT models trained with verl-omni.

The upstream ``verl.utils.flops_counter.FlopsCounter`` only understands
HuggingFace ``transformers`` LLM configs.  Diffusion training in verl-omni
uses ``diffusers`` transformer configs whose schema is different
(``num_layers``, ``attention_head_dim``, ``num_attention_heads``,
``joint_attention_dim`` etc.) and additionally:

- runs ``num_timesteps`` denoising steps per ``train_batch`` /
  ``infer_batch`` call,
- optionally runs two forward passes per step for True-CFG, and
- attends jointly over the image-latent stream and the text-encoder stream
  with no causal mask.

This module provides a small, architecture-aware counter that mirrors the
shape of the upstream LLM counter so it can be slotted into the same
``TrainingWorker._postprocess_output`` path.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Mapping, Optional, Sequence

from verl.utils.flops_counter import get_device_flops

__all__ = [
    "DiffusionFlopsCounter",
    "estimate_qwen_image_flops",
    "image_seqlens_from_latent_shape",
    "register_diffusion_flops_estimator",
    "resolve_cfg_passes",
]


# ---------------------------------------------------------------------------
# Architecture-agnostic meta extraction helpers
# ---------------------------------------------------------------------------
#
# These are intentionally pure-Python (no torch, no Ray, no diffusers) so they
# can be unit-tested without a worker fixture.  ``TrainingWorker.
# _collect_diffusion_flops_meta`` is a thin wrapper that calls them.


def image_seqlens_from_latent_shape(
    shape: Sequence[int],
    *,
    all_latents_layout: bool = False,
) -> int:
    """Return the per-sample image-latent token count for a latent tensor.

    Supports the common diffusers layouts that flow through ``train_batch``:

    - ``(B, L, C)`` (3D)            -> seqlen ``L``  (already-patched tokens).
    - ``(B, C, H, W)`` (4D)         -> seqlen ``H*W`` (image latents).
    - ``(B, T_steps, L, C)`` (4D, ``all_latents_layout=True``) -> ``L``
      (FlowGRPO/MixGRPO image rollouts).
    - ``(B, C, T, H, W)`` (5D)      -> seqlen ``T*H*W`` (video latents:
      Wan, HunyuanVideo, LTX, CogVideoX).
    - ``(B, T_steps, C, T, H, W)`` (6D, ``all_latents_layout=True``)
      -> ``T*H*W`` (FlowGRPO video rollouts).

    Returns ``0`` for shapes the caller cannot interpret, matching the
    "no info -> contribute 0 FLOPs" graceful-degradation contract of the
    upstream LLM counter.
    """
    if shape is None:
        return 0
    try:
        ndim = len(shape)
    except TypeError:
        return 0

    if all_latents_layout:
        # (B, T_steps, ...) — first dim is batch, second is denoising step
        # count, rest is the per-step latent shape.
        if ndim == 4:
            # (B, T_steps, L, C)
            return int(shape[2])
        if ndim == 5:
            # (B, T_steps, C, H, W)
            return int(shape[3]) * int(shape[4])
        if ndim == 6:
            # (B, T_steps, C, T_lat, H, W)
            return int(shape[3]) * int(shape[4]) * int(shape[5])
        return 0

    if ndim == 3:
        # (B, L, C)
        return int(shape[1])
    if ndim == 4:
        # (B, C, H, W)
        return int(shape[2]) * int(shape[3])
    if ndim == 5:
        # (B, C, T, H, W) — video latents (Wan, Hunyuan, LTX, CogVideoX).
        return int(shape[2]) * int(shape[3]) * int(shape[4])
    return 0


def resolve_cfg_passes(
    pipeline_config: Optional[Mapping[str, Any]] = None,
    transformer_config: Optional[Mapping[str, Any]] = None,
) -> int:
    """Decide how many denoiser forward passes a training call runs per step.

    Three sources are consulted in priority order so contributors can
    override the heuristic per pipeline without touching this module:

    1. Explicit ``pipeline_config.cfg_passes`` (int).  Use this when neither
       ``true_cfg_scale`` nor ``guidance_scale`` matches the pipeline's
       runtime behaviour (e.g. a custom rollout adapter that always batches
       cond + uncond into one tensor).
    2. ``transformer_config.guidance_embeds == True`` -> 1 pass.  Models
       like Flux are guidance-distilled: the guidance scalar is consumed
       inside the model and only one forward is run regardless of
       ``guidance_scale``.
    3. ``true_cfg_scale > 1.0`` -> 2 passes.  Qwen-Image-style "true CFG":
       the pipeline runs separate cond / uncond forwards.
    4. ``guidance_scale > 1.0`` -> 2 passes.  Standard CFG (Wan2.2, SD3,
       most non-distilled DiT pipelines).
    5. Otherwise (unconditional / class-conditioned / explicit no-CFG)
       -> 1 pass.

    Both arguments are accepted as ``None`` so this helper survives the
    bring-up window before ``architecture`` is wired into a given pipeline.
    """
    pcfg: Mapping[str, Any] = pipeline_config or {}
    tcfg: Mapping[str, Any] = transformer_config or {}

    explicit = pcfg.get("cfg_passes", None) if isinstance(pcfg, Mapping) else None
    if explicit is not None:
        passes = int(explicit)
        return max(passes, 1)

    if bool(tcfg.get("guidance_embeds", False)):
        return 1

    def _gt_one(value: Any) -> bool:
        try:
            return value is not None and float(value) > 1.0
        except (TypeError, ValueError):
            return False

    if _gt_one(pcfg.get("true_cfg_scale")):
        return 2
    if _gt_one(pcfg.get("guidance_scale")):
        return 2
    return 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# A FLOPs estimator returns the achieved TFLOPS (= FLOPs / 1e12 / delta_time)
# for a single ``train_batch`` / ``infer_batch`` call assuming the standard
# verl ``fwd+bwd = 6 * N * tokens`` convention.  The caller divides by 3 when
# only a forward pass was executed.
DiffusionFlopsEstimator = Callable[..., float]


_REGISTRY: dict[str, DiffusionFlopsEstimator] = {}


def register_diffusion_flops_estimator(
    *architectures: str,
) -> Callable[[DiffusionFlopsEstimator], DiffusionFlopsEstimator]:
    """Register a FLOPs estimator for one or more pipeline architectures.

    ``architectures`` should match ``DiffusionModelConfig.architecture`` which
    in turn is read from ``model_index.json``'s ``_class_name`` field, e.g.
    ``"QwenImagePipeline"`` or ``"StableDiffusion3Pipeline"``.
    """

    def decorator(func: DiffusionFlopsEstimator) -> DiffusionFlopsEstimator:
        for name in architectures:
            _REGISTRY[name] = func
        return func

    return decorator


# ---------------------------------------------------------------------------
# Per-architecture estimators
# ---------------------------------------------------------------------------


def _coerce_config(config: Any) -> Mapping[str, Any]:
    """Return a dict-like view of a ``diffusers`` model config.

    ``diffusers.ConfigMixin`` exposes a ``FrozenDict`` accessible via
    ``module.config``, which behaves like a plain ``dict``.  We additionally
    accept raw dicts for testing.
    """
    if isinstance(config, Mapping):
        return config
    # diffusers FrozenDict / OmegaConf DictConfig and similar containers
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "items"):
        return dict(config.items())
    raise TypeError(f"DiffusionFlopsCounter expects a dict-like transformer config, got {type(config).__name__}.")


def _sum_seqlens(seqlens: Optional[Sequence[int]]) -> int:
    if not seqlens:
        return 0
    return int(sum(seqlens))


def _sum_seqlen_squared(image_seqlens: Sequence[int], prompt_seqlens: Sequence[int]) -> int:
    """Per-sample :math:`(img_s + txt_s)^2`, summed across the batch."""
    if not image_seqlens and not prompt_seqlens:
        return 0
    # Pad the shorter of the two with zeros so the user can pass one of them
    # as ``None`` / empty (e.g. for SD3 there is always text; for tests we
    # sometimes use image-only batches).
    if not image_seqlens:
        return sum(int(s) ** 2 for s in prompt_seqlens)
    if not prompt_seqlens:
        return sum(int(s) ** 2 for s in image_seqlens)
    if len(image_seqlens) != len(prompt_seqlens):
        raise ValueError(
            f"image_seqlens and prompt_seqlens must have the same length, "
            f"got {len(image_seqlens)} and {len(prompt_seqlens)}."
        )
    return sum((int(i) + int(t)) ** 2 for i, t in zip(image_seqlens, prompt_seqlens, strict=False))


@register_diffusion_flops_estimator(
    "QwenImagePipeline",
    # Aliases sometimes seen in custom forks / vllm-omni shims.
    "QwenImagePipelineWithLogProb",
)
def estimate_qwen_image_flops(
    config: Mapping[str, Any],
    image_seqlens: Sequence[int],
    prompt_seqlens: Sequence[int],
    delta_time: float,
    *,
    num_timesteps: int,
    cfg_passes: int,
) -> float:
    """Estimate per-call TFLOPS for ``QwenImageTransformer2DModel``.

    Returns the achieved compute in **TFLOPS** assuming the standard
    ``fwd+bwd = 6 * N * tokens`` convention used by upstream verl. For
    forward-only calls the caller divides by 3 (see
    ``TrainingWorker._postprocess_output``).

    Qwen-Image is a double-stream DiT: image-latent tokens flow through the
    image-side linears (``to_q/to_k/to_v``, ``to_out``, ``img_mlp``) and
    text-encoder tokens flow through the text-side linears
    (``add_q_proj/add_k_proj/add_v_proj``, ``to_add_out``, ``txt_mlp``).
    The two streams meet only in the joint full attention. The formula
    below therefore attributes per-stream linear FLOPs to each stream's
    own token count, and uses the upstream full-attention factor 12 (vs
    the causal factor 6 used in ``_estimate_qwen2_flops``), matching the
    convention in ``_estimate_qwen3_vit_flop``.

    Per-block token-scaling params per stream:

      - QKV projections (3 x dim -> dim) :  3 * dim**2
      - attention output (dim -> dim)    :      dim**2
      - GELU FeedForward (dim -> 4*dim   :  8 * dim**2
        -> dim, no GLU)
      => 12 * dim**2 per stream per layer

    Modulation linears (``img_mod``, ``txt_mod``) act on the per-sample
    timestep embedding, so they are counted as ``batch_size * 12*dim**2``
    rather than per-token. Their contribution is typically << 1% of the
    block FLOPs and is included for completeness.
    """
    cfg = _coerce_config(config)

    num_attention_heads = int(cfg["num_attention_heads"])
    attention_head_dim = int(cfg["attention_head_dim"])
    num_layers = int(cfg["num_layers"])
    in_channels = int(cfg["in_channels"])
    joint_attention_dim = int(cfg["joint_attention_dim"])
    patch_size = int(cfg.get("patch_size", 2))
    out_channels = int(cfg.get("out_channels") or in_channels)

    dim = num_attention_heads * attention_head_dim

    # Per-stream, per-layer token-scaling params (see docstring).
    dense_block_n_per_stream = (3 + 1 + 8) * dim * dim  # 12 * dim**2

    # Patch / encoder / proj-out linears: each applies to one stream only.
    img_in_n = in_channels * dim
    txt_in_n = joint_attention_dim * dim
    proj_out_n = patch_size * patch_size * out_channels * dim

    img_tot = _sum_seqlens(image_seqlens)
    txt_tot = _sum_seqlens(prompt_seqlens)
    batch_size = max(len(image_seqlens) if image_seqlens else 0, len(prompt_seqlens) if prompt_seqlens else 0)

    # Dense FLOPs per stream; factor 6 = 2 FLOPs/MAC * 3 (fwd+bwd).
    img_dense_flops = 6 * (num_layers * dense_block_n_per_stream + img_in_n + proj_out_n) * img_tot
    txt_dense_flops = 6 * (num_layers * dense_block_n_per_stream + txt_in_n) * txt_tot

    # Modulation linears applied to the per-sample timestep embedding
    # (one token per sample), so they scale with batch_size, not tokens.
    mod_block_n = 12 * dim * dim
    mod_flops = 6 * num_layers * mod_block_n * batch_size

    # Joint full-attention FLOPs: non-causal Q@K^T + attn@V across the
    # combined img+txt stream; factor 12 = 2 FLOPs/MAC * 2 matmuls *
    # 3 (fwd+bwd), matching upstream's _estimate_qwen3_vit_flop.
    seqlen_square_sum = _sum_seqlen_squared(image_seqlens, prompt_seqlens)
    attn_flops = 12 * num_layers * num_attention_heads * attention_head_dim * seqlen_square_sum

    flops_all_steps = (img_dense_flops + txt_dense_flops + mod_flops + attn_flops) * num_timesteps * cfg_passes
    flops_achieved = flops_all_steps * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_unknown_diffusion_flops(*_args, **_kwargs) -> float:
    return 0.0


# ---------------------------------------------------------------------------
# Public counter
# ---------------------------------------------------------------------------


class DiffusionFlopsCounter:
    """Diffusion-aware counterpart of ``verl.utils.flops_counter.FlopsCounter``.

    Usage::

        counter = DiffusionFlopsCounter(architecture, transformer_config)
        estimated, promised = counter.estimate_flops(
            image_seqlens=[1024, 1024],
            prompt_seqlens=[256, 192],
            delta_time=4.2,
            num_timesteps=10,
            cfg_passes=2,
        )
        mfu = estimated / promised / world_size
        if forward_only:
            mfu /= 3.0
    """

    def __init__(self, architecture: Optional[str], transformer_config: Any):
        self.architecture = architecture
        self.config = _coerce_config(transformer_config) if transformer_config is not None else {}

        if architecture not in _REGISTRY:
            warnings.warn(
                f"DiffusionFlopsCounter: no FLOPs estimator registered for "
                f"architecture {architecture!r}. MFU will report 0. Register one with "
                f"@register_diffusion_flops_estimator({architecture!r}).",
                RuntimeWarning,
                stacklevel=2,
            )

    def estimate_flops(
        self,
        image_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int = 1,
        cfg_passes: int = 1,
    ) -> tuple[float, float]:
        """Return ``(estimated_tflops, promised_tflops)`` for one call.

        Args:
            image_seqlens: per-sample image-latent token counts.
            prompt_seqlens: per-sample text-encoder token counts after the
                attention mask (the same length as ``image_seqlens``).
            delta_time: wall-clock seconds for the entire
                ``train_batch`` / ``infer_batch`` call (covering all
                denoising steps and both CFG passes).
            num_timesteps: number of denoising steps executed per call.
                ``1`` for diffusion DPO; ``data["all_timesteps"].shape[1]``
                for FlowGRPO-family algorithms.
            cfg_passes: ``2`` when True-CFG is enabled
                (``true_cfg_scale > 1.0``), otherwise ``1``.
        """
        if delta_time <= 0:
            return 0.0, get_device_flops()
        if num_timesteps <= 0 or cfg_passes <= 0:
            return 0.0, get_device_flops()

        estimator = _REGISTRY.get(self.architecture, _estimate_unknown_diffusion_flops)
        estimated = estimator(
            self.config,
            image_seqlens,
            prompt_seqlens,
            delta_time,
            num_timesteps=num_timesteps,
            cfg_passes=cfg_passes,
        )
        promised = get_device_flops()
        return float(estimated), float(promised)
