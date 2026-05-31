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
- attends over a **latent stream** (VAE-encoded image / video / audio
  latents being denoised) and a **prompt stream** (text-encoder tokens,
  and any encoded reference tokens that share the cross-attention
  path), jointly with no causal mask for MM-DiT pipelines.

The counter is **modality-agnostic**: the latent stream can be image
latents (Qwen-Image, SD3, Flux), video latents (Wan2.2, Hunyuan, LTX),
audio latents (AudioLDM-style), or anything else a future diffusion
pipeline puts on the denoiser's primary input. The prompt stream is
likewise whatever the cross-attention or text-side linears consume.
"Latent" here follows diffusers' convention (``latents``,
``image_latents``, ``all_latents``) — the tokens flowing through the
main-side linears of the denoiser, noisy or not.

The two stream totals are the right granularity for FLOPs accounting
because they map directly onto where the per-block linears apply:

- **Latent stream** tokens flow through the image-side (or
  "main-side") linears of each block (``to_q/to_k/to_v``, ``to_out``,
  ``img_mlp`` in the Qwen-Image / SD3 family; ``attn1`` and ``ffn`` in
  Wan).
- **Prompt stream** tokens flow through the text-side / cross-attention
  linears (``add_q/k/v_proj``, ``to_add_out``, ``txt_mlp`` in MM-DiT;
  ``attn2``'s KV projections in Wan-style cross-attention).
- **Joint attention** sees the concatenation of both streams; its cost
  ``∝ (latent_s + prompt_s)²`` is computed *inside* ``estimate_flops``
  from the two totals, so a separate "joint_seqlens" field is not
  needed.

This implies a precise rule for variants that introduce extra latents:

- **Img2Img / Edit / Inpaint / ControlNet** concatenate reference
  latents to the denoise-target latents along the sequence dim
  *before* the transformer block, so the reference latents go through
  the same image-side linears as the denoise targets. They contribute
  to ``latent_seqlens``, not to a third bucket. ``latent_seqlens`` for
  an Edit pipeline is therefore
  ``denoise_target_token_count + reference_token_count`` per sample.
- **Img2Vid** with a vision-encoded reference image (Wan2.2-I2V style)
  concatenates the encoded reference tokens to the text encoder output
  before the cross-attention; those tokens flow through the cross-attn
  KV linears. They contribute to ``prompt_seqlens``.

Adding a new architecture is a single class with one required method.
Subclass :class:`DiffusionArchitectureFlops`, decorate with
:func:`register_diffusion_architecture`, and implement
``estimate_flops``. ``get_latent_seqlens`` and ``get_prompt_seqlens``
have sensible defaults that cover vanilla T2I / T2V / T2A models in
either training or FlowGRPO rollout-stacked layout; override them
only for variants whose data layout genuinely differs (Edit / Img2Vid
/ ControlNet that concatenate extra latents, models with flat
``(B, L, C)`` patched latents, NaViT-style ragged packing, etc.).
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping, Optional, Sequence

from verl.utils.flops_counter import get_device_flops

__all__ = [
    "DiffusionArchitectureFlops",
    "DiffusionFlopsCounter",
    "QwenImageFlops",
    "register_diffusion_architecture",
    "resolve_cfg_passes",
]


# ---------------------------------------------------------------------------
# CFG-pass resolution (architecture-independent helper)
# ---------------------------------------------------------------------------


def resolve_cfg_passes(
    pipeline_config: Optional[Mapping[str, Any]] = None,
    transformer_config: Optional[Mapping[str, Any]] = None,
) -> int:
    """Decide how many denoiser forward passes a training call runs per step.

    Five sources are consulted in priority order so contributors can override
    the heuristic per pipeline without touching this module:

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


_REGISTRY: dict[str, type[DiffusionArchitectureFlops]] = {}


def register_diffusion_architecture(
    *architectures: str,
):
    """Class decorator that registers a :class:`DiffusionArchitectureFlops`
    subclass under one or more pipeline architecture names.

    ``architectures`` should match ``DiffusionModelConfig.architecture``
    which in turn is read from ``model_index.json``'s ``_class_name``
    field, e.g. ``"QwenImagePipeline"`` or ``"StableDiffusion3Pipeline"``.
    Pass multiple names when the same FLOPs formula is shared by aliases
    (e.g. a custom vllm-omni rollout wrapper).
    """

    def decorator(cls: type[DiffusionArchitectureFlops]) -> type[DiffusionArchitectureFlops]:
        if not isinstance(cls, type) or not issubclass(cls, DiffusionArchitectureFlops):
            raise TypeError(
                f"@register_diffusion_architecture expects a DiffusionArchitectureFlops subclass, got {cls!r}."
            )
        for name in architectures:
            _REGISTRY[name] = cls
        return cls

    return decorator


# ---------------------------------------------------------------------------
# Helpers shared by built-in architectures and useful for subclasses
# ---------------------------------------------------------------------------


def _coerce_config(config: Any) -> Mapping[str, Any]:
    """Return a dict-like view of a ``diffusers`` model config.

    ``diffusers.ConfigMixin`` exposes a ``FrozenDict`` accessible via
    ``module.config``, which behaves like a plain ``dict``.  We additionally
    accept raw dicts for testing.
    """
    if isinstance(config, Mapping):
        return config
    if hasattr(config, "to_dict"):
        return config.to_dict()
    if hasattr(config, "items"):
        return dict(config.items())
    raise TypeError(f"DiffusionFlopsCounter expects a dict-like transformer config, got {type(config).__name__}.")


def _sum_seqlens(seqlens: Optional[Sequence[int]]) -> int:
    if not seqlens:
        return 0
    return int(sum(seqlens))


def _sum_seqlen_squared(latent_seqlens: Sequence[int], prompt_seqlens: Sequence[int]) -> int:
    """Per-sample :math:`(img_s + txt_s)^2`, summed across the batch."""
    if not latent_seqlens and not prompt_seqlens:
        return 0
    if not latent_seqlens:
        return sum(int(s) ** 2 for s in prompt_seqlens)
    if not prompt_seqlens:
        return sum(int(s) ** 2 for s in latent_seqlens)
    if len(latent_seqlens) != len(prompt_seqlens):
        raise ValueError(
            f"latent_seqlens and prompt_seqlens must have the same length, "
            f"got {len(latent_seqlens)} and {len(prompt_seqlens)}."
        )
    return sum((int(i) + int(t)) ** 2 for i, t in zip(latent_seqlens, prompt_seqlens, strict=False))


# ---------------------------------------------------------------------------
# Architecture base class
# ---------------------------------------------------------------------------


class DiffusionArchitectureFlops:
    """Base class for per-architecture diffusion FLOPs / MFU support.

    Subclass this and decorate with :func:`register_diffusion_architecture`
    to add FLOPs / MFU reporting for a new diffusion architecture.

    Two concepts are used throughout this API:

    - **Latent stream** — the VAE-encoded tokens the model is
      denoising, flowing through the transformer's "main-side"
      linears. For image generation these are image latent tokens
      (``H*W`` after VAE-encoding and patching); for video they are
      spatiotemporal tokens (``T*H*W``); for audio they are audio
      latent tokens. For Edit / Img2Img / Inpaint / ControlNet
      variants this **also includes any reference latents**
      concatenated along the seq dim before the transformer,
      because they flow through the same linears as the denoise
      targets. The naming follows diffusers (``latents``,
      ``image_latents``, ``all_latents``).
    - **Prompt stream** — tokens flowing through the text-side /
      cross-attention linears. Typically text-encoder tokens
      (after attention masking); for Img2Vid-style models this
      **also includes any encoded reference-image tokens** the
      pipeline concatenates to the text encoder output, because
      they share the cross-attention KV path.

    A useful mental rule: *if two tensors are concatenated along the
    sequence dim before being fed to a linear, they belong to the same
    stream for counting purposes — there is no third bucket.*

    Methods
    -------
    estimate_flops (REQUIRED)
        The FLOPs formula. Receives global (DP-allgathered)
        ``latent_seqlens`` and ``prompt_seqlens`` plus
        ``num_timesteps``, ``cfg_passes``, and the elapsed wall time;
        returns achieved TFLOPS assuming the verl
        ``fwd+bwd = 6 * N * tokens`` convention. Forward-only callers
        divide by 3 in the worker.

    get_latent_seqlens (OPTIONAL)
        Per-sample latent-stream token count, dispatched by
        :meth:`DiffusionFlopsCounter.collect_meta`. The default reads
        ``data["image_latents"]`` (training-time) or
        ``data["all_latents"]`` (FlowGRPO rollout-stacked) and returns
        the product of the spatial dims (``H*W`` for image,
        ``T*H*W`` for video, etc.), replicated to match the batch
        size. See the method docstring for the full Args / Returns
        contract and when to override.

    get_prompt_seqlens (OPTIONAL)
        Per-sample prompt-stream token count, dispatched by
        :meth:`DiffusionFlopsCounter.collect_meta`. The default reads
        ``prompt_embeds_mask`` (nested or dense), falls back to the
        dense ``prompt_embeds.shape[1]``, or returns zeros for
        unconditional / class-conditioned models. See the method
        docstring for the full Args / Returns contract and when to
        override.
    """

    @staticmethod
    def estimate_flops(
        config: Mapping[str, Any],
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        cfg_passes: int,
    ) -> float:
        """Return achieved TFLOPS for one ``train_batch`` / ``infer_batch``
        call. Subclasses must override.
        """
        raise NotImplementedError("Subclass DiffusionArchitectureFlops and override estimate_flops().")

    @staticmethod
    def get_latent_seqlens(data: Any, config: Mapping[str, Any]) -> list[int]:
        """Extract per-sample latent-stream token counts from a batch.

        The latent stream is the VAE-encoded tokens flowing through the
        transformer's main-side linears (``to_q/to_k/to_v``, ``to_out``,
        the FFN; ``img_*`` in Qwen-Image / SD3, ``attn1`` / ``ffn`` in
        Wan). For Img2Img / Edit / Inpaint / ControlNet variants this
        count must include any reference latents concatenated to the
        denoise-target latents along the seq dim before the
        transformer, because those reference tokens flow through the
        same linears.

        This default implementation handles the two standard layouts
        that ship today:

        * Training-time tensors at ``data["image_latents"]`` with shape
          ``(B, C, H, W)`` (image DiT) or ``(B, C, T, H, W)`` (video DiT).
        * FlowGRPO/MixGRPO rollout-stacked tensors at
          ``data["all_latents"]`` with one extra time-step axis, e.g.
          ``(B, T_steps, C, H, W)``.

        Override in your subclass when:

        * your pipeline stores the denoise-target latents under a
          different key (e.g. ``data["audio_latents"]``);
        * extra reference latents are concatenated along the seq dim
          (Edit / Img2Img / Inpaint / ControlNet — add the reference
          token count to each per-sample entry);
        * the latents arrive already flattened to ``(B, L, C)`` (the
          default would treat ``C`` as a spatial dim and over-count);
        * multiple samples are packed into one ragged sequence
          (NaViT-style).

        Args:
            data: The TensorDict / mapping that ``TrainingWorker`` passes
                through the training or inference path. Architecture-
                specific keys (``image_latents``, ``all_latents``,
                ``reference_image_latents``, ...) are looked up here.
            config: The diffusers transformer config (parsed contents of
                ``<model>/transformer/config.json``). Available in case
                the extractor needs fields like ``patch_size`` or
                ``in_channels`` to interpret a tensor shape; the default
                does not use it.

        Returns:
            A list of length ``B``, one int per sample in the batch,
            giving that sample's latent-stream token count after any
            patchifying / VAE downsampling already reflected in the
            tensor shape. Returns ``[]`` (or ``[0] * B``) when no
            latent tensor is present or its rank is below the minimum
            expected for the standard layout, so the counter degrades
            to ``mfu=0`` rather than crashing — matching the upstream
            LLM counter's graceful-degradation contract.
        """
        del config  # Default doesn't need transformer config.
        latents, stacked = _read_latents(data)
        if latents is None:
            return []
        shape = getattr(latents, "shape", None)
        if shape is None:
            return []
        try:
            ndim = len(shape)
        except TypeError:
            return 0  # type: ignore[return-value]

        # Minimum rank for the default: (B, C, *spatial); +1 if stacked.
        min_rank = 4 if stacked else 3
        if ndim < min_rank:
            return [0] * int(shape[0]) if ndim >= 1 else []

        batch_size = int(shape[0])
        # Drop B (and T_steps if stacked) plus C; multiply the rest.
        spatial_start = 3 if stacked else 2
        per_sample = 1
        for d in shape[spatial_start:]:
            per_sample *= int(d)
        return [per_sample] * batch_size

    @staticmethod
    def get_prompt_seqlens(data: Any, config: Mapping[str, Any]) -> list[int]:
        """Extract per-sample prompt-stream token counts from a batch.

        The prompt stream is the tokens flowing through the text-side
        / cross-attention linears (``add_q/k/v_proj``, ``to_add_out``,
        ``txt_mlp`` in MM-DiT; ``attn2``'s KV projections in
        Wan-style cross-attention). For Img2Vid variants this also
        includes any encoded reference-image tokens the pipeline
        concatenates to the text-encoder output, because those tokens
        share the cross-attention KV path with the text tokens.

        This default implementation reads the standard diffusers
        encoder fields, in order of precedence:

        * ``data["prompt_embeds_mask"]`` — nested
          (``offsets().diff()``) or dense (``mask.sum(-1)``);
        * ``data["prompt_embeds"].shape[1]`` — the padded dense
          length, used when no mask is available;
        * ``[0] * B`` — for unconditional or class-conditioned
          models with no text-encoder stream at all.

        Override in your subclass when:

        * your pipeline stores text embeddings under different keys;
        * masking uses a non-standard convention (e.g. a 4-D mask
          shared across attention heads);
        * reference-image or reference-audio tokens are concatenated
          to the prompt stream before cross-attention (Img2Vid,
          audio-to-audio — add the encoded reference token count to
          each per-sample entry).

        Args:
            data: The TensorDict / mapping that ``TrainingWorker`` passes
                through the training or inference path. Standard keys
                (``prompt_embeds``, ``prompt_embeds_mask``) and any
                pipeline-specific encoded-reference keys are looked up
                here.
            config: The diffusers transformer config; available in case
                the extractor needs fields like ``joint_attention_dim``
                to interpret ragged shapes. The default does not use it.

        Returns:
            A list of length ``B``, one int per sample in the batch,
            giving that sample's prompt-stream token count *after*
            attention masking (i.e. the count actually presented to
            the cross-attention / joint-attention linears, not the
            padded length). Returns ``[0] * B`` for unconditional or
            class-conditioned pipelines that carry no prompt stream.
        """
        del config
        prompt_embeds_mask = _safe_get(data, "prompt_embeds_mask")
        prompt_embeds = _safe_get(data, "prompt_embeds")
        batch_size = _batch_size(data)

        if prompt_embeds_mask is not None and hasattr(prompt_embeds_mask, "is_nested"):
            if prompt_embeds_mask.is_nested:
                return [int(s) for s in prompt_embeds_mask.offsets().diff().tolist()]
            return prompt_embeds_mask.sum(dim=-1).long().tolist()
        if prompt_embeds is not None and hasattr(prompt_embeds, "shape"):
            if getattr(prompt_embeds, "is_nested", False):
                return [int(s) for s in prompt_embeds.offsets().diff().tolist()]
            return [int(prompt_embeds.shape[1])] * batch_size
        return [0] * batch_size


def _safe_get(data: Any, key: str) -> Any:
    """``data.get(key)`` that survives both TensorDict and plain dict inputs."""
    if data is None:
        return None
    getter = getattr(data, "get", None)
    if callable(getter):
        try:
            return getter(key, None)
        except TypeError:
            try:
                return getter(key)
            except Exception:
                return None
    try:
        return data[key]
    except (KeyError, IndexError, TypeError):
        return None


def _batch_size(data: Any) -> int:
    """Best-effort batch size for a TensorDict-like or dict-like input."""
    shape = getattr(data, "shape", None)
    if shape is not None and hasattr(data, "batch_size") and getattr(data, "batch_size", None):
        try:
            return int(shape[0])
        except (TypeError, IndexError):
            pass
    # Fall back: peek at any tensor-valued entry.
    for key in ("image_latents", "all_latents", "prompt_embeds", "prompt_embeds_mask"):
        value = _safe_get(data, key)
        value_shape = getattr(value, "shape", None)
        if value_shape is not None and len(value_shape) >= 1:
            return int(value_shape[0])
    return 0


def _read_latents(data: Any) -> tuple[Any, bool]:
    """Return ``(latents_tensor, is_rollout_stacked)``.

    FlowGRPO/MixGRPO store the entire denoising trajectory in
    ``all_latents`` with one extra time axis; DPO and one-shot training
    paths store the single-step latents in ``image_latents``.
    """
    latents = _safe_get(data, "image_latents")
    if latents is not None:
        return latents, False
    latents = _safe_get(data, "all_latents")
    if latents is not None:
        return latents, True
    return None, False


# ---------------------------------------------------------------------------
# Built-in architectures
# ---------------------------------------------------------------------------


@register_diffusion_architecture(
    "QwenImagePipeline",
    # Aliases sometimes seen in custom forks / vllm-omni shims.
    "QwenImagePipelineWithLogProb",
)
class QwenImageFlops(DiffusionArchitectureFlops):
    """FLOPs estimator for ``QwenImageTransformer2DModel``.

    Qwen-Image is a dual-stream DiT: image-latent tokens flow through the
    image-side linears (``to_q/to_k/to_v``, ``to_out``, ``img_mlp``) and
    text-encoder tokens flow through the text-side linears
    (``add_q_proj/add_k_proj/add_v_proj``, ``to_add_out``, ``txt_mlp``).
    The two streams meet only in the joint full attention. The formula
    therefore attributes per-stream linear FLOPs to each stream's own
    token count, and uses the upstream full-attention factor 12 (vs the
    causal factor 6 used in ``_estimate_qwen2_flops``), matching the
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

    @staticmethod
    def estimate_flops(
        config: Mapping[str, Any],
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        cfg_passes: int,
    ) -> float:
        cfg = _coerce_config(config)

        num_attention_heads = int(cfg["num_attention_heads"])
        attention_head_dim = int(cfg["attention_head_dim"])
        num_layers = int(cfg["num_layers"])
        in_channels = int(cfg["in_channels"])
        joint_attention_dim = int(cfg["joint_attention_dim"])
        patch_size = int(cfg.get("patch_size", 2))
        out_channels = int(cfg.get("out_channels") or in_channels)

        dim = num_attention_heads * attention_head_dim

        # Per-stream, per-layer token-scaling params (see class docstring).
        dense_block_n_per_stream = (3 + 1 + 8) * dim * dim  # 12 * dim**2

        # Patch / encoder / proj-out linears: each applies to one stream only.
        img_in_n = in_channels * dim
        txt_in_n = joint_attention_dim * dim
        proj_out_n = patch_size * patch_size * out_channels * dim

        img_tot = _sum_seqlens(latent_seqlens)
        txt_tot = _sum_seqlens(prompt_seqlens)
        batch_size = max(
            len(latent_seqlens) if latent_seqlens else 0,
            len(prompt_seqlens) if prompt_seqlens else 0,
        )

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
        seqlen_square_sum = _sum_seqlen_squared(latent_seqlens, prompt_seqlens)
        attn_flops = 12 * num_layers * num_attention_heads * attention_head_dim * seqlen_square_sum

        flops_all_steps = (img_dense_flops + txt_dense_flops + mod_flops + attn_flops) * num_timesteps * cfg_passes
        return flops_all_steps * (1.0 / delta_time) / 1e12


# ---------------------------------------------------------------------------
# Public counter
# ---------------------------------------------------------------------------


class DiffusionFlopsCounter:
    """Diffusion-aware counterpart of ``verl.utils.flops_counter.FlopsCounter``.

    Usage::

        counter = DiffusionFlopsCounter(architecture, transformer_config)

        # Per-rank: extract local seqlens via the registered architecture.
        local_meta = counter.collect_meta(data)            # dict of lists

        # Worker: all-gather seqlens across the DP group.
        global_meta = allgather(local_meta)

        # Compute MFU.
        estimated, promised = counter.estimate_flops(
            delta_time=delta_time,
            num_timesteps=num_timesteps,
            cfg_passes=cfg_passes,
            **global_meta,
        )
        mfu = estimated / promised / world_size
        if forward_only:
            mfu /= 3.0
    """

    def __init__(self, architecture: Optional[str], transformer_config: Any):
        self.architecture = architecture
        self.config = _coerce_config(transformer_config) if transformer_config is not None else {}
        self._arch_cls: Optional[type[DiffusionArchitectureFlops]] = _REGISTRY.get(architecture)

        if architecture not in _REGISTRY:
            warnings.warn(
                f"DiffusionFlopsCounter: no FLOPs estimator registered for "
                f"architecture {architecture!r}. MFU will report 0. Register one with "
                f"@register_diffusion_architecture({architecture!r}).",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def architecture_cls(self) -> Optional[type[DiffusionArchitectureFlops]]:
        return self._arch_cls

    def collect_meta(self, data: Any) -> dict[str, list[int]]:
        """Extract per-rank ``latent_seqlens`` and ``prompt_seqlens`` from a
        batch by dispatching to the registered architecture class's
        :meth:`DiffusionArchitectureFlops.get_latent_seqlens` and
        :meth:`DiffusionArchitectureFlops.get_prompt_seqlens`.

        Returns empty lists for unknown architectures so the caller can
        skip downstream work without special-casing.
        """
        if self._arch_cls is None:
            return {"latent_seqlens": [], "prompt_seqlens": []}
        return {
            "latent_seqlens": list(self._arch_cls.get_latent_seqlens(data, self.config)),
            "prompt_seqlens": list(self._arch_cls.get_prompt_seqlens(data, self.config)),
        }

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int = 1,
        cfg_passes: int = 1,
    ) -> tuple[float, float]:
        """Return ``(estimated_tflops, promised_tflops)`` for one call.

        Args:
            latent_seqlens: per-sample image-latent token counts.
            prompt_seqlens: per-sample text-encoder token counts after the
                attention mask (the same length as ``latent_seqlens``).
            delta_time: wall-clock seconds for the entire
                ``train_batch`` / ``infer_batch`` call (covering all
                denoising steps and both CFG passes).
            num_timesteps: number of denoising steps executed per call.
                ``1`` for diffusion DPO; ``data["all_timesteps"].shape[1]``
                for FlowGRPO-family algorithms.
            cfg_passes: ``2`` when True-CFG is enabled
                (``true_cfg_scale > 1.0``), otherwise ``1``.
        """
        promised = get_device_flops()
        if self._arch_cls is None or delta_time <= 0 or num_timesteps <= 0 or cfg_passes <= 0:
            return 0.0, promised

        estimated = self._arch_cls.estimate_flops(
            self.config,
            latent_seqlens,
            prompt_seqlens,
            delta_time,
            num_timesteps=num_timesteps,
            cfg_passes=cfg_passes,
        )
        return float(estimated), float(promised)
