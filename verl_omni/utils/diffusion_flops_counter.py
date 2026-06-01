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

"""FLOPs / Model FLOPs Utilization (MFU) counter for diffusion DiT models.

This module provides the core logic to estimate FLOPs and compute MFU for
various multi-modal diffusion architectures (T2I, T2V, T2A, Edit, ControlNet, etc.).

Unlike LLMs, diffusion models trained with verl-omni are modality-agnostic and
operate on two primary token streams:
1. **Latent stream** — VAE-encoded tokens (image, video, or audio latents) flowing
   through main-side linears.
2. **Prompt stream** — conditioning tokens (text-encoder or reference-image tokens)
   flowing through cross-attention or text-side linears.

Adding support for a new architecture requires inheriting from :class:`DiffusionModelFlops`,
decorating it with :func:`register_diffusion_architecture`, and implementing `estimate_flops`.

For the complete MFU design details, theoretical FLOPs formulas, stream allocation
conventions, and subclass integration guides, see:
`docs/perf/diffusion_mfu.md` (or run `make html` in the docs folder to view the built page).
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Mapping, Optional, Sequence

from verl.utils.flops_counter import get_device_flops

__all__ = [
    "DiffusionModelFlops",
    "DiffusionFlopsCounter",
    "QwenImageFlops",
    "register_diffusion_architecture",
    "resolve_cfg_passes",
    "resolve_device_peak_tflops",
]


_DEVICE_PEAK_OVERRIDE_ENV = "VERL_OMNI_DEVICE_FLOPS_TFLOPS"


def resolve_device_peak_tflops() -> float:
    """Return the per-device bf16-dense peak in TFLOPS.

    Honors the ``VERL_OMNI_DEVICE_FLOPS_TFLOPS`` env var as a manual
    override and otherwise falls back to upstream
    :func:`verl.utils.flops_counter.get_device_flops`.

    The override is needed on clusters where ``torch.cuda.get_device_name()``
    returns a string verl's built-in table doesn't recognize
    (e.g. relabeled H200 SKUs that report as ``"NVIDIA L20X"`` and would
    otherwise mis-match the ``"L20"`` entry, producing nonsensical
    MFU > 1.0).
    """
    raw = os.environ.get(_DEVICE_PEAK_OVERRIDE_ENV)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
            warnings.warn(
                f"{_DEVICE_PEAK_OVERRIDE_ENV}={raw!r} must be positive; falling back to "
                "verl.utils.flops_counter.get_device_flops().",
                stacklevel=2,
            )
        except ValueError:
            warnings.warn(
                f"{_DEVICE_PEAK_OVERRIDE_ENV}={raw!r} is not a valid float; falling back to "
                "verl.utils.flops_counter.get_device_flops().",
                stacklevel=2,
            )
    return float(get_device_flops("T"))


# ---------------------------------------------------------------------------
# CFG-pass resolution (architecture-independent helper)
# ---------------------------------------------------------------------------


def _gt_one(value: Any) -> bool:
    """Return True if value can be converted to float and is > 1.0."""
    try:
        return value is not None and float(value) > 1.0
    except (TypeError, ValueError):
        return False


def resolve_cfg_passes(
    pipeline_config: Optional[Mapping[str, Any]] = None,
    transformer_config: Optional[Mapping[str, Any]] = None,
) -> int:
    """Resolve the number of model forward passes run per denoising step."""
    pcfg = pipeline_config or {}
    tcfg = transformer_config or {}

    # Explicit override takes priority
    if (explicit := pcfg.get("cfg_passes")) is not None:
        try:
            return max(int(explicit), 1)
        except (TypeError, ValueError):
            pass

    # Guidance-distilled models (e.g. Flux) run only 1 pass
    if tcfg.get("guidance_embeds"):
        return 1

    # True-CFG or standard CFG scales > 1.0 require 2 passes
    if _gt_one(pcfg.get("true_cfg_scale")) or _gt_one(pcfg.get("guidance_scale")):
        return 2

    return 1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


_REGISTRY: dict[str, type[DiffusionModelFlops]] = {}


def register_diffusion_architecture(
    *architectures: str,
):
    """Class decorator that registers a :class:`DiffusionModelFlops`
    subclass under one or more pipeline architecture names.

    ``architectures`` should match ``DiffusionModelConfig.architecture``
    which in turn is read from ``model_index.json``'s ``_class_name``
    field, e.g. ``"QwenImagePipeline"`` or ``"StableDiffusion3Pipeline"``.
    Pass multiple names when the same FLOPs formula is shared by aliases
    (e.g. a custom vllm-omni rollout wrapper).
    """

    def decorator(cls: type[DiffusionModelFlops]) -> type[DiffusionModelFlops]:
        if not isinstance(cls, type) or not issubclass(cls, DiffusionModelFlops):
            raise TypeError(f"@register_diffusion_architecture expects a DiffusionModelFlops subclass, got {cls!r}.")
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


class DiffusionModelFlops:
    """Base class for per-architecture diffusion FLOPs and MFU estimators.

    To support a new model, subclass this class, decorate it with
    `@register_diffusion_architecture("<PipelineClassName>")`, and override
    :meth:`estimate_flops`.

    This framework models multi-modal diffusion (DiT) architectures using two streams:
    1. **Latent Stream:** The noisy/denoised targets flowing through main-side linears
       (e.g., image, video, or audio latents).
    2. **Prompt Stream:** Conditioning inputs flowing through cross-attention or
       text-side linears (e.g., text encoder or reference image embeddings).

    The main estimator calculates bidirectional attention FLOPs proportional to
    `(latent_seqlen + prompt_seqlen)**2` and stream-specific linear FLOPs.

    For detailed theoretical formulas, refer to: `docs/perf/diffusion_mfu.md`
    """

    LATENT_KEYS: Sequence[str] = ("image_latents", "audio_latents", "all_latents")
    PROMPT_KEYS: Sequence[str] = ("prompt_embeds", "prompt_embeds_mask")

    def __init__(self, config: Mapping[str, Any]):
        self.config = _coerce_config(config)

    @property
    def dim(self) -> int:
        """Return the model's hidden dimension (num_heads * head_dim)."""
        num_heads = int(self.config.get("num_attention_heads", 0))
        head_dim = int(self.config.get("attention_head_dim", 0))
        return num_heads * head_dim

    def compute_dense_flops(self, params_per_token: float, total_tokens: float) -> float:
        """Compute dense linear layer FLOPs (factor 6 = 2 FLOPs/MAC * 3 for fwd+bwd)."""
        return 6.0 * params_per_token * total_tokens

    def compute_attention_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        *,
        causal: bool = False,
    ) -> float:
        """Compute FLOPs for the attention dot-products (Q@K^T + attn@V).

        Multiplier factor is 12 (fwd + bwd) for non-causal attention, and 6 for causal.
        """
        num_heads = int(self.config.get("num_attention_heads", 0))
        head_dim = int(self.config.get("attention_head_dim", 0))
        num_layers = int(self.config.get("num_layers", 0))

        seqlen_square_sum = _sum_seqlen_squared(latent_seqlens, prompt_seqlens)
        factor = 6 if causal else 12  # 2 FLOPs/MAC * 2 matmuls * 3 (fwd+bwd), causal is halved
        return factor * num_layers * num_heads * head_dim * seqlen_square_sum

    def estimate_flops(
        self,
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
        raise NotImplementedError("Subclass DiffusionModelFlops and override estimate_flops().")

    def get_latent_seqlens(self, data: Any = None, config: Optional[Mapping[str, Any]] = None) -> list[int]:
        """Extract per-sample latent-stream (VAE-encoded) token counts from a batch.

        The latent stream is the primary tensor denoised by the model. This default
        implementation extracts spatial/spatiotemporal token counts from:
        - `data["image_latents"]` with shape `(B, C, H, W)` or `(B, C, T, H, W)`
        - `data["all_latents"]` with stacked shape `(B, Steps, C, H, W)` or `(B, Steps, C, T, H, W)`

        Override in subclasses when:
        - The model uses non-standard latent keys (e.g. `audio_latents`).
        - The pipeline is an Edit/Img2Img/Inpaint model where reference latents are
          concatenated along the sequence dimension (in which case, the count must
          include both target and reference tokens).
        - Latents are already flattened to sequence dimensions `(B, L, C)`.
        """
        if not isinstance(self, DiffusionModelFlops):
            # Static call backward-compatibility path: get_latent_seqlens(data, config)
            data_param = self
            config_param = data
            temp_inst = DiffusionModelFlops(config_param or {})
            return temp_inst.get_latent_seqlens(data_param)

        latents = None
        stacked = False
        for key in self.LATENT_KEYS:
            latents = _safe_get(data, key)
            if latents is not None:
                stacked = key == "all_latents"
                break

        if latents is None:
            return []
        shape = getattr(latents, "shape", None)
        if shape is None:
            return []
        try:
            ndim = len(shape)
        except TypeError:
            return []

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

    def get_prompt_seqlens(self, data: Any = None, config: Optional[Mapping[str, Any]] = None) -> list[int]:
        """Extract per-sample prompt-stream (conditioning) token counts from a batch.

        This handles text encoder conditioning tokens flowing through text-side/cross-attention
        paths. The default implementation reads:
        - `data["prompt_embeds_mask"]` (resolving nested offsets or dense sequence sums).
        - `data["prompt_embeds"]` (using raw sequence length if unmasked).
        - Defaults to `[0] * B` for class-conditioned or unconditional models.

        Override in subclasses when:
        - Text embeddings are stored under non-standard keys.
        - The pipeline is an Img2Vid model where vision encoder embeddings are
          concatenated to the text embeddings prior to cross-attention.
        """
        if not isinstance(self, DiffusionModelFlops):
            # Static call backward-compatibility path: get_prompt_seqlens(data, config)
            data_param = self
            config_param = data
            temp_inst = DiffusionModelFlops(config_param or {})
            return temp_inst.get_prompt_seqlens(data_param)

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
class QwenImageFlops(DiffusionModelFlops):
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

    def get_latent_seqlens(self, data: Any = None, config: Optional[Mapping[str, Any]] = None) -> list[int]:
        """Override to handle the diffusers ``_pack_latents`` layout.

        Qwen-Image (and the wider MM-DiT family: SD3, Flux, ...) calls
        ``_pack_latents`` *before* the transformer, reshaping
        ``(B, num_channels_latents, H_lat, W_lat)`` into
        ``(B, L, C')`` where ``L = (H_lat//2) * (W_lat//2)`` is the
        sequence length actually seen by attention and
        ``C' = num_channels_latents * 4`` happens to equal the
        transformer's ``in_channels``. FlowGRPO stacks these along a
        new time axis to produce ``(B, T_steps, L, C')``.

        The base default extractor assumes the unpacked
        ``(B, [T,] C, H, W)`` layout and would mis-identify ``L`` as
        a channel dim, returning ``per_sample = C'`` (a 16x undercount
        of the real attention seqlen at 512x512). Detect the packed
        layout by ``shape[-1] == in_channels`` and return ``L``;
        otherwise fall back to the base default for any caller that
        stores raw VAE-encoded latents.
        """
        if not isinstance(self, DiffusionModelFlops):
            # Static call backward-compatibility: QwenImageFlops.get_latent_seqlens(data, config).
            data_param = self
            config_param = data
            return QwenImageFlops(config_param or {}).get_latent_seqlens(data_param)

        latents, _ = _read_latents(data)
        if latents is not None and hasattr(latents, "shape"):
            shape = tuple(int(d) for d in latents.shape)
            in_channels = int(self.config.get("in_channels") or 0)
            if len(shape) >= 3 and in_channels > 0 and shape[-1] == in_channels:
                # Packed: (B, [T_steps,] L, C') with C' == in_channels.
                return [shape[-2]] * shape[0]
        return super().get_latent_seqlens(data, config)

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        cfg_passes: int,
    ) -> float:
        cfg = self.config
        dim = self.dim
        num_layers = int(cfg["num_layers"])
        in_channels = int(cfg["in_channels"])
        joint_attention_dim = int(cfg["joint_attention_dim"])
        patch_size = int(cfg.get("patch_size", 2))
        out_channels = int(cfg.get("out_channels") or in_channels)

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

        # Dense FLOPs per stream.
        img_dense_flops = self.compute_dense_flops(
            num_layers * dense_block_n_per_stream + img_in_n + proj_out_n, img_tot
        )
        txt_dense_flops = self.compute_dense_flops(num_layers * dense_block_n_per_stream + txt_in_n, txt_tot)

        # Modulation linears applied to the per-sample timestep embedding
        # (one token per sample), so they scale with batch_size, not tokens.
        mod_block_n = 12 * dim * dim
        mod_flops = self.compute_dense_flops(num_layers * mod_block_n, batch_size)

        # Joint full-attention FLOPs.
        attn_flops = self.compute_attention_flops(latent_seqlens, prompt_seqlens)

        flops_all_steps = (img_dense_flops + txt_dense_flops + mod_flops + attn_flops) * num_timesteps * cfg_passes
        return flops_all_steps / delta_time / 1e12


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
        self._arch_cls: Optional[type[DiffusionModelFlops]] = _REGISTRY.get(architecture)
        self._arch: Optional[DiffusionModelFlops] = self._arch_cls(self.config) if self._arch_cls is not None else None

        if architecture not in _REGISTRY:
            warnings.warn(
                f"DiffusionFlopsCounter: no FLOPs estimator registered for "
                f"architecture {architecture!r}. MFU will report 0. Register one with "
                f"@register_diffusion_architecture({architecture!r}).",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def architecture_cls(self) -> Optional[type[DiffusionModelFlops]]:
        return self._arch_cls

    def collect_meta(self, data: Any) -> dict[str, list[int]]:
        """Extract per-rank ``latent_seqlens`` and ``prompt_seqlens`` from a
        batch by dispatching to the registered architecture class's
        :meth:`DiffusionArchitectureFlops.get_latent_seqlens` and
        :meth:`DiffusionArchitectureFlops.get_prompt_seqlens`.

        Returns empty lists for unknown architectures so the caller can
        skip downstream work without special-casing.
        """
        if self._arch is None:
            return {"latent_seqlens": [], "prompt_seqlens": []}

        import inspect

        # get_latent_seqlens signature resolution
        latent_sig = inspect.signature(self._arch.get_latent_seqlens)
        latent_params = list(latent_sig.parameters.keys())
        if len(latent_params) >= 2 and latent_params[1] == "config":
            latent_seqlens = self._arch.get_latent_seqlens(data, self.config)
        else:
            latent_seqlens = self._arch.get_latent_seqlens(data)

        # get_prompt_seqlens signature resolution
        prompt_sig = inspect.signature(self._arch.get_prompt_seqlens)
        prompt_params = list(prompt_sig.parameters.keys())
        if len(prompt_params) >= 2 and prompt_params[1] == "config":
            prompt_seqlens = self._arch.get_prompt_seqlens(data, self.config)
        else:
            prompt_seqlens = self._arch.get_prompt_seqlens(data)

        return {
            "latent_seqlens": list(latent_seqlens),
            "prompt_seqlens": list(prompt_seqlens),
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
        promised = resolve_device_peak_tflops()
        if self._arch is None or delta_time <= 0 or num_timesteps <= 0 or cfg_passes <= 0:
            return 0.0, promised

        import inspect

        sig = inspect.signature(self._arch.estimate_flops)
        params = list(sig.parameters.keys())

        if params and params[0] == "config":
            # Legacy static/class method signature: (config, latent_seqlens, prompt_seqlens, delta_time, ...)
            estimated = self._arch.estimate_flops(
                self.config,
                latent_seqlens,
                prompt_seqlens,
                delta_time,
                num_timesteps=num_timesteps,
                cfg_passes=cfg_passes,
            )
        else:
            # Modern instance method signature: (latent_seqlens, prompt_seqlens, delta_time, ...)
            estimated = self._arch.estimate_flops(
                latent_seqlens,
                prompt_seqlens,
                delta_time,
                num_timesteps=num_timesteps,
                cfg_passes=cfg_passes,
            )

        return float(estimated), float(promised)
