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

"""Core diffusion FLOPs / MFU counter: registry, base class, and facade.

Architecture-specific estimators live in sibling modules (e.g.
:mod:`verl_omni.utils.mfu.qwen_image`) and register via
:func:`register_diffusion_architecture`.

See ``docs/perf/diffusion_mfu.md`` for formulas and integration guide.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Mapping, Optional, Sequence

import torch
from verl.utils.device import is_device_available
from verl.utils.flops_counter import get_device_flops

__all__ = [
    "DiffusionModelFlops",
    "DiffusionFlopsCounter",
    "register_diffusion_architecture",
    "get_forward_passes_per_step",
    "get_device_peak_tflops",
    "collect_diffusion_flops_meta",
    "allgather_diffusion_flops_meta",
    "_REGISTRY",
]

_DEVICE_PEAK_OVERRIDE_ENV = "VERL_OMNI_DEVICE_FLOPS_TFLOPS"


# TODO: drop after introducing VERL_DEVICE_FLOPS_TFLOPS in verl
def get_device_peak_tflops() -> float:
    """Return the per-device bf16-dense peak in TFLOPS.

    Honors the ``VERL_OMNI_DEVICE_FLOPS_TFLOPS`` env var as a manual
    override and otherwise falls back to upstream
    :func:`verl.utils.flops_counter.get_device_flops`.
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
    # verl 0.9+ may call torch.cuda.get_device_name() on CPU-only hosts; pass device_name explicitly.
    if not is_device_available():
        return float(get_device_flops("T", device_name="CPU"))
    return float(get_device_flops("T"))


def get_forward_passes_per_step(
    pipeline_config: Optional[Mapping[str, Any]] = None,
    transformer_config: Optional[Mapping[str, Any]] = None,
) -> int:
    """Resolve the number of model forward passes run per denoising step."""
    pcfg = pipeline_config or {}
    tcfg = transformer_config or {}

    if "num_forward_passes" in pcfg:
        val = pcfg["num_forward_passes"]
        if isinstance(val, int | float):
            return max(int(val), 1)

    if tcfg.get("guidance_embeds"):
        return 1

    true_cfg = pcfg.get("true_cfg_scale", 1.0)
    if isinstance(true_cfg, int | float) and true_cfg > 1.0:
        return 2

    guidance = pcfg.get("guidance_scale", 1.0)
    if isinstance(guidance, int | float) and guidance > 1.0:
        return 2

    return 1


_REGISTRY: dict[str, type[DiffusionModelFlops]] = {}


def register_diffusion_architecture(
    *architectures: str,
):
    """Class decorator that registers a :class:`DiffusionModelFlops`
    subclass under one or more pipeline architecture names.
    """

    def decorator(cls: type[DiffusionModelFlops]) -> type[DiffusionModelFlops]:
        if not isinstance(cls, type) or not issubclass(cls, DiffusionModelFlops):
            raise TypeError(f"@register_diffusion_architecture expects a DiffusionModelFlops subclass, got {cls!r}.")
        for name in architectures:
            _REGISTRY[name] = cls
        return cls

    return decorator


def sum_seqlens(seqlens: Optional[Sequence[int]]) -> int:
    if not seqlens:
        return 0
    return int(sum(seqlens))


def sum_seqlen_squared(latent_seqlens: Sequence[int], prompt_seqlens: Sequence[int]) -> int:
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


class DiffusionModelFlops:
    """Base class for per-architecture diffusion FLOPs and MFU estimators."""

    LATENT_KEYS: Sequence[str] = ("image_latents", "audio_latents", "all_latents", "latents_clean")
    PROMPT_KEYS: Sequence[str] = ("prompt_embeds", "prompt_embeds_mask")

    def __init__(self, config: Mapping[str, Any]):
        self.config = config

    @property
    def dim(self) -> int:
        num_heads = int(self.config.get("num_attention_heads", 0))
        head_dim = int(self.config.get("attention_head_dim", 0))
        return num_heads * head_dim

    def compute_dense_flops(self, params_per_token: float, total_tokens: float) -> float:
        return 6.0 * params_per_token * total_tokens

    def compute_attention_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        *,
        causal: bool = False,
    ) -> float:
        num_heads = int(self.config.get("num_attention_heads", 0))
        head_dim = int(self.config.get("attention_head_dim", 0))
        num_layers = int(self.config.get("num_layers", 0))

        seqlen_square_sum = sum_seqlen_squared(latent_seqlens, prompt_seqlens)
        factor = 6 if causal else 12
        return factor * num_layers * num_heads * head_dim * seqlen_square_sum

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        num_forward_passes: int,
    ) -> float:
        raise NotImplementedError("Subclass DiffusionModelFlops and override estimate_flops().")

    def get_latent_seqlens(self, data: Any) -> list[int]:
        latents = None
        stacked = False
        for key in self.LATENT_KEYS:
            latents = data.get(key)
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

        min_rank = 4 if stacked else 3
        if ndim < min_rank:
            return [0] * int(shape[0]) if ndim >= 1 else []

        batch_size = int(shape[0])
        spatial_start = 3 if stacked else 2
        per_sample = 1
        for d in shape[spatial_start:]:
            per_sample *= int(d)
        return [per_sample] * batch_size

    def get_prompt_seqlens(self, data: Any) -> list[int]:
        prompt_embeds_mask = data.get("prompt_embeds_mask")
        prompt_embeds = data.get("prompt_embeds")

        if prompt_embeds_mask is not None and hasattr(prompt_embeds_mask, "is_nested"):
            if prompt_embeds_mask.is_nested:
                return [int(s) for s in prompt_embeds_mask.offsets().diff().tolist()]
            return prompt_embeds_mask.sum(dim=-1).long().tolist()
        if prompt_embeds is not None and hasattr(prompt_embeds, "shape"):
            if getattr(prompt_embeds, "is_nested", False):
                return [int(s) for s in prompt_embeds.offsets().diff().tolist()]
            return [int(prompt_embeds.shape[1])] * int(prompt_embeds.shape[0])
        return []


def read_latents(data: Any) -> tuple[Any, bool]:
    """Return ``(latents_tensor, is_rollout_stacked)``."""
    if (latents := data.get("image_latents")) is not None:
        return latents, False
    if (latents := data.get("latents_clean")) is not None:
        return latents, False
    if (latents := data.get("all_latents")) is not None:
        return latents, True
    return None, False


class DiffusionFlopsCounter:
    """Diffusion-aware counterpart of ``verl.utils.flops_counter.FlopsCounter``."""

    def __init__(self, architecture: Optional[str], transformer_config: Any):
        self.architecture = architecture
        self.config = transformer_config if transformer_config is not None else {}
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
        if self._arch is None:
            return {"latent_seqlens": [], "prompt_seqlens": []}

        return {
            "latent_seqlens": list(self._arch.get_latent_seqlens(data)),
            "prompt_seqlens": list(self._arch.get_prompt_seqlens(data)),
        }

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int = 1,
        num_forward_passes: int = 1,
    ) -> tuple[float, float]:
        promised = get_device_peak_tflops()
        if self._arch is None or delta_time <= 0 or num_timesteps <= 0 or num_forward_passes <= 0:
            return 0.0, promised

        estimated = self._arch.estimate_flops(
            latent_seqlens,
            prompt_seqlens,
            delta_time,
            num_timesteps=num_timesteps,
            num_forward_passes=num_forward_passes,
        )

        return float(estimated), float(promised)


def collect_diffusion_flops_meta(
    flops_counter: DiffusionFlopsCounter | None,
    data: Any,
    *,
    pipeline_config: Any = None,
) -> dict | None:
    """Extract per-call FLOPs metadata for the diffusion counter.

    Returns ``None`` when ``flops_counter`` is not a :class:`DiffusionFlopsCounter`
    so the LLM code path can keep using ``FlopsCounter.estimate_flops(global_token_num, ...)``.
    """
    if not isinstance(flops_counter, DiffusionFlopsCounter):
        return None

    seqlens = flops_counter.collect_meta(data)

    num_timesteps = 1
    for key in ("all_timesteps", "train_timesteps"):
        timesteps = data.get(key, None)
        if timesteps is not None and hasattr(timesteps, "shape") and timesteps.ndim >= 2:
            num_timesteps = int(timesteps.shape[1])
            break

    transformer_config = getattr(flops_counter, "config", None)
    pcfg_view: dict[str, Any] = {}
    for cfg_key in ("num_forward_passes", "true_cfg_scale", "guidance_scale"):
        value = getattr(pipeline_config, cfg_key, None) if pipeline_config is not None else None
        if value is not None:
            pcfg_view[cfg_key] = value
    num_forward_passes = get_forward_passes_per_step(pcfg_view, transformer_config)

    return {
        "latent_seqlens": seqlens["latent_seqlens"],
        "prompt_seqlens": seqlens["prompt_seqlens"],
        "num_timesteps": num_timesteps,
        "num_forward_passes": num_forward_passes,
    }


def allgather_diffusion_flops_meta(meta: dict, dp_group) -> dict:
    """All-gather per-rank latent/prompt seqlens across the DP group.

    ``num_timesteps`` and ``num_forward_passes`` are scalars constant across the
    DP group, so they are kept as-is. Mirrors the ``global_token_num`` gather
    performed in ``TrainingWorker.train_mini_batch`` for the LLM counter.
    """
    if dp_group is None or not torch.distributed.is_initialized():
        return meta

    dp_world = torch.distributed.get_world_size(dp_group)
    if dp_world <= 1:
        return meta

    gathered_latent = [None] * dp_world
    gathered_prompt = [None] * dp_world
    torch.distributed.all_gather_object(gathered_latent, meta["latent_seqlens"], dp_group)
    torch.distributed.all_gather_object(gathered_prompt, meta["prompt_seqlens"], dp_group)
    return {
        "latent_seqlens": [x for xs in gathered_latent for x in xs],
        "prompt_seqlens": [x for xs in gathered_prompt for x in xs],
        "num_timesteps": meta["num_timesteps"],
        "num_forward_passes": meta["num_forward_passes"],
    }
