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

from types import MethodType

import torch

# TODO: Remove this shim, its call sites and tests once the required VeOmni includes
# ByteDance-Seed/VeOmni#1214 and #1236. VeOmni 0.1.12 rejects Hub attention names and Qwen-Image
# ignores attn_implementation. Only Hub varlen backends handle Qwen-Image's text padding correctly,
# so this table is the allowlist except for H3's instance-local attention bridge.
_DIFFUSERS_ATTENTION_BACKENDS = {
    "eager": "native",
    "flash_attention_2_hub": "flash_varlen_hub",
    "flash_attention_3_hub": "_flash_3_varlen_hub",
}


# TODO: Remove this bridge and its call site once the required VeOmni includes
# ByteDance-Seed/VeOmni#1239 and this integration uses its native attention setup.
_MINIMAX_H3_ATTENTION_BACKENDS = {
    "eager": "native",
    "flash_attention_2": "flash",
    "flash_attention_3": "_flash_3",
    "flash_attention_2_hub": "flash_hub",
    "flash_attention_3_hub": "_flash_3_hub",
}


def _veomni_attn_implementation(attn_implementation: str) -> str:
    """Build with eager when the installed VeOmni cannot parse Hub attention names."""
    from veomni.arguments import OpsImplementationConfig

    if attn_implementation.endswith("_hub") and not hasattr(OpsImplementationConfig, "normalize_hub_attention_backend"):
        return "eager"  # Qwen-Image and H3 get the Hub kernel in _apply_attention_backend
    return attn_implementation


def _apply_attention_backend(model: torch.nn.Module, attn_implementation: str) -> None:
    """Apply model-specific attention compatibility while retaining backend validation."""
    if getattr(getattr(model, "config", None), "model_type", None) == "MiniMaxH3DiTModel":
        _apply_minimax_h3_attention_backend(model, attn_implementation)
        return
    from veomni.models.diffusers.qwen_image.qwen_image_transformer.modeling_qwen_image_transformer import (
        QwenImageSPAttnProcessor,
    )

    backend = _DIFFUSERS_ATTENTION_BACKENDS.get(attn_implementation)
    if backend is None:
        raise ValueError(
            f"veomni_config.attn_implementation={attn_implementation!r} is not supported by the VeOmni "
            f"diffusion engine; use one of {sorted(_DIFFUSERS_ATTENTION_BACKENDS)}. Local flash_attention_2/3 "
            "map to diffusers varlen kernels that mishandle Qwen-Image's text padding."
        )
    if not any(isinstance(getattr(m, "processor", None), QwenImageSPAttnProcessor) for m in model.modules()):
        # e.g. Wan / LTX select attention inside VeOmni.
        if _veomni_attn_implementation(attn_implementation) != attn_implementation:
            raise ValueError(
                f"veomni_config.attn_implementation={attn_implementation!r} requires a VeOmni release "
                "with Hub attention support; verl-omni only supplies it for Qwen-Image and MiniMax H3."
            )
        return
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    # set_attention_backend also switches diffusers' process-wide default; keep it for other models.
    active_backend = _AttentionBackendRegistry._active_backend
    model.set_attention_backend(backend)
    _AttentionBackendRegistry.set_active_backend(active_backend)


def _minimax_h3_attention_forward(self, x, *, rope_cos, rope_sin, cu_seqlens, max_seqlen=None, use_ulysses=False):
    """Single-sample forward adapted from VeOmni's MiniMaxH3Attention."""
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import _apply_rope

    if use_ulysses or len(cu_seqlens) != 2:
        raise ValueError("The H3 VeOmni attention bridge requires one sample and Ulysses SP=1.")
    total = x.shape[0]
    q, k, v = self.qkv_proj(x).view(total, self.num_heads, 3, self.head_dim).unbind(2)
    q, k = self.q_norm(q), self.k_norm(k)
    if rope_cos is not None:
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
    out = dispatch_attention_fn(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        scale=self.softmax_scale,
        backend=self._h3_attention_backend,
    )
    return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))


def _apply_minimax_h3_attention_backend(module: torch.nn.Module, implementation: str) -> None:
    """Select H3 attention per instance without changing parameters or global dispatch."""
    if hasattr(module, "_load_attention_kernel"):
        return
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _check_attention_backend_requirements,
        _maybe_download_kernel_for_backend,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import MiniMaxH3Attention

    if implementation not in _MINIMAX_H3_ATTENTION_BACKENDS:
        raise ValueError(
            f"Unsupported H3 VeOmni attention {implementation!r}; use one of {sorted(_MINIMAX_H3_ATTENTION_BACKENDS)}."
        )
    backend = AttentionBackendName(_MINIMAX_H3_ATTENTION_BACKENDS[implementation])
    _check_attention_backend_requirements(backend)
    _maybe_download_kernel_for_backend(backend)
    for layer in module.modules():
        if isinstance(layer, MiniMaxH3Attention):
            layer._h3_attention_backend = backend
            layer.forward = MethodType(_minimax_h3_attention_forward, layer)
