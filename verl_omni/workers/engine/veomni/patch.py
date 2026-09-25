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

import torch

# TODO: Remove this shim, its call sites and tests once the required VeOmni includes
# ByteDance-Seed/VeOmni#1214 and #1236. VeOmni 0.1.12 rejects Hub attention names and Qwen-Image
# ignores attn_implementation. Only Hub varlen backends handle Qwen-Image's text padding correctly,
# so this table is the allowlist for every model; the diffusers backend is applied to Qwen-Image only.
_DIFFUSERS_ATTENTION_BACKENDS = {
    "eager": "native",
    "flash_attention_2_hub": "flash_varlen_hub",
    "flash_attention_3_hub": "_flash_3_varlen_hub",
}


def _veomni_attn_implementation(attn_implementation: str) -> str:
    """Build with eager when the installed VeOmni cannot parse Hub attention names."""
    from veomni.arguments import OpsImplementationConfig

    if attn_implementation.endswith("_hub") and not hasattr(OpsImplementationConfig, "normalize_hub_attention_backend"):
        return "eager"  # Qwen-Image gets the Hub kernel in _apply_attention_backend
    return attn_implementation


def _apply_attention_backend(model: torch.nn.Module, attn_implementation: str) -> None:
    """Check ``veomni_config.attn_implementation`` for any model; apply it to VeOmni's Qwen-Image transformer."""
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
        # e.g. Wan / MiniMax H3 / LTX select attention inside VeOmni.
        if _veomni_attn_implementation(attn_implementation) != attn_implementation:
            raise ValueError(
                f"veomni_config.attn_implementation={attn_implementation!r} requires a VeOmni release "
                "with Hub attention support; the installed VeOmni only gets it for Qwen-Image via verl-omni."
            )
        return
    from diffusers.models.attention_dispatch import _AttentionBackendRegistry

    # set_attention_backend also switches diffusers' process-wide default; keep it for other models.
    active_backend = _AttentionBackendRegistry._active_backend
    model.set_attention_backend(backend)
    _AttentionBackendRegistry.set_active_backend(active_backend)
