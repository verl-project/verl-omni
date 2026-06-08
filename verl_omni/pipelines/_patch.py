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
import logging

import diffusers
import torch
from packaging import version

logger = logging.getLogger(__name__)


def _apply_qwen_image_ulysses_mask_fix() -> None:
    if version.parse(diffusers.__version__) < version.parse("0.38.0"):
        return

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    _orig_forward = QwenImageTransformer2DModel.forward
    if getattr(_orig_forward, "_verl_omni_ulysses_mask_patched", False):
        return

    def _patched_forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        attention_kwargs=None,
        **kwargs,
    ):
        parallel_config = getattr(self, "_parallel_config", None)
        cp_config = parallel_config.context_parallel_config if parallel_config is not None else None
        ulysses_degree = cp_config.ulysses_degree if cp_config is not None else 1

        if ulysses_degree > 1 and encoder_hidden_states_mask is not None:
            if not _patched_forward._warned:
                logger.warning(
                    "verl_omni patch applied: QwenImageTransformer2DModel.forward has been monkey-patched to fix "
                    "the Ulysses SP joint-attention-mask layout bug (diffusers==0.38). "
                    "The joint mask is now built in interleaved [txt_0, img_0, txt_1, img_1, ...] order "
                    "to match the post-all-to-all sequence layout when ulysses_degree > 1. "
                    "Remove this patch once the fix is upstreamed to diffusers."
                )
                _patched_forward._warned = True
            # Build the joint mask in the interleaved layout that matches the
            # post-all-to-all sequence order: [txt_0, img_0, txt_1, img_1, ...]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            txt_chunks = encoder_hidden_states_mask.chunk(ulysses_degree, dim=1)
            img_chunks = image_mask.chunk(ulysses_degree, dim=1)
            joint_mask = torch.cat([x for pair in zip(txt_chunks, img_chunks, strict=False) for x in pair], dim=1)
            attention_kwargs = dict(attention_kwargs or {}, attention_mask=joint_mask[:, None, None, :])
            encoder_hidden_states_mask = None

        return _orig_forward(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            attention_kwargs=attention_kwargs,
            **kwargs,
        )

    _patched_forward._verl_omni_ulysses_mask_patched = True
    _patched_forward._warned = False
    QwenImageTransformer2DModel.forward = _patched_forward


def _apply_flash_3_varlen_hub_return_fix() -> None:
    """Fix diffusers FA3 varlen hub when kernels returns a Tensor instead of a tuple."""
    if version.parse(diffusers.__version__) < version.parse("0.38.0"):
        return

    import diffusers.models.attention_dispatch as attention_dispatch

    if getattr(attention_dispatch._flash_attention_3_varlen_hub, "_verl_omni_fa3_varlen_return_patched", False):
        return

    def _patched_flash_attention_3_varlen_hub(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        scale: float | None = None,
        is_causal: bool = False,
        return_lse: bool = False,
        _parallel_config=None,
    ) -> torch.Tensor:
        from diffusers.models.attention_dispatch import (
            _HUB_KERNELS_REGISTRY,
            AttentionBackendName,
            _normalize_attn_mask,
            _prepare_for_flash_attn_or_sage_varlen,
        )

        batch_size, seq_len_q, _, _ = query.shape
        _, seq_len_kv, _, _ = key.shape

        if attn_mask is not None:
            attn_mask = _normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

        (_, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )

        key_valid, value_valid = [], []
        for b in range(batch_size):
            valid_len = seqlens_k[b]
            key_valid.append(key[b, :valid_len])
            value_valid.append(value[b, :valid_len])

        query_packed = query.flatten(0, 1)
        key_packed = torch.cat(key_valid, dim=0)
        value_packed = torch.cat(value_valid, dim=0)

        func = _HUB_KERNELS_REGISTRY[AttentionBackendName._FLASH_3_VARLEN_HUB].kernel_fn
        kernel_result = func(
            q=query_packed,
            k=key_packed,
            v=value_packed,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=scale,
            causal=is_causal,
        )
        if isinstance(kernel_result, tuple):
            out, lse, *_ = kernel_result
        else:
            out = kernel_result
            lse = None
        out = out.unflatten(0, (batch_size, -1))
        return (out, lse) if return_lse else out

    _patched_flash_attention_3_varlen_hub._verl_omni_fa3_varlen_return_patched = True
    attention_dispatch._flash_attention_3_varlen_hub = _patched_flash_attention_3_varlen_hub
    backend = attention_dispatch.AttentionBackendName._FLASH_3_VARLEN_HUB
    attention_dispatch._AttentionBackendRegistry._backends[backend] = _patched_flash_attention_3_varlen_hub
    if not getattr(_apply_flash_3_varlen_hub_return_fix, "_warned", False):
        logger.warning(
            "verl_omni patch applied: fixed diffusers _flash_attention_3_varlen_hub return "
            "unpack for kernels>=0.14 (Tensor vs tuple). Remove once upstream diffusers "
            "matches _flash_varlen_attention_3 handling."
        )
        _apply_flash_3_varlen_hub_return_fix._warned = True


_apply_qwen_image_ulysses_mask_fix()
_apply_flash_3_varlen_hub_return_fix()
