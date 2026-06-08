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


def _apply_flash_attention_3_varlen_hub_fix() -> None:
    """Patch ``_flash_attention_3_varlen_hub`` to support non-contiguous attention masks.

    diffusers==0.38.0 packs the valid key/value tokens with ``key[b, :valid_len]``, which
    only selects the correct tokens when the valid positions form a contiguous prefix.
    Qwen-Image joint text/image masks are *not* contiguous, so the wrong key/value tokens
    get gathered. We instead select the actual valid positions via
    ``attn_mask.flatten().nonzero()`` (matching the upstream diffusers ``fa3`` branch fix).

    Remove this patch once the fix is upstreamed to diffusers.
    """
    if version.parse(diffusers.__version__) < version.parse("0.38.0"):
        return

    from diffusers.models import attention_dispatch as _ad

    registry = _ad._AttentionBackendRegistry
    backend = _ad.AttentionBackendName._FLASH_3_VARLEN_HUB

    current = registry._backends.get(backend)
    if current is None or getattr(current, "_verl_omni_fa3_varlen_patched", False):
        return

    def _patched_flash_attention_3_varlen_hub(
        query,
        key,
        value,
        attn_mask=None,
        scale=None,
        is_causal=False,
        return_lse=False,
        _parallel_config=None,
    ):
        if not _patched_flash_attention_3_varlen_hub._warned:
            logger.warning(
                "verl_omni patch applied: diffusers `_flash_attention_3_varlen_hub` has been "
                "monkey-patched to gather key/value tokens by non-contiguous mask indices "
                "instead of assuming a contiguous prefix (diffusers==0.38). "
                "Remove this patch once the fix is upstreamed to diffusers."
            )
            _patched_flash_attention_3_varlen_hub._warned = True

        batch_size, seq_len_q, _, _ = query.shape
        _, seq_len_kv, _, _ = key.shape

        if attn_mask is not None:
            attn_mask = _ad._normalize_attn_mask(attn_mask, batch_size, seq_len_kv)

        (_, _seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k) = (
            _ad._prepare_for_flash_attn_or_sage_varlen(
                batch_size, seq_len_q, seq_len_kv, attn_mask=attn_mask, device=query.device
            )
        )

        query_packed = query.flatten(0, 1)
        if attn_mask is not None:
            # Gather valid key/value tokens by their actual (possibly non-contiguous)
            # positions rather than assuming a contiguous prefix `key[b, :valid_len]`.
            indices_k = attn_mask.flatten().nonzero(as_tuple=False).flatten()
            key_packed = key.reshape(-1, *key.shape[2:])[indices_k]
            value_packed = value.reshape(-1, *value.shape[2:])[indices_k]
        else:
            key_packed = key.flatten(0, 1)
            value_packed = value.flatten(0, 1)

        func = _ad._HUB_KERNELS_REGISTRY[backend].kernel_fn
        out, lse, *_ = func(
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
        out = out.unflatten(0, (batch_size, -1))

        return (out, lse) if return_lse else out

    _patched_flash_attention_3_varlen_hub._verl_omni_fa3_varlen_patched = True
    _patched_flash_attention_3_varlen_hub._warned = False

    registry._backends[backend] = _patched_flash_attention_3_varlen_hub
    _ad._flash_attention_3_varlen_hub = _patched_flash_attention_3_varlen_hub


_apply_qwen_image_ulysses_mask_fix()
_apply_flash_attention_3_varlen_hub_fix()
