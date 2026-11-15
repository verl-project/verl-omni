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
"""Hub FlashAttention-3 varlen attention registered with Transformers.

The vLLM rollout runs FlashAttention-3 on Hopper; calling the same kernel in the
actor makes attention bitwise equal on both sides. The kernel is fetched from the
``kernels`` hub on first call, so processes without a GPU can register the name.
"""

import functools

import torch
from transformers import AttentionInterface
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface

HUB_FA3_ATTENTION = "hub_fa3"


@functools.cache
def _flash_attn_varlen_func():
    from kernels import get_kernel

    return get_kernel("kernels-community/vllm-flash-attn3", version=1).flash_attn_varlen_func


def packed_cu_seqlens(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Derive varlen boundaries from packed position ids.

    Args:
        position_ids: ``(N,)`` positions of sequences packed along one row, restarting at 0 per sequence.

    Returns:
        ``cu_seqlens`` ``(num_seqs + 1,)`` int32 and the longest sequence length.
    """
    starts = torch.nonzero(position_ids == 0).flatten()
    cu_seqlens = torch.cat([starts, starts.new_tensor([position_ids.numel()])]).to(torch.int32)
    return cu_seqlens, int((cu_seqlens[1:] - cu_seqlens[:-1]).max())


def hub_fa3_attention(module, query, key, value, attention_mask, scaling=None, position_ids=None, **kwargs):
    """Causal varlen attention over a packed batch.

    Args:
        query, key, value: ``(1, H, N, D)`` packed along ``N``.
        position_ids: ``(1, N)``, or ``(C, 1, N)`` M-RoPE positions whose first row is used.

    Returns:
        ``(attn_output, None)`` with ``attn_output`` of shape ``(1, N, H, D)``.

    Raises:
        ValueError: If called without position ids, as the vision encoder does.
    """
    if position_ids is None:
        raise ValueError(
            "hub_fa3 serves the text decoder only, which passes packed position ids; "
            "true_on_policy does not support multimodal inputs."
        )
    text_positions = position_ids[0] if position_ids.dim() == 2 else position_ids[0, 0]
    cu_seqlens, max_seqlen = packed_cu_seqlens(text_positions)
    out = _flash_attn_varlen_func()(
        query[0].transpose(0, 1).contiguous(),
        key[0].transpose(0, 1).contiguous(),
        value[0].transpose(0, 1).contiguous(),
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale=scaling,
        causal=True,
    )
    out = out[0] if isinstance(out, tuple) else out  # some kernel builds also return the softmax lse
    return out[None], None


def register_hub_fa3_attention() -> str:
    """Register :func:`hub_fa3_attention` and the FlashAttention-2 mask under ``hub_fa3``."""
    AttentionInterface.register(HUB_FA3_ATTENTION, hub_fa3_attention)
    AttentionMaskInterface.register(HUB_FA3_ATTENTION, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])
    return HUB_FA3_ATTENTION
