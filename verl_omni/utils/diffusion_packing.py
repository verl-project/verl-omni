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
"""Model-independent sequence boundaries for packed diffusion self-attention.

Inputs are [1, total_tokens, heads, head_dim]. Model adapters own multimodal
row ordering, positions and timestep expansion. Q/K/V share sample boundaries;
this helper does not implement cross-attention with unequal Q/K lengths.
"""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import accumulate

import torch
from torch.nn import functional as F


@lru_cache(maxsize=1)
def get_fa3_varlen():
    """Load the autograd-enabled FA3 varlen kernel without a fallback."""
    from kernels import get_kernel

    return get_kernel("kernels-community/flash-attn3", version=1).flash_attn_varlen_func


@dataclass(frozen=True)
class PackedSequenceLayout:
    """Per-sample attention boundaries for one packed micro-batch."""

    cu_seqlens: torch.Tensor
    max_seqlen: int
    lengths: tuple[int, ...]
    total_tokens: int

    @classmethod
    def from_lengths(cls, lengths: list[int], device: torch.device):
        """Construct int32 boundaries for positive per-sample row counts."""
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("Packed sequences must have positive lengths.")
        return cls(
            torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device),
            max(lengths),
            tuple(lengths),
            sum(lengths),
        )

    @cached_property
    def valid_mask(self):
        """Mask for the padded native-attention reference path."""
        device = self.cu_seqlens.device
        return torch.arange(self.max_seqlen, device=device)[None] < torch.tensor(self.lengths, device=device)[:, None]

    @cached_property
    def padded_indices(self):
        """Indices of valid rows in the flattened native reference batch."""
        return self.valid_mask.flatten().nonzero().flatten()

    def attention(self, query, key, value, backend):
        """Compute noncausal, sample-isolated attention on packed Q/K/V."""
        if backend == "_flash_3_varlen_hub":
            return get_fa3_varlen()(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                cu_seqlens_q=self.cu_seqlens,
                cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=self.max_seqlen,
                max_seqlen_k=self.max_seqlen,
                causal=False,
            ).unsqueeze(0)
        if backend == "torch_varlen":
            from torch.nn.attention.varlen import varlen_attn

            return varlen_attn(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                self.cu_seqlens,
                self.cu_seqlens,
                self.max_seqlen,
                self.max_seqlen,
            ).unsqueeze(0)
        if backend != "native":
            raise ValueError(f"Unsupported packed attention backend: {backend!r}.")

        batch = self.valid_mask.shape[0]

        def pad(tensor):
            padded = tensor.new_zeros((batch * self.max_seqlen, *tensor.shape[2:]))
            return (
                padded.index_copy(0, self.padded_indices, tensor.squeeze(0))
                .view(batch, self.max_seqlen, *tensor.shape[2:])
                .transpose(1, 2)
            )

        output = F.scaled_dot_product_attention(
            pad(query),
            pad(key),
            pad(value),
            attn_mask=self.valid_mask[:, None, None, :],
            dropout_p=0.0,
        )
        return output.transpose(1, 2).flatten(0, 1).index_select(0, self.padded_indices).unsqueeze(0)
