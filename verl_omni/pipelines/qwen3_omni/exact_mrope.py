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
"""Exact interleaved M-RoPE frequencies for the true on-policy actor.

Transformers computes the thinker's rotary frequencies either as a matmul of positions and inverse
frequencies or elementwise, depending on the release. Under vLLM's batch-invariant mode that matmul runs
on a Triton kernel that rounds fp32 operands to TF32, so the two ways no longer agree. The vLLM-Omni
rollout computes the frequencies elementwise; this module does the same in the actor, whichever way the
installed Transformers writes it. It also keeps the inverse frequencies in fp32 when the model is cast to
bf16 (``fsdp_config.model_dtype=bfloat16``), which would otherwise round the Transformers buffer.
"""

import torch
from torch import nn


class ExactMRotaryEmbedding(nn.Module):
    """Drop-in for ``Qwen3OmniMoeThinkerTextRotaryEmbedding`` with elementwise frequencies."""

    def __init__(self, inv_freq: torch.Tensor, mrope_section: list[int], attention_scaling: float) -> None:
        super().__init__()
        # Stored as int32 bits so casting the model to bf16 (fsdp_config.model_dtype) keeps fp32 frequencies.
        self.register_buffer("inv_freq_bits", inv_freq.float().view(torch.int32), persistent=False)
        self.mrope_section = list(mrope_section)
        self.attention_scaling = attention_scaling

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.inv_freq_bits.view(torch.float32)

    @classmethod
    def from_hf(cls, rotary: nn.Module) -> "ExactMRotaryEmbedding":
        return cls(rotary.inv_freq, rotary.mrope_section, rotary.attention_scaling)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        freqs = position_ids[..., None].float() * self.inv_freq
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            idx = slice(offset, self.mrope_section[dim] * 3, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(x.dtype), sin.to(x.dtype)
