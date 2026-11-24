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
"""Training kernels for the Qwen3-Omni Thinker: Ascend fusions and the true on-policy actor kernels."""

import functools
import logging
from copy import copy

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AttentionInterface
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS, ExpertsInterface, grouped_mm_experts_forward
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
from transformers.models.qwen3_omni_moe import modeling_qwen3_omni_moe as modeling
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerTextExperts,
    Qwen3OmniMoeThinkerTextModel,
)
from transformers.utils import is_torch_npu_available

logger = logging.getLogger(__name__)

__all__ = [
    "ExactMRotaryEmbedding",
    "enable_ascend_grouped_mm",
    "enable_ascend_rmsnorm_rope",
    "fused_experts",
    "hub_fa3_attention",
    "packed_cu_seqlens",
    "register_hub_fa3_attention",
    "register_vllm_fused_moe_experts",
    "use_tensor_parallel_linear",
    "vllm_fused_moe_experts_forward",
]

HUB_FA3_ATTENTION = "hub_fa3"
VLLM_FUSED_MOE_EXPERTS = "vllm_fused_moe"


def _ascend_grouped_mm(
    experts: Qwen3OmniMoeThinkerTextExperts,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.device.type != "npu" or hidden_states.dtype != torch.bfloat16:
        raise ValueError("ascend_grouped_mm requires BF16 Thinker activations on an Ascend NPU.")

    import torch_npu
    from verl.models.transformers.npu_patch import NPUGmmFunction

    gate_up_weight = experts.gate_up_proj.transpose(1, 2)
    down_weight = experts.down_proj.transpose(1, 2)
    permuted, row_ids = torch_npu.npu_moe_token_permute(hidden_states, top_k_index.to(torch.int32))
    counts = torch.histc(top_k_index, bins=experts.num_experts, min=0, max=experts.num_experts)
    gate_up = NPUGmmFunction.apply(permuted, gate_up_weight, counts)
    activated = torch_npu.npu_swiglu(gate_up, dim=-1)
    output = NPUGmmFunction.apply(activated, down_weight, counts)
    unpermuted = torch_npu.npu_moe_token_unpermute(output.to(top_k_weights.dtype), row_ids, probs=top_k_weights)
    return unpermuted.to(hidden_states.dtype)


def enable_ascend_grouped_mm(text_model: Qwen3OmniMoeThinkerTextModel) -> None:
    if not is_torch_npu_available():
        raise ValueError("ascend_grouped_mm requires an available Ascend NPU.")
    if text_model.config.hidden_act != "silu":
        raise ValueError("ascend_grouped_mm requires SiLU Thinker experts.")

    ALL_EXPERTS_FUNCTIONS.register("ascend_grouped_mm", _ascend_grouped_mm)
    text_model.config = copy(text_model.config)
    text_model.set_experts_implementation("ascend_grouped_mm")
    for experts in text_model.modules():
        if isinstance(experts, Qwen3OmniMoeThinkerTextExperts):
            experts.config = text_model.config


def enable_ascend_rmsnorm_rope() -> None:
    from verl.models.transformers.npu_patch import apply_rotary_pos_emb_npu, rms_norm_forward_npu

    if modeling.Qwen3OmniMoeThinkerTextRMSNorm.forward is rms_norm_forward_npu:
        return

    eager_rope = modeling.apply_rotary_pos_emb

    def apply_rotary_pos_emb(
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q.device.type == "npu" and q.dtype == torch.bfloat16 and k.dtype == torch.bfloat16:
            return apply_rotary_pos_emb_npu(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
        return eager_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)

    modeling.Qwen3OmniMoeThinkerTextRMSNorm.forward = rms_norm_forward_npu
    modeling.Qwen3OmniMoeTextRMSNorm.forward = rms_norm_forward_npu
    modeling.apply_rotary_pos_emb = apply_rotary_pos_emb
    logger.info("Enabled Ascend RMSNorm and RoPE fusions for Qwen3-Omni Thinker.")


class ExactMRotaryEmbedding(nn.Module):
    """Drop-in for ``Qwen3OmniMoeThinkerTextRotaryEmbedding`` with elementwise fp32 frequencies.

    Transformers computes the frequencies as a matmul of positions and inverse frequencies or elementwise,
    depending on the release. Under vLLM's batch-invariant mode that matmul runs on a Triton kernel that
    rounds fp32 operands to TF32, so the two no longer agree. The vLLM-Omni rollout computes them
    elementwise, and so does this module.
    """

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
    """Causal varlen attention over a packed batch, on the hub FlashAttention-3 kernel vLLM runs on Hopper.

    The kernel is fetched from the ``kernels`` hub on first call, so processes without a GPU can register
    the name.

    Args:
        query, key, value: ``(1, H, N, D)`` packed along ``N``.
        position_ids: ``(1, N)``, or ``(C, 1, N)`` M-RoPE positions whose first row is used.

    Returns:
        ``(attn_output, None)`` with ``attn_output`` of shape ``(1, N, H, D)``.
    """
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


def fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights):
    """Run vLLM's fused experts on HF-layout weights.

    Args:
        hidden_states: ``(N, H)`` routed tokens.
        gate_up_proj: ``(E, 2I, H)`` gate and up projections, gate first.
        down_proj: ``(E, H, I)`` down projections.
        top_k_index: ``(N, K)`` selected experts.
        top_k_weights: ``(N, K)`` routing weights in the activation dtype.

    Returns:
        ``(N, H)`` combined expert output in the activation dtype.
    """
    from vllm.model_executor.layers.fused_moe import fused_experts as vllm_fused_experts

    return vllm_fused_experts(
        hidden_states, gate_up_proj, down_proj, top_k_weights.float(), top_k_index.to(torch.int32)
    )


def _tensor_parallel_sum(partials: list[torch.Tensor]) -> torch.Tensor:
    """Add per-rank partial results as vLLM's batch-invariant all-reduce does: in fp32, in rank order, rounded once."""
    total = partials[0].float()
    for partial in partials[1:]:
        total = total + partial.float()
    return total.to(partials[0].dtype)


def use_tensor_parallel_linear(linear: nn.Linear, tp_size: int) -> None:
    """Make a bias-free linear return what the row-parallel layer of a rollout with ``tp_size`` ranks returns.

    Each rank multiplies its slice of the input features with its slice of the weight, and the partial outputs
    are added by :func:`_tensor_parallel_sum`. A single matmul over all features rounds differently.
    """
    linear.forward = functools.partial(_tensor_parallel_linear_forward, linear, tp_size)


def _tensor_parallel_linear_forward(linear: nn.Linear, tp_size: int, x: torch.Tensor) -> torch.Tensor:
    per_rank = linear.in_features // tp_size
    return _tensor_parallel_sum(
        [
            F.linear(
                x[..., start : start + per_rank].contiguous(), linear.weight[:, start : start + per_rank].contiguous()
            )
            for start in range(0, linear.in_features, per_rank)
        ]
    )


def _tensor_parallel_fused_experts(hidden_states, gate_up_proj, down_proj, top_k_index, top_k_weights, tp_size):
    """:func:`fused_experts` as a rollout with ``tp_size`` ranks runs it: each rank on its slice of the intermediate
    dimension, the partial outputs added by :func:`_tensor_parallel_sum`."""
    intermediate = down_proj.shape[-1]
    per_rank = intermediate // tp_size
    gate_proj, up_proj = gate_up_proj[:, :intermediate], gate_up_proj[:, intermediate:]
    partials = []
    for start in range(0, intermediate, per_rank):
        rank = slice(start, start + per_rank)
        rank_gate_up_proj = torch.cat([gate_proj[:, rank], up_proj[:, rank]], dim=1)
        rank_down_proj = down_proj[..., rank].contiguous()
        partials.append(fused_experts(hidden_states, rank_gate_up_proj, rank_down_proj, top_k_index, top_k_weights))
    return _tensor_parallel_sum(partials)


class _ValueWithGradientOf(torch.autograd.Function):
    """Returns ``value`` and sends its gradient to ``reference``."""

    @staticmethod
    def forward(ctx, reference, value):
        return value.view_as(value)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def vllm_fused_moe_experts_forward(self, hidden_states, top_k_index, top_k_weights, tp_size: int = 1):
    """Experts forward for ``ExpertsInterface``: vLLM ``fused_experts`` output with the ``grouped_mm`` gradient.

    ``fused_experts`` is the kernel the vLLM-Omni rollout runs and has no backward, so the gradient is taken
    from Transformers' ``grouped_mm`` experts evaluated on the same inputs. ``tp_size`` is the tensor-parallel
    degree of that rollout.
    """
    with torch.no_grad():
        if tp_size == 1:
            output = fused_experts(hidden_states, self.gate_up_proj, self.down_proj, top_k_index, top_k_weights)
        else:
            output = _tensor_parallel_fused_experts(
                hidden_states, self.gate_up_proj, self.down_proj, top_k_index, top_k_weights, tp_size
            )
    if not torch.is_grad_enabled():
        return output
    reference = grouped_mm_experts_forward(self, hidden_states, top_k_index, top_k_weights)
    return _ValueWithGradientOf.apply(reference, output)


def register_vllm_fused_moe_experts(tp_size: int = 1) -> str:
    """Register :func:`vllm_fused_moe_experts_forward` for a rollout with ``tp_size`` ranks under ``vllm_fused_moe``."""
    forward = functools.partial(vllm_fused_moe_experts_forward, tp_size=tp_size)
    ExpertsInterface.register(VLLM_FUSED_MOE_EXPERTS, forward)
    return VLLM_FUSED_MOE_EXPERTS
