"""Run the Diffusers-layout MiniMax H3 Actor on the rollout's own normalization kernels.

The rollout denoiser computes every RMSNorm, AdaLN modulation, gated residual and Q/K norm + RoPE in
FP32 inside fused kernels and rounds to BF16 once per kernel; the stock Diffusers forward rounds after
each elementwise op. The 50-block stack amplifies that difference to several percent of the velocity,
so the Actor calls the same kernels. The kernels are not differentiable: their output is the forward
value and the same formula in plain torch carries the gradient (straight-through across the rounding).

Linear projections, SwiGLU and SDPA attention are already bit-identical between the two layouts.
"""

from __future__ import annotations

import types
from typing import Any

import torch
from torch import nn

__all__ = ["install_rollout_kernels", "rollout_kernels_enabled"]

_FLAG = "_minimax_h3_rollout_kernels"
SWITCHES = {"adaln": True, "qk_norm_rope": True, "rms_norm": True, "lora_addmm": True, "attention": "TORCH_SDPA"}
CALLS = {"kernel": 0, "stock": 0}
ATTENTION_SEGMENTS: dict[int, list[tuple[int, int]]] = {}


def _rollout_attention_kernel(sequence_length: int, padded: bool):
    """The SDPA kernel the rollout ends up on for one packed image document.

    The rollout pads the packed sequence and, on its ``TORCH_SDPA`` backend, masks the padding keys; torch then
    leaves the flash kernel for the memory-efficient one. Its ``CUDNN_ATTN`` backend slices the padding off.
    The token refiner runs on the unpadded text rows (``padded=False``).
    """
    from contextlib import nullcontext

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backend = SWITCHES["attention"]
    if backend == "CUDNN_ATTN":
        return sdpa_kernel([SDPBackend.CUDNN_ATTENTION])
    if backend == "TORCH_SDPA" and padded:
        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import _image_packed_seq_len

        if _image_packed_seq_len(sequence_length) != sequence_length:
            return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    return nullcontext()


def _straight_through(value: torch.Tensor, reference_fn) -> torch.Tensor:
    """``value`` in the forward pass, gradient of ``reference_fn()``."""
    if not torch.is_grad_enabled():
        return value
    reference = reference_fn()
    if not reference.requires_grad:
        return value
    return value.detach() + (reference - reference.detach())


def _rms_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = x.float()
    return values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + eps) * weight.float()


def _kernel_ready(x: torch.Tensor) -> bool:
    return x.is_cuda and x.dtype == torch.bfloat16


def _rms_norm(norm: nn.RMSNorm, x: torch.Tensor) -> torch.Tensor:
    """RMSNorm over the last axis with the rollout's fused kernel."""
    if not (SWITCHES["rms_norm"] and _kernel_ready(x)):
        return norm._stock_forward(x) if hasattr(norm, "_stock_forward") else norm(x)
    from vllm._custom_ops import rms_norm

    rows = x.reshape(-1, x.shape[-1]).contiguous()
    with torch.no_grad():
        out = torch.empty_like(rows)
        rms_norm(out, rows, norm.weight.data, norm.eps)
    out = out.reshape(x.shape)
    return _straight_through(out, lambda: _rms_reference(x, norm.weight, norm.eps).to(x.dtype))


def _modulate_reference(normalized: torch.Tensor, shift, scale, indices) -> torch.Tensor:
    return normalized * (1.0 + scale.float().index_select(0, indices)) + shift.float().index_select(0, indices)


def _block_forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
    if not (SWITCHES["adaln"] and _kernel_ready(hidden_states) and hidden_states.shape[0] == 1):
        CALLS["stock"] += 1
        return self._stock_forward(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)
    from vllm_omni.diffusion.layers.indexed_modulation import (
        indexed_gate,
        indexed_gate_rms_norm_scale_shift,
        rms_norm_indexed_scale_shift,
    )

    CALLS["kernel"] += 1

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        table.contiguous() for table in self.adaln_proj(temb)
    )
    dtype = hidden_states.dtype
    x = hidden_states[0].contiguous()
    eps1, eps2 = self.norm1.eps, self.norm2.eps

    with torch.no_grad():
        modulated = rms_norm_indexed_scale_shift(x, self.norm1.weight, shift_msa, scale_msa, adaln_indices, eps1)
    modulated = _straight_through(
        modulated,
        lambda: _modulate_reference(_rms_reference(x, self.norm1.weight, eps1), shift_msa, scale_msa, adaln_indices).to(
            dtype
        ),
    )
    attn_output = self.attn(modulated.unsqueeze(0), rotary_emb, attention_mask)[0].contiguous()

    with torch.no_grad():
        residual, modulated = indexed_gate_rms_norm_scale_shift(
            x, gate_msa, attn_output, self.norm2.weight, shift_mlp, scale_mlp, adaln_indices, eps2
        )

    def residual_reference():
        return (x.float() + gate_msa.float().index_select(0, adaln_indices) * attn_output.float()).to(dtype)

    residual = _straight_through(residual, residual_reference)
    modulated = _straight_through(
        modulated,
        lambda: _modulate_reference(
            _rms_reference(residual, self.norm2.weight, eps2), shift_mlp, scale_mlp, adaln_indices
        ).to(dtype),
    )
    ff_output = self.ff(modulated.unsqueeze(0))[0].contiguous()

    with torch.no_grad():
        output = indexed_gate(residual, gate_mlp, ff_output, adaln_indices)
    output = _straight_through(
        output,
        lambda: (residual.float() + gate_mlp.float().index_select(0, adaln_indices) * ff_output.float()).to(dtype),
    )
    return output.unsqueeze(0)


def _refiner_block_forward(self, hidden_states):
    hidden_states = hidden_states + self.attn(_rms_norm(self.norm1, hidden_states))
    return hidden_states + self.ff(_rms_norm(self.norm2, hidden_states))


def _norm_forward(self, hidden_states):
    return _rms_norm(self, hidden_states)


def _norm_out_forward(self, hidden_states, temb, timestep_indices):
    if not (SWITCHES["adaln"] and _kernel_ready(hidden_states) and hidden_states.shape[0] == 1):
        return self._stock_forward(hidden_states, temb, timestep_indices)
    from diffusers.models.modeling_utils import get_parameter_dtype
    from vllm_omni.diffusion.layers.indexed_modulation import indexed_scale_shift_

    shift, scale = self.linear(nn.functional.silu(temb).to(get_parameter_dtype(self.linear))).chunk(2, dim=-1)
    shift, scale = shift.contiguous(), scale.contiguous()
    normalized = _rms_norm(self.norm, hidden_states)[0].contiguous()
    with torch.no_grad():
        output = indexed_scale_shift_(normalized.detach().clone(), shift, scale, timestep_indices)
    output = _straight_through(
        output,
        lambda: _modulate_reference(normalized.float(), shift, scale, timestep_indices).to(normalized.dtype),
    )
    return output.unsqueeze(0)


def _qk_norm_rope_reference(x, weight, eps, table):
    normalized = _rms_reference(x, weight, eps)
    half = table.shape[-1] // 2
    cos, sin = table[:, None, :half].float(), table[:, None, half:].float()
    first, second, rest = normalized[..., :half], normalized[..., half : 2 * half], normalized[..., 2 * half :]
    return torch.cat((first * cos - second * sin, second * cos + first * sin, rest), dim=-1)


class MiniMaxH3RolloutKernelAttnProcessor:
    """The stock processor with the rollout's Q/K RMSNorm (+ RoPE) kernels."""

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))

        fused = SWITCHES["qk_norm_rope"] and _kernel_ready(query) and query.shape[0] == 1
        if rotary_emb is None:
            query = _rms_norm(attn.norm_q, query) if fused else attn.norm_q(query)
            key = _rms_norm(attn.norm_k, key) if fused else attn.norm_k(key)
        elif fused:
            from vllm_omni.diffusion.layers.fused_qk_norm_rope import fused_qk_norm_rope

            cos, sin = rotary_emb
            half = cos.shape[-1] // 2
            table = torch.cat((cos[..., :half], sin[..., :half]), dim=-1).to(query.dtype).contiguous()
            raw_query, raw_key = query[0].contiguous(), key[0].contiguous()
            eps = attn.norm_q.eps
            with torch.no_grad():
                query, key = fused_qk_norm_rope(raw_query, raw_key, attn.norm_q.weight, attn.norm_k.weight, table, eps)
            query = _straight_through(
                query, lambda: _qk_norm_rope_reference(raw_query, attn.norm_q.weight, eps, table).to(raw_query.dtype)
            ).unsqueeze(0)
            key = _straight_through(
                key, lambda: _qk_norm_rope_reference(raw_key, attn.norm_k.weight, eps, table).to(raw_key.dtype)
            ).unsqueeze(0)
        else:
            from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb

            query = _apply_rotary_emb(attn.norm_q(query), *rotary_emb)
            key = _apply_rotary_emb(attn.norm_k(key), *rotary_emb)

        from contextlib import nullcontext

        segments = ATTENTION_SEGMENTS.get(int(query.shape[1])) or [(0, int(query.shape[1]))]
        attended = []
        for start, end in segments:
            kernel = _rollout_attention_kernel(end - start, padded=rotary_emb is not None) if fused else nullcontext()
            with kernel:
                attended.append(
                    dispatch_attention_fn(
                        query[:, start:end],
                        key[:, start:end],
                        value[:, start:end],
                        attn_mask=attention_mask,
                        dropout_p=0.0,
                        is_causal=False,
                        backend=self._attention_backend,
                        parallel_config=self._parallel_config,
                    )
                )
        hidden_states = attended[0] if len(attended) == 1 else torch.cat(attended, dim=1)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def _swap_processor(attention: nn.Module) -> None:
    stock = attention.processor
    processor = MiniMaxH3RolloutKernelAttnProcessor()
    processor._attention_backend = getattr(stock, "_attention_backend", None)
    processor._parallel_config = getattr(stock, "_parallel_config", None)
    attention.set_processor(processor)


def _lora_linear_forward(self, x, *args, **kwargs):
    """PEFT LoRA linear with the rollout's expand step: one ``addmm`` into the base output."""
    adapters = [name for name in self.active_adapters if name in self.lora_A]
    plain = (
        SWITCHES["lora_addmm"]
        and not args
        and not kwargs
        and not self.disable_adapters
        and not self.merged
        and len(adapters) == 1
        and isinstance(self.lora_dropout[adapters[0]], nn.Identity)
        and not self.use_dora.get(adapters[0], False)
    )
    if plain:
        lora_a, lora_b = self.lora_A[adapters[0]].weight, self.lora_B[adapters[0]].weight
        plain = _kernel_ready(x) and lora_a.dtype == lora_b.dtype == x.dtype
    if not plain:
        return self._stock_forward(x, *args, **kwargs)
    result = self.base_layer(x)
    buffer = x.reshape(-1, x.shape[-1]) @ lora_a.t()
    expanded = torch.addmm(
        result.reshape(-1, result.shape[-1]), buffer, lora_b.t(), alpha=float(self.scaling[adapters[0]])
    )
    return expanded.view(result.shape)


def _rebind(module: nn.Module, forward) -> None:
    if not hasattr(module, "_stock_forward"):
        module._stock_forward = module.forward
    module.forward = types.MethodType(forward, module)


def rollout_kernels_enabled(module: nn.Module) -> bool:
    return bool(getattr(module, _FLAG, False))


def install_rollout_kernels(module: Any) -> None:
    """Rebind the Actor's normalization sites to the rollout kernels (idempotent)."""
    if rollout_kernels_enabled(module):
        return
    for block in module.transformer_blocks:
        _rebind(block, _block_forward)
        _swap_processor(block.attn)
    for block in module.token_refiner.refiner_blocks:
        _rebind(block, _refiner_block_forward)
        _swap_processor(block.attn)
    _rebind(module.token_refiner.final_norm, _norm_forward)
    _rebind(module.norm_out, _norm_out_forward)
    for child in module.modules():
        if hasattr(child, "lora_A") and hasattr(child, "base_layer") and isinstance(child.base_layer, nn.Linear):
            _rebind(child, _lora_linear_forward)
    module.proj_out.rollout_row_padding = True
    setattr(module, _FLAG, True)
