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
"""Actor forward on the kernels of the batch-invariant vLLM rollout (Qwen3 text decoder).

Under ``VLLM_BATCH_INVARIANT=1`` the eager vLLM Qwen3 decoder runs the Triton persistent
matmul for every linear and the LM head, the Triton RMSNorm when there is no residual and
the CUDA ``fused_add_rms_norm`` when there is, the CUDA NeoX rotary kernel on a bf16
cos/sin cache, FA3 varlen with ``num_splits=1``, the CUDA ``silu_and_mul``, and the Triton
``log_softmax`` over fp32 logits divided by the temperature. The functions below call the
same kernels with the same dataflow (the residual add is deferred into the next norm) and
give each an ordinary torch backward. LoRA layers run one GEMM on the merged bf16 weight
that the weight sync also streams (``merge_lora_weight``).

Enabled by ``VERL_OMNI_VLLM_PARITY=1``.
"""

from __future__ import annotations

import logging
import os
import types

import torch

logger = logging.getLogger(__name__)

PARITY_SWITCH = "VERL_OMNI_VLLM_PARITY"

__all__ = [
    "PARITY_SWITCH",
    "build_rope_cache",
    "flash_attn",
    "fused_add_rms_norm",
    "install_qwen3_parity",
    "linear",
    "logprobs_from_logits",
    "lora_linear",
    "merge_lora_weight",
    "parity_enabled",
    "rms_norm",
    "rotary",
    "silu_and_mul",
]


def parity_enabled() -> bool:
    return os.environ.get(PARITY_SWITCH, "0") == "1"


def _bi():
    from vllm.model_executor.determinism import batch_invariant

    return batch_invariant


def _ops():
    import vllm._custom_ops as ops

    return ops


def _no_autocast():
    from verl.utils.device import get_device_name

    return torch.autocast(device_type=get_device_name(), enabled=False)


# ----------------------------------------------------------------------------- linear


def _bi_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = _bi().linear_batch_invariant(x2, weight)
    return out.reshape(*x.shape[:-1], weight.shape[0])


def merge_lora_weight(base: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """bf16(W + s * B @ A) from bf16-rounded operands, in fp32 with a deterministic Triton GEMM.

    The actor forward and the weight sync both call this, so the rollout serves exactly
    the weight the actor multiplies by.
    """
    with _no_autocast():
        a = lora_a.detach().to(torch.bfloat16).float()
        b = lora_b.detach().to(torch.bfloat16).float()
        delta = _bi().matmul_persistent(b.contiguous(), a.contiguous())
        merged = base.detach().to(torch.bfloat16).float() + delta * float(scaling)
        return merged.to(torch.bfloat16)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        w = weight if weight.dtype == torch.bfloat16 else weight.to(torch.bfloat16)
        ctx.save_for_backward(x, weight)
        with _no_autocast():
            return _bi_linear(x, w)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        gx = gw = None
        with _no_autocast():
            g2 = grad.reshape(-1, grad.shape[-1])
            if ctx.needs_input_grad[0]:
                gx = (g2 @ weight.to(g2.dtype)).reshape(x.shape).to(x.dtype)
            if ctx.needs_input_grad[1]:
                gw = (g2.t().float() @ x.reshape(-1, x.shape[-1]).float()).to(weight.dtype)
        return gx, gw


class _LoraLinear(torch.autograd.Function):
    """y = x @ merge(W, A, B, s)^T; backward is PEFT's LoRA gradient with dx = dy @ W_merged."""

    @staticmethod
    def forward(ctx, x, weight, lora_a, lora_b, scaling):
        ctx.scaling = scaling
        ctx.save_for_backward(x, weight, lora_a, lora_b)
        merged = merge_lora_weight(weight, lora_a, lora_b, scaling)
        with _no_autocast():
            return _bi_linear(x, merged)

    @staticmethod
    def backward(ctx, grad):
        x, weight, lora_a, lora_b = ctx.saved_tensors
        s = ctx.scaling
        gx = ga = gb = None
        with _no_autocast():
            g2 = grad.reshape(-1, grad.shape[-1])
            x2 = x.reshape(-1, x.shape[-1])
            if ctx.needs_input_grad[0]:
                merged = merge_lora_weight(weight, lora_a, lora_b, s)
                gx = (g2 @ merged.to(g2.dtype)).reshape(x.shape).to(x.dtype)
            if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
                a32 = lora_a.float()
                b32 = lora_b.float()
                g32 = g2.float()
                x32 = x2.float()
                if ctx.needs_input_grad[3]:
                    gb = (s * (g32.t() @ (x32 @ a32.t()))).to(lora_b.dtype)
                if ctx.needs_input_grad[2]:
                    ga = (s * ((g32 @ b32).t() @ x32)).to(lora_a.dtype)
        return gx, None, ga, gb, None


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _Linear.apply(x, weight)


def _lora_parts(module):
    """(base weight, A, B, scaling) of a plain or PEFT-LoRA linear; A/B None when no adapter is active."""
    base = getattr(module, "base_layer", None)
    if base is None:
        if getattr(module, "bias", None) is not None:
            raise NotImplementedError("vLLM parity linear does not support bias.")
        return module.weight, None, None, None
    if getattr(base, "bias", None) is not None:
        raise NotImplementedError("vLLM parity linear does not support bias.")
    if getattr(module, "disable_adapters", False) or getattr(module, "merged", False):
        return base.weight, None, None, None
    active = [name for name in module.active_adapters if name in module.lora_A]
    if not active:
        return base.weight, None, None, None
    if len(active) > 1:
        raise NotImplementedError("vLLM parity LoRA supports one active adapter.")
    name = active[0]
    dropout = module.lora_dropout[name]
    if not isinstance(dropout, torch.nn.Identity) and getattr(dropout, "p", 0.0) > 0 and module.training:
        raise NotImplementedError("vLLM parity LoRA requires lora_dropout == 0 (the rollout serves merged weights).")
    if getattr(module, "use_dora", {}).get(name, False):
        raise NotImplementedError("vLLM parity LoRA does not support DoRA.")
    return base.weight, module.lora_A[name].weight, module.lora_B[name].weight, module.scaling[name]


def iter_merged_lora_state_dict(module, device):
    """Yield ``module.state_dict()`` with every LoRA base weight replaced by ``merge_lora_weight``.

    Merges one layer at a time; LoRA A/B entries are yielded unchanged for the caller to drop.
    """
    from torch.distributed.tensor import DTensor

    def full(t):
        t = t.detach().to(device, non_blocking=True)
        return t.full_tensor() if isinstance(t, DTensor) else t

    modules = dict(module.named_modules())
    suffix = ".base_layer.weight"
    for name, param in module.state_dict().items():
        owner = modules.get(name[: -len(suffix)]) if name.endswith(suffix) else None
        if owner is not None:
            base, lora_a, lora_b, scaling = _lora_parts(owner)
            if lora_a is not None:
                yield name, merge_lora_weight(full(base), full(lora_a), full(lora_b), scaling)
                continue
        yield name, param


def lora_linear(x: torch.Tensor, module) -> torch.Tensor:
    weight, lora_a, lora_b, scaling = _lora_parts(module)
    if lora_a is None:
        return _Linear.apply(x, weight)
    return _LoraLinear.apply(x, weight, lora_a, lora_b, scaling)


# ----------------------------------------------------------------------------- norms


def _rms_backward(x, weight, grad, eps):
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    xhat = x32 * rstd
    gw = grad.float() * weight.float()
    gx = rstd * (gw - xhat * (gw * xhat).mean(-1, keepdim=True))
    gweight = (grad.float() * xhat).reshape(-1, x.shape[-1]).sum(0)
    return gx, gweight


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps):
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
        return _bi().rms_norm_batch_invariant(x, w, eps)

    @staticmethod
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        gx, gw = _rms_backward(x, weight, grad, ctx.eps)
        return gx.to(x.dtype), gw.to(weight.dtype) if ctx.needs_input_grad[1] else None, None


class _FusedAddRMSNorm(torch.autograd.Function):
    """(out, residual') = (norm(x + residual) * w, x + residual), the CUDA in-place kernel on copies."""

    @staticmethod
    def forward(ctx, x, residual, weight, eps):
        out = x.contiguous().clone()
        res = residual.contiguous().clone()
        w = weight if weight.dtype == x.dtype else weight.to(x.dtype)
        _ops().fused_add_rms_norm(out, res, w, eps)
        ctx.eps = eps
        ctx.save_for_backward(res, weight)
        return out, res

    @staticmethod
    def backward(ctx, grad_out, grad_res):
        res, weight = ctx.saved_tensors
        gx, gw = _rms_backward(res, weight, grad_out, ctx.eps)
        if grad_res is not None:
            gx = gx + grad_res.float()
        gx = gx.to(res.dtype)
        return gx, gx, gw.to(weight.dtype) if ctx.needs_input_grad[2] else None, None


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _RMSNorm.apply(x, weight, eps)


def fused_add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    return _FusedAddRMSNorm.apply(x, residual, weight, eps)


# ----------------------------------------------------------------------------- rotary


def build_rope_cache(head_size: int, max_position: int, base: float, device) -> torch.Tensor:
    """vLLM ``RotaryEmbedding`` cache: built in fp32 on the device, then cast to bf16."""
    with torch.device(device), _no_autocast():
        inv_freq = 1.0 / (base ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size))
        t = torch.arange(max_position, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache.to(torch.bfloat16)


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, q, k, cache, cache_inv, head_size):
        q2 = q.contiguous().clone()
        k2 = k.contiguous().clone()
        _ops().rotary_embedding(positions, q2, k2, head_size, cache, True)
        ctx.head_size = head_size
        ctx.save_for_backward(positions, cache_inv)
        return q2, k2

    @staticmethod
    def backward(ctx, gq, gk):
        positions, cache_inv = ctx.saved_tensors
        gq2 = gq.contiguous().clone()
        gk2 = gk.contiguous().clone()
        _ops().rotary_embedding(positions, gq2, gk2, ctx.head_size, cache_inv, True)
        return None, gq2, gk2, None, None, None


def rotary(positions, q, k, cache, cache_inv, head_size):
    return _Rotary.apply(positions, q, k, cache, cache_inv, head_size)


# ----------------------------------------------------------------------------- attention


class _FlashAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, scale):
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        out, lse = flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            max_seqlen_q=max_seqlen,
            cu_seqlens_q=cu_seqlens,
            max_seqlen_k=max_seqlen,
            cu_seqlens_k=cu_seqlens,
            softmax_scale=scale,
            causal=True,
            return_softmax_lse=True,
            num_splits=1,
            fa_version=3,
        )
        ctx.max_seqlen = max_seqlen
        ctx.scale = scale
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        return out

    @staticmethod
    def backward(ctx, grad):
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        q, k, v, out, lse, cu = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_varlen_backward(
            grad.contiguous(),
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            cu,
            cu,
            ctx.max_seqlen,
            ctx.max_seqlen,
            0.0,
            ctx.scale,
            True,
            -1,
            -1,
            0.0,
            None,
            False,
            None,
        )
        return dq, dk, dv, None, None, None


def flash_attn(q, k, v, cu_seqlens, max_seqlen, scale):
    return _FlashAttn.apply(q, k, v, cu_seqlens, max_seqlen, scale)


# ----------------------------------------------------------------------------- activation


class _SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        torch.ops._C.silu_and_mul(out, x.contiguous())
        ctx.save_for_backward(x)
        return out

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        d = x.shape[-1] // 2
        g = x[..., :d].float()
        u = x[..., d:].float()
        sig = torch.sigmoid(g)
        silu = g * sig
        gf = grad.float()
        dg = gf * u * (sig * (1.0 + g * (1.0 - sig)))
        du = gf * silu
        return torch.cat([dg, du], dim=-1).to(x.dtype)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    return _SiluAndMul.apply(x)


# ----------------------------------------------------------------------------- log-probs

_LOGPROB_CHUNK = 4096


class _LogProbs(torch.autograd.Function):
    """log_softmax(fp32(logits) / temperature)[label] on vLLM's Triton kernel, chunked over rows."""

    @staticmethod
    def forward(ctx, logits, labels, temperature):
        n = logits.shape[0]
        out = torch.empty(n, dtype=torch.float32, device=logits.device)
        with _no_autocast():
            for s in range(0, n, _LOGPROB_CHUNK):
                e = min(n, s + _LOGPROB_CHUNK)
                z = logits[s:e].float()
                if temperature is not None:
                    z = z / temperature[s:e].unsqueeze(-1)
                lsm = _bi().log_softmax(z, dim=-1)
                out[s:e] = lsm.gather(-1, labels[s:e].unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(logits, labels, temperature)
        return out

    @staticmethod
    def backward(ctx, grad):
        logits, labels, temperature = ctx.saved_tensors
        n = logits.shape[0]
        glogits = torch.empty_like(logits)
        with _no_autocast():
            for s in range(0, n, _LOGPROB_CHUNK):
                e = min(n, s + _LOGPROB_CHUNK)
                z = logits[s:e].float()
                if temperature is not None:
                    z = z / temperature[s:e].unsqueeze(-1)
                p = torch.softmax(z, dim=-1)
                g = grad[s:e].float()
                dz = p.mul_(-g.unsqueeze(-1))
                dz.scatter_add_(-1, labels[s:e].unsqueeze(-1), g.unsqueeze(-1))
                if temperature is not None:
                    dz = dz / temperature[s:e].unsqueeze(-1)
                glogits[s:e] = dz.to(logits.dtype)
        return glogits, None, None


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: torch.Tensor | None = None):
    """Rollout log-probs: ``temperature`` is a per-row fp32 tensor (or None for 1.0)."""
    shape = labels.shape
    logits2 = logits.reshape(-1, logits.shape[-1])
    labels2 = labels.reshape(-1)
    if temperature is not None:
        temperature = temperature.reshape(-1).to(torch.float32)
    return _LogProbs.apply(logits2, labels2, temperature).reshape(shape)


# ----------------------------------------------------------------------------- Qwen3 install


class _RopeCache:
    def __init__(self, head_size, max_position, base):
        self.head_size = head_size
        self.max_position = max_position
        self.base = base
        self._cache = {}

    def get(self, device):
        key = str(device)
        if key not in self._cache:
            cache = build_rope_cache(self.head_size, self.max_position, self.base, device)
            half = self.head_size // 2
            cache_inv = torch.cat([cache[:, :half], -cache[:, half:]], dim=-1)
            self._cache[key] = (cache, cache_inv)
        return self._cache[key]


def _packed_layout(position_ids: torch.Tensor, num_tokens: int, device):
    """Positions and cu_seqlens of the packed (rmpad) row: each sequence restarts at position 0."""
    if position_ids is None:
        positions = torch.arange(num_tokens, device=device)
    else:
        if position_ids.dim() == 3:
            position_ids = position_ids[0]
        if position_ids.dim() == 2:
            if position_ids.shape[0] != 1:
                raise NotImplementedError("vLLM parity forward expects the packed [1, T] layout (use_remove_padding).")
            position_ids = position_ids[0]
        positions = position_ids.to(device=device, dtype=torch.long)
    starts = torch.nonzero(positions == 0, as_tuple=False).flatten()
    if starts.numel() == 0 or int(starts[0]) != 0:
        starts = torch.cat([torch.zeros(1, dtype=starts.dtype, device=device), starts])
    cu = torch.cat([starts, torch.tensor([num_tokens], dtype=starts.dtype, device=device)]).to(torch.int32)
    max_seqlen = int((cu[1:] - cu[:-1]).max().item())
    return positions.contiguous(), cu, max_seqlen


def _decoder_layer_forward(self, hidden_states, residual, positions, cu_seqlens, max_seqlen):
    attn = self.self_attn
    mlp = self.mlp
    eps = self.input_layernorm.variance_epsilon
    num_tokens = hidden_states.shape[0]
    head_dim = attn.head_dim
    if residual is None:
        residual = hidden_states
        h = rms_norm(hidden_states, self.input_layernorm.weight, eps)
    else:
        h, residual = fused_add_rms_norm(hidden_states, residual, self.input_layernorm.weight, eps)

    q = lora_linear(h, attn.q_proj)
    k = lora_linear(h, attn.k_proj)
    v = lora_linear(h, attn.v_proj)
    num_heads = q.shape[-1] // head_dim
    num_kv_heads = k.shape[-1] // head_dim
    q = rms_norm(q.view(num_tokens, num_heads, head_dim), attn.q_norm.weight, attn.q_norm.variance_epsilon)
    k = rms_norm(k.view(num_tokens, num_kv_heads, head_dim), attn.k_norm.weight, attn.k_norm.variance_epsilon)
    cache, cache_inv = self._vllm_parity_rope.get(hidden_states.device)
    q, k = rotary(positions, q.view(num_tokens, -1), k.view(num_tokens, -1), cache, cache_inv, head_dim)
    o = flash_attn(
        q.view(num_tokens, num_heads, head_dim),
        k.view(num_tokens, num_kv_heads, head_dim),
        v.view(num_tokens, num_kv_heads, head_dim),
        cu_seqlens,
        max_seqlen,
        attn.scaling,
    )
    o = lora_linear(o.reshape(num_tokens, num_heads * head_dim), attn.o_proj)

    h, residual = fused_add_rms_norm(o, residual, self.post_attention_layernorm.weight, eps)
    gate_up = torch.cat([lora_linear(h, mlp.gate_proj), lora_linear(h, mlp.up_proj)], dim=-1)
    out = lora_linear(silu_and_mul(gate_up), mlp.down_proj)
    return out, residual


def _model_forward(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    from transformers.modeling_outputs import BaseModelOutputWithPast

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    if inputs_embeds.dim() == 3:
        if inputs_embeds.shape[0] != 1:
            raise NotImplementedError("vLLM parity forward expects the packed [1, T] layout (use_remove_padding).")
        hidden = inputs_embeds[0]
    else:
        hidden = inputs_embeds
    if hidden.dtype != torch.bfloat16:
        hidden = hidden.to(torch.bfloat16)
    num_tokens = hidden.shape[0]
    positions, cu_seqlens, max_seqlen = _packed_layout(position_ids, num_tokens, hidden.device)

    residual = None
    for layer in self.layers[: self.config.num_hidden_layers]:
        hidden, residual = layer(hidden, residual, positions, cu_seqlens, max_seqlen)
    hidden, _ = fused_add_rms_norm(hidden, residual, self.norm.weight, self.norm.variance_epsilon)
    return BaseModelOutputWithPast(last_hidden_state=hidden.unsqueeze(0), past_key_values=None)


def _lm_head_forward(self, x):
    return _Linear.apply(x, self.weight)


def install_qwen3_parity(causal_lm) -> None:
    """Route an HF ``Qwen3ForCausalLM`` (decoder layers, final norm, LM head) through the rollout kernels.

    Layers keep their module identity, so FSDP2 unsharding and gradient checkpointing still
    wrap each decoder layer call; PEFT LoRA wrappers are resolved at call time.
    """
    model = causal_lm.model
    if getattr(model, "_vllm_parity_installed", False):
        return
    cfg = model.config
    rope_params = getattr(cfg, "rope_parameters", None) or {}
    rope_type = rope_params.get("rope_type", "default")
    if rope_type != "default":
        raise NotImplementedError(f"vLLM parity supports default RoPE only, got {rope_type}.")
    rope_theta = float(rope_params.get("rope_theta", getattr(cfg, "rope_theta", 10000.0)))
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    if getattr(cfg, "attention_bias", False) or getattr(cfg, "use_sliding_window", False):
        raise NotImplementedError("vLLM parity supports Qwen3 without attention bias or sliding window.")
    if getattr(cfg, "hidden_act", "silu") != "silu":
        raise NotImplementedError("vLLM parity supports SiLU MLPs only.")
    rope = _RopeCache(head_dim, cfg.max_position_embeddings, rope_theta)
    for layer in model.layers:
        layer._vllm_parity_rope = rope
        layer.forward = types.MethodType(_decoder_layer_forward, layer)
    model.forward = types.MethodType(_model_forward, model)
    causal_lm.lm_head.forward = types.MethodType(_lm_head_forward, causal_lm.lm_head)
    model._vllm_parity_installed = True
    logger.info("Installed vLLM parity forward on %s (%d layers).", type(causal_lm).__name__, len(model.layers))
