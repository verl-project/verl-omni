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
"""Compatibility helpers for Diffusers context parallelism.

The uneven Ulysses transform is adapted from Hugging Face Diffusers:
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py
"""

import inspect

import torch
import torch.distributed as dist
import torch.nn.functional as F

_UPSTREAM_NOT_IMPLEMENTED = "Backward pass for Ulysses Anything Attention in diffusers is not implemented yet."


class _UlyssesUnevenSequenceAttention(torch.autograd.Function):
    """Ulysses attention with uneven sequence lengths and divisible head counts."""

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        return_lse,
        forward_op,
        backward_op,
        _parallel_config=None,
        **kwargs,
    ):
        from diffusers.models import attention_dispatch as attention

        group = _parallel_config.context_parallel_config._ulysses_mesh.get_group()
        world_size = dist.get_world_size(group=group)
        if query.shape[2] % world_size or key.shape[2] % world_size or value.shape[2] % world_size:
            raise ValueError(
                "Uneven-sequence Ulysses training requires query/key/value head counts divisible by the SP size."
            )

        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config
        ctx.query_metadata = attention.ulysses_anything_metadata(query)
        ctx.key_metadata = attention.ulysses_anything_metadata(key)

        query_wait = attention.all_to_all_single_any_qkv_async(query, group, **ctx.query_metadata)
        key_wait = attention.all_to_all_single_any_qkv_async(key, group, **ctx.key_metadata)
        value_wait = attention.all_to_all_single_any_qkv_async(value, group, **ctx.key_metadata)
        query, key, value = query_wait(), key_wait(), value_wait()

        if attn_mask is not None:
            local_kv_size = ctx.key_metadata["Q_S_LOCAL"]
            if attn_mask.shape[-1] == local_kv_size:
                mask_local_sizes = attention.gather_size_by_comm(local_kv_size, group)
                max_local_size = max(mask_local_sizes)
                if local_kv_size < max_local_size:
                    attn_mask = F.pad(attn_mask, (0, max_local_size - local_kv_size))
                mask_list = [torch.empty_like(attn_mask) for _ in range(world_size)]
                dist.all_gather(mask_list, attn_mask, group=group)
                attn_mask = torch.cat(mask_list, dim=-1)[..., : sum(mask_local_sizes)]

        output = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            output, lse, *_ = output

        output_wait = attention.all_to_all_single_any_o_async(output, group, **ctx.query_metadata)
        if return_lse:
            lse_wait = attention.all_to_all_single_any_o_async(lse.unsqueeze(-1), group, **ctx.query_metadata)
            output = output_wait()
            lse = lse_wait().squeeze(-1).contiguous()
        else:
            output = output_wait()
            lse = None
        return (output, lse) if return_lse else output

    @staticmethod
    def backward(ctx, grad_output, *args):
        from diffusers.models import attention_dispatch as attention

        group = ctx._parallel_config.context_parallel_config._ulysses_mesh.get_group()
        grad_output = attention.all_to_all_single_any_qkv_async(grad_output, group, **ctx.query_metadata)()
        grad_query, grad_key, grad_value, *_ = ctx.backward_op(ctx, grad_output)
        grad_query = attention.all_to_all_single_any_o_async(grad_query, group, **ctx.query_metadata)()
        grad_key = attention.all_to_all_single_any_o_async(grad_key, group, **ctx.key_metadata)()
        grad_value = attention.all_to_all_single_any_o_async(grad_value, group, **ctx.key_metadata)()
        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None, None


def ensure_ulysses_uneven_sequence_backward() -> None:
    """Install the missing Diffusers 0.40 Ulysses-Anything backward implementation."""
    from diffusers.models import attention_dispatch as attention

    upstream = attention.TemplatedUlyssesAnythingAttention
    if upstream is _UlyssesUnevenSequenceAttention:
        return
    try:
        source = inspect.getsource(upstream.backward)
    except (OSError, TypeError) as exc:
        raise RuntimeError("Cannot verify the installed Diffusers Ulysses-Anything backward implementation.") from exc
    if _UPSTREAM_NOT_IMPLEMENTED not in source:
        if "NotImplementedError" in source:
            raise RuntimeError("The installed Diffusers Ulysses-Anything backward implementation is unsupported.")
        return
    attention.TemplatedUlyssesAnythingAttention = _UlyssesUnevenSequenceAttention
