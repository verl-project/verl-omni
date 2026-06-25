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

"""BagelForTraining – FSDP-compatible BAGEL MoT module for flow-matching training.

Ported from vllm-omni/BAGEL.  MoT dual pathways (text vs generation),
start/end-of-image boundary tokens, shared RoPE for latent tokens, and
QK-norm + RoPE in float32 (cast to bf16 only for SDPA).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from verl_omni.pipelines.non_diffusers_model_base import NonDiffusersModelBase

# ===================================================================
#  Config
# ===================================================================


@dataclass
class BagelTrainingConfig:
    hidden_size: int = 3584
    intermediate_size: int = 18944
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768
    # Bagel-specific
    latent_patch_size: int = 2
    max_latent_size: int = 32
    latent_channel: int = 16
    vae_downsample: int = 8
    start_of_image_id: int = 151652  # <|vision_start|>
    end_of_image_id: int = 151653  # <|vision_end|>

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_latent_dim(self) -> int:
        return self.latent_patch_size**2 * self.latent_channel

    def save_pretrained(self, save_directory: str):
        """Save config as JSON.

        Args:
            save_directory: Target directory.
        """
        from dataclasses import asdict

        output_path = os.path.join(save_directory, "config.json")
        os.makedirs(save_directory, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(asdict(self), f, indent=4, sort_keys=True)

    @classmethod
    def from_model_path(cls, model_path: str) -> BagelTrainingConfig:
        """Parse BAGEL config from ``config.json`` in *model_path*.

        Args:
            model_path: Directory containing ``config.json``.

        Returns:
            BagelTrainingConfig with values from the checkpoint config.
        """
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path) as f:
            root_cfg = json.load(f)
        llm = root_cfg.get("llm_config", {})
        vae = root_cfg.get("vae_config", {})
        return cls(
            hidden_size=llm.get("hidden_size", 3584),
            intermediate_size=llm.get("intermediate_size", 18944),
            num_hidden_layers=llm.get("num_hidden_layers", 28),
            num_attention_heads=llm.get("num_attention_heads", 28),
            num_key_value_heads=llm.get("num_key_value_heads", 4),
            vocab_size=llm.get("vocab_size", 152064),
            rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
            rope_theta=llm.get("rope_theta", 1_000_000.0),
            max_position_embeddings=llm.get("max_position_embeddings", 32768),
            latent_patch_size=root_cfg.get("latent_patch_size", 2),
            max_latent_size=root_cfg.get("max_latent_size", 32),
            latent_channel=vae.get("z_channels", 16),
            vae_downsample=vae.get("downsample", 8),
        )


def get_flattened_position_ids(img_h: int, img_w: int, patch_size: int, max_num_patches_per_side: int) -> torch.Tensor:
    """Compute flattened 2-D position IDs for latent patches.

    Args:
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        patch_size: Latent patch size (VAE downsample × latent_patch_size).
        max_num_patches_per_side: Max grid size for position embedding.

    Returns:
        Flattened position IDs of shape ``(num_patches,)``.
    """
    num_patches_h = img_h // patch_size
    num_patches_w = img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


# ===================================================================
#  Transformer building blocks
# ===================================================================


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class BagelMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ===================================================================
#  RoPE helpers
# ===================================================================


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_emb(q, k, cos, sin):
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_position_embeddings: int = 32768, theta: float = 1_000_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor):
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq.to(position_ids.device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


# ===================================================================
#  MoT Attention & Layer
# ===================================================================


class BagelMoTAttention(nn.Module):
    """MoT attention with separate standard and generation projections.

    Args:
        config: ``BagelTrainingConfig`` with head dimensions and MoT settings.
    """

    def __init__(self, config: BagelTrainingConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj_moe_gen = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_moe_gen = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, L, _ = hidden_states.shape
        text_idx = text_mask.nonzero(as_tuple=True)
        latent_idx = latent_mask.nonzero(as_tuple=True)

        q = hidden_states.new_zeros(B, L, self.num_heads * self.head_dim)
        k = hidden_states.new_zeros(B, L, self.num_kv_heads * self.head_dim)
        v = hidden_states.new_zeros(B, L, self.num_kv_heads * self.head_dim)

        text_hs = hidden_states[text_idx]
        q[text_idx] = self.q_proj(text_hs)
        k[text_idx] = self.k_proj(text_hs)
        v[text_idx] = self.v_proj(text_hs)

        latent_hs = hidden_states[latent_idx]
        q[latent_idx] = self.q_proj_moe_gen(latent_hs)
        k[latent_idx] = self.k_proj_moe_gen(latent_hs)
        v[latent_idx] = self.v_proj_moe_gen(latent_hs)

        q = q.view(B, L, self.num_heads, self.head_dim)
        k = k.view(B, L, self.num_kv_heads, self.head_dim)
        v = v.view(B, L, self.num_kv_heads, self.head_dim)

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        q_normed = q.new_zeros(q.shape)
        k_normed = k.new_zeros(k.shape)
        q_normed[text_idx] = self.q_norm(q[text_idx])
        k_normed[text_idx] = self.k_norm(k[text_idx])
        q_normed[latent_idx] = self.q_norm_moe_gen(q[latent_idx])
        k_normed[latent_idx] = self.k_norm_moe_gen(k[latent_idx])

        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        q_normed, k_normed = _apply_rotary_emb(q_normed, k_normed, cos, sin)

        q_normed = q_normed.to(torch.bfloat16)
        k_normed = k_normed.to(torch.bfloat16)
        v = v.to(torch.bfloat16)

        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k_normed = k_normed.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, L, self.num_heads, self.head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, rep, -1).reshape(B, L, self.num_heads, self.head_dim)

        # Padded SDPA requires a key padding mask so that zero-padded text
        # tokens don't leak into image-region attention (log_prob invariance).
        q_normed = q_normed.transpose(1, 2)  # (B, H, L, D)
        k_normed = k_normed.transpose(1, 2)
        v = v.transpose(1, 2)

        if L_ctx > 0:
            if key_padding_mask is not None and not key_padding_mask.all():
                # Official BAGEL packs prompts, so padded prompt keys do not exist there.
                # Mask them in both branches to match packed attention semantics.
                text_key_mask = key_padding_mask[:, :L_ctx].view(B, 1, 1, L_ctx)
                causal_mask = torch.ones(L_ctx, L_ctx, dtype=torch.bool, device=hidden_states.device).tril()
                text_out = F.scaled_dot_product_attention(
                    q_normed[:, :, :L_ctx],
                    k_normed[:, :, :L_ctx],
                    v[:, :, :L_ctx],
                    attn_mask=text_key_mask & causal_mask.view(1, 1, L_ctx, L_ctx),
                    is_causal=False,
                )
                # key_padding_mask: True = valid key, broadcast as (B,1,1,L).
                img_attn_mask = key_padding_mask.view(B, 1, 1, L)
                img_out = F.scaled_dot_product_attention(
                    q_normed[:, :, L_ctx:],
                    k_normed,
                    v,
                    attn_mask=img_attn_mask,
                    is_causal=False,
                )
            else:
                text_out = F.scaled_dot_product_attention(
                    q_normed[:, :, :L_ctx],
                    k_normed[:, :, :L_ctx],
                    v[:, :, :L_ctx],
                    is_causal=True,
                )
                img_out = F.scaled_dot_product_attention(
                    q_normed[:, :, L_ctx:],
                    k_normed,
                    v,
                    is_causal=False,
                )
            attn_out = torch.cat([text_out, img_out], dim=2)
        else:
            attn_out = F.scaled_dot_product_attention(
                q_normed,
                k_normed,
                v,
                is_causal=False,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, -1)

        out = hidden_states.new_zeros(B, L, self.hidden_size)
        out[text_idx] = self.o_proj(attn_out[text_idx].to(self.o_proj.weight.dtype))
        out[latent_idx] = self.o_proj_moe_gen(attn_out[latent_idx].to(self.o_proj_moe_gen.weight.dtype))
        return out


class BagelMoTLayer(nn.Module):
    def __init__(self, config: BagelTrainingConfig):
        super().__init__()
        self.self_attn = BagelMoTAttention(config)
        self.mlp = BagelMLP(config.hidden_size, config.intermediate_size)
        self.mlp_moe_gen = BagelMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with MoT-routed layernorm, attention, and MLP.

        Args:
            hidden_states: ``(B, L, D)`` input sequence.
            cos: RoPE cosine embedding.
            sin: RoPE sine embedding.
            text_mask: Bool mask — True for text pathway.
            latent_mask: Bool mask — True for gen pathway.
            L_ctx: Text context length for causal split.
            key_padding_mask: ``(B, L)`` — True at valid keys.

        Returns:
            Output of shape ``(B, L, D)``.
        """
        text_idx = text_mask.nonzero(as_tuple=True)
        latent_idx = latent_mask.nonzero(as_tuple=True)

        normed = hidden_states.new_zeros(hidden_states.shape)
        normed[text_idx] = self.input_layernorm(hidden_states[text_idx])
        normed[latent_idx] = self.input_layernorm_moe_gen(hidden_states[latent_idx])

        attn_out = self.self_attn(
            normed,
            cos,
            sin,
            text_mask,
            latent_mask,
            L_ctx,
            key_padding_mask=key_padding_mask,
        )
        hidden_states = hidden_states + attn_out

        residual = hidden_states
        mlp_out = hidden_states.new_zeros(hidden_states.shape)
        mlp_out[text_idx] = self.mlp(self.post_attention_layernorm(hidden_states[text_idx]))
        mlp_out[latent_idx] = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(hidden_states[latent_idx]))
        hidden_states = residual + mlp_out
        return hidden_states


# ===================================================================
#  Position embedding helpers
# ===================================================================


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.freq_dim = freq_dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        emb = emb.to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class PositionEmbedding(nn.Module):
    def __init__(self, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        pos_embed = _get_2d_sincos_pos_embed(hidden_size, max_num_patch_per_side)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float(), requires_grad=False)

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.pos_embed[position_ids]


# ===================================================================
#  Main module: BagelForTraining
# ===================================================================


class BagelForTraining(NonDiffusersModelBase):
    """Standalone Bagel MoT module for FlowGRPO FSDP training.

    ``_no_split_modules`` enables layer-level FSDP sharding so that
    ``layered_summon`` finds ``layers.N`` for rollout weight sync.
    """

    _no_split_modules = ["BagelMoTLayer"]
    _supports_gradient_checkpointing = True

    def __init__(self, config: BagelTrainingConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([BagelMoTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        text_token_ids: Optional[Tensor],
        latent_pos_ids: Tensor,
        **kwargs,
    ) -> tuple[Tensor]:
        """Forward pass.

        Args:
            hidden_states: ``(B, L_latent, patch_latent_dim)`` noisy latent patches.
            timestep: ``(B,)`` diffusion timestep.
            text_token_ids: ``(B, L_text)`` token IDs, or ``None`` for CFG unconditional.
            latent_pos_ids: ``(B, L_latent)`` 2-D position indices.
            text_attention_mask: ``(B, L_text)`` bool mask (via ``**kwargs``).

        Returns:
            Tuple of ``(velocity,)`` — noise prediction of shape
            ``(B, L_latent, patch_latent_dim)``.
        """
        text_attention_mask = kwargs.pop("text_attention_mask", None)
        if text_token_ids is not None and text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(device=text_token_ids.device, dtype=torch.bool)
            text_lengths = text_attention_mask.sum(dim=-1)
            if text_lengths.numel() > 0:
                text_length = int(text_lengths.max().item())
                if text_length > 0:
                    text_token_ids = text_token_ids[:, :text_length]
                    text_attention_mask = text_attention_mask[:, :text_length]
                else:
                    text_token_ids = None
                    text_attention_mask = None

        B = hidden_states.shape[0]
        L_latent = hidden_states.shape[1]
        dev = hidden_states.device

        # 1. Embed text context
        if text_token_ids is not None:
            text_embeds = self.embed_tokens(text_token_ids)
            L_ctx = text_embeds.shape[1]
        else:
            L_ctx = 0
            text_attention_mask = None

        # 2. SOI / EOI boundary tokens
        soi_ids = torch.full((B, 1), self.config.start_of_image_id, dtype=torch.long, device=dev)
        eoi_ids = torch.full((B, 1), self.config.end_of_image_id, dtype=torch.long, device=dev)
        soi_emb = self.embed_tokens(soi_ids)
        eoi_emb = self.embed_tokens(eoi_ids)

        # 3. Latent projection
        t_emb = self.time_embedder(timestep)
        pos_emb = self.latent_pos_embed(latent_pos_ids)
        latent_embeds = self.vae2llm(hidden_states) + t_emb.unsqueeze(1) + pos_emb
        latent_embeds = latent_embeds.to(soi_emb.dtype)

        # 4. Sequence: [text?, soi, latent_0..N, eoi]
        L_total = L_ctx + 1 + L_latent + 1
        if L_ctx > 0:
            sequence = torch.cat([text_embeds, soi_emb, latent_embeds, eoi_emb], dim=1)
        else:
            sequence = torch.cat([soi_emb, latent_embeds, eoi_emb], dim=1)

        # 5. MoT routing masks
        #    text pathway: text_ctx + soi + eoi
        #    gen pathway:  latent tokens only
        text_mask = torch.zeros(B, L_total, dtype=torch.bool, device=dev)
        text_mask[:, : L_ctx + 1] = True  # text + soi
        text_mask[:, -1] = True  # eoi
        latent_mask = ~text_mask

        # 6. RoPE positions
        if L_ctx > 0:
            ctx_pos = torch.arange(L_ctx, device=dev)
            img_pos = ctx_pos.new_full((1 + L_latent + 1,), L_ctx)
            position_ids = torch.cat([ctx_pos, img_pos]).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.zeros(1, L_total, dtype=torch.long, device=dev).expand(B, -1)
        cos, sin = self.rotary_emb(position_ids)

        # Key padding mask: zero-padded text tokens in uneven micro-batches
        # must not attend to image queries.  ``None`` keeps the flash backend.
        if L_ctx > 0 and text_attention_mask is not None and not bool(text_attention_mask.all()):
            key_padding_mask = text_attention_mask.new_ones(B, L_total, dtype=torch.bool)
            key_padding_mask[:, :L_ctx] = text_attention_mask
        else:
            key_padding_mask = None

        # 7. Transformer layers (split attention: text causal + image full)
        for layer in self.layers:

            def _layer_fn(seq, cos_, sin_, text_mask_, latent_mask_, kpm, *, _layer=layer):
                return _layer(seq, cos_, sin_, text_mask_, latent_mask_, L_ctx, key_padding_mask=kpm)

            sequence = self._checkpointed_call(_layer_fn, sequence, cos, sin, text_mask, latent_mask, key_padding_mask)

        # 8. Final norm with MoT routing
        normed = sequence.new_zeros(sequence.shape)
        t_idx = text_mask.nonzero(as_tuple=True)
        l_idx = latent_mask.nonzero(as_tuple=True)
        normed[t_idx] = self.norm(sequence[t_idx])
        normed[l_idx] = self.norm_moe_gen(sequence[l_idx])

        # 9. Extract latent output
        latent_output = normed[:, L_ctx + 1 : L_ctx + 1 + L_latent, :]
        velocity = self.llm2vae(latent_output)

        return (velocity,)

    # ------------------------------------------------------------------
    #  Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(cls, model_path: str, torch_dtype=torch.bfloat16) -> BagelForTraining:
        """Load pretrained weights from ``ema.safetensors``.

        Args:
            model_path: Directory containing ``config.json`` and ``ema.safetensors``.
            torch_dtype: Target dtype (default ``bfloat16``).

        Returns:
            BagelForTraining instance with loaded weights.
        """
        config = BagelTrainingConfig.from_model_path(model_path)
        ckpt_path = os.path.join(model_path, "ema.safetensors")
        from safetensors.torch import load_file

        state_dict = load_file(ckpt_path)

        if "latent_pos_embed.pos_embed" in state_dict:
            actual_len = state_dict["latent_pos_embed.pos_embed"].shape[0]
            grid = int(actual_len**0.5)
            if grid * grid == actual_len and grid != config.max_latent_size:
                config.max_latent_size = grid

        model = cls(config)
        mapped = _map_checkpoint_to_training(state_dict, config)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            import logging

            logging.getLogger(__name__).warning(f"Missing keys when loading BagelForTraining: {len(missing)} keys")

        model = model.to(torch_dtype)
        return model


def _map_checkpoint_to_training(state_dict: dict[str, Tensor], config: BagelTrainingConfig) -> dict:
    """Map ``ema.safetensors`` keys to ``BagelForTraining`` parameter names.

    Args:
        state_dict: Raw checkpoint state dict.
        config: Training config (unused, reserved for future key remapping).

    Returns:
        Dict with keys matching ``BagelForTraining`` parameters.
    """
    mapped: dict[str, Tensor] = {}
    for src_key, tensor in state_dict.items():
        dst_key: str | None = None
        if src_key.startswith("language_model.model."):
            dst_key = src_key[len("language_model.model.") :]
        elif src_key.startswith("language_model."):
            continue
        elif src_key.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
            dst_key = src_key
        if dst_key is not None:
            mapped[dst_key] = tensor
    return mapped
