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
"""Repo-native BAGEL atomic SFT primitives.

This module adds the tokenizer-independent model boundary used by the visual
reflection RFC while preserving :class:`BagelForTraining` as the existing
FlowGRPO compatibility surface.  The three forwards consume already-tokenized
homogeneous micro-batches and already-normalized image tensors; data loading,
tokenization, and task sampling remain trainer responsibilities.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .bagel_model import (
    BagelForTraining,
    BagelMLP,
    BagelMoTAttention,
    BagelMoTLayer,
    BagelTrainingConfig,
    PositionEmbedding,
    RMSNorm,
    RotaryEmbedding,
    TimestepEmbedder,
    _apply_rotary_emb,
)

SFTTask = Literal["t2i", "reflect", "edit"]


@dataclass
class BagelSFTConfig(BagelTrainingConfig):
    """BAGEL SFT-only configuration without widening the FlowGRPO loader."""

    timestep_shift: float = 1.0
    vit_patch_size: int = 14
    vit_hidden_size: int = 1152
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"
    text_start_id: int = 151644  # <|im_start|>
    text_end_id: int = 151645  # <|im_end|>

    @classmethod
    def from_model_path(cls, model_path: str) -> BagelSFTConfig:
        """Load generation and understanding settings from BAGEL artifacts."""
        with open(os.path.join(model_path, "config.json")) as f:
            root_cfg = json.load(f)

        llm: dict[str, Any] = {}
        llm_path = os.path.join(model_path, "llm_config.json")
        if os.path.isfile(llm_path):
            with open(llm_path) as f:
                llm.update(json.load(f))
        llm.update(root_cfg.get("llm_config", {}))

        vit: dict[str, Any] = {}
        vit_path = os.path.join(model_path, "vit_config.json")
        if os.path.isfile(vit_path):
            with open(vit_path) as f:
                vit.update(json.load(f))
        vit.update(root_cfg.get("vit_config", {}))

        if not llm:
            raise ValueError("BAGEL SFT requires llm_config.json or config.json.llm_config")
        if not vit:
            raise ValueError("BAGEL SFT requires vit_config.json or config.json.vit_config")
        if bool(llm.get("tie_word_embeddings", False)):
            raise ValueError("BAGEL SFT requires an untied lm_head checkpoint")

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
            timestep_shift=root_cfg.get("timestep_shift", 1.0),
            vit_patch_size=vit.get("patch_size", 14),
            vit_hidden_size=vit.get("hidden_size", 1152),
            vit_max_num_patch_per_side=root_cfg.get("vit_max_num_patch_per_side", 70),
            connector_act=root_cfg.get("connector_act", "gelu_pytorch_tanh"),
        )


@dataclass
class BagelSFTOutput:
    """Output shared by the three atomic SFT primitives."""

    task: SFTTask
    loss: Tensor
    loss_per_sample: Tensor
    logits: Tensor | None = None
    velocity: Tensor | None = None
    target: Tensor | None = None
    valid_mask: Tensor | None = None
    hidden_states: Tensor | None = None
    context_length: int | None = None


@dataclass(frozen=True)
class BagelCheckpointLoadReport:
    """Fail-closed summary of the tensors loaded from BAGEL checkpoints."""

    loaded_model_keys: tuple[str, ...]
    ignored_model_keys: tuple[str, ...]
    loaded_vae_keys: tuple[str, ...]


class BagelMLPConnector(nn.Module):
    """Two-layer visual connector with checkpoint-compatible parameter names."""

    def __init__(self, input_dim: int, output_dim: int, activation: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "gelu_pytorch_tanh":
            self.act = nn.GELU(approximate="tanh")
        else:
            raise ValueError(f"unsupported BAGEL connector activation: {activation!r}")
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(hidden_states)))


class BagelFrozenVAEEncoder:
    """Non-registered frozen VAE boundary for image-to-latent convenience paths."""

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad = False

    def eval(self) -> BagelFrozenVAEEncoder:
        self.module.eval()
        return self

    def _apply(self, fn) -> BagelFrozenVAEEncoder:
        def preserve_dtype(tensor: Tensor) -> Tensor:
            original_dtype = tensor.dtype
            converted = fn(tensor)
            if (tensor.is_floating_point() or tensor.is_complex()) and converted.dtype != original_dtype:
                converted = tensor.to(device=converted.device, dtype=original_dtype)
            return converted

        self.module._apply(preserve_dtype)
        self.eval()
        return self

    def parameters(self):
        return self.module.parameters()

    def state_dict(self) -> dict[str, Tensor]:
        return self.module.state_dict()

    def encode(self, images: Tensor) -> Tensor:
        with torch.no_grad():
            encoded = self.module.encode(images)
        if not isinstance(encoded, Tensor):
            raise TypeError("frozen VAE encode must return a Tensor")
        return encoded


class BagelSiglipVisionTower(nn.Module):
    """NaViT-style packed wrapper around a Transformers SigLIP vision model."""

    def __init__(self, vit_model: nn.Module, *, patch_size: int, max_num_patch_per_side: int) -> None:
        super().__init__()
        self.vision_model = vit_model.vision_model if hasattr(vit_model, "vision_model") else vit_model
        self.patch_size = patch_size
        self.max_num_patch_per_side = max_num_patch_per_side

    def forward(self, pixel_values: Tensor, patch_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape (B, 3, H, W)")
        batch_size, _, height, width = pixel_values.shape
        patch_size = self.patch_size
        if height % patch_size or width % patch_size:
            raise ValueError("vision image dimensions must be divisible by vit_patch_size")
        grid_h, grid_w = height // patch_size, width // patch_size
        if grid_h > self.max_num_patch_per_side or grid_w > self.max_num_patch_per_side:
            raise ValueError("vision image exceeds vit_max_num_patch_per_side")

        vision_model = self.vision_model
        patch_embed = vision_model.embeddings.patch_embedding
        raw_patches = pixel_values.reshape(batch_size, 3, grid_h, patch_size, grid_w, patch_size)
        if patch_embed.weight.ndim == 4:
            patches = torch.einsum("bchpwq->bhwcpq", raw_patches)
            weight = patch_embed.weight.reshape(patch_embed.weight.shape[0], -1)
        elif patch_embed.weight.ndim == 2:
            patches = torch.einsum("bchpwq->bhwpqc", raw_patches)
            weight = patch_embed.weight
        else:
            raise ValueError("SigLIP patch embedding weight must be rank 2 or 4")
        packed_patches = patches.reshape(batch_size * grid_h * grid_w, -1).to(weight.dtype)
        hidden_states = F.linear(packed_patches, weight, patch_embed.bias)

        one_position_ids = (
            torch.arange(grid_h, device=pixel_values.device)[:, None] * self.max_num_patch_per_side
            + torch.arange(grid_w, device=pixel_values.device)[None, :]
        ).reshape(-1)
        packed_position_ids = one_position_ids.repeat(batch_size)
        if not hasattr(vision_model.embeddings, "position_embedding"):
            raise ValueError("BAGEL SFT currently requires SigLIP absolute position embeddings")
        hidden_states = hidden_states + vision_model.embeddings.position_embedding(packed_position_ids)
        hidden_states = hidden_states.unsqueeze(0)

        tokens_per_image = grid_h * grid_w
        valid_mask = _normalize_patch_mask(
            patch_mask,
            expected=(batch_size, tokens_per_image),
            device=pixel_values.device,
            name="patch_mask",
            role="",
        )
        total_tokens = batch_size * tokens_per_image
        attention_mask = torch.full(
            (1, 1, total_tokens, total_tokens),
            torch.finfo(hidden_states.dtype).min,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        for batch_index in range(batch_size):
            start = batch_index * tokens_per_image
            end = start + tokens_per_image
            valid_keys = valid_mask[batch_index]
            allowed = valid_keys.unsqueeze(0).expand(tokens_per_image, -1).clone()
            invalid_queries = ~valid_keys
            if bool(invalid_queries.any()):
                invalid_indexes = invalid_queries.nonzero(as_tuple=True)[0]
                allowed[invalid_indexes] = False
                allowed[invalid_indexes, invalid_indexes] = True
            attention_mask[..., start:end, start:end].masked_fill_(allowed, 0)
        outputs = vision_model.encoder(inputs_embeds=hidden_states, attention_mask=attention_mask)
        encoded = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if not hasattr(vision_model, "post_layernorm"):
            raise ValueError("SigLIP vision model is missing post_layernorm")
        encoded = vision_model.post_layernorm(encoded)
        embeddings = encoded.squeeze(0).reshape(batch_size, tokens_per_image, -1)
        return embeddings, one_position_ids.unsqueeze(0).expand(batch_size, -1)


class _BagelSFTMoTAttention(BagelMoTAttention):
    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is None:
            return super().forward(
                hidden_states,
                cos,
                sin,
                text_mask,
                latent_mask,
                L_ctx,
                key_padding_mask=key_padding_mask,
            )
        if L_ctx or key_padding_mask is not None:
            raise ValueError("SFT attention_mask cannot be combined with FlowGRPO split attention")

        batch_size, sequence_length, _ = hidden_states.shape
        text_index = text_mask.nonzero(as_tuple=True)
        latent_index = latent_mask.nonzero(as_tuple=True)

        query = hidden_states.new_zeros(batch_size, sequence_length, self.num_heads * self.head_dim)
        key = hidden_states.new_zeros(batch_size, sequence_length, self.num_kv_heads * self.head_dim)
        value = hidden_states.new_zeros(batch_size, sequence_length, self.num_kv_heads * self.head_dim)

        text_hidden = hidden_states[text_index]
        query[text_index] = self.q_proj(text_hidden)
        key[text_index] = self.k_proj(text_hidden)
        value[text_index] = self.v_proj(text_hidden)

        latent_hidden = hidden_states[latent_index]
        query[latent_index] = self.q_proj_moe_gen(latent_hidden)
        key[latent_index] = self.k_proj_moe_gen(latent_hidden)
        value[latent_index] = self.v_proj_moe_gen(latent_hidden)

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).float()
        key = key.view(batch_size, sequence_length, self.num_kv_heads, self.head_dim).float()
        value = value.view(batch_size, sequence_length, self.num_kv_heads, self.head_dim)
        normalized_query = query.new_zeros(query.shape)
        normalized_key = key.new_zeros(key.shape)
        normalized_query[text_index] = self.q_norm(query[text_index])
        normalized_key[text_index] = self.k_norm(key[text_index])
        normalized_query[latent_index] = self.q_norm_moe_gen(query[latent_index])
        normalized_key[latent_index] = self.k_norm_moe_gen(key[latent_index])

        normalized_query, normalized_key = _apply_rotary_emb(
            normalized_query,
            normalized_key,
            cos.unsqueeze(2),
            sin.unsqueeze(2),
        )
        normalized_query = normalized_query.to(torch.bfloat16)
        normalized_key = normalized_key.to(torch.bfloat16)
        value = value.to(torch.bfloat16)

        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            normalized_key = normalized_key.unsqueeze(3).expand(-1, -1, -1, repeats, -1)
            normalized_key = normalized_key.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
            value = value.unsqueeze(3).expand(-1, -1, -1, repeats, -1)
            value = value.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)

        if attention_mask.ndim == 3:
            attention_mask = attention_mask.unsqueeze(1)
        if attention_mask.ndim != 4 or attention_mask.shape[0] != batch_size:
            raise ValueError("attention_mask must have shape (B, L, L) or (B, 1, L, L)")
        if attention_mask.shape[-2:] != (sequence_length, sequence_length):
            raise ValueError(
                f"attention_mask has trailing shape {tuple(attention_mask.shape[-2:])}, "
                f"expected {(sequence_length, sequence_length)}"
            )

        attention_output = F.scaled_dot_product_attention(
            normalized_query.transpose(1, 2),
            normalized_key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=attention_mask,
            is_causal=False,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)

        output = hidden_states.new_zeros(batch_size, sequence_length, self.hidden_size)
        output[text_index] = self.o_proj(attention_output[text_index].to(self.o_proj.weight.dtype))
        output[latent_index] = self.o_proj_moe_gen(attention_output[latent_index].to(self.o_proj_moe_gen.weight.dtype))
        return output


class _BagelSFTMoTLayer(BagelMoTLayer):
    def __init__(self, config: BagelTrainingConfig) -> None:
        nn.Module.__init__(self)
        self.self_attn = _BagelSFTMoTAttention(config)
        self.mlp = BagelMLP(config.hidden_size, config.intermediate_size)
        self.mlp_moe_gen = BagelMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        L_ctx: int = 0,
        key_padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if attention_mask is None:
            return super().forward(
                hidden_states,
                position_ids,
                text_mask,
                latent_mask,
                L_ctx,
                key_padding_mask=key_padding_mask,
            )

        cos, sin = self.rotary_emb(position_ids)
        text_index = text_mask.nonzero(as_tuple=True)
        latent_index = latent_mask.nonzero(as_tuple=True)
        normalized = hidden_states.new_zeros(hidden_states.shape)
        normalized[text_index] = self.input_layernorm(hidden_states[text_index])
        normalized[latent_index] = self.input_layernorm_moe_gen(hidden_states[latent_index])
        attention_output = self.self_attn(
            normalized,
            cos,
            sin,
            text_mask,
            latent_mask,
            L_ctx,
            key_padding_mask=key_padding_mask,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + attention_output

        residual = hidden_states
        mlp_output = hidden_states.new_zeros(hidden_states.shape)
        mlp_output[text_index] = self.mlp(self.post_attention_layernorm(hidden_states[text_index]))
        mlp_output[latent_index] = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(hidden_states[latent_index]))
        return residual + mlp_output


class BagelForSFT(BagelForTraining):
    """BAGEL MoT model exposing atomic T2I, reflection, and edit losses."""

    _no_split_modules = ["_BagelSFTMoTLayer"]

    def __init__(
        self,
        config: BagelSFTConfig,
        *,
        vision_tower: nn.Module | None = None,
        vae_encoder: BagelFrozenVAEEncoder | nn.Module | None = None,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([_BagelSFTMoTLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_moe_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.time_embedder = TimestepEmbedder(config.hidden_size)
        self.vae2llm = nn.Linear(config.patch_latent_dim, config.hidden_size)
        self.llm2vae = nn.Linear(config.hidden_size, config.patch_latent_dim)
        self.latent_pos_embed = PositionEmbedding(config.max_latent_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vit_model = vision_tower
        self.connector = BagelMLPConnector(config.vit_hidden_size, config.hidden_size, config.connector_act)
        self.vit_pos_embed = PositionEmbedding(config.vit_max_num_patch_per_side, config.hidden_size)
        if isinstance(vae_encoder, nn.Module):
            vae_encoder = BagelFrozenVAEEncoder(vae_encoder)
        self.vae_encoder = vae_encoder
        self.checkpoint_load_report: BagelCheckpointLoadReport | None = None

    def _apply(self, fn, recurse: bool = True) -> BagelForSFT:
        super()._apply(fn, recurse=recurse)
        if self.vae_encoder is not None:
            self.vae_encoder._apply(fn)
        return self

    def train(self, mode: bool = True) -> BagelForSFT:
        super().train(mode)
        if self.vae_encoder is not None:
            self.vae_encoder.eval()
        return self

    def forward(self, task: SFTTask, **kwargs: Any) -> BagelSFTOutput:
        """Dispatch one homogeneous micro-batch to its typed primitive."""
        if task == "t2i":
            return self.forward_t2i(**kwargs)
        if task == "reflect":
            return self.forward_reflect(**kwargs)
        if task == "edit":
            return self.forward_edit(**kwargs)
        raise ValueError(f"unsupported BAGEL SFT task: {task!r}")

    def forward_t2i(
        self,
        *,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor | None,
        timestep_logits: Tensor,
        target_images: Tensor | None = None,
        target_latents: Tensor | None = None,
        noise: Tensor | None = None,
        target_patch_mask: Tensor | None = None,
    ) -> BagelSFTOutput:
        """Compute T2I loss from normal logits mapped through sigmoid and timestep shift."""
        prompt_mask = _framed_token_mask(prompt_input_ids, prompt_attention_mask, self.config, name="prompt")
        target_latents = self._resolve_latents(target_images, target_latents, name="target")
        target_patches, latent_pos_ids = self._patchify_latents(target_latents)
        valid_mask = _patch_mask(target_patch_mask, target_patches)
        noisy_patches, target_velocity, shifted_timesteps = self._prepare_flow_target(
            target_patches, timestep_logits=timestep_logits, noise=noise
        )
        model_dtype = self.vae2llm.weight.dtype
        noisy_patches = noisy_patches.to(model_dtype)
        if bool(prompt_mask.all()) and bool(valid_mask.all()):
            velocity = BagelForTraining.forward(
                self,
                hidden_states=noisy_patches,
                timestep=shifted_timesteps,
                text_token_ids=prompt_input_ids,
                text_attention_mask=prompt_mask,
                latent_pos_ids=latent_pos_ids,
            )[0]
        else:
            velocity = self._forward_t2i_sft_sequence(
                prompt_input_ids=prompt_input_ids,
                prompt_mask=prompt_mask,
                noisy_patches=noisy_patches,
                shifted_timesteps=shifted_timesteps,
                latent_pos_ids=latent_pos_ids,
                valid_mask=valid_mask,
            )
        loss, per_sample = _normalized_mse(velocity, target_velocity, valid_mask)
        return BagelSFTOutput(
            task="t2i",
            loss=loss,
            loss_per_sample=per_sample,
            velocity=velocity,
            target=target_velocity,
            valid_mask=valid_mask,
        )

    def _forward_t2i_sft_sequence(
        self,
        *,
        prompt_input_ids: Tensor,
        prompt_mask: Tensor,
        noisy_patches: Tensor,
        shifted_timesteps: Tensor,
        latent_pos_ids: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        batch_size, latent_length, _ = noisy_patches.shape
        if prompt_input_ids.shape[0] != batch_size:
            raise ValueError("T2I prompt and target tensors must have the same batch size")
        prompt_length = prompt_input_ids.shape[1]
        latent_start = prompt_length + 1
        sequence_length = latent_start + latent_length + 1

        prompt_embeddings = self.embed_tokens(prompt_input_ids)
        latent_embeddings = (
            self.vae2llm(noisy_patches)
            + self.time_embedder(shifted_timesteps).unsqueeze(1)
            + self.latent_pos_embed(latent_pos_ids)
        ).to(prompt_embeddings.dtype)
        sequence = torch.cat(
            [
                prompt_embeddings,
                self._marker_embedding(self.config.start_of_image_id, batch_size, noisy_patches.device),
                latent_embeddings,
                self._marker_embedding(self.config.end_of_image_id, batch_size, noisy_patches.device),
            ],
            dim=1,
        )

        text_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=sequence.device)
        text_mask[:, :prompt_length] = prompt_mask
        text_mask[:, prompt_length] = True
        text_mask[:, -1] = True
        latent_mask = torch.zeros_like(text_mask)
        latent_mask[:, latent_start : latent_start + latent_length] = valid_mask
        sequence_valid = text_mask | latent_mask

        prompt_positions = torch.arange(prompt_length, device=sequence.device).unsqueeze(0).expand(batch_size, -1)
        image_positions = prompt_mask.sum(dim=-1, dtype=torch.long).unsqueeze(1)
        image_positions = image_positions.expand(-1, latent_length + 2)
        position_ids = torch.cat([prompt_positions, image_positions], dim=1)
        attention_mask = _segment_attention_mask(
            sequence_valid,
            [(0, prompt_length, "causal"), (prompt_length, sequence_length, "noise")],
        )
        hidden_states = self._run_sft_sequence(
            sequence,
            position_ids=position_ids,
            text_mask=text_mask,
            latent_mask=latent_mask,
            valid_mask=sequence_valid,
            attention_mask=attention_mask,
        )
        return self.llm2vae(hidden_states[:, latent_start : latent_start + latent_length])

    def forward_reflect(
        self,
        *,
        prefix_input_ids: Tensor,
        prefix_attention_mask: Tensor | None,
        response_input_ids: Tensor,
        response_labels: Tensor,
        response_loss_mask: Tensor | None,
        current_vit_pixel_values: Tensor,
        current_vit_patch_mask: Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> BagelSFTOutput:
        """Predict only the serialized reflection response and EOS."""
        prefix_mask = _framed_token_mask(prefix_input_ids, prefix_attention_mask, self.config, name="prefix")
        response_mask = _validate_response_shift(
            response_input_ids,
            response_labels,
            response_loss_mask,
            self.config,
        )
        vision_embeddings, _, vision_valid_mask = self._encode_vision(
            current_vit_pixel_values,
            current_vit_patch_mask,
        )
        batch_size, vision_length, _ = vision_embeddings.shape
        if prefix_input_ids.shape[0] != batch_size or response_input_ids.shape[0] != batch_size:
            raise ValueError("reflect tensors must have the same batch size")

        prefix_length = prefix_input_ids.shape[1]
        response_length = response_input_ids.shape[1]
        image_start = prefix_length
        vision_start = image_start + 1
        image_end = vision_start + vision_length
        response_start = image_end + 1
        sequence_length = response_start + response_length

        sequence = self.embed_tokens.weight.new_zeros(batch_size, sequence_length, self.config.hidden_size)
        sequence[:, :prefix_length] = self.embed_tokens(prefix_input_ids)
        sequence[:, image_start : image_start + 1] = self._marker_embedding(
            self.config.start_of_image_id, batch_size, sequence.device
        )
        sequence[:, vision_start:image_end] = vision_embeddings.to(sequence.dtype)
        sequence[:, image_end : image_end + 1] = self._marker_embedding(
            self.config.end_of_image_id, batch_size, sequence.device
        )
        sequence[:, response_start:] = self.embed_tokens(response_input_ids)

        valid_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool, device=sequence.device)
        valid_mask[:, :prefix_length] = prefix_mask
        valid_mask[:, image_start] = True
        valid_mask[:, vision_start:image_end] = vision_valid_mask
        valid_mask[:, image_end] = True
        valid_mask[:, response_start:] = response_mask

        position_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=sequence.device)
        for batch_index in range(batch_size):
            prefix_tokens = int(prefix_mask[batch_index].sum().item())
            position_ids[batch_index, :prefix_length] = torch.arange(prefix_length, device=sequence.device)
            position_ids[batch_index, image_start : image_end + 1] = prefix_tokens
            position_ids[batch_index, response_start:] = (
                prefix_tokens + 1 + torch.arange(response_length, device=sequence.device)
            )

        segments = [
            (0, image_start, "causal"),
            (image_start, response_start, "full"),
            (response_start, sequence_length, "causal"),
        ]
        attention_mask = _segment_attention_mask(valid_mask, segments)
        hidden_states = self._run_sft_sequence(
            sequence,
            position_ids=position_ids,
            text_mask=valid_mask,
            latent_mask=torch.zeros_like(valid_mask),
            valid_mask=valid_mask,
            attention_mask=attention_mask,
        )
        predictor_hidden = hidden_states[:, response_start:]
        logits = self.lm_head(predictor_hidden)
        loss, per_sample = _normalized_cross_entropy(logits, response_labels, response_mask)
        return BagelSFTOutput(
            task="reflect",
            loss=loss,
            loss_per_sample=per_sample,
            logits=logits,
            target=response_labels,
            valid_mask=response_mask,
            hidden_states=hidden_states if return_hidden_states else None,
            context_length=response_start,
        )

    def forward_edit(
        self,
        *,
        prompt_input_ids: Tensor,
        prompt_attention_mask: Tensor | None,
        edit_input_ids: Tensor,
        edit_attention_mask: Tensor | None,
        current_vit_pixel_values: Tensor,
        timestep_logits: Tensor,
        current_vae_images: Tensor | None = None,
        current_latents: Tensor | None = None,
        current_patch_mask: Tensor | None = None,
        current_vit_patch_mask: Tensor | None = None,
        target_images: Tensor | None = None,
        target_latents: Tensor | None = None,
        noise: Tensor | None = None,
        target_patch_mask: Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> BagelSFTOutput:
        """Condition a target image on prompt, clean source image, and delta Edit."""
        prompt_mask = _framed_token_mask(prompt_input_ids, prompt_attention_mask, self.config, name="prompt")
        edit_mask = _framed_token_mask(edit_input_ids, edit_attention_mask, self.config, name="edit")
        clean_latents = self._resolve_latents(current_vae_images, current_latents, name="current")
        target_latents = self._resolve_latents(target_images, target_latents, name="target")
        clean_patches, clean_position_ids = self._patchify_latents(clean_latents)
        target_patches, target_position_ids = self._patchify_latents(target_latents)
        clean_valid_mask = _patch_mask(
            current_patch_mask,
            clean_patches,
            name="current_patch_mask",
            role="current",
        )
        valid_mask = _patch_mask(target_patch_mask, target_patches)
        noisy_target, target_velocity, shifted_timesteps = self._prepare_flow_target(
            target_patches, timestep_logits=timestep_logits, noise=noise
        )
        vision_embeddings, _, vision_valid_mask = self._encode_vision(
            current_vit_pixel_values,
            current_vit_patch_mask,
        )

        batch_size = prompt_input_ids.shape[0]
        if any(
            tensor.shape[0] != batch_size
            for tensor in (edit_input_ids, clean_patches, target_patches, vision_embeddings)
        ):
            raise ValueError("edit tensors must have the same batch size")

        prompt_length = prompt_input_ids.shape[1]
        clean_length = clean_patches.shape[1]
        vision_length = vision_embeddings.shape[1]
        edit_length = edit_input_ids.shape[1]
        target_length = target_patches.shape[1]

        clean_block_start = prompt_length
        clean_latent_start = clean_block_start + 1
        clean_block_end = clean_latent_start + clean_length + 1
        vision_block_start = clean_block_end
        vision_token_start = vision_block_start + 1
        vision_block_end = vision_token_start + vision_length + 1
        edit_start = vision_block_end
        target_block_start = edit_start + edit_length
        target_latent_start = target_block_start + 1
        target_block_end = target_latent_start + target_length + 1

        sequence = self.embed_tokens.weight.new_zeros(batch_size, target_block_end, self.config.hidden_size)
        sequence[:, :prompt_length] = self.embed_tokens(prompt_input_ids)
        sequence[:, clean_block_start : clean_block_start + 1] = self._marker_embedding(
            self.config.start_of_image_id, batch_size, sequence.device
        )
        clean_time = torch.zeros(batch_size, device=sequence.device, dtype=shifted_timesteps.dtype)
        clean_embeddings = (
            self.vae2llm(clean_patches.to(self.vae2llm.weight.dtype))
            + self.time_embedder(clean_time).unsqueeze(1)
            + self.latent_pos_embed(clean_position_ids)
        )
        sequence[:, clean_latent_start : clean_latent_start + clean_length] = clean_embeddings.to(sequence.dtype)
        sequence[:, clean_block_end - 1 : clean_block_end] = self._marker_embedding(
            self.config.end_of_image_id, batch_size, sequence.device
        )
        sequence[:, vision_block_start : vision_block_start + 1] = self._marker_embedding(
            self.config.start_of_image_id, batch_size, sequence.device
        )
        sequence[:, vision_token_start : vision_token_start + vision_length] = vision_embeddings.to(sequence.dtype)
        sequence[:, vision_block_end - 1 : vision_block_end] = self._marker_embedding(
            self.config.end_of_image_id, batch_size, sequence.device
        )
        sequence[:, edit_start : edit_start + edit_length] = self.embed_tokens(edit_input_ids)
        sequence[:, target_block_start : target_block_start + 1] = self._marker_embedding(
            self.config.start_of_image_id, batch_size, sequence.device
        )
        target_embeddings = (
            self.vae2llm(noisy_target.to(self.vae2llm.weight.dtype))
            + self.time_embedder(shifted_timesteps).unsqueeze(1)
            + self.latent_pos_embed(target_position_ids)
        )
        sequence[:, target_latent_start : target_latent_start + target_length] = target_embeddings.to(sequence.dtype)
        sequence[:, target_block_end - 1 : target_block_end] = self._marker_embedding(
            self.config.end_of_image_id, batch_size, sequence.device
        )

        sequence_valid = torch.zeros(batch_size, target_block_end, dtype=torch.bool, device=sequence.device)
        sequence_valid[:, :prompt_length] = prompt_mask
        sequence_valid[:, clean_block_start] = True
        sequence_valid[:, clean_latent_start : clean_latent_start + clean_length] = clean_valid_mask
        sequence_valid[:, clean_block_end - 1] = True
        sequence_valid[:, vision_block_start] = True
        sequence_valid[:, vision_token_start : vision_token_start + vision_length] = vision_valid_mask
        sequence_valid[:, vision_block_end - 1] = True
        sequence_valid[:, edit_start : edit_start + edit_length] = edit_mask
        sequence_valid[:, target_block_start] = True
        sequence_valid[:, target_latent_start : target_latent_start + target_length] = valid_mask
        sequence_valid[:, target_block_end - 1] = True

        text_mask = sequence_valid.clone()
        latent_mask = torch.zeros_like(sequence_valid)
        latent_mask[:, clean_latent_start : clean_latent_start + clean_length] = clean_valid_mask
        latent_mask[:, target_latent_start : target_latent_start + target_length] = valid_mask
        text_mask[latent_mask] = False

        position_ids = torch.zeros(batch_size, target_block_end, dtype=torch.long, device=sequence.device)
        for batch_index in range(batch_size):
            prompt_tokens = int(prompt_mask[batch_index].sum().item())
            edit_tokens = int(edit_mask[batch_index].sum().item())
            position_ids[batch_index, :prompt_length] = torch.arange(prompt_length, device=sequence.device)
            position_ids[batch_index, clean_block_start:clean_block_end] = prompt_tokens
            position_ids[batch_index, vision_block_start:vision_block_end] = prompt_tokens + 1
            position_ids[batch_index, edit_start : edit_start + edit_length] = (
                prompt_tokens + 2 + torch.arange(edit_length, device=sequence.device)
            )
            position_ids[batch_index, target_block_start:target_block_end] = prompt_tokens + 2 + edit_tokens

        segments = [
            (0, prompt_length, "causal"),
            (clean_block_start, clean_block_end, "full"),
            (vision_block_start, vision_block_end, "full"),
            (edit_start, edit_start + edit_length, "causal"),
            (target_block_start, target_block_end, "noise"),
        ]
        attention_mask = _segment_attention_mask(sequence_valid, segments)
        hidden_states = self._run_sft_sequence(
            sequence,
            position_ids=position_ids,
            text_mask=text_mask,
            latent_mask=latent_mask,
            valid_mask=sequence_valid,
            attention_mask=attention_mask,
        )
        target_hidden = hidden_states[:, target_latent_start : target_latent_start + target_length]
        velocity = self.llm2vae(target_hidden)
        loss, per_sample = _normalized_mse(velocity, target_velocity, valid_mask)
        return BagelSFTOutput(
            task="edit",
            loss=loss,
            loss_per_sample=per_sample,
            velocity=velocity,
            target=target_velocity,
            valid_mask=valid_mask,
            hidden_states=hidden_states if return_hidden_states else None,
            context_length=target_block_start,
        )

    def _run_sft_sequence(
        self,
        sequence: Tensor,
        *,
        position_ids: Tensor,
        text_mask: Tensor,
        latent_mask: Tensor,
        valid_mask: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if text_mask.shape != sequence.shape[:2] or latent_mask.shape != text_mask.shape:
            raise ValueError("routing masks must match the first two sequence dimensions")
        if valid_mask.shape != text_mask.shape:
            raise ValueError("valid_mask must match routing masks")
        if bool((text_mask & latent_mask).any()):
            raise ValueError("text and latent routing masks must be disjoint")
        if not torch.equal(text_mask | latent_mask, valid_mask):
            raise ValueError("each valid token must use exactly one MoT route")
        if attention_mask.shape != (sequence.shape[0], 1, sequence.shape[1], sequence.shape[1]):
            raise ValueError("attention_mask must have shape (B, 1, L, L)")
        for layer in self.layers:

            def _layer_fn(seq, pos, text, latent, mask, *, _layer=layer):
                return _layer(seq, pos, text, latent, 0, attention_mask=mask)

            sequence = self._checkpointed_call(
                _layer_fn, sequence, position_ids, text_mask, latent_mask, attention_mask
            )

        normalized = sequence.new_zeros(sequence.shape)
        text_index = text_mask.nonzero(as_tuple=True)
        latent_index = latent_mask.nonzero(as_tuple=True)
        normalized[text_index] = self.norm(sequence[text_index])
        normalized[latent_index] = self.norm_moe_gen(sequence[latent_index])
        return normalized

    def _marker_embedding(self, token_id: int, batch_size: int, device: torch.device) -> Tensor:
        token_ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)
        return self.embed_tokens(token_ids)

    def _resolve_latents(self, images: Tensor | None, latents: Tensor | None, *, name: str) -> Tensor:
        if (images is None) == (latents is None):
            raise ValueError(f"exactly one of {name}_images and {name}_latents must be provided")
        if latents is not None:
            return _validate_latents(latents, self.config.latent_channel, name=name)
        if self.vae_encoder is None:
            raise RuntimeError(f"{name}_images require a configured frozen VAE")
        encoded = self.vae_encoder.encode(images)
        return _validate_latents(encoded, self.config.latent_channel, name=name)

    def _patchify_latents(self, latents: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, channels, height, width = latents.shape
        patch_size = self.config.latent_patch_size
        if height % patch_size or width % patch_size:
            raise ValueError("latent dimensions must be divisible by latent_patch_size")
        grid_h, grid_w = height // patch_size, width // patch_size
        if grid_h > self.config.max_latent_size or grid_w > self.config.max_latent_size:
            raise ValueError("latent grid exceeds max_latent_size")
        patches = latents.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
        patches = torch.einsum("bchpwq->bhwpqc", patches).reshape(
            batch_size, grid_h * grid_w, self.config.patch_latent_dim
        )
        one_position_ids = (
            torch.arange(grid_h, device=latents.device)[:, None] * self.config.max_latent_size
            + torch.arange(grid_w, device=latents.device)[None, :]
        ).reshape(-1)
        return patches, one_position_ids.unsqueeze(0).expand(batch_size, -1)

    def _prepare_flow_target(
        self,
        clean_patches: Tensor,
        *,
        timestep_logits: Tensor,
        noise: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = clean_patches.shape[0]
        if timestep_logits.ndim != 1 or timestep_logits.shape[0] != batch_size:
            raise ValueError("timestep_logits must have shape (B,)")
        timestep_logits = timestep_logits.to(device=clean_patches.device, dtype=torch.float32)
        if not bool(torch.isfinite(timestep_logits).all()):
            raise ValueError("timestep_logits must be finite")
        if not math.isfinite(self.config.timestep_shift) or self.config.timestep_shift <= 0:
            raise ValueError("timestep_shift must be finite and positive")
        unshifted = timestep_logits.sigmoid()
        shifted = self.config.timestep_shift * unshifted / (1 + (self.config.timestep_shift - 1) * unshifted)
        if noise is None:
            noise = torch.randn_like(clean_patches)
        if noise.shape != clean_patches.shape:
            raise ValueError("noise must have the same shape as patchified target latents")
        noisy = (1 - shifted[:, None, None]) * clean_patches + shifted[:, None, None] * noise
        return noisy, noise - clean_patches, shifted

    def _encode_vision(
        self,
        pixel_values: Tensor,
        patch_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.vit_model is None:
            raise RuntimeError("visual understanding requires a configured vision tower")
        if patch_mask is None:
            output = self.vit_model(pixel_values)
        else:
            output = self.vit_model(pixel_values, patch_mask=patch_mask)
        if not isinstance(output, tuple) or len(output) != 2:
            raise TypeError("vit_model must return (embeddings, position_ids)")
        embeddings, position_ids = output
        if embeddings.ndim != 3 or embeddings.shape[-1] != self.config.vit_hidden_size:
            raise ValueError("vision embeddings have an unexpected shape")
        if position_ids.shape != embeddings.shape[:2]:
            raise ValueError("vision position_ids must match the first two embedding dimensions")
        valid_mask = _patch_mask(
            patch_mask,
            embeddings,
            name="current_vit_patch_mask",
            role="",
        )
        connected = self.connector(embeddings.to(self.connector.fc1.weight.dtype))
        connected = connected + self.vit_pos_embed(position_ids)
        return connected, position_ids, valid_mask

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs: Any,
    ) -> BagelForSFT:
        """Audit and load BAGEL SFT components from the two published checkpoints."""
        config = BagelSFTConfig.from_model_path(model_path)
        vision_tower = kwargs.pop("vision_tower", None)
        vae_model = kwargs.pop("vae_model", None)
        marker_token_ids = kwargs.pop("marker_token_ids", None)
        if kwargs:
            raise TypeError(f"unexpected BagelForSFT.from_pretrained arguments: {sorted(kwargs)}")

        ema_path = os.path.join(model_path, "ema.safetensors")
        ae_path = os.path.join(model_path, "ae.safetensors")
        if not os.path.isfile(ema_path):
            raise FileNotFoundError(f"BAGEL model checkpoint not found: {ema_path}")
        if not os.path.isfile(ae_path):
            raise FileNotFoundError(f"BAGEL VAE checkpoint not found: {ae_path}")

        from safetensors.torch import load_file

        model_state = load_file(ema_path)
        _configure_position_grids(config, model_state)
        markers = marker_token_ids or _load_marker_token_ids(model_path, config.vocab_size)
        _apply_marker_token_ids(config, markers)

        if vision_tower is None or vae_model is None:
            default_vision, default_vae = _build_default_image_components(model_path, config, model_state)
            vision_tower = vision_tower or default_vision
            vae_model = vae_model or default_vae
        model = cls(config, vision_tower=vision_tower, vae_encoder=vae_model)

        loaded_model_keys, ignored_model_keys = _load_model_checkpoint(model, model_state)
        if model.vae_encoder is None:
            raise ValueError("BAGEL SFT requires a frozen VAE encoder")
        loaded_vae_keys = _load_vae_checkpoint(model.vae_encoder.module, load_file(ae_path))
        model.checkpoint_load_report = BagelCheckpointLoadReport(
            loaded_model_keys=tuple(sorted(loaded_model_keys)),
            ignored_model_keys=tuple(sorted(ignored_model_keys)),
            loaded_vae_keys=tuple(sorted(loaded_vae_keys)),
        )
        model = model.to(torch_dtype)
        return model


def _token_mask(token_ids: Tensor, mask: Tensor | None, *, name: str) -> Tensor:
    if token_ids.ndim != 2 or token_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name}_token_ids must be a rank-2 integer tensor")
    if mask is None:
        normalized = torch.ones_like(token_ids, dtype=torch.bool)
    else:
        if mask.shape != token_ids.shape:
            raise ValueError(f"{name}_attention_mask must match token ids")
        normalized = mask.to(device=token_ids.device, dtype=torch.bool)
    if bool((normalized[:, 1:] & ~normalized[:, :-1]).any()):
        raise ValueError(f"{name}_attention_mask must use right padding")
    return normalized


def _framed_token_mask(
    token_ids: Tensor,
    mask: Tensor | None,
    config: BagelSFTConfig,
    *,
    name: str,
) -> Tensor:
    normalized = _token_mask(token_ids, mask, name=name)
    lengths = normalized.sum(dim=-1)
    if not bool((lengths >= 2).all()):
        raise ValueError(f"each {name} segment must contain BAGEL text start and end tokens")
    if not bool((token_ids[:, 0] == config.text_start_id).all()):
        raise ValueError(f"each {name} segment must start with BAGEL text_start_id")
    end_ids = token_ids.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    if not bool((end_ids == config.text_end_id).all()):
        raise ValueError(f"each {name} segment must end with BAGEL text_end_id")
    return normalized


def _validate_response_shift(
    input_ids: Tensor,
    labels: Tensor,
    loss_mask: Tensor | None,
    config: BagelSFTConfig,
) -> Tensor:
    mask = _token_mask(input_ids, loss_mask, name="response")
    if labels.shape != input_ids.shape or labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("response_labels must be a rank-2 integer tensor matching response_input_ids")
    lengths = mask.sum(dim=-1)
    if not bool((lengths >= 2).all()):
        raise ValueError("each response must contain a start token, protocol text, and supervised EOS")
    if not bool((input_ids[:, 0] == config.text_start_id).all()):
        raise ValueError("response_input_ids must start with BAGEL text_start_id")
    end_labels = labels.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
    if not bool((end_labels == config.text_end_id).all()):
        raise ValueError("the final supervised response label must be BAGEL text_end_id")
    for batch_index, length in enumerate(lengths.tolist()):
        if not torch.equal(input_ids[batch_index, 1:length], labels[batch_index, : length - 1]):
            raise ValueError("response_input_ids and response_labels must be one-token shifted")
    return mask


def _validate_latents(latents: Tensor, expected_channels: int, *, name: str) -> Tensor:
    if latents.ndim != 4 or latents.shape[1] != expected_channels:
        raise ValueError(f"{name}_latents must have shape (B, {expected_channels}, H, W)")
    return latents


def _normalize_patch_mask(
    mask: Tensor | None,
    *,
    expected: tuple[int, int],
    device: torch.device,
    name: str,
    role: str,
) -> Tensor:
    if mask is None:
        return torch.ones(expected, dtype=torch.bool, device=device)
    if mask.shape != expected:
        raise ValueError(f"{name} must have shape (B, num_patches)")
    normalized = mask.to(device=device, dtype=torch.bool)
    if not bool(normalized.any(dim=1).all()):
        qualifier = f"{role} " if role else ""
        raise ValueError(f"each sample must contain at least one valid {qualifier}patch")
    return normalized


def _patch_mask(
    mask: Tensor | None,
    patches: Tensor,
    *,
    name: str = "target_patch_mask",
    role: str = "target",
) -> Tensor:
    return _normalize_patch_mask(
        mask,
        expected=patches.shape[:2],
        device=patches.device,
        name=name,
        role=role,
    )


def _normalized_mse(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape or prediction.shape[:2] != valid_mask.shape:
        raise ValueError("MSE prediction, target, and mask shapes do not match")
    token_loss = (prediction.float() - target.float()).square().mean(dim=-1)
    counts = valid_mask.sum(dim=-1).clamp_min(1)
    masked_loss = torch.where(valid_mask, token_loss, torch.zeros_like(token_loss))
    per_sample = masked_loss.sum(dim=-1) / counts
    return per_sample.mean(), per_sample


def _normalized_cross_entropy(logits: Tensor, labels: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    if logits.shape[:2] != labels.shape or labels.shape != valid_mask.shape:
        raise ValueError("CE logits, labels, and mask shapes do not match")
    token_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1).long(), reduction="none"
    )
    token_loss = token_loss.reshape(labels.shape)
    counts = valid_mask.sum(dim=-1).clamp_min(1)
    masked_loss = torch.where(valid_mask, token_loss, torch.zeros_like(token_loss))
    per_sample = masked_loss.sum(dim=-1) / counts
    return per_sample.mean(), per_sample


def _segment_attention_mask(
    valid_mask: Tensor,
    segments: list[tuple[int, int, Literal["causal", "full", "noise"]]],
) -> Tensor:
    """Build BAGEL packed-style attention with no future target leakage."""
    batch_size, sequence_length = valid_mask.shape
    attention = torch.zeros(batch_size, sequence_length, sequence_length, dtype=torch.bool, device=valid_mask.device)
    cursor = 0
    for start, end, mode in segments:
        if start != cursor or not (start <= end <= sequence_length):
            raise ValueError("attention segments must be contiguous and cover the sequence")
        segment_length = end - start
        if segment_length:
            if start:
                attention[:, start:end, :start] = valid_mask[:, None, :start]
            local_keys = valid_mask[:, None, start:end]
            if mode == "causal":
                causal = torch.ones(segment_length, segment_length, dtype=torch.bool, device=valid_mask.device).tril()
                attention[:, start:end, start:end] = causal.unsqueeze(0) & local_keys
            elif mode in {"full", "noise"}:
                attention[:, start:end, start:end] = local_keys.expand(-1, segment_length, -1)
            else:
                raise ValueError(f"unsupported attention mode: {mode}")
        cursor = end
    if cursor != sequence_length:
        raise ValueError("attention segments must cover the full sequence")

    invalid_queries = ~valid_mask
    for batch_index in range(batch_size):
        indexes = invalid_queries[batch_index].nonzero(as_tuple=True)[0]
        if indexes.numel():
            attention[batch_index, indexes] = False
            attention[batch_index, indexes, indexes] = True
    return attention.unsqueeze(1)


_FIXED_POSITION_KEYS = {"latent_pos_embed.pos_embed", "vit_pos_embed.pos_embed"}
_MARKER_TOKENS = {
    "text_start_id": "<|im_start|>",
    "text_end_id": "<|im_end|>",
    "vision_start_id": "<|vision_start|>",
    "vision_end_id": "<|vision_end|>",
}


def _strip_checkpoint_prefix(name: str) -> str:
    for prefix in ("module.", "model."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _find_checkpoint_tensor(state_dict: dict[str, Tensor], target_name: str) -> Tensor | None:
    matches = [tensor for name, tensor in state_dict.items() if _strip_checkpoint_prefix(name) == target_name]
    if len(matches) > 1:
        raise ValueError(f"multiple checkpoint tensors map to {target_name}")
    return matches[0] if matches else None


def _configure_position_grids(config: BagelSFTConfig, state_dict: dict[str, Tensor]) -> None:
    for checkpoint_name, config_name in (
        ("latent_pos_embed.pos_embed", "max_latent_size"),
        ("vit_pos_embed.pos_embed", "vit_max_num_patch_per_side"),
    ):
        tensor = _find_checkpoint_tensor(state_dict, checkpoint_name)
        if tensor is None:
            continue
        if tensor.ndim != 2 or tensor.shape[1] != config.hidden_size:
            raise ValueError(f"fixed position tensor {checkpoint_name} has an invalid shape: {tuple(tensor.shape)}")
        side = math.isqrt(tensor.shape[0])
        if side * side != tensor.shape[0]:
            raise ValueError(f"fixed position tensor {checkpoint_name} must contain a square grid")
        setattr(config, config_name, side)


def _load_marker_token_ids(model_path: str, vocab_size: int) -> dict[str, int]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("loading BAGEL marker IDs requires Transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    marker_ids: dict[str, int] = {}
    for name, token in _MARKER_TOKENS.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        if not isinstance(token_id, int) or token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"BAGEL tokenizer marker {token!r} has invalid ID {token_id!r}")
        if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
            raise ValueError(f"BAGEL tokenizer marker {token!r} resolves to the unknown token")
        if tokenizer.encode(token, add_special_tokens=False) != [token_id]:
            raise ValueError(f"BAGEL tokenizer marker {token!r} must encode to exactly one token")
        marker_ids[name] = token_id
    return marker_ids


def _apply_marker_token_ids(config: BagelSFTConfig, marker_ids: dict[str, int]) -> None:
    missing = sorted(set(_MARKER_TOKENS) - set(marker_ids))
    unexpected = sorted(set(marker_ids) - set(_MARKER_TOKENS))
    if missing or unexpected:
        raise ValueError(f"BAGEL marker ID mismatch: missing={missing}, unexpected={unexpected}")
    values = list(marker_ids.values())
    if any(not isinstance(value, int) or value < 0 or value >= config.vocab_size for value in values):
        raise ValueError("BAGEL marker IDs must be integer vocabulary indexes")
    if len(set(values)) != len(values):
        raise ValueError("BAGEL text and vision markers must use distinct token IDs")
    config.text_start_id = marker_ids["text_start_id"]
    config.text_end_id = marker_ids["text_end_id"]
    config.start_of_image_id = marker_ids["vision_start_id"]
    config.end_of_image_id = marker_ids["vision_end_id"]


def _checkpoint_vit_layer_count(state_dict: dict[str, Tensor]) -> int:
    prefix = "vit_model.vision_model.encoder.layers."
    indexes: set[int] = set()
    for source_name in state_dict:
        name = _strip_checkpoint_prefix(source_name)
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix) :]
        index_text = suffix.split(".", 1)[0]
        if not index_text.isdigit():
            raise ValueError(f"invalid ViT layer key in BAGEL checkpoint: {source_name}")
        indexes.add(int(index_text))
    if not indexes or indexes != set(range(max(indexes) + 1)):
        raise ValueError("BAGEL checkpoint must contain contiguous ViT encoder layers starting at zero")
    return max(indexes) + 1


def _build_default_image_components(
    model_path: str,
    config: BagelSFTConfig,
    state_dict: dict[str, Tensor],
) -> tuple[BagelSiglipVisionTower, nn.Module]:
    try:
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from vllm_omni.diffusion.models.bagel.autoencoder import AutoEncoder, AutoEncoderParams
    except ImportError as exc:
        raise RuntimeError(
            "loading BAGEL image components requires the pinned Transformers and vLLM-Omni environment"
        ) from exc

    vit_config_path = os.path.join(model_path, "vit_config.json")
    vit_config = SiglipVisionConfig.from_json_file(vit_config_path)
    checkpoint_layers = _checkpoint_vit_layer_count(state_dict)
    configured_layers = vit_config.num_hidden_layers
    if checkpoint_layers != configured_layers:
        if checkpoint_layers != configured_layers - 1:
            raise ValueError(f"ViT layer count mismatch: config={configured_layers}, checkpoint={checkpoint_layers}")
        vit_config.num_hidden_layers = checkpoint_layers
    vit_config.vision_use_head = False
    vision_tower = BagelSiglipVisionTower(
        SiglipVisionModel(vit_config),
        patch_size=config.vit_patch_size,
        max_num_patch_per_side=config.vit_max_num_patch_per_side,
    )
    vae_model = AutoEncoder(
        AutoEncoderParams(
            resolution=256,
            in_channels=3,
            downsample=config.vae_downsample,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=config.latent_channel,
            scale_factor=0.3611,
            shift_factor=0.1159,
        )
    )
    return vision_tower, vae_model


def _model_checkpoint_target_name(source_name: str) -> str | None:
    name = _strip_checkpoint_prefix(source_name)
    if name.startswith("language_model.model."):
        return name[len("language_model.model.") :]
    if name.startswith("language_model.lm_head."):
        return "lm_head." + name[len("language_model.lm_head.") :]
    if name.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
        return name
    if name.startswith(("connector.", "vit_pos_embed.")):
        return name
    if name.startswith("vit_model."):
        return name
    return None


def _checkpoint_tensor_for_target(source: Tensor, target: Tensor, target_name: str) -> Tensor:
    if tuple(source.shape) == tuple(target.shape):
        return source
    if target_name.endswith(".embeddings.patch_embedding.weight"):
        if source.ndim == 2 and target.ndim == 4:
            out_channels, in_channels, patch_h, patch_w = target.shape
            if source.shape != (out_channels, patch_h * patch_w * in_channels):
                raise ValueError("published SigLIP patch weight has an invalid flattened shape")
            return source.reshape(out_channels, patch_h, patch_w, in_channels).permute(0, 3, 1, 2).contiguous()
        if source.ndim == 4 and target.ndim == 2:
            out_channels, in_channels, patch_h, patch_w = source.shape
            if target.shape != (out_channels, patch_h * patch_w * in_channels):
                raise ValueError("SigLIP Linear patch target has an invalid flattened shape")
            return source.permute(0, 2, 3, 1).reshape(target.shape).contiguous()
    raise ValueError(f"checkpoint shape mismatch for {target_name}: {tuple(source.shape)} != {tuple(target.shape)}")


def _validate_fixed_position_tensor(source_name: str, tensor: Tensor, hidden_size: int) -> None:
    if tensor.ndim != 2 or tensor.shape[1] != hidden_size:
        raise ValueError(f"fixed position tensor {source_name} has an invalid shape: {tuple(tensor.shape)}")
    side = math.isqrt(tensor.shape[0])
    if side * side != tensor.shape[0]:
        raise ValueError(f"fixed position tensor {source_name} must contain a square grid")


def _load_model_checkpoint(model: BagelForSFT, state_dict: dict[str, Tensor]) -> tuple[set[str], set[str]]:
    target_state = model.state_dict()
    mapped: dict[str, Tensor] = {}
    ignored: set[str] = set()
    for source_name, tensor in state_dict.items():
        target_name = _model_checkpoint_target_name(source_name)
        if target_name in _FIXED_POSITION_KEYS:
            _validate_fixed_position_tensor(source_name, tensor, model.config.hidden_size)
            ignored.add(source_name)
            continue
        if target_name is None:
            raise ValueError(f"unsupported BAGEL SFT checkpoint tensor: {source_name}")
        if target_name not in target_state:
            raise ValueError(f"checkpoint tensor {source_name} maps to unknown target {target_name}")
        if target_name in mapped:
            raise ValueError(f"multiple checkpoint tensors map to {target_name}")
        try:
            mapped[target_name] = _checkpoint_tensor_for_target(tensor, target_state[target_name], target_name)
        except ValueError as exc:
            raise ValueError(f"checkpoint shape mismatch for {source_name} -> {target_name}: {exc}") from exc

    required = set(target_state) - _FIXED_POSITION_KEYS
    missing = sorted(required - set(mapped))
    if missing:
        raise ValueError(f"BAGEL SFT checkpoint is missing {len(missing)} required tensors: {missing[:8]}")
    incompatible = model.load_state_dict(mapped, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    required_missing = sorted(name for name in incompatible.missing_keys if name not in _FIXED_POSITION_KEYS)
    if unexpected or required_missing:
        raise ValueError(f"BAGEL SFT model load mismatch: missing={required_missing[:8]}, unexpected={unexpected[:8]}")
    return set(mapped), ignored


def _load_vae_checkpoint(vae_model: nn.Module | None, state_dict: dict[str, Tensor]) -> set[str]:
    if vae_model is None:
        raise ValueError("cannot load ae.safetensors without a VAE module")
    target_state = vae_model.state_dict()
    mapped: dict[str, Tensor] = {}
    for source_name, tensor in state_dict.items():
        target_name = source_name
        for prefix in ("module.", "model.", "vae_model.", "vae."):
            if target_name.startswith(prefix):
                target_name = target_name[len(prefix) :]
        if target_name not in target_state:
            raise ValueError(f"unsupported BAGEL VAE checkpoint tensor: {source_name}")
        if target_name in mapped:
            raise ValueError(f"multiple VAE checkpoint tensors map to {target_name}")
        if tuple(tensor.shape) != tuple(target_state[target_name].shape):
            raise ValueError(
                f"VAE checkpoint shape mismatch for {source_name}: "
                f"{tuple(tensor.shape)} != {tuple(target_state[target_name].shape)}"
            )
        mapped[target_name] = tensor
    missing = sorted(set(target_state) - set(mapped))
    if missing:
        raise ValueError(f"BAGEL VAE checkpoint is missing {len(missing)} tensors: {missing[:8]}")
    vae_model.load_state_dict(mapped, strict=True)
    return set(mapped)


__all__ = [
    "BagelForSFT",
    "BagelSFTConfig",
    "BagelSFTOutput",
]
