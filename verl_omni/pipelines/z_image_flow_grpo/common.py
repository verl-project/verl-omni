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

"""Shared utilities for Z-Image FlowGRPO adapters."""

import torch

Z_IMAGE_VAE_SCALE_FACTOR = 8


def build_img_shapes(
    height: int, width: int, batch_size: int, vae_scale_factor: int
) -> list[list[tuple[int, int, int]]]:
    """Build per-sample image shape metadata used by Z-Image transformer patchify."""
    latent_height = height // vae_scale_factor // 2
    latent_width = width // vae_scale_factor // 2
    return [[(1, latent_height, latent_width)]] * batch_size


def apply_standard_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    guidance_scale: float,
    cfg_normalization: float = 0.0,
) -> torch.Tensor:
    """Apply standard CFG (pos + scale * (pos - neg)) with optional renormalization.

    This matches the Z-Image pipeline's CFG logic: pred = pos + scale * (pos - neg),
    with optional renormalization via cfg_normalization.

    Args:
        noise_pred: Positive prompt prediction of shape (B, ...).
        negative_noise_pred: Negative prompt prediction of shape (B, ...).
        guidance_scale: CFG scale (typically > 1.0 to enable CFG).
        cfg_normalization: Max multiplier for renormalization; 0.0 disables.

    Returns:
        Combined prediction of shape (B, ...).
    """
    pred = negative_noise_pred + guidance_scale * (noise_pred - negative_noise_pred)

    if cfg_normalization > 0.0:
        ori_pos_norm = torch.linalg.vector_norm(noise_pred.float(), dim=-1, keepdim=True)
        new_pos_norm = torch.linalg.vector_norm(pred.float(), dim=-1, keepdim=True)
        max_new_norm = ori_pos_norm * float(cfg_normalization)
        scale = torch.where(
            new_pos_norm > max_new_norm,
            (max_new_norm / new_pos_norm.clamp(min=1e-12)).to(pred.dtype),
            torch.ones_like(new_pos_norm),
        )
        pred = pred * scale

    return pred


def pad_and_unpad_prompt_embeds(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Convert padded prompt embeddings (B, L, D) to list of unpadded tensors.

    Z-Image's transformer expects per-sample text embeddings as a list.

    Args:
        prompt_embeds: Padded embeddings of shape (B, L, D).
        prompt_embeds_mask: Boolean mask of shape (B, L) where True = valid token.

    Returns:
        List of tensors, each of shape (L_i, D) where L_i is the unpadded length.
    """
    embeddings_list = []
    for i in range(prompt_embeds.shape[0]):
        mask_i = prompt_embeds_mask[i].bool()
        embeddings_list.append(prompt_embeds[i][mask_i])
    return embeddings_list
