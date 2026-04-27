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

import torch

# Default VAE downsampling factor for Z-Image
# (vae_scale_factor as defined by diffusers ZImagePipeline; the effective
# pixel-to-latent stride is ``vae_scale_factor * 2`` because the transformer
# patchifies latents with patch size 2).
Z_IMAGE_VAE_SCALE_FACTOR = 8


def split_padded_embeds_to_list(
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
) -> list[torch.Tensor]:
    """Convert a padded ``(B, L, D)`` tensor + ``(B, L)`` mask pair into the
    per-sample variable-length list expected by the Z-Image transformer.

    Args:
        prompt_embeds (torch.Tensor): Padded prompt embeddings of shape ``(B, L, D)``.
        prompt_embeds_mask (torch.Tensor): Boolean / 0-1 mask of shape ``(B, L)``
            where ``1`` marks a valid (non-padding) token.

    Returns:
        list[torch.Tensor]: ``B`` tensors of shape ``(L_i, D)``, each containing
            the valid prefix of the corresponding row.
    """
    bool_mask = prompt_embeds_mask.bool()
    return [prompt_embeds[i][bool_mask[i]] for i in range(prompt_embeds.shape[0])]


def latents_to_transformer_input(latents: torch.Tensor) -> list[torch.Tensor]:
    """Convert a ``(B, C, H, W)`` latent tensor to the per-sample list of
    4-D ``(C, F=1, H, W)`` tensors expected by ``ZImageTransformer2DModel``.

    The transformer treats videos as 4-D ``(C, F, H, W)`` per sample and images
    as a video with a single frame ``F=1``.

    Args:
        latents (torch.Tensor): Latents of shape ``(B, C, H, W)``.

    Returns:
        list[torch.Tensor]: ``B`` tensors of shape ``(C, 1, H, W)``.
    """
    return list(latents.unsqueeze(2).unbind(dim=0))


def stack_transformer_output(model_out_list: list[torch.Tensor]) -> torch.Tensor:
    """Stack the per-sample list returned by ``ZImageTransformer2DModel`` back
    into a ``(B, C, H, W)`` tensor and apply the Z-Image ``-x`` convention.

    The diffusers ZImagePipeline negates the model output before passing it to
    the scheduler (``noise_pred = -noise_pred``); we mirror that here.

    Args:
        model_out_list (list[torch.Tensor]): ``B`` tensors of shape ``(C, 1, H, W)``.

    Returns:
        torch.Tensor: ``(B, C, H, W)`` noise prediction with the Z-Image sign
            convention already applied.
    """
    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)
    noise_pred = noise_pred.squeeze(2)
    return -noise_pred


def apply_z_image_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    cfg_scale: float,
    cfg_normalization: float = 0.0,
) -> torch.Tensor:
    """Apply Z-Image style classifier-free guidance with optional renormalization.

    Implements ``pred = pos + cfg_scale * (pos - neg)``; when
    ``cfg_normalization > 0`` the resulting per-sample norm is clipped to
    ``cfg_normalization * ||pos||``.

    Args:
        noise_pred (torch.Tensor): Positive (conditional) noise prediction of
            shape ``(B, C, H, W)``.
        negative_noise_pred (torch.Tensor): Negative (unconditional) noise
            prediction with the same shape.
        cfg_scale (float): Classifier-free guidance scale.
        cfg_normalization (float, *optional*): Maximum allowed ratio between the
            CFG-combined norm and the positive-only norm. ``0`` disables
            renormalization.

    Returns:
        torch.Tensor: CFG-combined noise prediction of shape ``(B, C, H, W)``.
    """
    pos = noise_pred.float()
    neg = negative_noise_pred.float()
    pred = pos + cfg_scale * (pos - neg)

    if cfg_normalization and float(cfg_normalization) > 0.0:
        flat_pos = pos.flatten(1)
        flat_pred = pred.flatten(1)
        ori_pos_norm = torch.linalg.vector_norm(flat_pos, dim=1, keepdim=True)
        new_pos_norm = torch.linalg.vector_norm(flat_pred, dim=1, keepdim=True)
        max_new_norm = ori_pos_norm * float(cfg_normalization)
        scale = torch.where(
            new_pos_norm > max_new_norm,
            (max_new_norm / new_pos_norm.clamp(min=1e-12)).to(pred.dtype),
            pred.new_tensor(1.0),
        )
        pred = pred * scale.view(-1, *([1] * (pred.dim() - 1)))
    return pred
