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

FLUX_VAE_SCALE_FACTOR = 8


def maybe_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def coalesce_not_none(value, default):
    return default if value is None else value


def getattr_not_none(obj, name: str, default):
    return coalesce_not_none(getattr(obj, name, None), default)


def calculate_shift(
    image_seq_len: int,
    base_image_seq_len: int = 256,
    max_image_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b


def packed_latent_seq_len(height: int, width: int, vae_scale_factor: int = FLUX_VAE_SCALE_FACTOR) -> int:
    latent_height = 2 * (int(height) // (vae_scale_factor * 2))
    latent_width = 2 * (int(width) // (vae_scale_factor * 2))
    return (latent_height // 2) * (latent_width // 2)


def squeeze_batch_position_ids(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is not None and value.ndim == 3:
        return value[0]
    return value


def batched_position_ids(value: torch.Tensor, batch_size: int) -> torch.Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0).expand(batch_size, -1, -1)
    return value


def apply_true_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    return negative_noise_pred + true_cfg_scale * (noise_pred - negative_noise_pred)
