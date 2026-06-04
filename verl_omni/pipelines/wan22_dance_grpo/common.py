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

"""
Shared utilities for Wan2.2 DanceGRPO adapters.
"""

import hashlib

import torch

WAN_VAE_SCALE_FACTOR_SPATIAL = 8
WAN_VAE_SCALE_FACTOR_TEMPORAL = 4


def sd3_time_shift(shift: float, t: torch.Tensor) -> torch.Tensor:
    """
    Time shift function from SD3 / Mochi / Wan, also used by DanceGRPO.

    Args:
        shift (float): The shift factor (e.g. 3.0 or 5.0).
        t (torch.Tensor): Input tensor in [0, 1] range.

    Returns:
        torch.Tensor: Shifted tensor.
    """
    return (shift * t) / (1 + (shift - 1) * t)


def apply_cfg(
    noise_pred: torch.Tensor,
    negative_noise_pred: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """
    Standard classifier-free guidance (no renormalisation).

    Args:
        noise_pred (torch.Tensor): Positive-condition noise prediction.
        negative_noise_pred (torch.Tensor): Negative/uncond noise prediction.
        cfg_scale (float): Guidance scale; > 1 enables CFG.

    Returns:
        torch.Tensor: Combined noise prediction.
    """
    return negative_noise_pred + cfg_scale * (noise_pred - negative_noise_pred)


def flatten(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


def seed_from_prompt_ids(prompt_ids):
    """
    Generate a deterministic seed from prompt token ids.

    Args:
        prompt_ids: Can be one of:
            - torch.Tensor (e.g., [1, 77])
            - flat list (e.g., [49406, 320, ...])
            - nested list (e.g., [[49406, 320, ...]])

    Returns:
        int: A 64-bit integer seed deterministically derived from the ids.

    Raises:
        TypeError: If prompt_ids is not a Tensor or list.
    """
    # Flatten input to a 1D tuple, handling both Tensor and list types
    if isinstance(prompt_ids, torch.Tensor):
        ids_flat = prompt_ids.flatten().tolist()
    elif isinstance(prompt_ids, list):
        ids_flat = list(flatten(prompt_ids))
    else:
        raise TypeError(f"Unsupported type for prompt_ids: {type(prompt_ids)}")

    ids_tuple = tuple(ids_flat)

    # Hash the tuple to produce a deterministic seed
    hash_bytes = hashlib.md5(str(ids_tuple).encode()).digest()
    seed = int.from_bytes(hash_bytes[:8], byteorder="big")
    return seed
