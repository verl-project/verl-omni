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
"""Reshape collated media into the container shapes MiniCPMO.forward expects.

Splits what the collator batched together back into the per-sample slices, mel
stack, and clip lengths the remote towers index into.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

__all__ = [
    "batch_audio_feature_lens",
    "normalize_audio_features",
    "sample_pixel_slices",
    "sample_tgt_sizes",
    "unwrap_collated",
]


def batch_audio_feature_lens(value: Any, device: torch.device) -> list[torch.Tensor]:
    """Return one 1-D length tensor per clip, as ``hstack`` in the remote towers requires.

    Args:
        value: Per-sample feature lengths, possibly collated.
        device: Device for the returned tensors.

    Returns:
        List of 1-D ``torch.long`` tensors; empty for an audio-free batch.
    """
    value = unwrap_collated(value)
    if _is_empty_audio_features(value):
        return []
    if isinstance(value, torch.Tensor) and value.ndim <= 1:
        return [_one_sample_audio_feature_lens(value, device)]
    if isinstance(value, list | tuple):
        return [_one_sample_audio_feature_lens(sample, device) for sample in value]
    return [_one_sample_audio_feature_lens(value, device)]


def normalize_audio_features(value: Any) -> torch.Tensor | list:
    """Stack the clips into ``(n_clips, 80, frames)``, zero-padding to the longest.

    Args:
        value: Mel features, possibly collated and/or nested per sample.

    Returns:
        ``[]`` for an audio-free batch, otherwise a padded ``(n_clips, n_mels, frames)``
        tensor.
    """
    value = unwrap_collated(value)
    if _is_empty_audio_features(value):
        return []
    if isinstance(value, torch.Tensor):
        if value.ndim == 2:
            return value.unsqueeze(0).contiguous()
        return value.contiguous()
    if isinstance(value, np.ndarray):
        return normalize_audio_features(torch.as_tensor(value))
    if isinstance(value, list | tuple):
        clips: list[torch.Tensor] = []
        for item in value:
            if _is_empty_audio_features(item):
                continue
            packed = normalize_audio_features(item)
            if isinstance(packed, list):
                continue
            if packed.ndim == 2:
                clips.append(packed)
            else:
                clips.extend(clip.contiguous() for clip in packed)
        if not clips:
            return []
        max_frames = max(int(clip.shape[-1]) for clip in clips)
        padded = []
        for clip in clips:
            if clip.ndim != 2:
                raise ValueError(f"MiniCPM audio clip must be (n_mels, frames), got shape {tuple(clip.shape)}.")
            pad = max_frames - int(clip.shape[-1])
            if pad:
                clip = torch.nn.functional.pad(clip, (0, pad))
            padded.append(clip)
        return torch.stack(padded, dim=0)
    return normalize_audio_features(_as_tensor(value))


def sample_pixel_slices(pixel_values: Any) -> list[torch.Tensor]:
    """Return this sample's patch slices as a flat list of tensors.

    Args:
        pixel_values: One sample's pixel values, possibly collated or nested
            per slice.

    Returns:
        Flat list of slice tensors; empty when the sample has no images.
    """
    pixel_values = unwrap_collated(pixel_values)
    if pixel_values is None:
        return []
    if isinstance(pixel_values, torch.Tensor):
        if pixel_values.numel() == 0:
            return []
        # Only (n_slices, C, H, W) iterates; a single (C, H, W) slice is one slice,
        # not C feature maps.
        if pixel_values.ndim == 4:
            return [slice_tensor.contiguous() for slice_tensor in pixel_values]
        return [pixel_values.contiguous()]
    if isinstance(pixel_values, np.ndarray):
        return sample_pixel_slices(torch.as_tensor(pixel_values))
    if isinstance(pixel_values, list | tuple):
        if not pixel_values:
            return []
        first = pixel_values[0]
        if isinstance(first, list | tuple):
            if len(pixel_values) == 1:
                return sample_pixel_slices(first)
            slices: list[torch.Tensor] = []
            for item in pixel_values:
                slices.extend(sample_pixel_slices(item))
            return slices
        return [_as_tensor(item).contiguous() for item in pixel_values]
    return [_as_tensor(pixel_values).contiguous()]


def sample_tgt_sizes(tgt_sizes: Any, *, device: torch.device) -> torch.Tensor:
    """Return this sample's ``(n_slices, 2)`` target sizes.

    Args:
        tgt_sizes: One sample's target sizes, possibly collated or nested.
        device: Device for the returned tensor.

    Returns:
        ``int32`` tensor of shape ``(n_slices, 2)``, or ``(0, 2)`` when absent.
    """
    tgt_sizes = unwrap_collated(tgt_sizes)
    if tgt_sizes is None or (isinstance(tgt_sizes, (list | tuple)) and not tgt_sizes):
        return torch.zeros(0, 2, dtype=torch.int32, device=device)
    if isinstance(tgt_sizes, list | tuple) and len(tgt_sizes) == 1 and not isinstance(tgt_sizes[0], int | float):
        inner = tgt_sizes[0]
        if isinstance(inner, (list | tuple | np.ndarray | torch.Tensor)):
            return sample_tgt_sizes(inner, device=device)
    sizes = torch.as_tensor(tgt_sizes, dtype=torch.int32, device=device)
    if sizes.numel() == 0:
        return torch.zeros(0, 2, dtype=torch.int32, device=device)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(0)
    return sizes.reshape(-1, 2).contiguous()


def unwrap_collated(value: Any) -> Any:
    """Normalize DataProto's ragged-collation artifacts to plain containers.

    Ragged media (per-slice pixel arrays, variable-length clip lists) is collated
    into object-dtype ndarrays and short slots are padded with ``None``, neither of
    which ``torch.as_tensor`` accepts. Recurses through nested containers, drops the
    ``None`` padding, and keeps everything else in order.
    """
    if isinstance(value, np.ndarray) and value.dtype == object:
        return unwrap_collated(value.tolist())
    if isinstance(value, list | tuple):
        return [item for item in (unwrap_collated(member) for member in value) if item is not None]
    return value


def _as_tensor(value: Any) -> torch.Tensor:
    """Tensor for ``value``, unwrapping a collated object array first."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray) and value.dtype == object:
        return _as_tensor(unwrap_collated(value))
    return torch.as_tensor(np.asarray(value))


def _is_empty_audio_features(value: Any) -> bool:
    """True when there are no mel frames, including collated empty placeholders.

    MiniCPMO decides "has audio" with ``len(data['audio_features']) > 0``, but a
    collated image-only batch is often ``[[], [], ...]`` — length equal to the text
    batch size with no clip in it.
    """
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        return value.numel() == 0
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, list | tuple):
        return len(value) == 0 or all(_is_empty_audio_features(item) for item in value)
    return False


def _one_sample_audio_feature_lens(sample: Any, device: torch.device) -> torch.Tensor:
    """Flatten one sample's per-clip lengths into a single 1-D ``torch.long`` tensor."""
    if sample is None or (isinstance(sample, list | tuple) and not sample):
        return torch.zeros(0, dtype=torch.long, device=device)
    if isinstance(sample, torch.Tensor):
        return sample.to(device=device, dtype=torch.long).reshape(-1).contiguous()
    if isinstance(sample, np.ndarray):
        return torch.as_tensor(sample, dtype=torch.long, device=device).reshape(-1).contiguous()
    if isinstance(sample, list | tuple):
        if len(sample) == 1 and isinstance(sample[0], list | tuple | torch.Tensor | np.ndarray):
            return _one_sample_audio_feature_lens(sample[0], device)
        if sample and not isinstance(sample[0], int | float | np.integer | np.floating):
            return torch.cat([_one_sample_audio_feature_lens(item, device) for item in sample], dim=0)
        return torch.as_tensor(sample, dtype=torch.long, device=device).reshape(-1).contiguous()
    return torch.as_tensor(sample, dtype=torch.long, device=device).reshape(-1).contiguous()
