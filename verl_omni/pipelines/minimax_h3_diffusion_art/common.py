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
"""Token layout shared by the MiniMax H3 DiffusionART rollout and training adapters.

H3 keeps video rows of width 96 and audio rows of width 32. ``DiffusionARTLoss`` splits one unified
``(B, tokens, channels)`` sequence at ``video_seq_len`` along the token axis, so both streams are
re-expressed as rows of width 32: every video row becomes three consecutive 32-wide tokens, audio
rows stay as they are. The loss is a per-element quantity (per-item MSE, unit-RMS guidance,
drift energy), so regrouping elements into tokens leaves it unchanged, and
``art_rows.reshape(batch, -1)`` is exactly the ``pack_video_audio_rows`` layout the H3 NFT adapter
already consumes.
"""

from __future__ import annotations

import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import AUDIO_ROW_WIDTH, VIDEO_ROW_WIDTH

__all__ = ["ART_ROW_WIDTH", "art_rows_to_packed", "art_video_seq_len", "packed_to_art_rows", "rows_to_art_rows"]

#: Width of one DiffusionART token for H3: the audio row width, which divides the video row width.
ART_ROW_WIDTH = AUDIO_ROW_WIDTH
_VIDEO_TOKENS_PER_ROW = VIDEO_ROW_WIDTH // ART_ROW_WIDTH
if _VIDEO_TOKENS_PER_ROW * ART_ROW_WIDTH != VIDEO_ROW_WIDTH:
    raise RuntimeError("MiniMax H3 video row width must be a multiple of the audio row width.")


def art_video_seq_len(num_video_rows: int) -> int:
    """Number of width-32 tokens the video stream occupies (the ``video_seq_len`` the loss splits at)."""
    return int(num_video_rows) * _VIDEO_TOKENS_PER_ROW


def rows_to_art_rows(video_rows: torch.Tensor, audio_rows: torch.Tensor) -> torch.Tensor:
    """``(rows, 96)`` + ``(arows, 32)`` -> ``(3 * rows + arows, 32)``; leading batch dims are kept."""
    if video_rows.shape[-1] != VIDEO_ROW_WIDTH or audio_rows.shape[-1] != AUDIO_ROW_WIDTH:
        raise ValueError(
            f"MiniMax H3 rows must be ({VIDEO_ROW_WIDTH}, {AUDIO_ROW_WIDTH}) wide, "
            f"got {video_rows.shape[-1]} / {audio_rows.shape[-1]}."
        )
    lead = video_rows.shape[:-2]
    video = video_rows.reshape(*lead, -1, ART_ROW_WIDTH)
    return torch.cat([video, audio_rows], dim=-2)


def art_rows_to_packed(art_rows: torch.Tensor) -> torch.Tensor:
    """``(B, 3 * rows + arows, 32)`` -> the flat ``pack_video_audio_rows`` layout ``(B, rows * 96 + arows * 32)``."""
    if art_rows.ndim != 3 or art_rows.shape[-1] != ART_ROW_WIDTH:
        raise ValueError(
            f"DiffusionART H3 latents must have shape (B, tokens, {ART_ROW_WIDTH}), got {tuple(art_rows.shape)}."
        )
    return art_rows.reshape(art_rows.shape[0], -1)


def packed_to_art_rows(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`art_rows_to_packed` for any ``(B, rows * 96 + arows * 32)`` tensor."""
    if packed.ndim != 2 or packed.shape[1] % ART_ROW_WIDTH:
        raise ValueError(f"Packed H3 rows must be (B, k * {ART_ROW_WIDTH}), got {tuple(packed.shape)}.")
    return packed.reshape(packed.shape[0], -1, ART_ROW_WIDTH)
