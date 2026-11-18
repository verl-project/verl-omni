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
"""Image latent metadata shared by H3 Image rollout and replay."""

import torch

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    LATENT_META_WIDTH,
)


def image_latent_meta(num_rows: int, latent_h: int, latent_w: int) -> torch.Tensor:
    """H3 latent metadata of one image: no audio rows, one latent frame."""
    return torch.tensor([[num_rows, 0, 1, latent_h, latent_w, 0]], dtype=torch.long)


def parse_image_latent_meta(latent_meta: torch.Tensor) -> tuple[int, int, int]:
    """Return ``(num_rows, latent_h, latent_w)`` of a micro-batch that shares one canvas."""
    meta = latent_meta.reshape(latent_meta.shape[0], -1)
    if meta.shape[1] != LATENT_META_WIDTH:
        raise ValueError(f"MiniMax H3 image latent_meta must have {LATENT_META_WIDTH} fields, got {meta.shape[1]}.")
    if not bool((meta == meta[:1]).all()):
        raise ValueError("MiniMax H3 image micro-batches must share one generation canvas.")
    num_rows, num_audio_rows, latent_t, latent_h, latent_w, audio_t = (int(value) for value in meta[0].tolist())
    if num_audio_rows != 0 or audio_t != 0 or latent_t != 1:
        raise ValueError(
            "MiniMax H3 image latents are audio-free single frames; got "
            f"audio_rows={num_audio_rows}, audio_t={audio_t}, latent_t={latent_t}."
        )
    if num_rows != (latent_h // 2) * (latent_w // 2):
        raise ValueError(f"MiniMax H3 image latent_meta rows {num_rows} do not match grid {latent_h}x{latent_w}.")
    return num_rows, latent_h, latent_w
