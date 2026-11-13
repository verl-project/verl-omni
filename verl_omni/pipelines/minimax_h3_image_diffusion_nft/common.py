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
"""Shared MiniMax H3 image policies: PDD step grid, per-step output heads and the audio-free layout.

The image release is a 6-step PDD student: one backbone and one fused output head per denoise
step. Head ``k`` is only defined at grid point ``sigma_k``, so both sides address a step by its
index and the Actor never trains at any other noise level.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import (
    LATENT_META_WIDTH,
    VIDEO_ROW_WIDTH,
    _deserialize_ref_blocks,
)

__all__ = [
    "IMAGE_ROW_WIDTH",
    "IMAGE_AUDIO_ROW_WIDTH",
    "IMAGE_CONDITION_TIMESTEP",
    "PDD_HEADS_FILENAME",
    "PDD_CONFIG_FILENAME",
    "PDD_TIMESTEP_SCALE",
    "MiniMaxH3PDDHeadLinear",
    "image_latent_meta",
    "parse_image_latent_meta",
    "pdd_train_timesteps",
    "pdd_step_indices",
    "pdd_dit_timestep",
    "resolve_pdd_heads_path",
    "load_pdd_heads",
    "install_pdd_heads",
    "ensure_pdd_heads",
    "build_image_layout",
]

IMAGE_ROW_WIDTH = VIDEO_ROW_WIDTH
IMAGE_AUDIO_ROW_WIDTH = 32
IMAGE_CONDITION_TIMESTEP = 0.999
PDD_HEADS_FILENAME = "pdd_fused_heads.safetensors"
PDD_CONFIG_FILENAME = "pdd_config.json"
PDD_TIMESTEP_SCALE = 1000.0
_PDD_LOOKUP_TOLERANCE = 1.0


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


def pdd_train_timesteps(run_sigmas: Sequence[float] | torch.Tensor) -> torch.Tensor:
    """``sigma * 1000`` of the grid points the rollout evaluated; ``run_sigmas`` excludes the final 0."""
    sigmas = torch.as_tensor(run_sigmas, dtype=torch.float32).reshape(-1)
    if sigmas.numel() < 1:
        raise ValueError("MiniMax H3 image PDD grid has no denoise step.")
    if float(sigmas[-1]) == 0.0:
        raise ValueError("pdd_train_timesteps takes the evaluated sigmas only; drop the final 0.")
    return sigmas * PDD_TIMESTEP_SCALE


def pdd_step_indices(timesteps: torch.Tensor, pdd_sigmas: torch.Tensor) -> torch.Tensor:
    """Map each ``sigma * 1000`` timestep (exact or ``.long()``-truncated) to its PDD step index.

    Args:
        timesteps: ``(B,)`` train timesteps.
        pdd_sigmas: ``(B, K)`` or ``(K,)`` sigmas the rollout evaluated, in step order.

    Returns:
        ``(B,)`` long step indices; raises when a timestep is not within one unit of a grid point
        or the grid is too dense for the lookup to be unambiguous.
    """
    timesteps = timesteps.reshape(-1)
    grid = pdd_sigmas.to(device=timesteps.device, dtype=torch.float64) * PDD_TIMESTEP_SCALE
    if grid.ndim == 1:
        grid = grid.unsqueeze(0).expand(timesteps.shape[0], -1)
    if grid.ndim != 2 or grid.shape[0] != timesteps.shape[0] or grid.shape[1] < 1:
        raise ValueError(
            f"pdd_sigmas must be (B, K) or (K,), got {tuple(pdd_sigmas.shape)} for B={timesteps.shape[0]}."
        )
    if grid.shape[1] > 1 and float((grid[:, :-1] - grid[:, 1:]).min()) <= 2.0 * _PDD_LOOKUP_TOLERANCE:
        raise ValueError("MiniMax H3 image PDD sigmas must decrease by more than 0.002 per step.")
    distance = (grid - timesteps.to(torch.float64).unsqueeze(1)).abs()
    nearest, indices = distance.min(dim=1)
    if bool((nearest >= _PDD_LOOKUP_TOLERANCE).any()):
        raise ValueError(
            "MiniMax H3 image trains only on the rollout's PDD sigmas; got timesteps "
            f"{timesteps.tolist()} for grid {grid[0].tolist()}."
        )
    return indices


def pdd_dit_timestep(sigmas: torch.Tensor) -> torch.Tensor:
    """Model time ``1 - sigma`` rounded exactly as the rollout loop rounds it (FP64 subtract, FP32 store)."""
    return (1.0 - sigmas.to(torch.float64)).to(torch.float32)


class MiniMaxH3PDDHeadLinear(torch.nn.Linear):
    """``proj_out`` of a PDD release: one frozen FP32 head per denoise step, selected by step index.

    The checkpoint's own ``weight`` / ``bias`` stay in the state dict but are bypassed, as the
    serving model bypasses ``final_layer.video_out`` once fused heads are loaded.
    """

    pdd_step: int | None = None
    rollout_row_padding: bool = False

    pdd_segments: list[tuple[int, int, int]] | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.pdd_segments is not None:
            segments = self.pdd_segments
            self.pdd_segments = None
            try:
                outputs = []
                for start, end, step in segments:
                    self.pdd_step = step
                    outputs.append(self.forward(input[:, start:end]))
                return torch.cat(outputs, dim=1)
            finally:
                self.pdd_segments = segments
                self.pdd_step = None
        step = self.pdd_step
        num_steps = int(self.pdd_weight.shape[0])
        if step is None or not 0 <= int(step) < num_steps:
            raise ValueError(f"MiniMax H3 image PDD has {num_steps} fused heads; got pdd_step={step!r}.")
        weight = self.pdd_weight[int(step)].to(device=input.device, dtype=input.dtype)
        bias = self.pdd_bias[int(step)].to(device=input.device, dtype=input.dtype)
        if self.rollout_row_padding and input.is_cuda and input.ndim == 3 and input.shape[0] == 1:
            from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_image import _image_packed_seq_len

            rows = input[0]
            padding = _image_packed_seq_len(rows.shape[0]) - rows.shape[0]
            projected = F.linear(F.pad(rows, (0, 0, 0, padding)), weight, bias)
            return projected[: rows.shape[0]].unsqueeze(0)
        return F.linear(input, weight, bias)


def resolve_pdd_heads_path(model_config: Any) -> Path:
    """Find the fused heads next to the Actor transformer or in the native release root."""
    candidates: list[Path] = []
    for attr in ("config_path", "local_path", "path"):
        value = getattr(model_config, attr, None)
        if isinstance(value, str | os.PathLike) and str(value):
            root = Path(value)
            candidates.extend([root / PDD_HEADS_FILENAME, root.parent / PDD_HEADS_FILENAME])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"MiniMax H3 image requires {PDD_HEADS_FILENAME}; searched {[str(path) for path in candidates]}."
    )


def load_pdd_heads(path: str | os.PathLike) -> tuple[torch.Tensor, torch.Tensor]:
    """Read ``video_out.{weight,bias}`` of an audio-free fused-head file as FP32."""
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        heads = {key: handle.get_tensor(key) for key in handle.keys()}
    if set(heads) != {"video_out.weight", "video_out.bias"}:
        raise ValueError(f"MiniMax H3 image PDD heads must hold video_out.weight/bias only, got {sorted(heads)}.")
    weight, bias = heads["video_out.weight"], heads["video_out.bias"]
    if weight.ndim != 3 or bias.ndim != 2 or weight.shape[:2] != bias.shape or weight.shape[0] < 1:
        raise ValueError(f"Invalid MiniMax H3 image PDD head shapes: {tuple(weight.shape)}, {tuple(bias.shape)}.")
    return weight.to(torch.float32), bias.to(torch.float32)


def install_pdd_heads(module: torch.nn.Module, weight: torch.Tensor, bias: torch.Tensor) -> MiniMaxH3PDDHeadLinear:
    """Turn the transformer's ``proj_out`` into a per-step head selector (idempotent)."""
    proj_out = module.proj_out
    if isinstance(proj_out, MiniMaxH3PDDHeadLinear):
        return proj_out
    if type(proj_out) is not torch.nn.Linear:
        raise TypeError(f"MiniMax H3 image PDD heads replace an nn.Linear proj_out, got {type(proj_out).__name__}.")
    expected = (proj_out.out_features, proj_out.in_features)
    if tuple(weight.shape[1:]) != expected or tuple(bias.shape) != (weight.shape[0], expected[0]):
        raise ValueError(
            f"MiniMax H3 image PDD heads {tuple(weight.shape)} / {tuple(bias.shape)} do not match proj_out {expected}."
        )
    device = proj_out.weight.device
    proj_out.register_buffer("pdd_weight", weight.to(device=device, dtype=torch.float32), persistent=False)
    proj_out.register_buffer("pdd_bias", bias.to(device=device, dtype=torch.float32), persistent=False)
    proj_out.__class__ = MiniMaxH3PDDHeadLinear
    proj_out.pdd_step = None
    return proj_out


def ensure_pdd_heads(module: torch.nn.Module, model_config: Any) -> MiniMaxH3PDDHeadLinear:
    """Install the release's fused heads on first use."""
    proj_out = module.proj_out
    if isinstance(proj_out, MiniMaxH3PDDHeadLinear):
        return proj_out
    return install_pdd_heads(module, *load_pdd_heads(resolve_pdd_heads_path(model_config)))


def build_image_layout(
    latent_h: int,
    latent_w: int,
    num_text_tokens: int,
    ref_block_meta: torch.Tensor | None = None,
    ref_block_count: int = 0,
    text_token_tags: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build the Actor layout with the packer the rollout denoiser uses, without its padding rows.

    Returns ``(position_ids, token_tags, image_indices, text_indices, num_condition_rows)``. The
    rollout pads the sequence as a separate attention document, which real rows never attend to.
    """
    from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence_ref2va_blocks

    blocks = _deserialize_ref_blocks(ref_block_meta, ref_block_count) if ref_block_count else []
    if any(block["kind"] != "image" for block in blocks):
        raise ValueError("MiniMax H3 image edit accepts image references only.")
    packed = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=num_text_tokens,
        latent_t=1,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=0,
        ref_blocks=blocks,
    )
    used = int(packed["cu_seqlens"][1])
    token_tags = packed["token_tags"][:used].clone()
    if text_token_tags is not None:
        text_token_tags = text_token_tags.reshape(-1)[:num_text_tokens].to(dtype=torch.long, device="cpu")
        if text_token_tags.numel() != num_text_tokens:
            raise ValueError(f"Expected {num_text_tokens} MiniMax H3 text token tags, got {text_token_tags.numel()}.")
        token_tags[packed["text_pos"]] = text_token_tags
    if int(packed["audio_pos"].numel()) != 0:
        raise ValueError("MiniMax H3 image layout must not contain audio rows.")
    position_ids = packed["img_position_ids"]
    position_ids = position_ids.reshape(-1, 3)[:used]
    num_condition_rows = int((~packed["update_mask"]).sum())
    return position_ids, token_tags, packed["img_pos"], packed["text_pos"], num_condition_rows


def debug_dump(kind: str, payload: dict[str, Any]) -> None:
    """Write one diagnostic record when ``MINIMAX_H3_IMAGE_DEBUG_DIR`` is set (end-to-end evidence runs only)."""
    directory = os.environ.get("MINIMAX_H3_IMAGE_DEBUG_DIR")
    if not directory:
        return
    import time

    os.makedirs(directory, exist_ok=True)
    torch.save(payload, os.path.join(directory, f"{kind}-{os.getpid()}-{time.time_ns()}.pt"))


def debug_enabled() -> bool:
    return bool(os.environ.get("MINIMAX_H3_IMAGE_DEBUG_DIR"))
