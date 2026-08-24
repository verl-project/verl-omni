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
"""Shared MiniMax H3 latent-layout and weight-sync helpers."""

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch

VIDEO_ROW_WIDTH = 96
AUDIO_ROW_WIDTH = 32
LATENT_META_WIDTH = 6
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32

__all__ = [
    "VIDEO_ROW_WIDTH",
    "AUDIO_ROW_WIDTH",
    "LATENT_META_WIDTH",
    "VIDEO_TAG",
    "TEXT_TAG",
    "AUDIO_TAG",
    "pack_video_audio_rows",
    "unpack_video_audio_rows",
    "split_dual_velocity",
    "h3_dit_timestep",
    "h3_velocity_to_flow_match",
    "build_packed_sequence",
    "build_layout_from_meta",
    "build_row_timesteps",
    "MiniMaxH3RolloutWeightSyncMixin",
]


MINIMAX_H3_TOKEN_ID_NATIVE_KEY = "minimax_h3_token_id_native"


def messages_to_text(messages: Any) -> str:
    """Extract plain text items from chat messages without rendering a template."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        messages = [messages]

    parts: list[str] = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for item in content or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


def h3_dit_timestep(timesteps: torch.Tensor) -> torch.Tensor:
    """Convert ``sigma * 1000`` to H3's data-fraction timestep."""
    return 1.0 - timesteps / 1000.0


def h3_velocity_to_flow_match(velocity: torch.Tensor) -> torch.Tensor:
    """Convert H3 velocity to the diffusers flow-match sign."""
    return -velocity


def pack_video_audio_rows(video_rows: torch.Tensor, audio_rows: torch.Tensor) -> torch.Tensor:
    """Flatten and concatenate video and audio rows."""
    if video_rows.ndim == 2:
        video_rows = video_rows.unsqueeze(0)
    if audio_rows.ndim == 2:
        audio_rows = audio_rows.unsqueeze(0)
    batch = video_rows.shape[0]
    return torch.cat([video_rows.reshape(batch, -1), audio_rows.reshape(batch, -1)], dim=1)


def unpack_video_audio_rows(
    packed: torch.Tensor,
    num_video_rows: int,
    num_audio_rows: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack flattened video and audio rows."""
    batch = packed.shape[0]
    split = num_video_rows * VIDEO_ROW_WIDTH
    video_rows = packed[:, :split].reshape(batch, num_video_rows, VIDEO_ROW_WIDTH)
    audio_rows = packed[:, split:].reshape(batch, num_audio_rows, AUDIO_ROW_WIDTH)
    return video_rows, audio_rows


def split_dual_velocity(result) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a transformer output into video and audio velocity rows."""
    if isinstance(result, tuple | list):
        return result[0], result[1]
    if hasattr(result, "sample") and hasattr(result, "audio_sample"):
        return result.sample, result.audio_sample
    raise TypeError(f"Unexpected MiniMax H3 transformer output type: {type(result).__name__}")


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> torch.Tensor:
    """Build one aspect-normalized spatial rotary axis."""
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    grid = np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE
    return torch.from_numpy(grid).to(torch.float64)


def _temporal_position_grid(num_latent_frames: int, origin: float) -> torch.Tensor:
    """Rotary time of every latent frame, starting at ``origin``. Spacing is ``5/3 * (1, 4, 4, 4, 4)``."""
    spans = torch.tensor(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[index % len(_ROPE_FRAMES_PER_LATENT)]
            for index in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    return origin + torch.cat([torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)])


def _frame_position_grid(
    latent_height: int, latent_width: int, patch_h: int, patch_w: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(h, w)`` rotary coordinates of one latent frame, and the width axis they were built from."""
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, patch_h, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def build_packed_sequence(
    text_token_tags: torch.Tensor,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    audio_channels: int,
    audio_tag: int = AUDIO_TAG,
    video_tag: int = VIDEO_TAG,
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build the packed H3 text, condition, audio, and video layout."""
    _, patch_h, patch_w = patch_size
    rows_per_frame = (latent_height // patch_h) * (latent_width // patch_w)
    num_text_tokens = text_token_tags.shape[0]
    num_condition_rows = len(keyframe_anchors) * rows_per_frame
    num_audio_rows = num_audio_latents * audio_channels
    num_video_rows = num_latent_frames * rows_per_frame
    sequence_length = num_text_tokens + num_condition_rows + num_audio_rows + num_video_rows

    condition_start = num_text_tokens
    audio_start = condition_start + num_condition_rows
    video_start = audio_start + num_audio_rows

    position_ids = torch.zeros(sequence_length, 3, dtype=torch.float64)
    position_ids[:num_text_tokens, 0] = torch.arange(num_text_tokens, dtype=torch.float64)

    frame_grid, width_grid = _frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = float(num_text_tokens)
        elif anchor == "last":
            spans = np.ones(num_latent_frames, dtype=np.float64) * _ROPE_FRAME_RESCALE
            for offset in range(len(_ROPE_FRAMES_PER_LATENT)):
                spans[offset :: len(_ROPE_FRAMES_PER_LATENT)] *= _ROPE_FRAMES_PER_LATENT[offset]
            anchor_time = float(num_text_tokens) + float(spans.sum()) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"A keyframe anchor must be 'first' or 'last', got {anchor!r}.")
        rows = slice(condition_start + index * rows_per_frame, condition_start + (index + 1) * rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    audio_time = float(num_text_tokens) + torch.arange(num_audio_latents, dtype=torch.float64)
    position_ids[audio_start:video_start, 0] = audio_time.repeat(audio_channels)
    position_ids[audio_start:video_start, 2] = torch.cat(
        [
            torch.full((num_audio_latents,), float(width_grid[0]), dtype=torch.float64),
            torch.full((num_audio_rows - num_audio_latents,), float(width_grid[-1]), dtype=torch.float64),
        ]
    )

    video_position_ids = torch.empty(num_latent_frames, rows_per_frame, 3, dtype=torch.float64)
    video_position_ids[:, :, 0] = _temporal_position_grid(num_latent_frames, float(num_text_tokens))[:, None]
    video_position_ids[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video_position_ids.reshape(-1, 3)

    video_indices = torch.cat([torch.arange(condition_start, audio_start), torch.arange(video_start, sequence_length)])
    audio_indices = torch.arange(audio_start, video_start)
    text_indices = torch.arange(num_text_tokens)

    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = text_token_tags.to(torch.long)
    token_tags[audio_indices] = audio_tag
    token_tags[video_indices] = video_tag

    return position_ids, token_tags, video_indices, audio_indices, text_indices, num_condition_rows, 0


def build_layout_from_meta(
    meta: Sequence[int],
    num_text_tokens: int,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    keyframe_anchors: tuple[str, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Build an H3 layout from latent metadata and text length."""
    num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
    num_latent_frames, latent_height, latent_width = int(meta[2]), int(meta[3]), int(meta[4])
    num_audio_latents = int(meta[5])
    if num_audio_latents <= 0:
        raise ValueError(f"latent_meta audio_t must be positive, got {num_audio_latents}.")
    audio_channels = num_audio_rows // num_audio_latents

    layout = build_packed_sequence(
        text_token_tags=torch.full((num_text_tokens,), TEXT_TAG, dtype=torch.long),
        num_latent_frames=num_latent_frames,
        latent_height=latent_height,
        latent_width=latent_width,
        num_audio_latents=num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
        keyframe_anchors=keyframe_anchors,
    )
    _, _, video_indices, audio_indices, *_ = layout
    if audio_indices.shape[0] != num_audio_rows:
        raise ValueError(f"Derived {audio_indices.shape[0]} audio rows, latent_meta says {num_audio_rows}.")
    if not keyframe_anchors and video_indices.shape[0] != num_video_rows:
        raise ValueError(f"Derived {video_indices.shape[0]} video rows, latent_meta says {num_video_rows}.")
    return layout


def build_row_timesteps(
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    num_text_tokens: int,
    video_timestep: float,
    audio_timestep: float,
    condition_video_timestep: float,
    condition_audio_timestep: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build distinct H3 timesteps and per-row indices."""
    sequence_length = int(video_indices.numel() + audio_indices.numel() + num_text_tokens)
    row_timesteps = torch.full((sequence_length,), video_timestep, dtype=torch.float32)
    row_timesteps[video_indices[:num_condition_video_rows]] = condition_video_timestep
    row_timesteps[audio_indices[num_condition_audio_rows:]] = audio_timestep
    row_timesteps[audio_indices[:num_condition_audio_rows]] = condition_audio_timestep
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


_TOPLEVEL_RENAMES = (
    ("audio_proj_in", "audio_patch_proj"),
    ("audio_proj_out", "final_layer.audio_out"),
    ("proj_in", "video_patch_proj"),
    ("proj_out", "final_layer.video_out"),
    ("context_embedder", "condition_proj"),
    ("time_embedder.linear_1", "time_embedder.proj_in"),
    ("time_embedder.linear_2", "time_embedder.proj_out"),
    ("norm_out.linear", "final_layer.adaln_proj.linear"),
    ("norm_out.norm", "final_layer.norm"),
)


def _diffusers_to_vllm_name(name: str) -> str:
    """Rename a diffusers transformer param to its fused-vllm counterpart (no reshape)."""
    name = name.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    name = name.replace("transformer_blocks.", "blocks.")
    name = name.replace(".attn.norm_q.", ".attn.q_norm.")
    name = name.replace(".attn.norm_k.", ".attn.k_norm.")
    name = name.replace(".attn.to_out.0.", ".attn.out_proj.")
    name = name.replace(".ff.net.2.", ".mlp.fc2.")
    for old, new in _TOPLEVEL_RENAMES:  # audio_* listed first so they win over proj_in/out
        if name.startswith(old + "."):
            return new + name[len(old) :]
    return name


_LORA_VLLM_TARGET_MODULES = ["to_q", "to_k", "to_v", "out_proj", "fc1_0", "fc1_1", "fc2"]
_SUPPORTED_DIFFUSERS_LORA_TARGETS = frozenset({"to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2"})

_LORA_STACKED_PARAMS_MAPPING = [
    (".qkv_proj", ".to_q", "q"),
    (".qkv_proj", ".to_k", "k"),
    (".qkv_proj", ".to_v", "v"),
    (".fc1", ".fc1_0", "0"),
    (".fc1", ".fc1_1", "1"),
]


def validate_lora_target_modules(target_modules) -> set[str]:
    """Validate H3 LoRA targets against the sync-safe whitelist (single source of truth).

    ``all-linear`` and top-level modules are rejected because FSDP layered-summon does
    not transport them to rollout. Returns the normalized set; raises on violations.
    """
    if isinstance(target_modules, str):
        requested = {target_modules}
    elif isinstance(target_modules, list | tuple | set | frozenset):
        requested = {str(target) for target in target_modules}
    else:
        raise ValueError(f"MiniMax H3 LoRA requires an explicit target_modules list; got {target_modules!r}.")
    unsupported = requested - _SUPPORTED_DIFFUSERS_LORA_TARGETS
    if not requested or unsupported:
        raise ValueError(
            "MiniMax H3 LoRA supports only transformer/refiner block targets "
            f"{sorted(_SUPPORTED_DIFFUSERS_LORA_TARGETS)}, got {sorted(requested)}. "
            "`all-linear` and other top-level modules are not synced to rollout "
            "(FSDP layered-summon does not transport them)."
        )
    return requested


def _map_lora_module_to_vllm(module: str) -> str:
    """Map a diffusers LoRA target module path to its fused-vllm path (fc1 handled separately)."""
    return _diffusers_to_vllm_name(module + ".")[:-1]


class MiniMaxH3RolloutWeightSyncMixin:
    """Map Diffusers H3 weights and token-id-native prompts to vLLM-Omni."""

    def encode_prompt(
        self,
        *,
        task: str,
        prompt: str,
        image=None,
        prepared_videos: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        """Encode pre-tokenized T2VA IDs without decode/re-tokenize drift."""
        prompt_ids = getattr(self, "_h3_prompt_ids", None)
        if prompt_ids is None or task != "t2va":
            return super().encode_prompt(
                task=task,
                prompt=prompt,
                image=image,
                prepared_videos=prepared_videos,
                **kwargs,
            )

        from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import _broadcast_tensor, _dit_rank_world

        _, rank, _ = _dit_rank_world()
        hidden = None
        tags = None
        ids = None
        vision_kwargs: dict[str, torch.Tensor] = {}
        if rank == 0:
            ids = prompt_ids
            tags = torch.ones(ids.shape[0], dtype=torch.long)

        if rank < self.text_encoder_tp_size:
            ids = self._distribute_encode_inputs(ids, vision_kwargs)
            hidden = self._encode_text_hidden(ids, vision_kwargs)
        hidden = _broadcast_tensor(hidden, dtype=torch.bfloat16, device=self.device)
        tags = _broadcast_tensor(tags, dtype=torch.long, device=self.device)
        return hidden, tags

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Translate Diffusers weights into the fused vLLM H3 layout."""
        arch = self.transformer.arch
        heads, head_dim, ff_half = arch.num_attention_heads, arch.attention_head_dim, arch.ffn_hidden_size
        partials = getattr(self, "_qkv_buffer", None)
        if partials is None:
            partials = self._qkv_buffer = {}
        translated: list[tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            if not name.startswith("transformer."):
                translated.append((name, tensor))
                continue
            inner = name[len("transformer.") :].replace(".base_layer", "")
            if "lora_" in inner:
                continue
            if inner.endswith((".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight")):
                block, comp = inner.rsplit(".attn.to_", 1)
                slot = partials.setdefault(block, {})
                slot[comp[0]] = tensor
                if len(slot) == 3:
                    heads_qkv = [slot[c].view(heads, head_dim, -1) for c in ("q", "k", "v")]
                    qkv = torch.stack(heads_qkv, dim=1).reshape(heads * 3 * head_dim, -1)
                    translated.append((f"transformer.{_diffusers_to_vllm_name(block)}.attn.qkv_proj.weight", qkv))
                    del partials[block]
                continue
            if inner.endswith(".ff.net.0.proj.weight"):
                swapped = torch.cat([tensor[ff_half:], tensor[:ff_half]], dim=0)
                vname = _diffusers_to_vllm_name(inner).replace(".ff.net.0.proj.", ".mlp.fc1.")
                translated.append((f"transformer.{vname}", swapped))
                continue
            translated.append((f"transformer.{_diffusers_to_vllm_name(inner)}", tensor))
        needs_rope = not getattr(self, "_rope_inv_freq_loaded", False)
        if needs_rope:
            rope_len = arch.rope_inv_freq_len
            inv_freq = 10000.0 ** (-(torch.arange(0, 2 * rope_len, 2, dtype=torch.float32) / (2 * rope_len)))
            # vllm-omni 0.27 requires each weight-prefix group to be contiguous;
            # the rope table must stay inside the "transformer." run.
            insert_at = next(
                (i for i, (n, _) in enumerate(translated) if not n.startswith("transformer.")),
                len(translated),
            )
            translated.insert(insert_at, ("transformer.rope.inv_freq", inv_freq))

        loaded = super().load_weights(translated)
        if needs_rope and "transformer.rope.inv_freq" in loaded:
            self._rope_inv_freq_loaded = True
        return loaded

    def _install_lora_layout(self) -> None:
        """Install H3 QKV and FC1 LoRA slice metadata."""
        transformer = getattr(self, "transformer", None)
        if transformer is not None and not getattr(transformer, "stacked_params_mapping", None):
            transformer.stacked_params_mapping = list(_LORA_STACKED_PARAMS_MAPPING)

    def map_lora_update_to_engine(
        self, tensors: dict[str, torch.Tensor], peft_config: dict
    ) -> tuple[dict[str, torch.Tensor], dict]:
        """Translate LoRA deltas to the fused vLLM H3 layout."""
        target_modules = peft_config.get("target_modules") if peft_config is not None else None
        validate_lora_target_modules(target_modules)

        ff_half = self.transformer.arch.ffn_hidden_size
        mapped: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            is_lora_a = name.endswith(".lora_A.weight")
            is_lora_b = name.endswith(".lora_B.weight")
            if not (is_lora_a or is_lora_b):
                mapped[name] = tensor
                continue
            suffix = ".lora_A.weight" if is_lora_a else ".lora_B.weight"
            module = name[: -len(suffix)]
            anchors = [
                a for a in (module.find("transformer_blocks."), module.find("token_refiner.refiner_blocks.")) if a >= 0
            ]
            if not anchors:
                mapped[name] = tensor
                continue
            module = module[min(anchors) :]
            if ".ff.net.0.proj" in module:
                base = _diffusers_to_vllm_name(module + ".")[:-1].replace(".ff.net.0.proj", ".mlp.fc1")
                if is_lora_b:
                    swapped = torch.cat([tensor[ff_half:], tensor[:ff_half]], dim=0)
                    mapped[f"transformer.{base}_0{suffix}"] = swapped[:ff_half].contiguous()
                    mapped[f"transformer.{base}_1{suffix}"] = swapped[ff_half:].contiguous()
                else:
                    mapped[f"transformer.{base}_0{suffix}"] = tensor
                    mapped[f"transformer.{base}_1{suffix}"] = tensor
                continue
            vllm_module = _map_lora_module_to_vllm(module)
            mapped[f"transformer.{vllm_module}{suffix}"] = tensor

        new_config = dict(peft_config) if peft_config is not None else {}
        new_config["target_modules"] = list(_LORA_VLLM_TARGET_MODULES)
        return mapped, new_config

    def _ensure_prompt_text(self, request: Any) -> None:
        """Expose pre-tokenized IDs and satisfy the upstream non-empty-text check."""
        self._h3_prompt_ids = None
        prompts = getattr(request, "prompts", None)
        if not prompts or not isinstance(prompts[0], dict):
            return
        custom_prompt = prompts[0]
        token_ids = custom_prompt.get("prompt_token_ids")
        if token_ids is None:
            return
        sampling_params = getattr(request, "sampling_params", None)
        extra_args = getattr(sampling_params, "extra_args", None) or {}
        if extra_args.get(MINIMAX_H3_TOKEN_ID_NATIVE_KEY) is not True:
            raise ValueError(
                "MiniMax H3 token-ID-native rollout requires "
                "actor_rollout_ref.rollout.agent.default_agent_loop="
                "minimax_h3_diffusion_single_turn_agent."
            )
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        self._h3_prompt_ids = torch.as_tensor([int(token) for token in token_ids], dtype=torch.long)
        if self._h3_prompt_ids.numel() == 0:
            raise ValueError("MiniMax H3 requires non-empty prompt_token_ids.")
        custom_prompt["prompt"] = "[pretokenized]"
