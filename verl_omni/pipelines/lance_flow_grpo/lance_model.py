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


"""Lance checkpoint loading, latent geometry, and three-axis image RoPE for MoT replay."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import (
    BagelForTraining,
    BagelTrainingConfig,
    RotaryEmbedding,
    _get_1d_sincos_pos_embed_from_grid,
)

from .common import (
    LANCE_LATENT_PATCH_SIZE,
    LANCE_MAX_LATENT_SIZE,
    LANCE_MROPE_SECTION,
    LANCE_TEMPORAL_ROPE_SCALE,
    LANCE_VAE_DOWNSAMPLE_SPATIAL,
    LANCE_VAE_DOWNSAMPLE_TEMPORAL,
    LANCE_VAE_Z_CHANNELS,
)

logger = logging.getLogger(__name__)

#: Checkpoint sub-directories inside the ``bytedance-research/Lance`` bundle.
IMAGE_CKPT_DIR = "Lance_3B"
VIDEO_CKPT_DIR = "Lance_3B_Video"


@dataclass
class LanceTrainingConfig(BagelTrainingConfig):
    """Lance variant of :class:`BagelTrainingConfig`.

    Same fields; the defaults describe Lance_3B's Wan2.2 latent geometry
    instead of BAGEL's.  The LLM dimensions are always read from the
    checkpoint's ``llm_config.json``.
    """

    hidden_size: int = 2048
    latent_patch_size: int = LANCE_LATENT_PATCH_SIZE
    max_latent_size: int = LANCE_MAX_LATENT_SIZE
    #: Latent frames covered by ``latent_pos_embed``.  One frame is the image
    #: table; Lance_3B_Video ships a 3-D table of 31.  ``from_pretrained`` takes
    #: the value from the checkpoint, so a caller does not set this by hand.
    max_num_frames: int = 1
    latent_channel: int = LANCE_VAE_Z_CHANNELS
    vae_downsample: int = LANCE_VAE_DOWNSAMPLE_SPATIAL
    #: Video VAE temporal stride; ``latent_frames = (num_frames - 1) // this + 1``.
    vae_downsample_temporal: int = LANCE_VAE_DOWNSAMPLE_TEMPORAL
    #: Head-dimension split across the (t, h, w) rotary axes.  Read from
    #: ``rope_scaling`` when the checkpoint carries it, so the trainer uses
    #: the split the rollout configures rather than a copy of it.
    mrope_section: tuple[int, ...] = LANCE_MROPE_SECTION

    @classmethod
    def from_model_path(cls, model_path: str, *, trust_remote_code: bool = False) -> LanceTrainingConfig:
        """Build a config from a Lance checkpoint directory.

        Args:
            model_path: Either the bundle root (which owns ``Lance_3B/``) or a
                checkpoint directory containing ``llm_config.json``.
            trust_remote_code: Whether the tokenizer fallback may load custom code.

        Returns:
            Config with LLM dimensions from ``llm_config.json`` and Lance's
            latent geometry.
        """
        ckpt_dir = resolve_checkpoint_dir(model_path)
        with open(os.path.join(ckpt_dir, "llm_config.json")) as f:
            llm = json.load(f)

        start_of_image_id, end_of_image_id = _resolve_boundary_token_ids(
            ckpt_dir, llm, trust_remote_code=trust_remote_code
        )
        rope_scaling = llm.get("rope_scaling") or {}
        mrope_section = rope_scaling.get("mrope_section") or LANCE_MROPE_SECTION
        return cls(
            mrope_section=tuple(int(x) for x in mrope_section),
            hidden_size=llm["hidden_size"],
            intermediate_size=llm["intermediate_size"],
            num_hidden_layers=llm["num_hidden_layers"],
            num_attention_heads=llm["num_attention_heads"],
            num_key_value_heads=llm["num_key_value_heads"],
            vocab_size=llm["vocab_size"],
            rms_norm_eps=llm.get("rms_norm_eps", 1e-6),
            rope_theta=llm.get("rope_theta", 1_000_000.0),
            max_position_embeddings=llm.get("max_position_embeddings", 32768),
            start_of_image_id=start_of_image_id,
            end_of_image_id=end_of_image_id,
        )


def resolve_checkpoint_dir(model_path: str) -> str:
    """Return the directory that holds ``llm_config.json``.

    Accepts the bundle root (``.../Lance``) as well as a checkpoint
    directory (``.../Lance/Lance_3B``), so ``model.path`` can point at either.

    Args:
        model_path: Bundle root or checkpoint directory.

    Returns:
        Directory containing ``llm_config.json``.

    Raises:
        FileNotFoundError: If no ``llm_config.json`` is found.
    """
    if os.path.isfile(os.path.join(model_path, "llm_config.json")):
        return model_path
    for sub in (IMAGE_CKPT_DIR, VIDEO_CKPT_DIR):
        candidate = os.path.join(model_path, sub)
        if os.path.isfile(os.path.join(candidate, "llm_config.json")):
            return candidate
    raise FileNotFoundError(
        f"No llm_config.json under {model_path!r} or its {IMAGE_CKPT_DIR}/{VIDEO_CKPT_DIR} subdirectories."
    )


def _resolve_boundary_token_ids(ckpt_dir: str, llm_config: dict, *, trust_remote_code: bool) -> tuple[int, int]:
    """Resolve the image boundary token IDs for a Lance checkpoint.

    ``llm_config.json`` carries ``vision_start_token_id`` /
    ``vision_end_token_id``; prefer those.  Otherwise fall back to the
    tokenizer, matching the rollout. Invalid tokenizer data must fail rather
    than silently selecting unrelated token IDs.

    Args:
        ckpt_dir: Checkpoint directory, which also holds the tokenizer.
        llm_config: Parsed ``llm_config.json``.
        trust_remote_code: Whether the tokenizer may load custom code.

    Returns:
        ``(start_of_image_id, end_of_image_id)``.
    """
    start = llm_config.get("vision_start_token_id")
    end = llm_config.get("vision_end_token_id")
    if start is not None and end is not None:
        return int(start), int(end)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=trust_remote_code)
    start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
    if start is None or end is None or tokenizer.unk_token_id in (start, end):
        raise ValueError(f"Lance checkpoint {ckpt_dir!r} is missing vision boundary tokens.")
    return int(start), int(end)


def map_lance_checkpoint_to_training(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map ``Lance_3B/model.safetensors`` keys to training parameter names.

    Lance uses BAGEL's key layout: ``language_model.model.*`` for the MoT
    transformer plus top-level ``time_embedder.`` / ``vae2llm.`` / ``llm2vae.``
    / ``latent_pos_embed.``.  Everything else - ``language_model.lm_head.*``
    (unused for flow matching) and ``vit_model.*`` (understanding only) - is
    dropped.

    Args:
        state_dict: Raw checkpoint tensors.

    Returns:
        Tensors keyed by :class:`LanceForTraining` parameter names.
    """
    mapped: dict[str, Tensor] = {}
    for src_key, tensor in state_dict.items():
        if src_key.startswith("language_model.model."):
            mapped[src_key[len("language_model.model.") :]] = tensor
        elif src_key.startswith(("time_embedder.", "vae2llm.", "llm2vae.", "latent_pos_embed.")):
            mapped[src_key] = tensor
    return mapped


def load_lance_state_dict(ckpt_dir: str) -> dict[str, Tensor]:
    """Load ``model.safetensors``, following the shard index when present.

    Args:
        ckpt_dir: Directory holding the checkpoint.

    Returns:
        The merged state dict.

    Raises:
        FileNotFoundError: If neither a single file nor a shard index exists.
    """
    from safetensors.torch import load_file

    single = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.isfile(single):
        return load_file(single)

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        state_dict: dict[str, Tensor] = {}
        for shard in sorted(set(index["weight_map"].values())):
            state_dict.update(load_file(os.path.join(ckpt_dir, shard)))
        return state_dict

    raise FileNotFoundError(f"No model.safetensors or model.safetensors.index.json in {ckpt_dir!r}.")


class LanceRotaryEmbedding(RotaryEmbedding):
    """Qwen2.5-VL multimodal RoPE, as the Lance rollout applies it.

    The rollout keeps ``rope_scaling = {"rope_type": "mrope", "mrope_section":
    [16, 24, 24]}`` on the language model and feeds the generation block
    per-token ``(t, h, w)`` positions, so ``BagelRotaryEmbedding`` takes its
    multimodal branch.  The section split below is that branch: per-axis
    frequencies, then a basis assembled by cycling ``axis = i % 3`` over the
    doubled ``mrope_section``.  Scalar positions still go through the 1-D path,
    which keeps the text prefix identical to BAGEL's.
    """

    def __init__(
        self,
        head_dim: int,
        theta: float = 1_000_000.0,
        mrope_section: Sequence[int] | None = None,
    ):
        super().__init__(head_dim, theta=theta)
        self.mrope_section = list(mrope_section or LANCE_MROPE_SECTION)
        if sum(self.mrope_section) * 2 != head_dim:
            raise ValueError(
                f"mrope_section {self.mrope_section} sums to {sum(self.mrope_section)}, "
                f"which is not half of head_dim {head_dim}"
            )

    def forward(self, position_ids: torch.Tensor):
        if position_ids.ndim != 3:
            return super().forward(position_ids)

        batch = position_ids.shape[0]
        inv_freq = self.inv_freq.to(position_ids.device).float()
        inv_freq_expanded = inv_freq[None, None, :, None].expand(batch, 3, -1, 1)
        positions = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ positions).transpose(2, 3)
        emb = torch.cat((freqs, freqs), dim=-1)
        # ``mrope_section`` sums to head_dim / 2; doubled it spans the head and
        # the axis cycles t, h, w, t, h, w.
        sections = self.mrope_section * 2
        cos = torch.cat([c[:, i % 3] for i, c in enumerate(emb.cos().split(sections, dim=-1))], dim=-1)
        sin = torch.cat([s[:, i % 3] for i, s in enumerate(emb.sin().split(sections, dim=-1))], dim=-1)
        return cos, sin


def get_3d_sincos_pos_embed(embed_dim: int, t: int, h: int, w: int) -> np.ndarray:
    """3-D sin-cos positional embedding over ``(t, h, w)``.

    The dimension split matches vllm-omni's
    ``lance_transformer.get_3d_sincos_pos_embed`` and upstream Lance
    ``modeling/lance/modeling_utils.py``; the checkpoint supplies the values, so
    this only has to build the table at the shape the checkpoint uses.
    """
    tt, hh, ww = np.meshgrid(
        np.arange(t, dtype=np.float32),
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    d = embed_dim // 3
    d = d if d % 2 == 0 else d - 1
    emb_t = _get_1d_sincos_pos_embed_from_grid(d, tt)
    emb_h = _get_1d_sincos_pos_embed_from_grid(d, hh)
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim - 2 * d, ww)
    return np.concatenate([emb_t, emb_h, emb_w], axis=1)


class LancePositionEmbedding3D(nn.Module):
    """Frozen 3-D latent position embedding, matching the rollout's table.

    BAGEL ships a 2-D table for image latents; ``Lance_3B_Video`` adds a
    temporal axis and stores ``(max_num_frames * side**2, hidden)`` rows, which
    the trainer indexes with the same flattened ``t * side**2 + h * side + w``
    ids the rollout uses.  The image checkpoint's table is the ``t = 1`` case.
    """

    def __init__(self, max_num_frames: int, max_num_patch_per_side: int, hidden_size: int):
        super().__init__()
        n = max_num_frames * max_num_patch_per_side * max_num_patch_per_side
        self.pos_embed = nn.Parameter(torch.zeros(n, hidden_size), requires_grad=False)
        table = get_3d_sincos_pos_embed(hidden_size, max_num_frames, max_num_patch_per_side, max_num_patch_per_side)
        self.pos_embed.data.copy_(torch.from_numpy(table).float())

    def forward(self, position_ids: Tensor) -> Tensor:
        return self.pos_embed[position_ids]


class LanceForTraining(BagelForTraining):
    """Lance MoT module for FlowGRPO FSDP training.

    Reuses BAGEL's layers and parameter names with Lance's checkpoint layout,
    latent geometry, and three-axis rotary positions.
    """

    def __init__(self, config: LanceTrainingConfig):
        super().__init__(config)
        if config.max_num_frames > 1:
            # The base built BAGEL's image table; a video checkpoint's table
            # covers every frame, so rebuild it at the checkpoint's row count.
            self.latent_pos_embed = LancePositionEmbedding3D(
                config.max_num_frames, config.max_latent_size, config.hidden_size
            )
        # The rollout runs the generation block through Qwen2.5-VL mRoPE.  The
        # BAGEL layer builds a 1-D rotary, so swap the module per layer rather
        # than duplicate the layer class; everything else about the layer is
        # shared with BAGEL.
        for layer in self.layers:
            layer.rotary_emb = LanceRotaryEmbedding(
                config.head_dim,
                theta=config.rope_theta,
                mrope_section=config.mrope_section,
            )

    def build_position_ids(
        self,
        batch: int,
        num_text: int,
        num_latent: int,
        latent_pos_ids: Tensor,
        device,
        *,
        text_attention_mask: Tensor | None = None,
        position_anchor: Tensor | None = None,
    ) -> Tensor:
        """Per-token ``(t, h, w)`` positions, matching what the rollout feeds.

        ``LanceBagel.prepare_vae_latent`` replaces the scalar positions BAGEL
        would use with, for a latent grid anchored at ``P``::

            start_of_image  -> (P,            P,            P)
            latent (hi, wi) -> (P + 1,        P + 1 + hi,   P + 1 + wi)
            end_of_image    -> (P + max_hw + 1, same, same)

        with ``P`` the number of valid text tokens in each sample. Padding
        occupies sequence slots but must not advance the image's positions.

        The grid coordinates come from ``latent_pos_ids``, which
        ``get_flattened_position_ids`` builds as ``hi * max_latent_size + wi``.

        Args:
            batch: Batch size.
            num_text: Text context length.
            num_latent: Number of latent tokens.
            latent_pos_ids: ``(L_latent,)`` or ``(B, L_latent)`` grid indices.
            device: Device for the returned tensor.
            text_attention_mask: ``(B, num_text)`` mask for right-padded text.

        Returns:
            ``(B, 3, L_total)`` long tensor, rows ordered ``(t, h, w)``.
        """
        grid = latent_pos_ids.to(device=device, dtype=torch.long)
        if grid.ndim == 1:
            grid = grid.unsqueeze(0).expand(batch, -1)
        if grid.shape != (batch, num_latent):
            raise ValueError(f"latent_pos_ids has shape {tuple(grid.shape)}, expected {(batch, num_latent)}")
        side = int(self.config.max_latent_size)
        stride = side * side
        frames = torch.div(grid, stride, rounding_mode="floor")
        within = grid % stride
        rows = torch.div(within, side, rounding_mode="floor")
        cols = within % side
        if position_anchor is not None:
            # The rollout's own anchor for this block (Lance's edit modes place the
            # noise latents at the reference VAE block's positions).
            anchor = position_anchor.to(device=device, dtype=torch.long).reshape(batch, 1)
        elif text_attention_mask is None:
            anchor = grid.new_full((batch, 1), num_text)
        else:
            anchor = text_attention_mask.to(device=device, dtype=torch.bool).sum(dim=-1, keepdim=True)

        # Extents and the end marker, mirroring
        # ``LanceBagel._per_token_mrope_for_video_latent``.  A single frame leaves
        # the temporal term at zero, so the image path is unchanged: max_thw
        # collapses to max(h, w) and latent_t to anchor + 1.
        t_lat = frames.amax(dim=-1, keepdim=True) + 1
        h_lat = rows.amax(dim=-1, keepdim=True) + 1
        w_lat = cols.amax(dim=-1, keepdim=True) + 1
        max_thw = torch.maximum((t_lat - 1) * LANCE_TEMPORAL_ROPE_SCALE, torch.maximum(h_lat, w_lat) - 1) + 1

        text = torch.arange(num_text, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        end = anchor + max_thw + 1
        latent_t = anchor + 1 + frames * LANCE_TEMPORAL_ROPE_SCALE

        axes = [
            torch.cat([text, anchor, latent_t, end], dim=-1),
            torch.cat([text, anchor, anchor + 1 + rows, end], dim=-1),
            torch.cat([text, anchor, anchor + 1 + cols, end], dim=-1),
        ]
        return torch.stack(axes, dim=1)

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        text_token_ids: Optional[Tensor] = None,
        latent_pos_ids: Optional[Tensor] = None,
        condition: dict | None = None,
        **kwargs,
    ) -> tuple[Tensor]:
        """Replay a step, with or without rollout-exported conditioning rows.

        ``condition`` carries what ``LancePipeline._forward_image_edit`` prefilled
        through its ViT and VAE: this model has neither tower, so those rows and
        the rotary basis they were placed on travel with the trajectory.  Without
        it the plain text-to-image / text-to-video path runs unchanged.
        """
        if condition is None:
            return super().forward(hidden_states, timestep, text_token_ids, latent_pos_ids, **kwargs)
        return self._forward_conditioned(hidden_states, timestep, condition)

    def _forward_conditioned(self, hidden_states: Tensor, timestep: Tensor, condition: dict) -> tuple[Tensor]:
        """Assemble the edit context from exported rows and predict the velocity."""
        B = hidden_states.shape[0]
        L_latent = hidden_states.shape[1]
        dev = hidden_states.device
        rows_dtype = self.embed_tokens.weight.dtype

        prefix = self.embed_tokens(condition["condition_prefix_ids"].to(dev))
        tail = self.embed_tokens(condition["condition_tail_ids"].to(dev))
        ref_rows = condition["condition_ref_rows"].to(device=dev, dtype=rows_dtype)
        context = torch.cat([prefix, ref_rows, tail], dim=1)
        L_prefix = prefix.shape[1]
        L_ref = ref_rows.shape[1]
        L_ctx = context.shape[1]

        # Context rows carry one scalar rope id broadcast across (t, h, w); the
        # gen latent block's own positions - markers included - come verbatim.
        def broadcast(positions: Tensor) -> Tensor:
            positions = positions.to(dev).reshape(B, -1)
            return positions.unsqueeze(1).expand(B, 3, -1)

        def as_mrope(positions: Tensor) -> Tensor:
            positions = positions.to(dev)
            if positions.dim() == 3:
                # Real per-axis (t, h, w) positions for a reference's image rows.
                return positions if positions.shape[1] == 3 else positions.transpose(1, 2)
            positions = positions.reshape(B, -1)
            return positions.unsqueeze(1).expand(B, 3, -1)

        position_ids = torch.cat(
            [
                broadcast(condition["condition_prefix_positions"]),
                as_mrope(condition["condition_ref_positions"]),
                broadcast(condition["condition_tail_positions"]),
                condition["condition_latent_positions"].to(dev).expand(B, -1, -1),
            ],
            dim=-1,
        )

        soi_emb = self.embed_tokens(torch.full((B, 1), self.config.start_of_image_id, dtype=torch.long, device=dev))
        eoi_emb = self.embed_tokens(torch.full((B, 1), self.config.end_of_image_id, dtype=torch.long, device=dev))
        t_emb = self.time_embedder(timestep)
        pos_emb = self.latent_pos_embed(condition["condition_latent_grid"].to(dev))
        if t_emb.ndim == 2:
            t_emb = t_emb.unsqueeze(1)
        latent_embeds = self.vae2llm(hidden_states) + t_emb + pos_emb
        sequence = torch.cat([context, soi_emb, latent_embeds.to(soi_emb.dtype), eoi_emb], dim=1)

        # Routing: the reference's VAE rows and the noise latents use the
        # generation expert, everything else the understanding one.
        is_gen_ctx = condition["condition_ref_is_gen"].to(dev)
        text_mask = torch.ones(B, L_ctx + 2 + L_latent, dtype=torch.bool, device=dev)
        text_mask[:, L_prefix : L_prefix + L_ref] = ~is_gen_ctx
        text_mask[:, L_ctx + 1 : L_ctx + 1 + L_latent] = False  # noise latents are gen
        text_mask[:, -1] = True  # end marker
        latent_mask = ~text_mask

        # Padded tail slots travel only to keep the batch rectangular; they must
        # not be attended to.
        keep = torch.ones(B, L_ctx, dtype=torch.bool, device=dev)
        keep[:, L_prefix + L_ref :] = condition["condition_tail_mask"].to(dev)
        key_padding_mask = torch.cat(
            [
                keep,
                torch.ones(B, 2 + L_latent, dtype=torch.bool, device=dev),
            ],
            dim=1,
        )

        # The rollout prefilled this sequence segment by segment: causal text, a
        # fully-visible reference, causal text, then the fully-visible latent
        # block.  A single causal split cannot express that, so hand the layers
        # the block ranges (query_start, query_end, key_end, causal bound).
        # The reference VAE and generated latents share one grid, including two markers.
        L_vit = L_ref - L_latent - 2
        if L_vit < 1:
            raise ValueError("Lance edit replay requires ViT rows followed by a VAE block matching the latent grid.")
        attn_plan = [
            (0, L_prefix, L_prefix, 0),
            (L_prefix, L_prefix + L_vit, L_prefix + L_vit, None),
            (L_prefix + L_vit, L_prefix + L_ref, L_prefix + L_ref, None),
            (L_prefix + L_ref, L_ctx, L_ctx, L_prefix + L_ref),
            (L_ctx, L_ctx + 2 + L_latent, L_ctx + 2 + L_latent, None),
        ]

        for layer in self.layers:

            def _layer_fn(seq, pos_ids, text_mask_, latent_mask_, kpm, *, _layer=layer):
                return _layer(
                    seq,
                    pos_ids,
                    text_mask_,
                    latent_mask_,
                    L_ctx,
                    key_padding_mask=kpm,
                    attn_plan=attn_plan,
                )

            sequence = self._checkpointed_call(
                _layer_fn, sequence, position_ids, text_mask, latent_mask, key_padding_mask
            )

        normed = sequence.new_zeros(sequence.shape)
        t_idx = text_mask.nonzero(as_tuple=True)
        l_idx = latent_mask.nonzero(as_tuple=True)
        normed[t_idx] = self.norm(sequence[t_idx])
        normed[l_idx] = self.norm_moe_gen(sequence[l_idx])
        latent_output = normed[:, L_ctx + 1 : L_ctx + 1 + L_latent, :]
        return (self.llm2vae(latent_output),)

    @classmethod
    def from_pretrained(
        cls, model_path: str, torch_dtype=torch.bfloat16, *, trust_remote_code: bool = False
    ) -> LanceForTraining:
        """Load a Lance checkpoint.

        Shapes that the released checkpoint decides - the extended vocabulary
        and the latent position table - are taken from the tensors instead of
        the config, so a checkpoint whose tokenizer added tokens still loads.

        Args:
            model_path: Bundle root or checkpoint directory.
            torch_dtype: Target dtype.
            trust_remote_code: Whether the tokenizer fallback may load custom code.

        Returns:
            A ``LanceForTraining`` with the checkpoint applied.
        """
        ckpt_dir = resolve_checkpoint_dir(model_path)
        config = LanceTrainingConfig.from_model_path(ckpt_dir, trust_remote_code=trust_remote_code)
        state_dict = load_lance_state_dict(ckpt_dir)
        mapped = map_lance_checkpoint_to_training(state_dict)

        embed_weight = mapped.get("embed_tokens.weight")
        if embed_weight is not None and embed_weight.shape[0] != config.vocab_size:
            logger.info(
                "Lance checkpoint carries vocab_size=%d, overriding llm_config.json's %d.",
                embed_weight.shape[0],
                config.vocab_size,
            )
            config.vocab_size = int(embed_weight.shape[0])

        pos_embed = mapped.get("latent_pos_embed.pos_embed")
        if pos_embed is not None:
            rows = int(pos_embed.shape[0])
            side = int(config.max_latent_size)
            if rows % (side * side) == 0:
                # ``(max_num_frames * side**2, hidden)``.  One frame is the image
                # table; the video checkpoint carries 31 frames (``126976`` rows).
                frames = rows // (side * side)
                config.max_num_frames = frames
                if frames > 1:
                    logger.info(
                        "Lance checkpoint carries a %d-frame video position table (%d rows, side %d).",
                        frames,
                        rows,
                        side,
                    )
            else:
                grid = int(round(rows**0.5))
                if grid * grid != rows:
                    raise ValueError(
                        f"latent_pos_embed.pos_embed has {rows} rows, which is neither a square grid nor "
                        f"a whole number of frames at max_latent_size={side}."
                    )
                if grid != config.max_latent_size:
                    logger.info(
                        "Lance checkpoint carries max_latent_size=%d, overriding %d.", grid, config.max_latent_size
                    )
                    config.max_latent_size = grid

        model = cls(config)
        model.load_state_dict(mapped, strict=True)
        return model.to(torch_dtype)
