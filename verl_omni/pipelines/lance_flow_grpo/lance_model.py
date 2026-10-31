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

import torch
from torch import Tensor

from verl_omni.pipelines.bagel_flow_grpo.bagel_model import (
    BagelForTraining,
    BagelTrainingConfig,
    RotaryEmbedding,
)

from .common import (
    LANCE_LATENT_PATCH_SIZE,
    LANCE_MAX_LATENT_SIZE,
    LANCE_MROPE_SECTION,
    LANCE_VAE_DOWNSAMPLE_SPATIAL,
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
    latent_channel: int = LANCE_VAE_Z_CHANNELS
    vae_downsample: int = LANCE_VAE_DOWNSAMPLE_SPATIAL
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


class LanceForTraining(BagelForTraining):
    """Lance MoT module for FlowGRPO FSDP training.

    Reuses BAGEL's layers and parameter names with Lance's checkpoint layout,
    latent geometry, and three-axis rotary positions.
    """

    def __init__(self, config: LanceTrainingConfig):
        super().__init__(config)
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
        rows = torch.div(grid, side, rounding_mode="floor")
        cols = grid % side
        if text_attention_mask is None:
            anchor = grid.new_full((batch, 1), num_text)
        else:
            anchor = text_attention_mask.to(device=device, dtype=torch.bool).sum(dim=-1, keepdim=True)
        max_hw = torch.maximum(rows.amax(dim=-1, keepdim=True), cols.amax(dim=-1, keepdim=True)) + 1

        text = torch.arange(num_text, device=device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
        end = anchor + max_hw + 1
        latent_t = (anchor + 1).expand(-1, num_latent)

        axes = [
            torch.cat([text, anchor, latent_t, end], dim=-1),
            torch.cat([text, anchor, anchor + 1 + rows, end], dim=-1),
            torch.cat([text, anchor, anchor + 1 + cols, end], dim=-1),
        ]
        return torch.stack(axes, dim=1)

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
            grid = int(round(pos_embed.shape[0] ** 0.5))
            if grid * grid != pos_embed.shape[0]:
                raise ValueError(
                    f"latent_pos_embed.pos_embed has {pos_embed.shape[0]} rows, which is not a square "
                    "grid; this is the 3-D video table, which the image path does not support."
                )
            if grid != config.max_latent_size:
                logger.info("Lance checkpoint carries max_latent_size=%d, overriding %d.", grid, config.max_latent_size)
                config.max_latent_size = grid

        model = cls(config)
        model.load_state_dict(mapped, strict=True)
        return model.to(torch_dtype)
