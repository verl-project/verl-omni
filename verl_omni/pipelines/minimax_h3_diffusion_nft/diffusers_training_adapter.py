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
"""MiniMax H3 training adapter for DiffusionNFT."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from itertools import accumulate
from types import MethodType
from typing import Optional

import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Attention,
    MiniMaxH3AttnProcessor,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerOutput,
    _apply_rotary_emb,
)
from diffusers.utils import apply_lora_scale
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    build_layout_from_meta,
    build_ref2va_layout_from_meta,
    build_row_timesteps,
    h3_dit_timestep,
    h3_velocity_to_flow_match,
    keyframe_indices_to_anchors,
    pack_video_audio_rows,
    prepare_h3_processor_files,
    split_dual_velocity,
    unpack_video_audio_rows,
    validate_lora_target_modules,
)

__all__ = ["MiniMaxH3DiffusionNFT"]


@lru_cache(maxsize=1)
def _get_fa3_varlen():
    from kernels import get_kernel

    return get_kernel("kernels-community/flash-attn3", version=1).flash_attn_varlen_func


@dataclass(frozen=True)
class PackedSequenceLayout:
    """Per-sample attention boundaries for one packed micro-batch."""

    cu_seqlens: torch.Tensor
    max_seqlen: int
    lengths: tuple[int, ...]
    total_tokens: int

    @classmethod
    def from_lengths(cls, lengths: list[int], device: torch.device):
        if not lengths or any(length <= 0 for length in lengths):
            raise ValueError("Packed H3 sequences must have positive lengths.")
        return cls(
            torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device=device),
            max(lengths),
            tuple(lengths),
            sum(lengths),
        )

    @cached_property
    def valid_mask(self):
        device = self.cu_seqlens.device
        return torch.arange(self.max_seqlen, device=device)[None] < torch.tensor(self.lengths, device=device)[:, None]

    @cached_property
    def padded_indices(self):
        return self.valid_mask.flatten().nonzero().flatten()

    def attention(self, query, key, value, backend):
        if backend == "_flash_3_varlen_hub":
            return _get_fa3_varlen()(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                cu_seqlens_q=self.cu_seqlens,
                cu_seqlens_k=self.cu_seqlens,
                max_seqlen_q=self.max_seqlen,
                max_seqlen_k=self.max_seqlen,
                causal=False,
            ).unsqueeze(0)
        if backend == "torch_varlen":
            from torch.nn.attention.varlen import varlen_attn

            return varlen_attn(
                query.squeeze(0),
                key.squeeze(0),
                value.squeeze(0),
                self.cu_seqlens,
                self.cu_seqlens,
                self.max_seqlen,
                self.max_seqlen,
            ).unsqueeze(0)
        if backend != "native":
            raise ValueError(f"Unsupported packed H3 attention backend: {backend!r}.")

        batch = self.valid_mask.shape[0]

        def pad(tensor):
            padded = tensor.new_zeros((batch * self.max_seqlen, *tensor.shape[2:]))
            return (
                padded.index_copy(0, self.padded_indices, tensor.squeeze(0))
                .view(batch, self.max_seqlen, *tensor.shape[2:])
                .transpose(1, 2)
            )

        output = F.scaled_dot_product_attention(
            pad(query),
            pad(key),
            pad(value),
            attn_mask=self.valid_mask[:, None, None, :],
            dropout_p=0.0,
        )
        return output.transpose(1, 2).flatten(0, 1).index_select(0, self.padded_indices).unsqueeze(0)


def pack_model_inputs(samples: list[dict]) -> dict:
    """Merge H3 samples while retaining their attention and timestep boundaries."""
    if not samples:
        raise ValueError("Cannot pack an empty H3 batch.")
    device = samples[0]["hidden_states"].device
    lengths = [sample["token_tags"].numel() for sample in samples]
    text_lengths = [sample["text_indices"].numel() for sample in samples]
    offsets = [0, *accumulate(lengths[:-1])]
    packed = {
        key: torch.cat([sample[key] for sample in samples], dim=1)
        for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states")
    }
    for key in ("video_indices", "audio_indices", "text_indices"):
        packed[key] = torch.cat([sample[key] + offset for sample, offset in zip(samples, offsets, strict=True)]).to(
            device
        )
    for key in ("position_ids", "token_tags"):
        packed[key] = torch.cat([sample[key] for sample in samples]).to(device)
    timesteps, table_indices = torch.unique(
        torch.cat([sample["timestep"] for sample in samples]), sorted=True, return_inverse=True
    )
    tables = table_indices.split([sample["timestep"].numel() for sample in samples])
    indices = torch.cat([table[sample["timestep_indices"]] for table, sample in zip(tables, samples, strict=True)])
    packed["timestep"], packed["timestep_indices"] = timesteps.to(device), indices.to(device)
    packed["sequence_layout"] = PackedSequenceLayout.from_lengths(lengths, device)
    packed["text_sequence_layout"] = PackedSequenceLayout.from_lengths(text_lengths, device)
    packed["return_dict"] = False
    return packed


class _PackedAttnProcessor(MiniMaxH3AttnProcessor):
    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if not isinstance(attention_mask, PackedSequenceLayout):
            raise ValueError("Packed H3 attention requires explicit sample boundaries.")
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if rotary_emb is not None:
            query, key = _apply_rotary_emb(query, *rotary_emb), _apply_rotary_emb(key, *rotary_emb)
        output = attention_mask.attention(query, key, value, self._attention_backend)
        return attn.to_out[1](attn.to_out[0](output.flatten(2, 3).type_as(query)))


def _packed_refiner_block_forward(self, hidden_states, sequence_layout):
    hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=sequence_layout)
    return hidden_states + self.ff(self.norm2(hidden_states))


def _packed_refiner_forward(self, hidden_states, sequence_layout):
    for block in self.refiner_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(block, hidden_states, sequence_layout)
        else:
            hidden_states = block(hidden_states, sequence_layout)
    return self.final_norm(hidden_states)


def _set_packed_attention_backend(self, backend):
    if backend not in {"native", "torch_varlen", "_flash_3_varlen_hub"}:
        raise ValueError("Packed H3 requires attn_backend=_flash_3_varlen_hub, torch_varlen or native.")
    if backend == "_flash_3_varlen_hub":
        _get_fa3_varlen()
    elif backend == "torch_varlen":
        from torch.nn.attention.varlen import varlen_attn  # noqa: F401
    for module in self.modules():
        if isinstance(module, MiniMaxH3Attention):
            module.processor._attention_backend = backend


@apply_lora_scale("attention_kwargs")
def _packed_transformer_forward(
    self,
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    timestep,
    timestep_indices,
    token_tags,
    position_ids,
    video_indices,
    audio_indices,
    text_indices,
    sequence_layout,
    text_sequence_layout,
    attention_kwargs=None,
    return_dict=True,
):
    if hidden_states.shape[0] != 1 or audio_hidden_states.shape[0] != 1 or encoder_hidden_states.shape[0] != 1:
        raise ValueError("Packed H3 expects a singleton outer dimension and concatenated sequence rows.")
    sequence_length = position_ids.shape[0]
    if (
        position_ids.shape != (sequence_length, 3)
        or token_tags.shape != (sequence_length,)
        or timestep_indices.shape != (sequence_length,)
    ):
        raise ValueError("Packed H3 positions, token tags and timesteps must describe the same sequence.")
    if sequence_layout.total_tokens != sequence_length:
        raise ValueError("Packed H3 attention boundaries do not match the sequence length.")
    if text_sequence_layout.total_tokens != encoder_hidden_states.shape[1]:
        raise ValueError("Packed H3 text attention boundaries do not match the text length.")

    rotary_emb = self.rope(position_ids)
    video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
    audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
    text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
    text_embeds = self.token_refiner(text_embeds, text_sequence_layout)
    hidden_states = text_embeds.new_zeros((1, sequence_length, text_embeds.shape[-1]))
    hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
    hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
    hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

    temb = self.time_embedder(self.time_proj(timestep).to(get_parameter_dtype(self.time_embedder)))
    adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block, hidden_states, temb, adaln_indices, rotary_emb, sequence_layout
            )
        else:
            hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb, sequence_layout)
    hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(get_parameter_dtype(self.proj_out))
    video_output = self.proj_out(hidden_states).index_select(1, video_indices)
    audio_output = self.audio_proj_out(hidden_states).index_select(1, audio_indices)
    if not return_dict:
        return video_output, audio_output
    return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)


def _find_h3_transformer(module):
    queue, seen = [module], set()
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, MiniMaxH3Transformer3DModel):
            return candidate
        if isinstance(candidate, nn.Module):
            queue.extend(candidate._modules.values())
    raise TypeError("MiniMax H3 packed forward requires a MiniMaxH3Transformer3DModel.")


def enable_packed_forward(module):
    """Install sample-isolated packed execution on an AutoModel-loaded H3 transformer."""
    transformer = _find_h3_transformer(module)
    if getattr(transformer, "supports_packed_batch", False):
        return transformer
    attentions = [item for item in transformer.modules() if isinstance(item, MiniMaxH3Attention)]
    if any(getattr(item.processor, "_parallel_config", None) is not None for item in attentions):
        raise NotImplementedError("Packed H3 batch forward does not yet support context/sequence parallelism.")
    backends = {getattr(item.processor, "_attention_backend", None) for item in attentions}
    if len(backends) != 1:
        raise ValueError(f"MiniMax H3 attention modules use inconsistent backends: {sorted(map(str, backends))}.")
    backend = backends.pop() or "native"
    for item in attentions:
        processor = _PackedAttnProcessor()
        processor._attention_backend = backend
        item.set_processor(processor)
    for block in transformer.token_refiner.refiner_blocks:
        block.forward = MethodType(_packed_refiner_block_forward, block)
    transformer.token_refiner.forward = MethodType(_packed_refiner_forward, transformer.token_refiner)
    transformer.forward = MethodType(_packed_transformer_forward, transformer)
    transformer.set_attention_backend = MethodType(_set_packed_attention_backend, transformer)
    transformer.supports_packed_batch = True
    transformer.set_attention_backend(backend)
    return transformer


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="diffusion_nft")
class MiniMaxH3DiffusionNFT(DiffusionModelBase):
    """Forward-process MiniMax H3 adapter used by DiffusionNFT."""

    @classmethod
    def validate_lora_config(cls, model_config: DiffusionModelConfig) -> None:
        """Reject LoRA targets the rollout weight sync cannot transport (shares common.py whitelist)."""
        if model_config.lora_rank > 0:
            validate_lora_target_modules(model_config.target_modules)

    @classmethod
    def prepare_processor_files(cls, model_path: str) -> str:
        """Make the official Qwen3-VL processor discoverable by AutoProcessor."""
        return prepare_h3_processor_files(model_path)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        """Build the video-shifted rectified-flow scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler

        pipeline = model_config.pipeline
        scheduler = FlowMatchEulerDiscreteScheduler(shift=pipeline.get("video_flow_shift", 12.0))
        cls.set_timesteps(scheduler, model_config, device="cpu")
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler, model_config: DiffusionModelConfig, device: str):
        """Set video-stream timesteps."""
        scheduler.set_timesteps(model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        """Unpack joint latents and prepare H3 transformer inputs."""
        del step, negative_prompt_embeds, negative_prompt_embeds_mask
        if not torch.all(micro_batch["latent_meta"] == micro_batch["latent_meta"][0]):
            raise ValueError("Packed H3 currently requires a shared target latent layout within each micro-batch.")
        keyframes = micro_batch.get("keyframe_frame_indices", None)
        if keyframes is not None and not torch.all(keyframes == keyframes[0]):
            raise ValueError("Packed H3 currently requires shared FL2VA keyframe anchors within each micro-batch.")
        meta = micro_batch["latent_meta"][0].reshape(-1).tolist()
        num_video_rows, num_audio_rows = int(meta[0]), int(meta[1])
        video_rows, audio_rows = unpack_video_audio_rows(latents, num_video_rows, num_audio_rows)
        condition_video_rows = micro_batch.get("condition_video_rows", None)
        if condition_video_rows is None:
            condition_video_rows = video_rows.new_zeros((video_rows.shape[0], 0, video_rows.shape[-1]))
        condition_audio_rows = micro_batch.get("condition_audio_rows", None)
        if condition_audio_rows is None:
            condition_audio_rows = audio_rows.new_zeros((audio_rows.shape[0], 0, audio_rows.shape[-1]))
        frame_indices = micro_batch.get("keyframe_frame_indices", None)
        frame_indices = [] if frame_indices is None else frame_indices[0].reshape(-1).tolist()
        prompt_token_tags = micro_batch.get("prompt_token_tags", None)

        model_inputs = {
            "video_rows": video_rows,
            "audio_rows": audio_rows,
            "condition_video_rows": condition_video_rows,
            "condition_audio_rows": condition_audio_rows,
            # Per-sample counts undo the cross-worker padding; None keeps the pre-multi-reference behavior.
            "condition_video_row_count": micro_batch.get("condition_video_row_count", None),
            "condition_audio_row_count": micro_batch.get("condition_audio_row_count", None),
            "keyframe_anchors": keyframe_indices_to_anchors(frame_indices),
            "ref_block_meta": micro_batch.get("ref_block_meta", None),
            "ref_block_count": micro_batch.get("ref_block_count", None),
            "prompt_token_tags": prompt_token_tags,
            "encoder_hidden_states": prompt_embeds,
            "encoder_mask": prompt_embeds_mask,
            "timestep": h3_dit_timestep(timesteps.float()),
            "latent_meta": meta,
        }
        return model_inputs, None

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict,
        negative_model_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Return target-only flow-match velocities from one packed H3 forward."""
        del negative_model_inputs
        if isinstance(module, nn.Module):
            enable_packed_forward(module)
        samples = list(cls._iter_sample_inputs(module, model_inputs))
        inputs = [sample[0] for sample in samples]
        video, audio = split_dual_velocity(module(**pack_model_inputs(inputs)))
        results = zip(
            video.split([item["hidden_states"].shape[1] for item in inputs], dim=1),
            audio.split([item["audio_hidden_states"].shape[1] for item in inputs], dim=1),
            strict=True,
        )

        packed_velocities = []
        for (v_video, v_audio), (_, num_cond_video, num_cond_audio) in zip(results, samples, strict=True):
            packed_velocities.append(
                pack_video_audio_rows(
                    h3_velocity_to_flow_match(v_video[:, num_cond_video:]),
                    h3_velocity_to_flow_match(v_audio[:, num_cond_audio:]),
                )
            )
        return torch.cat(packed_velocities, dim=0)

    @classmethod
    def _iter_sample_inputs(cls, module, model_inputs):
        video_rows = model_inputs["video_rows"]
        audio_rows = model_inputs["audio_rows"]
        condition_video_rows = model_inputs["condition_video_rows"]
        condition_audio_rows = model_inputs["condition_audio_rows"]
        condition_video_row_count = model_inputs["condition_video_row_count"]
        condition_audio_row_count = model_inputs["condition_audio_row_count"]
        keyframe_anchors = model_inputs["keyframe_anchors"]
        ref_block_meta = model_inputs["ref_block_meta"]
        ref_block_count = model_inputs["ref_block_count"]
        prompt_token_tags = model_inputs["prompt_token_tags"]
        encoder_hidden_states = model_inputs["encoder_hidden_states"]
        encoder_mask = model_inputs["encoder_mask"]
        timestep = model_inputs["timestep"]
        meta = model_inputs["latent_meta"]
        device = video_rows.device
        raw_patch = getattr(getattr(module, "config", None), "patch_size", (1, 2, 2))
        patch_size = (int(raw_patch[0]), int(raw_patch[1]), int(raw_patch[2]))

        batch = video_rows.shape[0]
        if encoder_mask is not None:
            text_lengths = encoder_mask.long().sum(dim=1).tolist()  # one host sync per micro-batch
        else:
            text_lengths = [encoder_hidden_states.shape[1]] * batch

        for index in range(batch):
            num_text_tokens = int(text_lengths[index])
            sample_text_tags = None if prompt_token_tags is None else prompt_token_tags[index, :num_text_tokens]
            if ref_block_meta is None:
                layout = build_layout_from_meta(
                    meta,
                    num_text_tokens,
                    patch_size,
                    keyframe_anchors=keyframe_anchors,
                    text_token_tags=sample_text_tags,
                )
            else:
                if ref_block_count is None:
                    raise ValueError("MiniMax H3 Ref2VA requires ref_block_count.")
                layout = build_ref2va_layout_from_meta(
                    meta,
                    num_text_tokens,
                    ref_block_meta[index],
                    int(ref_block_count[index].reshape(-1)[0]),
                    text_token_tags=sample_text_tags,
                )
            position_ids, token_tags, video_indices, audio_indices, text_indices, num_cond_video, num_cond_audio = (
                layout
            )
            sample_video_condition = condition_video_rows[index]
            if condition_video_row_count is not None:
                sample_video_condition = sample_video_condition[: int(condition_video_row_count[index].reshape(-1)[0])]
            sample_audio_condition = condition_audio_rows[index]
            if condition_audio_row_count is not None:
                sample_audio_condition = sample_audio_condition[: int(condition_audio_row_count[index].reshape(-1)[0])]
            if sample_video_condition.shape[0] != num_cond_video:
                raise ValueError(
                    f"MiniMax H3 condition video rows {sample_video_condition.shape[0]} "
                    f"do not match layout rows {num_cond_video}."
                )
            if sample_audio_condition.shape[0] != num_cond_audio:
                raise ValueError(
                    f"MiniMax H3 condition audio rows {sample_audio_condition.shape[0]} "
                    f"do not match layout rows {num_cond_audio}."
                )
            full_video_rows = torch.cat([sample_video_condition, video_rows[index]], dim=0).unsqueeze(0)
            full_audio_rows = torch.cat([sample_audio_condition, audio_rows[index]], dim=0).unsqueeze(0)
            video_t = float(timestep[index])
            unique_timesteps, timestep_indices = build_row_timesteps(
                video_indices,
                audio_indices,
                num_cond_video,
                num_cond_audio,
                num_text_tokens,
                video_timestep=video_t,
                audio_timestep=video_t,
                condition_video_timestep=max(video_t, 0.999),
                condition_audio_timestep=1.0 if ref_block_meta is not None else video_t,
            )
            yield (
                dict(
                    hidden_states=full_video_rows,
                    audio_hidden_states=full_audio_rows,
                    encoder_hidden_states=encoder_hidden_states[index : index + 1, :num_text_tokens],
                    timestep=unique_timesteps.to(device),
                    timestep_indices=timestep_indices.to(device),
                    token_tags=token_tags.to(device),
                    position_ids=position_ids.to(device),
                    video_indices=video_indices.to(device),
                    audio_indices=audio_indices.to(device),
                    text_indices=text_indices.to(device),
                    return_dict=False,
                ),
                num_cond_video,
                num_cond_audio,
            )

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs,
        step: int,
    ):
        """Reject reverse-SDE sampling for the forward-process objective."""
        raise NotImplementedError(
            "MiniMaxH3DiffusionNFT is a forward-process objective and does not "
            "sample the reverse SDE. Reverse-sampling (flow_grpo) is a separate milestone."
        )
