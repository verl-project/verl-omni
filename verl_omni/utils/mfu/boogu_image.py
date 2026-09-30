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

"""FLOPs estimator for Boogu-Image and its log-prob rollout variant."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from verl_omni.utils.mfu.diffusion_flops_counter import (
    DiffusionModelFlops,
    read_latents,
    register_diffusion_architecture,
    sum_seqlens,
)

__all__ = ["BooguImageFlops"]


def _instruction_feat_dim(config: Mapping[str, Any]) -> int:
    """Instruction width entering the caption embedder.

    Mirrors ``BooguImageTransformer2DModel.cal_preprocessed_instruction_feat_dim``:
    ``cat`` stacks ``num_instruction_feat_layers`` encoder layers (it is clamped to
    at least 1 by the model), ``mean`` collapses them. Anything unrecognised is
    treated as ``mean`` rather than raising, so a surprising config degrades the
    estimate instead of breaking a training run.
    """
    instruction = config.get("instruction_feature_configs") or {}
    num_layers = max(int(instruction.get("num_instruction_feat_layers", 1) or 1), 1)
    feat_dim = int(instruction.get("instruction_feat_dim", 4096) or 4096)
    reduce_type = str(instruction.get("reduce_type", "concat") or "concat").lower()
    return num_layers * feat_dim if "cat" in reduce_type else feat_dim


def _ffn_inner_dim(dim: int, config: Mapping[str, Any]) -> int:
    """``LuminaFeedForward`` hidden width.

    ``inner_dim = 4 * dim``, optionally scaled by ``ffn_dim_multiplier``, then
    rounded **up** to a multiple of ``multiple_of`` -- see
    ``LuminaFeedForward.__init__``. The rounding is why the frozen ``ffn_dim``
    cannot be read straight off the config the way ``WanFlops`` does.
    """
    multiple_of = int(config.get("multiple_of", 256) or 256)
    if multiple_of <= 0:
        multiple_of = 1
    inner = 4 * dim
    multiplier = config.get("ffn_dim_multiplier")
    if multiplier:
        inner = int(float(multiplier) * inner)
    return multiple_of * ((inner + multiple_of - 1) // multiple_of)


def _sum_paired_squares(a: Sequence[int], b: Sequence[int]) -> int:
    """``sum((a_i + b_i)**2)``, tolerating unequal lengths.

    The base class's :func:`sum_seqlen_squared` raises on a length mismatch; the
    diffusion batches here always carry one entry per sample for both streams, but
    a caller that passes only one of them should not take down a training step.
    """
    total = 0
    for index in range(max(len(a), len(b))):
        left = int(a[index]) if index < len(a) else 0
        right = int(b[index]) if index < len(b) else 0
        total += (left + right) ** 2
    return total


@register_diffusion_architecture(
    "BooguImagePipeline",
    "BooguImagePipelineWithLogProb",
)
class BooguImageFlops(DiffusionModelFlops):
    """FLOPs estimator for ``BooguImageTransformer2DModel``."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        config = config or {}

        # The base class's ``dim`` property reads ``attention_head_dim``, which
        # Boogu's transformer config does not carry -- it ships ``hidden_size``
        # instead (and requires ``hidden_size // num_attention_heads ==
        # sum(axes_dim_rope)``). Derive the width locally rather than shadowing
        # the read-only property.
        num_heads = int(config.get("num_attention_heads", 0) or 0)
        dim = int(config.get("hidden_size", 0) or 0) or self.dim
        head_dim = dim // num_heads if num_heads else 0
        self.num_attention_heads = num_heads
        self.head_dim = head_dim
        num_kv_heads = int(config.get("num_kv_heads", 0) or 0) or num_heads
        self.kv_dim = head_dim * num_kv_heads

        inner = _ffn_inner_dim(dim, config)
        self.ffn_n = 3 * dim * inner

        self.num_double_stream_layers = int(config.get("num_double_stream_layers", 0) or 0)
        self.num_single_stream_layers = max(int(config.get("num_layers", 0) or 0) - self.num_double_stream_layers, 0)
        self.num_refiner_layers = int(config.get("num_refiner_layers", 0) or 0)

        self.patch_size = int(config.get("patch_size", 2) or 2)
        in_channels = int(config.get("in_channels", 0) or 0)
        out_channels = int(config.get("out_channels") or in_channels)

        # ``LuminaRMSNormZero`` projects the timestep embedding through
        # ``min(hidden_size, 1024)`` and expands it 4x (scale/gate for attn and MLP).
        modulation_dim = min(dim, 1024)

        # QKV + output projection, per token. GQA: K/V project to
        # head_dim * num_kv_heads rather than to `dim`.
        qkv_out_n = 2 * dim * dim + 2 * dim * self.kv_dim
        # One single-stream block: attention linears + SwiGLU FFN (linear_1/2/3).
        self.single_stream_n = qkv_out_n + self.ffn_n
        # Double-stream image token: joint QKV/out on the image side, the
        # image-only self-attention, and the image FFN.
        img_joint_n = 3 * dim * dim + 2 * dim * self.kv_dim
        img_self_attn_n = 2 * dim * dim + 2 * dim * self.kv_dim
        self.double_stream_img_n = img_joint_n + img_self_attn_n + self.ffn_n
        # Double-stream instruction token: joint QKV/out on the instruction side
        # plus the instruction FFN (no self-attention).
        instruct_joint_n = 3 * dim * dim + 2 * dim * self.kv_dim
        self.double_stream_instruct_n = instruct_joint_n + self.ffn_n

        # LuminaRMSNormZero: (min(dim, 1024) -> 4*dim) per sample, per block.
        self.modulation_n = 4 * dim * modulation_dim

        # Once-per-call linears.
        self.x_embedder_n = self.patch_size * self.patch_size * in_channels * dim
        self.caption_embedder_n = _instruction_feat_dim(config) * dim
        self.norm_out_n = dim * self.patch_size * self.patch_size * out_channels
        # Timestep MLP plus norm_out's conditioning projection, both per sample.
        # 256 is ``Lumina2CombinedTimestepCaptionEmbedding.time_proj``'s frequency width.
        self.per_sample_embed_n = 256 * modulation_dim + modulation_dim * modulation_dim + modulation_dim * dim

    def get_reference_seqlens(self, data: Any) -> list[int]:
        """Per-sample reference-latent token counts (``[]`` when absent).

        The reference latents are patchified by the transformer exactly like the
        noise latents, so they divide by the same patch volume.
        """
        reference = data.get("condition_image_latents") if hasattr(data, "get") else None
        if reference is None:
            return []
        shape = getattr(reference, "shape", None)
        if shape is None:
            return []
        shape = [int(d) for d in shape]
        # ``(B, C, H, W)``; tolerate a per-reference axis ``(B, N, C, H, W)`` by
        # collapsing everything between the batch dim and the last three dims.
        if len(shape) < 4:
            return [0] * shape[0]
        per_sample = 1
        for dim in shape[-2:]:
            per_sample *= dim
        patch_volume = self.patch_size * self.patch_size
        return [per_sample // patch_volume] * shape[0]

    def get_latent_seqlens(self, data: Any) -> list[int]:
        """Patchified noise latents **plus** concatenated reference latents."""
        latents, stacked = read_latents(data)
        base: list[int] = []
        if latents is not None and hasattr(latents, "shape"):
            shape = [int(d) for d in latents.shape]
            # ``stacked`` (FlowGRPO ``all_latents``) carries a leading time axis:
            # ``(B, T, C, H, W)``; otherwise ``(B, C, H, W)``.
            spatial_start = 3 if stacked else 2
            if len(shape) > spatial_start:
                per_sample = 1
                for dim in shape[spatial_start:]:
                    per_sample *= dim
                base = [per_sample // (self.patch_size * self.patch_size)] * shape[0]
        if not base:
            base = super().get_latent_seqlens(data)

        reference = self.get_reference_seqlens(data)
        if not reference:
            return base
        listed = list(base)
        for index in range(len(listed)):
            if index < len(reference):
                listed[index] += reference[index]
        return listed

    def collect_meta(self, data: Any) -> dict[str, list[int]]:
        """Standard seqlens plus the reference split the refiners need.

        ``ref_seqlens`` is an extra architecture-specific list. It rides the
        generic DP all-gather in :func:`allgather_diffusion_flops_meta` (which
        gathers every list-valued field), so it stays consistent across ranks
        without any extra plumbing.
        """
        meta = super().collect_meta(data)
        reference = self.get_reference_seqlens(data)
        if not reference:
            reference = [0] * len(meta["latent_seqlens"])
        meta["ref_seqlens"] = reference
        return meta

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        num_forward_passes: int,
        ref_seqlens: Optional[Sequence[int]] = None,
    ) -> float:
        """FLOPs incurred by one denoising-loop call, as achieved TFLOPS.

        ``latent_seqlens`` is ``noise + reference`` per the stream rule above;
        ``ref_seqlens`` isolates the reference part so the three refiners can be
        charged their own (smaller) sequence lengths.
        """
        latents = [int(s) for s in latent_seqlens]
        prompts = [int(s) for s in prompt_seqlens]
        reference = [int(s) for s in ref_seqlens] if ref_seqlens else [0] * len(latents)
        if len(reference) != len(latents):
            reference = (reference + [0] * len(latents))[: len(latents)]
        noise = [max(latent - ref, 0) for latent, ref in zip(latents, reference, strict=False)]

        img_tot = sum_seqlens(latents)
        noise_tot = sum_seqlens(noise)
        reference_tot = sum_seqlens(reference)
        prompt_tot = sum_seqlens(prompts)
        fused_tot = img_tot + prompt_tot
        batch_size = max(len(latents), len(prompts), 1)

        # --- dense (per-token linear) terms ---
        # noise_refiner + ref_image_refiner: single-stream blocks, one on each
        # subset of the image stream.
        dense = self.compute_dense_flops(self.num_refiner_layers * self.single_stream_n, noise_tot + reference_tot)
        # context_refiner: same linears, instruction stream only.
        dense += self.compute_dense_flops(self.num_refiner_layers * self.single_stream_n, prompt_tot)
        # Double-stream layer: separate image / instruction sides.
        dense += self.compute_dense_flops(self.num_double_stream_layers * self.double_stream_img_n, img_tot)
        dense += self.compute_dense_flops(self.num_double_stream_layers * self.double_stream_instruct_n, prompt_tot)
        # Single-stream layers run on the fused sequence.
        dense += self.compute_dense_flops(self.num_single_stream_layers * self.single_stream_n, fused_tot)

        # --- per-sample modulation (timestep embedding) ---
        modulation_blocks = self.num_single_stream_layers + 2 * self.num_refiner_layers
        modulation_blocks += 5 * self.num_double_stream_layers  # img_norm1/2/3 + instruct_norm1/2
        mod_flops = self.compute_dense_flops(modulation_blocks * self.modulation_n, batch_size)

        # --- once-per-call projections ---
        embed_flops = self.compute_dense_flops(self.x_embedder_n, noise_tot)
        embed_flops += self.compute_dense_flops(self.x_embedder_n, reference_tot)
        embed_flops += self.compute_dense_flops(self.caption_embedder_n, prompt_tot)
        embed_flops += self.compute_dense_flops(self.norm_out_n, fused_tot)
        embed_flops += self.compute_dense_flops(self.per_sample_embed_n, batch_size)

        # --- attention ---
        # 12 = 2 FLOPs/MAC * 2 matmuls (QK^T and softmax @ V) * 3 (fwd + bwd).
        # GQA does not reduce these matmuls, so the head count stays at
        # num_attention_heads (the Q side), matching the rest of the registry.
        attn_flops = (
            12.0
            * self.num_attention_heads
            * self.head_dim
            * (
                # single-stream (fused) + double-stream (joint) full attention.
                (self.num_single_stream_layers + self.num_double_stream_layers) * _sum_paired_squares(latents, prompts)
                # double-stream image-only self-attention.
                + self.num_double_stream_layers * sum(s * s for s in latents)
                # the three refiners: instruction, noise, reference respectively.
                + self.num_refiner_layers * sum(s * s for s in prompts)
                + self.num_refiner_layers * sum(s * s for s in noise)
                + self.num_refiner_layers * sum(s * s for s in reference)
            )
        )

        flops_per_call = (dense + mod_flops + embed_flops + attn_flops) * num_timesteps * num_forward_passes
        return flops_per_call / delta_time / 1e12
