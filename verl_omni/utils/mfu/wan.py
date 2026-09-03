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

"""FLOPs estimator for Wan2.1/2.2 (``WanTransformer3DModel``) and aliases."""

from __future__ import annotations

from typing import Any, Sequence

from verl_omni.utils.mfu.diffusion_flops_counter import (
    DiffusionModelFlops,
    register_diffusion_architecture,
    sum_seqlens,
)

__all__ = ["WanFlops"]


@register_diffusion_architecture("WanPipeline")
class WanFlops(DiffusionModelFlops):
    """FLOPs estimator for ``WanTransformer3DModel``.

    Wan is a single-stream video DiT: video-latent tokens self-attend
    (``attn1``, cost proportional to ``latent_s**2``), then cross-attend to
    the text encoder stream (``attn2``, cost proportional to
    ``latent_s * prompt_s``). This is unlike Qwen-Image/SD3's joint full
    attention over the concatenated sequence.

    Latents arrive as raw ``(B, C, T, H, W)`` (or FlowGRPO-stacked
    ``(B, T_steps, C, T, H, W)``); the base extractor's spatial-product
    default already returns the post-patchify token count only if the
    tensor were pre-patched, which Wan's pipeline does not do before this
    point, so ``get_latent_seqlens`` divides by the 3D patch volume.
    """

    def get_latent_seqlens(self, data: Any) -> list[int]:
        raw = super().get_latent_seqlens(data)
        patch_size = self.config.get("patch_size", (1, 2, 2))
        divisor = 1
        for p in patch_size:
            divisor *= int(p)
        return [s // divisor for s in raw]

    def estimate_flops(
        self,
        latent_seqlens: Sequence[int],
        prompt_seqlens: Sequence[int],
        delta_time: float,
        *,
        num_timesteps: int,
        num_forward_passes: int,
    ) -> float:
        cfg = self.config
        dim = self.dim
        num_heads = int(cfg.get("num_attention_heads", 0))
        head_dim = int(cfg.get("attention_head_dim", 0))
        num_layers = int(cfg["num_layers"])
        ffn_dim = int(cfg["ffn_dim"])
        added_kv = int(cfg.get("added_kv_proj_dim") or dim)
        in_channels = int(cfg["in_channels"])
        out_channels = int(cfg.get("out_channels") or in_channels)
        text_dim = int(cfg.get("text_dim", dim))
        patch_size = cfg.get("patch_size", (1, 2, 2))
        patch_prod = 1
        for p in patch_size:
            patch_prod *= int(p)

        # Self-attn (QKV + out, 4*dim^2) + cross-attn Q/out on the latent
        # side (2*dim^2) + FFN (dim->ffn_dim->dim, explicit ffn_dim, not a
        # 4x-dim assumption).
        img_block_n = 4 * dim * dim + 2 * dim * dim + 2 * dim * ffn_dim
        # Cross-attn K/V projected from the text stream.
        txt_block_n = 2 * dim * added_kv

        img_in_n = in_channels * patch_prod * dim  # patch_embedding (Conv3d)
        proj_out_n = patch_prod * out_channels * dim
        txt_in_n = text_dim * dim + dim * dim  # PixArtAlphaTextProjection (2-layer MLP)

        img_tot = sum_seqlens(latent_seqlens)
        txt_tot = sum_seqlens(prompt_seqlens)
        batch_size = max(
            len(latent_seqlens) if latent_seqlens else 0,
            len(prompt_seqlens) if prompt_seqlens else 0,
        )

        img_dense_flops = self.compute_dense_flops(num_layers * img_block_n + img_in_n + proj_out_n, img_tot)
        txt_dense_flops = self.compute_dense_flops(num_layers * txt_block_n + txt_in_n, txt_tot)

        # condition_embedder.time_proj: a single dim -> 6*dim linear shared
        # across all layers (not one per layer); modulation itself is an
        # additive `scale_shift_table` parameter, not a matmul.
        time_proj_flops = self.compute_dense_flops(6 * dim * dim, batch_size)

        if latent_seqlens and prompt_seqlens and len(latent_seqlens) != len(prompt_seqlens):
            raise ValueError(
                f"latent_seqlens and prompt_seqlens must have the same length, "
                f"got {len(latent_seqlens)} and {len(prompt_seqlens)}."
            )

        # Self-attention over the latent stream only.
        self_attn_flops = self.compute_attention_flops(latent_seqlens, [])
        # Cross-attention: latent queries against text keys/values, cost
        # proportional to latent_s * prompt_s (not the joint (a+b)^2 term).
        cross_seqlen_sum = sum(
            int(latent) * int(prompt) for latent, prompt in zip(latent_seqlens, prompt_seqlens, strict=True)
        )
        cross_attn_flops = 12 * num_layers * num_heads * head_dim * cross_seqlen_sum

        flops_all_steps = (
            (img_dense_flops + txt_dense_flops + time_proj_flops + self_attn_flops + cross_attn_flops)
            * num_timesteps
            * num_forward_passes
        )
        return flops_all_steps / delta_time / 1e12
