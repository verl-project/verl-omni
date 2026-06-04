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

"""FLOPs estimator for Qwen-Image and aliases."""

from __future__ import annotations

from typing import Any, Sequence

from verl_omni.utils.mfu.diffusion_flops_counter import (
    DiffusionModelFlops,
    read_latents,
    register_diffusion_architecture,
    sum_seqlens,
)

__all__ = ["QwenImageFlops"]


@register_diffusion_architecture(
    "QwenImagePipeline",
    "QwenImagePipelineWithLogProb",
)
class QwenImageFlops(DiffusionModelFlops):
    """FLOPs estimator for ``QwenImageTransformer2DModel``.

    Qwen-Image is a dual-stream DiT: image-latent tokens flow through the
    image-side linears and text-encoder tokens through the text-side linears.
    The two streams meet only in joint full attention.

    Per-block token-scaling params per stream: ``12 * dim**2`` (QKV + out +
    GELU FFN). Modulation linears scale with batch size, not token count.
    """

    def get_latent_seqlens(self, data: Any) -> list[int]:
        """Handle diffusers ``_pack_latents`` layout ``(B, [T,] L, C')``."""
        latents, stacked = read_latents(data)
        if latents is not None and hasattr(latents, "shape"):
            shape = tuple(int(d) for d in latents.shape)
            in_channels = int(self.config.get("in_channels") or 0)
            if in_channels > 0 and shape[-1] == in_channels:
                # Packed: (B, L, C') or FlowGRPO-stacked (B, T, L, C').
                # Unpacked NFT/VAE latents use NCHW `(B, C, H, W)` (no packed `L` dim),
                # so even if `shape[-1]` happens to equal `in_channels` we defer to the base
                # spatial-product extractor instead of treating `H` as the seqlen.
                if stacked and len(shape) == 4:
                    return [shape[-2]] * shape[0]
                if not stacked and len(shape) == 3:
                    return [shape[-2]] * shape[0]
        return super().get_latent_seqlens(data)

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
        num_layers = int(cfg["num_layers"])
        in_channels = int(cfg["in_channels"])
        joint_attention_dim = int(cfg["joint_attention_dim"])
        patch_size = int(cfg.get("patch_size", 2))
        out_channels = int(cfg.get("out_channels") or in_channels)

        dense_block_n_per_stream = (3 + 1 + 8) * dim * dim

        img_in_n = in_channels * dim
        txt_in_n = joint_attention_dim * dim
        proj_out_n = patch_size * patch_size * out_channels * dim

        img_tot = sum_seqlens(latent_seqlens)
        txt_tot = sum_seqlens(prompt_seqlens)
        batch_size = max(
            len(latent_seqlens) if latent_seqlens else 0,
            len(prompt_seqlens) if prompt_seqlens else 0,
        )

        img_dense_flops = self.compute_dense_flops(
            num_layers * dense_block_n_per_stream + img_in_n + proj_out_n, img_tot
        )
        txt_dense_flops = self.compute_dense_flops(num_layers * dense_block_n_per_stream + txt_in_n, txt_tot)

        mod_block_n = 12 * dim * dim
        mod_flops = self.compute_dense_flops(num_layers * mod_block_n, batch_size)

        attn_flops = self.compute_attention_flops(latent_seqlens, prompt_seqlens)

        flops_all_steps = (
            (img_dense_flops + txt_dense_flops + mod_flops + attn_flops) * num_timesteps * num_forward_passes
        )
        return flops_all_steps / delta_time / 1e12
