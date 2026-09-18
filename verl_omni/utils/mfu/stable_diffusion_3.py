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

"""FLOPs estimator for Stable Diffusion 3 (SD3 / SD3.5) and aliases."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

from verl_omni.utils.mfu.diffusion_flops_counter import (
    DiffusionModelFlops,
    register_diffusion_architecture,
    sum_seqlens,
)

__all__ = ["StableDiffusion3Flops"]


@register_diffusion_architecture(
    "StableDiffusion3Pipeline",
    "StableDiffusion3PipelineWithLogProb",
)
class StableDiffusion3Flops(DiffusionModelFlops):
    """FLOPs estimator for ``SD3Transformer2DModel``.

    SD3 is a dual-stream MM-DiT: image and text tokens each go through their
    own per-block linears (``JointTransformerBlock``) and meet only in joint
    full attention, same topology as Qwen-Image. Two differences from
    Qwen-Image matter for the formula:

    - The last block is ``context_pre_only``: the text stream drops its
      output projection (``to_add_out``) and its FFN (``ff_context``),
      keeping only the ``add_q/k/v_proj`` needed to feed joint attention.
    - The transformer consumes raw ``(B, C, H, W)`` latents and patchifies
      internally (``PatchEmbed``), so the default latent-seqlens extractor
      overcounts by ``patch_size**2``.

    ``dual_attention_layers`` (SD3.5's extra image-only ``attn2``) is not
    modeled; a warning fires if the config declares any, since the formula
    below would undercount those layers.
    """

    def get_latent_seqlens(self, data: Any) -> list[int]:
        raw = super().get_latent_seqlens(data)
        patch_size = int(self.config.get("patch_size", 2))
        divisor = patch_size * patch_size
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
        num_layers = int(cfg["num_layers"])
        in_channels = int(cfg["in_channels"])
        joint_attention_dim = int(cfg["joint_attention_dim"])
        caption_projection_dim = int(cfg.get("caption_projection_dim") or dim)
        patch_size = int(cfg.get("patch_size", 2))
        out_channels = int(cfg.get("out_channels") or in_channels)

        dual_attention_layers = cfg.get("dual_attention_layers") or ()
        if len(dual_attention_layers) > 0:
            warnings.warn(
                "StableDiffusion3Flops does not model SD3.5's dual_attention_layers "
                f"(got {dual_attention_layers!r}); the estimate undercounts those layers' "
                "extra image-only attn2 block.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Attn (QKV + out, 4*dim^2) + FFN (dim->4dim->dim, gelu-approximate, 8*dim^2).
        block_n_per_stream = 12 * dim * dim
        # Last block is context_pre_only: text stream keeps only add_q/k/v_proj (3*dim^2),
        # dropping to_add_out (1*dim^2) and ff_context (8*dim^2).
        txt_last_block_n = 3 * dim * dim

        img_in_n = in_channels * patch_size * patch_size * dim
        txt_in_n = joint_attention_dim * caption_projection_dim
        proj_out_n = patch_size * patch_size * out_channels * dim

        img_tot = sum_seqlens(latent_seqlens)
        txt_tot = sum_seqlens(prompt_seqlens)
        batch_size = max(
            len(latent_seqlens) if latent_seqlens else 0,
            len(prompt_seqlens) if prompt_seqlens else 0,
        )

        img_dense_flops = self.compute_dense_flops(num_layers * block_n_per_stream + img_in_n + proj_out_n, img_tot)
        txt_dense_n = max(num_layers - 1, 0) * block_n_per_stream + (txt_last_block_n if num_layers > 0 else 0)
        txt_dense_flops = self.compute_dense_flops(txt_dense_n + txt_in_n, txt_tot)

        # Modulation (AdaLN) linears are batch-scaled, not token-scaled: one
        # timestep embedding per sample, not per token. Image stream uses
        # AdaLayerNormZero (dim -> 6*dim) every layer; text stream uses the
        # same for all but the last layer, which uses AdaLayerNormContinuous
        # (dim -> 2*dim, no gate) since it is context_pre_only.
        img_mod_n = num_layers * 6 * dim * dim
        txt_mod_n = max(num_layers - 1, 0) * 6 * dim * dim + (2 * dim * dim if num_layers > 0 else 0)
        mod_flops = self.compute_dense_flops(img_mod_n + txt_mod_n, batch_size)

        # SD3 attention is joint full attention over the concatenated
        # (img, txt) sequence, same topology as Qwen-Image.
        attn_flops = self.compute_attention_flops(latent_seqlens, prompt_seqlens)

        flops_all_steps = (
            (img_dense_flops + txt_dense_flops + mod_flops + attn_flops) * num_timesteps * num_forward_passes
        )
        return flops_all_steps / delta_time / 1e12
