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
"""MiniMax H3 training adapter for DiffusionART."""

from __future__ import annotations

from typing import Optional

import torch
from tensordict import TensorDict

from verl_omni.pipelines.minimax_h3_diffusion_nft.common import h3_dit_timestep
from verl_omni.pipelines.minimax_h3_diffusion_nft.diffusers_training_adapter import MiniMaxH3DiffusionNFT
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.workers.config import DiffusionModelConfig

from .common import art_rows_to_packed, packed_to_art_rows

__all__ = ["MiniMaxH3DiffusionART"]


@DiffusionModelBase.register("MiniMaxH3Pipeline", algorithm="diffusion_art")
class MiniMaxH3DiffusionART(MiniMaxH3DiffusionNFT):
    """Joint video/audio velocity at one cached ODE state, in the DiffusionART token layout.

    ``latents`` is ``(B, 3 * video_rows + audio_rows, 32)`` (see :mod:`.common`); the forward returns
    the same layout so ``DiffusionARTLoss`` can split it at ``video_seq_len``. The video timestep is
    the engine's ``train_timesteps`` (video sigma x 1000); the audio stream is denoised on its own
    sigma grid, read from ``train_segment_timesteps[:, step, 1]`` so that ``v_old`` is recomputed at
    exactly the ``(t_video, t_audio)`` pair the rollout used. No CFG: H3 ships cfg-distilled weights
    and the recipe runs ``true_cfg_scale=1.0``.
    """

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
        """Re-pack the ART rows into the H3 layout and attach the per-sample audio timestep."""
        if "train_segment_timesteps" not in micro_batch.keys():
            raise KeyError(
                "MiniMax H3 DiffusionART needs `train_segment_timesteps` (video/audio sigma per cached state); "
                "the diffusion_art rollout adapter publishes `all_segment_timesteps`."
            )
        segment_timesteps = micro_batch["train_segment_timesteps"]
        if segment_timesteps.ndim != 3 or segment_timesteps.shape[-1] != 2:
            raise ValueError(
                f"train_segment_timesteps must be (B, S, 2) = [video, audio], got {tuple(segment_timesteps.shape)}."
            )
        audio_timesteps = segment_timesteps[:, step, 1].float()
        if not torch.allclose(segment_timesteps[:, step, 0].float(), timesteps.float(), atol=1e-3):
            raise ValueError("train_segment_timesteps[:, step, 0] must equal the video timestep of this state.")
        model_inputs, negative = super().prepare_model_inputs(
            module,
            model_config,
            art_rows_to_packed(latents),
            timesteps,
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            micro_batch,
            step,
        )
        model_inputs["audio_timestep"] = h3_dit_timestep(audio_timesteps)
        return model_inputs, negative

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict,
        negative_model_inputs: Optional[dict] = None,
    ) -> torch.Tensor:
        """Packed FP32 flow-match velocities, returned in the ART token layout."""
        packed = super().forward(module, model_config, model_inputs, negative_model_inputs)
        return packed_to_art_rows(packed.float())

    @classmethod
    def forward_and_sample_previous_step(cls, *args, **kwargs):
        """Reject the reverse-transition API; DiffusionART regresses velocities, it does not replay transitions."""
        raise NotImplementedError("MiniMax H3 DiffusionART does not sample reverse transitions or compute log-probs.")
