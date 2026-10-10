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
"""SD3.5 vLLM-Omni rollout adapter for DGPO."""

from __future__ import annotations

import torch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

__all__ = ["StableDiffusion3DGPOPipeline"]


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="dgpo")
class StableDiffusion3DGPOPipeline(StableDiffusion3PipelineWithLogProb):
    """Deterministic SD3.5 rollout that returns the final clean latents and the timestep schedule.

    DGPO trains with a forward-process objective on the final sample, so the whole
    trajectory is an ODE (noise level 0) and no per-step latents or log-probabilities
    leave the pipeline.
    """

    supports_request_batch = True
    emit_clean_latents = True

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        do_cfg: bool,
        guidance_scale: float,
        noise_level: float,
        sde_window: tuple[int, int] | list[tuple[int, int]],
        sde_type: str,
        generator: torch.Generator | list[torch.Generator] | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor]:
        del noise_level, sde_window, sde_type, logprobs
        # The parent stores fp32 latents for every step inside the window; a one-step window keeps that to two.
        latents, _, _, _ = super().diffuse(
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            latents,
            timesteps,
            do_cfg,
            guidance_scale,
            0.0,
            (len(timesteps) - 1, len(timesteps)),
            "sde",
            generator,
            False,
        )
        return latents, None, None, timesteps.unsqueeze(0).expand(latents.shape[0], -1)
