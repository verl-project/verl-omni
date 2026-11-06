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
"""SD3.5 vLLM-Omni rollout adapter for DiffusionOPSD."""

from __future__ import annotations

import torch

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.sd3_flow_grpo.vllm_omni_rollout_adapter import StableDiffusion3PipelineWithLogProb

__all__ = ["StableDiffusion3OPSDPipeline"]


@VllmOmniPipelineBase.register("StableDiffusion3Pipeline", algorithm="opsd")
class StableDiffusion3OPSDPipeline(StableDiffusion3PipelineWithLogProb):
    """Deterministic SD3.5 rollout that also returns the low-noise query states.

    DiffusionOPSD fits detached clean-output targets at the low-noise state ``z_q`` of each
    behaviour-policy trajectory. The rollout stays the plain deterministic ODE used by the
    forward-process family: the final sample is decoded and scored as usual, so the endpoint
    reward is still available for the group-relative fitting weight. The only addition is
    that the trailing ``query_window_steps`` states and their scheduler timesteps are kept
    in ``all_latents`` / ``all_timesteps``; the trainer then selects the state whose sigma is
    closest to ``opsd_query_sigma``. The parent loop already stores fp32 latents for an SDE
    window, so this costs one small state tensor per captured step and no extra forward pass.

    The window must cover the requested query sigma, and which sigmas land in it depends on the
    step count. For SD3.5-Medium's shift-3 schedule at 10 steps the scheduler yields sigmas
    ``1.0, 0.960, 0.913, 0.858, 0.790, 0.707, 0.602, 0.465, 0.278, 0.009``, so the four trailing
    states span ``0.602, 0.465, 0.278, 0.009`` and contain the paper's sigma* = 0.278 exactly
    (note the last captured state sits at the final sigma ~= 0.009 and is never selected because
    it is far outside the tolerance). The covered range for sigma* = 0.278 is roughly 7 to 69
    steps; ``OPSDLoss.validate_trainer_config`` checks the configured step count against this
    window at startup. The value is a class attribute (not a pipeline config field) because the
    rollout engine forwards sampling-time tensors, not trainer-side loss config.
    """

    supports_request_batch = True
    emit_clean_latents = True

    #: Number of trailing denoising steps kept for query selection.
    query_window_steps = 4

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
    ) -> tuple[torch.Tensor, torch.Tensor, None, torch.Tensor]:
        """Run the deterministic ODE and keep the trailing states for query selection."""
        del noise_level, sde_window, sde_type, logprobs
        num_steps = len(timesteps)
        window_start = max(0, num_steps - int(self.query_window_steps))
        latents, all_latents, _, all_timesteps = super().diffuse(
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            latents,
            timesteps,
            do_cfg,
            guidance_scale,
            0.0,
            (window_start, num_steps),
            "sde",
            generator,
            False,
        )
        return latents, all_latents, None, all_timesteps
