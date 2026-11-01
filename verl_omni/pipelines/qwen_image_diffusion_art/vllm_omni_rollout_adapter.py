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
"""Qwen-Image vLLM-Omni rollout adapter for DiffusionART."""

from typing import Any

import torch
from vllm_omni.diffusion.worker.utils import StepRequestState

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.qwen_image_flow_grpo.vllm_omni_rollout_adapter import QwenImagePipelineWithLogProb

__all__ = ["QwenImageDiffusionARTPipeline"]


@VllmOmniPipelineBase.register("QwenImagePipeline", algorithm="diffusion_art")
class QwenImageDiffusionARTPipeline(QwenImagePipelineWithLogProb):
    """Deterministic Qwen-Image rollout that keeps every visited latent.

    DiffusionART regresses the velocity at the states an ODE rollout visited, so the whole
    trajectory runs at noise level 0 with the SDE window spanning all steps: ``all_latents``
    holds the ``T + 1`` fp32 states (the last one is the clean endpoint) and ``all_timesteps``
    the schedule. No log-probabilities are computed; the old-policy velocity at each state is
    recomputed by the actor's ``old`` LoRA adapter.
    """

    supports_request_batch = QwenImagePipelineWithLogProb.supports_request_batch

    def diffuse(
        self,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        latents,
        img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        timesteps,
        do_true_cfg,
        guidance,
        true_cfg_scale,
        noise_level,
        sde_window,
        sde_type,
        generator,
        logprobs,
    ):
        del noise_level, sde_window, logprobs
        return super().diffuse(
            prompt_embeds,
            prompt_embeds_mask,
            negative_prompt_embeds,
            negative_prompt_embeds_mask,
            latents,
            img_shapes,
            txt_seq_lens,
            negative_txt_seq_lens,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            0.0,
            (0, len(timesteps)),
            sde_type,
            generator,
            False,
        )

    def prepare_encode(self, state: StepRequestState, **kwargs: Any) -> StepRequestState:
        """Force the step-execution path onto the same full-window ODE as :meth:`diffuse`."""
        state = super().prepare_encode(state, **kwargs)
        state.noise_level = 0.0
        state.logprobs = False
        state.sde_window = (0, len(state.timesteps))
        return state

    def step_scheduler(self, state: StepRequestState, noise_pred: torch.Tensor, **kwargs: Any) -> None:
        if state.noise_level != 0.0 or state.logprobs:
            raise RuntimeError("DiffusionART rollouts must be deterministic ODE trajectories without log-probs.")
        super().step_scheduler(state, noise_pred, **kwargs)
