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
"""LTX-2.3 joint audio-video rollout for DiffusionART: a pure ODE whose every visited state is kept."""

from __future__ import annotations

import torch
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.ltx2_flow_grpo.common import normalize_ltx_output_type
from verl_omni.pipelines.ltx2_flow_grpo.vllm_omni_rollout_adapter import LTX23PipelineWithLogProb
from verl_omni.pipelines.model_base import VllmOmniPipelineBase

__all__ = ["LTX23DiffusionARTPipeline"]


@VllmOmniPipelineBase.register("LTX2Pipeline", algorithm="diffusion_art")
class LTX23DiffusionARTPipeline(LTX23PipelineWithLogProb):
    """Deterministic LTX-2.3 rollout that keeps every visited joint video/audio state.

    Reuses the FlowGRPO trajectory capture (``all_latents`` / ``all_next_latents`` /
    ``all_timesteps`` / ``video_seq_len``) and the base adapter's named media artifacts
    (``video_preview`` with the resolved frame rate, ``audio`` with the vocoder sample rate,
    ``video_latent`` / ``audio_latent``; see ``docs/advanced/diffusion_media_artifacts.md``), which the
    audio-video scorers select by name. This class only changes the sampler: noise level 0 at every
    step, all ``T`` transitions recorded, no log-probabilities. Per-modality CFG (``video_cfg_scale`` /
    ``audio_cfg_scale``) is allowed because the DiffusionART training adapter applies the same
    clean-space guidance when it recomputes ``v_old``. STG, modality and rescale guidance are passed
    through to the native LTX sampler untouched: validation rollouts (``val_kwargs``) may use them,
    while ``DiffusionARTLoss.validate_trainer_config`` keeps them at their neutral values for the
    training rollout, where the recomputed ``v_old`` has to match the sampled velocity.
    """

    supports_request_batch = False

    def _configure_flow_grpo(self, req: OmniDiffusionRequest) -> None:
        req.sampling_params.output_type = normalize_ltx_output_type(req.sampling_params.output_type)
        extra_args = dict(req.sampling_params.extra_args or {})
        guidance_scale = float(getattr(req.sampling_params, "guidance_scale", None) or 1.0)
        for key in ("video_cfg_scale", "audio_cfg_scale"):
            configured = extra_args.get(key, None)
            extra_args[key] = float(configured) if configured is not None else guidance_scale
        task = extra_args.get("task")
        if task not in (None, "t2va"):
            raise NotImplementedError(f"LTX-2.3 DiffusionART supports text-to-audio-video only, got task={task!r}.")
        req.sampling_params.extra_args = extra_args
        self._flow_grpo_noise_level = 0.0
        self._flow_grpo_sde_type = "sde"
        self._flow_grpo_window_size = None
        self._flow_grpo_window_range = None
        self._flow_grpo_sde_contiguous = True
        self._flow_grpo_logprobs = False
        self._flow_grpo_task = task
        self._flow_grpo_seed = 0

    def _select_sde_steps(self, num_steps: int, device: torch.device) -> list[int]:
        del device
        return list(range(num_steps))
