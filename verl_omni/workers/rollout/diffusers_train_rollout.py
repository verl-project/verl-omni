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
"""Train-side diffusers rollouter.

Prototype for the Q3 roadmap item "Support diffusers rollouter ... via
train-side engine" (verl-project/verl-omni#97). Runs the SDE denoising loop
directly against the training FSDP transformer, in-process, with no
vLLM-Omni server/engine involved.

Scope note: this module intentionally covers only the sampling core shared
by every SD3-family rollout adapter (``StableDiffusion3PipelineWithLogProb``
in ``verl_omni.pipelines.sd3_flow_grpo``) -- the fixed-window SDE loop over
``FlowMatchSDEDiscreteScheduler``. It is a numerically-verified building
block, not a drop-in ``AsyncRollout`` implementation: request batching,
async server plumbing, and rollout config schema wiring are follow-up work
once the sampling core itself is trusted. Do not register this in
``_ROLLOUT_REGISTRY`` until that wiring exists -- an incomplete registry
entry would silently break rollout backend selection for every other model.

Why a separate train-side path at all, instead of routing everything
through vLLM-Omni: #97 also names an already-merged alternative, vLLM-Omni's
generic ``DiffusersAdapterPipeline`` (``--diffusion-load-format diffusers``,
vllm-project/vllm-omni#2724). That adapter is a black-box wrapper around
any diffusers pipeline's own ``__call__()`` -- it works today, but by design
exposes none of the per-step trajectory data (latents, timesteps, log-probs)
that FlowGRPO-style RL training needs, and explicitly does not support CFG
parallel, sequence parallel, or step-wise/continuous batching. Calling the
transformer directly, as this module does, is what makes the SDE-window
bookkeeping and log-prob collection in ``sde_denoise_loop`` possible in the
first place; a black-box adapter has no hook for either. Once wired up,
this path is meant to sit alongside the vLLM-Omni rollout as another
``_ROLLOUT_REGISTRY`` backend for models/algorithms that need in-process
sampling (e.g. no vLLM-Omni rollout adapter exists yet, or the trajectory
data must come from the exact weights currently being trained rather than a
periodically-synced inference copy) -- not a replacement for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler


@dataclass
class TrainSideDiffusionOutput:
    """Trajectory data produced by :func:`sde_denoise_loop`.

    Field names mirror ``verl_omni.workers.rollout.replica.DiffusionOutput``'s
    ``extra_fields`` (``all_latents`` / ``all_timesteps``) and ``log_probs`` so
    downstream FlowGRPO log-prob recomputation code needs no branching on
    which rollout backend produced the trajectory.
    """

    final_latents: torch.Tensor
    all_latents: torch.Tensor
    all_timesteps: torch.Tensor
    log_probs: torch.Tensor | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)


def predict_noise_with_cfg(
    transformer: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_projections: torch.Tensor,
    do_cfg: bool,
    guidance_scale: float,
    negative_encoder_hidden_states: torch.Tensor | None = None,
    negative_pooled_projections: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the transformer once (or twice, for CFG) and combine the noise predictions.

    Matches the plain SD3 guidance formula used by
    ``StableDiffusion3PipelineWithLogProb.predict_noise_maybe_with_cfg``:
    ``uncond + guidance_scale * (cond - uncond)``.
    """
    noise_pred_cond = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        return_dict=False,
    )[0]
    if not do_cfg:
        return noise_pred_cond

    if negative_encoder_hidden_states is None or negative_pooled_projections is None:
        raise ValueError("do_cfg=True requires negative_encoder_hidden_states/negative_pooled_projections.")

    noise_pred_uncond = transformer(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=negative_encoder_hidden_states,
        pooled_projections=negative_pooled_projections,
        return_dict=False,
    )[0]
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


@torch.no_grad()
def sde_denoise_loop(
    transformer: torch.nn.Module,
    scheduler: FlowMatchSDEDiscreteScheduler,
    *,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    do_cfg: bool,
    guidance_scale: float,
    noise_level: float,
    sde_window_range: tuple[int, int],
    sde_type: Literal["sde", "cps", "dance_sde"] = "sde",
    generator: torch.Generator | None = None,
    logprobs: bool = True,
    model_dtype: torch.dtype | None = None,
) -> TrainSideDiffusionOutput:
    """Run the full SDE diffusion loop directly against ``transformer``.

    This is a device/engine-agnostic port of
    ``StableDiffusion3PipelineWithLogProb.diffuse`` (see
    ``verl_omni/pipelines/sd3_flow_grpo/vllm_omni_rollout_adapter.py``): same
    scheduler class, same per-step SDE-window bookkeeping, same trajectory
    fields. ``transformer`` may be a plain ``nn.Module`` or an FSDP-wrapped
    training module -- both expose the same ``SD3Transformer2DModel.forward``
    signature, so the loop body is identical either way.
    """
    batch_size = latents.shape[0]
    start, end = sde_window_range
    if not (0 <= start < end <= len(timesteps)):
        raise ValueError(f"sde_window_range {sde_window_range} out of bounds for {len(timesteps)} timesteps.")

    model_dtype = model_dtype if model_dtype is not None else prompt_embeds.dtype
    device = latents.device

    all_latents: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []
    all_timesteps: list[torch.Tensor] = []

    scheduler.set_begin_index(0)

    for i, timestep_value in enumerate(timesteps):
        if i == start:
            all_latents.append(latents.detach().float().clone())

        cur_noise_level = float(noise_level) if start <= i < end else 0.0
        timestep = timestep_value.expand(batch_size).to(device=device, dtype=model_dtype)
        x = latents.to(model_dtype)

        noise_pred = predict_noise_with_cfg(
            transformer,
            hidden_states=x,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            do_cfg=do_cfg,
            guidance_scale=guidance_scale,
            negative_encoder_hidden_states=negative_prompt_embeds,
            negative_pooled_projections=negative_pooled_prompt_embeds,
        )

        latents, log_prob, _, _ = scheduler.step(
            noise_pred.float(),
            timestep_value,
            latents,
            generator=generator,
            noise_level=cur_noise_level,
            sde_type=sde_type,
            return_logprobs=logprobs,
            return_dict=False,
        )

        if start <= i < end:
            all_latents.append(latents.detach().float().clone())
            all_timesteps.append(timestep_value)
            if log_prob is not None:
                all_log_probs.append(log_prob)

    all_latents_t = torch.stack(all_latents, dim=0)
    all_timesteps_t = torch.stack(all_timesteps, dim=0) if all_timesteps else torch.empty(0)
    all_log_probs_t = torch.stack(all_log_probs, dim=0) if all_log_probs else None

    return TrainSideDiffusionOutput(
        final_latents=latents,
        all_latents=all_latents_t,
        all_timesteps=all_timesteps_t,
        log_probs=all_log_probs_t,
    )
