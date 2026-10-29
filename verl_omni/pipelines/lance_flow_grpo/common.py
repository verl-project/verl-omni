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


"""Lance geometry and schedule constants shared by rollout and training."""

from verl_omni.pipelines.bagel_flow_grpo.common import (
    setup_bagel_sigmas,
)
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

#: Sigma-schedule shift.  Lance's rollout defaults to 3.5 where BAGEL uses
#: 3.0; a mismatch silently moves every sigma and breaks log-prob parity.
LANCE_TIMESTEP_SHIFT = 3.5

#: Wan2.2 VAE latent geometry.  Lance feeds the latent straight into
#: ``vae2llm`` instead of unfolding a 2x2 patch the way BAGEL does, so
#: ``patch_latent_dim = latent_patch_size**2 * z_channels = 1 * 48``, which
#: matches the released ``vae2llm.weight`` of shape ``(2048, 48)``.
LANCE_LATENT_PATCH_SIZE = 1
LANCE_VAE_Z_CHANNELS = 48
LANCE_VAE_DOWNSAMPLE_SPATIAL = 16

#: Wan2.2 video VAE temporal stride.  The rollout reads it from the checkpoint's
#: ``vae_config.downsample_temporal``; the trainer loads no VAE, so it carries the
#: value as a constant next to the spatial one.  Latent frames for ``T`` pixel
#: frames are ``(T - 1) // LANCE_VAE_DOWNSAMPLE_TEMPORAL + 1``.
LANCE_VAE_DOWNSAMPLE_TEMPORAL = 4

#: ``latent_pos_embed.pos_embed`` ships as ``(4096, 2048) = (64 * 64, hidden)``.
LANCE_MAX_LATENT_SIZE = 64

#: The temporal rotary axis is amplified by this factor so neighbouring latent
#: frames stay well separated, matching the upstream Qwen2.5-VL rope convention.
#: ``LanceBagel._per_token_mrope_for_video_latent`` computes it as
#: ``LANCE_TOKENS_PER_SECOND * LANCE_SECONDS_PER_GRID`` (2 * 1.0) in vllm-omni;
#: the trainer has to place the same values, so keep the product here.
LANCE_TEMPORAL_ROPE_SCALE = 2

#: Schedule points Lance samples beyond ``num_inference_steps``; mirrors
#: ``LanceBagel._denoise_schedule_extra_step``, which a test pins.
LANCE_SCHEDULE_EXTRA_POINTS = 1

#: Qwen2.5-VL head-dimension split across the (t, h, w) rotary axes; the
#: rollout sets the same list on the language model's ``rope_scaling``.
LANCE_MROPE_SECTION = (16, 24, 24)


def setup_lance_sigmas(
    scheduler: FlowMatchSDEDiscreteScheduler,
    num_steps: int,
    shift: float = LANCE_TIMESTEP_SHIFT,
    device: str | None = None,
) -> list[float]:
    """Configure *scheduler* with Lance's shifted sigma schedule.

    Lance differs from BAGEL twice over.  The shift is 3.5 rather than 3.0,
    and the schedule carries one more point: ``LanceBagel`` sets
    ``_denoise_schedule_extra_step``, so ``Bagel.generate_image`` samples
    ``num_steps + 1`` points and runs ``num_steps`` Euler steps, where BAGEL
    samples ``num_steps`` and runs one fewer.  Building BAGEL's count here
    would leave the trainer one sigma short of the trajectory the rollout
    recorded, and every sigma would sit on a different grid.

    Args:
        scheduler: Scheduler to configure in place.
        num_steps: Number of denoising steps the rollout performs.
        shift: Timestep shift; defaults to Lance's 3.5.
        device: Device for the scheduler buffers.

    Returns:
        The sigma list, terminal zero dropped; ``num_steps`` entries.
    """
    return setup_bagel_sigmas(scheduler, num_steps + LANCE_SCHEDULE_EXTRA_POINTS, shift=shift, device=device)
