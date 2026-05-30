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

from typing import Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from diffusers.training_utils import compute_density_for_timestep_sampling
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.workers.config import DiffusionModelConfig

from .model_base import DiffusionModelBase


def prepare_model_inputs(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    negative_prompt_embeds_mask: torch.Tensor,
    micro_batch: TensorDict,
    step: int,
) -> tuple[dict, dict]:
    """Build architecture-specific model inputs for the forward pass.
    Dispatches to the registered DiffusionModelBase subclass for the current architecture.

    Args:
        module (ModelMixin): the diffusion transformer module.
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
        latents (torch.Tensor): full latent tensor from the micro-batch, shape (B, T, ...).
        timesteps (torch.Tensor): full timestep tensor from the micro-batch, shape (B, T).
        prompt_embeds (torch.Tensor): dense positive prompt embeddings, shape (B, L, D).
        prompt_embeds_mask (torch.Tensor): attention mask for prompt_embeds, shape (B, L).
        negative_prompt_embeds (torch.Tensor): dense negative prompt embeddings, shape (B, L, D).
        negative_prompt_embeds_mask (torch.Tensor): attention mask for negative_prompt_embeds.
        micro_batch (TensorDict): the full micro-batch, available for architecture-specific
            metadata (e.g. height, width, vae_scale_factor).
        step (int): the current denoising step index.
    """
    return DiffusionModelBase.get_class(model_config).prepare_model_inputs(
        module,
        model_config,
        latents,
        timesteps,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    )


def build_scheduler(model_config: DiffusionModelConfig) -> SchedulerMixin:
    """Build and configure the scheduler for the diffusion model.
    The returned scheduler has timesteps and sigmas already set.

    Args:
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
    """
    return DiffusionModelBase.get_class(model_config).build_scheduler(model_config)


def set_timesteps(scheduler: SchedulerMixin, model_config: DiffusionModelConfig):
    """Set correct timesteps and sigmas for diffusion model schedulers.

    Args:
        scheduler (SchedulerMixin): the scheduler used for the diffusion process.
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
    """
    DiffusionModelBase.get_class(model_config).set_timesteps(scheduler, model_config, get_device_name())


def sample_noise_and_timesteps(
    latents: torch.Tensor,
    scheduler: SchedulerMixin,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample pairwise flow-matching noise and timesteps for adjacent DPO pairs."""
    batch_size = latents.shape[0]
    if batch_size % 2 != 0:
        raise ValueError("DPO flow training expects an even batch laid out as [chosen0, rejected0, ...].")

    pair_count = batch_size // 2
    pair_noise = torch.randn_like(latents[:pair_count])

    # Sample a random timestep for each image
    # for weighting schemes where we sample timesteps non-uniformly
    u = compute_density_for_timestep_sampling(
        weighting_scheme="logit_normal",
        batch_size=pair_count,
        logit_mean=0,
        logit_std=1,
        mode_scale=1.29,
    )
    indices = (u * scheduler.config.num_train_timesteps).long()
    pair_timesteps = scheduler.timesteps[indices].to(device=latents.device)

    noise = pair_noise.repeat_interleave(2, dim=0)
    timesteps = pair_timesteps.repeat_interleave(2, dim=0)
    return noise, timesteps


def _validate_adjacent_pair_values(values: torch.Tensor, name: str) -> None:
    if values.shape[0] % 2 != 0:
        raise ValueError(f"DPO flow training expects `{name}` to have an even batch dimension.")
    if not torch.allclose(values[0::2], values[1::2]):
        raise ValueError(f"DPO flow training expects adjacent chosen/rejected samples to share `{name}`.")


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def prepare_noisy_latents(
    latents: torch.Tensor,
    scheduler: SchedulerMixin,
    noise: torch.Tensor | None = None,
    timesteps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build noisy latents with shared noise/timesteps for adjacent DPO pairs."""
    if (noise is None) != (timesteps is None):
        raise KeyError("Diffusion flow training requires `noise` and `timesteps` to be provided together.")

    if noise is None:
        noise, timesteps = sample_noise_and_timesteps(latents, scheduler)
    else:
        noise = noise.to(device=latents.device, dtype=latents.dtype)
        timesteps = timesteps.to(device=latents.device)
    _validate_adjacent_pair_values(noise, "noise")
    _validate_adjacent_pair_values(timesteps, "timesteps")

    if hasattr(scheduler, "scale_noise"):
        noisy_latents = scheduler.scale_noise(latents, timesteps, noise)
    else:
        sigmas = get_sigmas(scheduler, timesteps, latents.device, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

    return noisy_latents, noise, timesteps


def forward_and_sample_previous_step(
    module: ModelMixin,
    scheduler: SchedulerMixin,
    model_config: DiffusionModelConfig,
    model_inputs: dict,
    negative_model_inputs: Optional[dict],
    scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
    step: int,
):
    """Forward the model and sample previous step.
    This method is usually used for RL-algorithms based on reversed-sampling process.
    Such as FlowGRPO, DanceGRPO, etc.

    Args:
        module (ModelMixin): the diffusion model to be forwarded.
        scheduler (SchedulerMixin): the scheduler used for the diffusion process.
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
        model_inputs (dict[str, torch.Tensor]): the inputs to the diffusion model.
        negative_model_inputs (Optional[dict[str, torch.Tensor]]): the negative inputs for guidance.
        scheduler_inputs (Optional[TensorDict | dict[str, torch.Tensor]]): the extra inputs for the scheduler,
            which may contain the latents and timesteps.
        step (int): the current step in the diffusion process.
    """
    return DiffusionModelBase.get_class(model_config).forward_and_sample_previous_step(
        module, scheduler, model_config, model_inputs, negative_model_inputs, scheduler_inputs, step
    )


def prepare_forward_diffusion_inputs(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    xt: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: Optional[torch.Tensor],
    negative_prompt_embeds_mask: Optional[torch.Tensor],
    micro_batch: TensorDict,
    step: int,
) -> tuple[dict, Optional[dict]]:
    """Build architecture-specific inputs for forward-process diffusion losses."""
    return DiffusionModelBase.get_class(model_config).prepare_forward_diffusion_inputs(
        module,
        model_config,
        xt,
        timestep,
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        micro_batch,
        step,
    )


def forward_velocity(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    model_inputs: dict,
    negative_model_inputs: Optional[dict],
) -> torch.Tensor:
    """Forward the model for velocity/prediction-space objectives."""
    return DiffusionModelBase.get_class(model_config).forward_velocity(
        module, model_config, model_inputs, negative_model_inputs
    )
