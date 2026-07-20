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


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch
from diffusers import ModelMixin, SchedulerMixin
from diffusers.training_utils import compute_density_for_timestep_sampling
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.workers.config import DiffusionModelConfig

from .model_base import DiffusionI2IModelBase, DiffusionModelBase

__all__ = [
    "ImageGenerationRequest",
    "build_scheduler",
    "forward",
    "forward_and_sample_previous_step",
    "get_sigmas",
    "prepare_model_inputs",
    "prepare_noisy_latents",
    "sample_noise_and_timesteps",
    "set_timesteps",
]


@dataclass
class ImageGenerationRequest:
    """Image generation request shared by t2i and image-conditioned backends.

    Parses prompt / prompt token ids and condition images from verl-omni's
    rollout request payload (``custom_prompt`` dict), checking multiple image
    candidate keys: ``images``, ``image``, ``multi_modal_data.image``,
    ``extra_args.multi_modal_data.image``, ``additional_information.condition_images``.
    """

    prompt: Any
    images: list[Any] = field(default_factory=list)
    """Condition images: empty for t2i, single-element for image editing, multi-element for multi-image conditioning."""

    negative_prompt: Any | None = None
    metadata: Mapping[str, Any] | None = None

    @classmethod
    def from_request_payload(cls, request_payload: Mapping[str, Any]) -> ImageGenerationRequest:
        """Build a request from verl-omni's rollout request payload (``custom_prompt``)."""
        prompt = request_payload.get("prompt")
        if prompt is None:
            prompt = request_payload.get("prompt_token_ids")
        if prompt is None:
            raise ValueError(
                "ImageGenerationRequest missing required 'prompt' or 'prompt_token_ids' field. "
                "The rollout request payload must carry one of them in custom_prompt."
            )

        multi_modal_data = request_payload.get("multi_modal_data")
        extra_args = request_payload.get("extra_args")
        extra_multi_modal_data = extra_args.get("multi_modal_data") if isinstance(extra_args, Mapping) else None
        additional_information = request_payload.get("additional_information")

        image_candidates = [
            request_payload.get("images"),
            request_payload.get("image"),
            multi_modal_data.get("image") if isinstance(multi_modal_data, Mapping) else None,
            extra_multi_modal_data.get("image") if isinstance(extra_multi_modal_data, Mapping) else None,
            additional_information.get("condition_images") if isinstance(additional_information, Mapping) else None,
        ]
        images = []
        # Select the first image source explicitly present in the request.
        for candidate in image_candidates:
            if candidate is not None:
                images = candidate
                break
        if isinstance(images, tuple):
            images = list(images)
        elif not isinstance(images, list):
            images = [images]

        metadata_candidates = [
            request_payload.get("metadata"),
            request_payload.get("extra_info"),
            additional_information,
        ]
        metadata = None
        # Select the first metadata source explicitly present in the request.
        for candidate in metadata_candidates:
            if candidate is not None:
                metadata = candidate
                break
        if not isinstance(metadata, Mapping):
            metadata = None

        return cls(
            prompt=prompt,
            images=images,
            negative_prompt=request_payload.get("negative_prompt"),
            metadata=metadata,
        )


def _to_prompt_row(value: Any, *, device: torch.device, field_name: str) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    if tensor.ndim == 1:
        return tensor
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor[0]
    raise ValueError(f"Request-batch {field_name} must be 1D or single-row 2D, got shape={tuple(tensor.shape)}.")


def _get_prompt_field(prompt: Any, aliases: tuple[str, ...]) -> Any:
    if isinstance(prompt, str) or not hasattr(prompt, "get"):
        return None
    for name in aliases:
        value = prompt.get(name)
        if value is None:
            additional = prompt.get("additional_information")
            if isinstance(additional, dict):
                value = additional.get(name)
        if value is not None:
            return value
    return None


def _rows_from_default(
    value: torch.Tensor | list[int] | None,
    *,
    device: torch.device,
    field_name: str,
) -> tuple[torch.Tensor | None, list[int] | None]:
    if value is None:
        return None, None
    tensor = value.to(device=device) if isinstance(value, torch.Tensor) else torch.tensor(value, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{field_name} must be 1D or 2D, got shape={tuple(tensor.shape)}.")
    return tensor, [int(tensor.shape[1])] * int(tensor.shape[0])


def collate_prompt_rows(
    prompts: list[Any],
    aliases: tuple[str, ...],
    default_value: torch.Tensor | list[int] | None,
    *,
    device: torch.device,
    field_name: str,
    pad_value: int = 0,
) -> tuple[torch.Tensor | None, list[int] | None]:
    default_rows, default_lengths = _rows_from_default(default_value, device=device, field_name=field_name)
    if default_rows is not None:
        if len(prompts) > 1 and default_rows.shape[0] != len(prompts):
            raise ValueError(
                f"Batched {field_name} default must have one row per request; "
                f"got {default_rows.shape[0]} rows for {len(prompts)} requests."
            )
        return default_rows, default_lengths

    rows = [
        _to_prompt_row(
            _get_prompt_field(prompt, aliases),
            device=device,
            field_name=field_name,
        )
        for prompt in prompts
    ]
    if not any(row is not None for row in rows):
        return None, None
    if not all(row is not None for row in rows):
        raise ValueError(f"Cannot batch requests with a mix of provided and missing {field_name}.")

    typed_rows = [row for row in rows if row is not None]
    target_len = max(int(row.shape[0]) for row in typed_rows)
    result = torch.full(
        (len(typed_rows), target_len),
        pad_value,
        dtype=typed_rows[0].dtype,
        device=typed_rows[0].device,
    )
    lengths: list[int] = []
    for idx, row in enumerate(typed_rows):
        row_len = int(row.shape[0])
        result[idx, :row_len] = row
        lengths.append(row_len)
    return result, lengths


def collate_prompt_mask(
    prompts: list[Any],
    aliases: tuple[str, ...],
    default_value: torch.Tensor | list[int] | None,
    *,
    device: torch.device,
    field_name: str,
    token_lengths: list[int] | None,
    target_seq_len: int | None,
) -> torch.Tensor | None:
    mask, _ = collate_prompt_rows(
        prompts,
        aliases,
        default_value,
        device=device,
        field_name=field_name,
        pad_value=0,
    )
    if mask is not None:
        mask = mask != 0
        if target_seq_len is not None:
            if mask.shape[1] < target_seq_len:
                padded = torch.zeros((mask.shape[0], target_seq_len), dtype=torch.bool, device=mask.device)
                padded[:, : mask.shape[1]] = mask
                mask = padded
            elif mask.shape[1] > target_seq_len:
                mask = mask[:, :target_seq_len]
        return mask

    if token_lengths is None or target_seq_len is None:
        return None

    mask = torch.zeros((len(token_lengths), target_seq_len), dtype=torch.bool, device=device)
    for idx, row_len in enumerate(token_lengths):
        mask[idx, :row_len] = True
    return mask


def _slice_batch_value(value: Any, start: int, stop: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value[start:stop] if value.ndim > 0 and value.shape[0] >= stop else value
    if isinstance(value, tuple):
        return tuple(_slice_batch_value(item, start, stop) for item in value)
    if isinstance(value, list):
        return value[start:stop] if len(value) >= stop else value
    return value


def split_diffusion_output_by_request(
    result: Any,
    req: Any,
    *,
    num_outputs_per_prompt: int,
) -> list[Any]:
    outputs: list[Any] = []
    custom_output = result.custom_output or {}
    for idx in range(req.num_reqs):
        start = idx * num_outputs_per_prompt
        stop = (idx + 1) * num_outputs_per_prompt
        outputs.append(
            result.__class__(
                output=_slice_batch_value(result.output, start, stop),
                trajectory_timesteps=_slice_batch_value(
                    result.trajectory_timesteps,
                    start,
                    stop,
                ),
                trajectory_latents=_slice_batch_value(
                    result.trajectory_latents,
                    start,
                    stop,
                ),
                trajectory_log_probs=_slice_batch_value(
                    result.trajectory_log_probs,
                    start,
                    stop,
                ),
                trajectory_decoded=_slice_batch_value(
                    result.trajectory_decoded,
                    start,
                    stop,
                ),
                error=result.error,
                error_status_code=result.error_status_code,
                error_type=result.error_type,
                aborted=result.aborted,
                abort_message=result.abort_message,
                post_process_func=result.post_process_func,
                custom_output={key: _slice_batch_value(value, start, stop) for key, value in custom_output.items()},
                finished=result.finished,
                chunk_index=result.chunk_index,
                total_chunks=result.total_chunks,
                stage_durations=dict(result.stage_durations),
                peak_memory_mb=result.peak_memory_mb,
                to_cpu=result.to_cpu,
            )
        )
    return outputs


def prepare_model_inputs(
    module: ModelMixin,
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
    """Build architecture-specific model inputs for the forward pass.
    Dispatches to the registered DiffusionModelBase subclass for the current architecture.

    Args:
        module (ModelMixin): the diffusion transformer module.
        model_config (DiffusionModelConfig): the configuration of the diffusion model.
        latents (torch.Tensor): latent tensor from the micro-batch. This can be a full trajectory
            or an already selected/noised latent, depending on the algorithm.
        timesteps (torch.Tensor): timestep tensor from the micro-batch. This can be a full trajectory
            or an already selected timestep, depending on the algorithm.
        prompt_embeds (torch.Tensor): dense positive prompt embeddings, shape (B, L, D).
        prompt_embeds_mask (torch.Tensor): attention mask for prompt_embeds, shape (B, L).
        negative_prompt_embeds (torch.Tensor): dense negative prompt embeddings, shape (B, L, D).
        negative_prompt_embeds_mask (torch.Tensor): attention mask for negative_prompt_embeds.
        micro_batch (TensorDict): the full micro-batch, available for architecture-specific
            metadata (e.g. height, width, vae_scale_factor).
        step (int): the current denoising step index.

    Returns:
        tuple[dict, Optional[dict]]: A pair of ``(model_inputs, negative_model_inputs)``
        dicts ready to be unpacked into the transformer forward call. When the
        registered adapter is an I2I subclass, condition tensors are injected
        after the T2I ``prepare_model_inputs`` call.
    """
    model_cls = DiffusionModelBase.get_class(model_config)

    # T2I original logic (unchanged)
    model_inputs, negative_model_inputs = model_cls.prepare_model_inputs(
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

    # I2I adapters prepare and inject their model-specific condition tensors.
    if issubclass(model_cls, DiffusionI2IModelBase):
        condition = model_cls.prepare_condition(micro_batch, latents, step)
        if condition is None:
            available_keys = [str(key) for key in micro_batch.keys()]
            raise ValueError(
                f"{model_cls.__name__}.prepare_condition returned None. "
                f"Available micro-batch keys: {available_keys}. Check that the "
                "rollout output contains the condition fields expected by this adapter."
            )
        if not isinstance(condition, dict) or len(condition) == 0:
            condition_keys = list(condition) if isinstance(condition, dict) else None
            raise TypeError(
                f"prepare_condition returned {type(condition).__name__}; "
                f"expected a non-empty dict, keys={condition_keys}."
            )
        model_inputs, negative_model_inputs = model_cls.inject_condition(model_inputs, negative_model_inputs, condition)

    return model_inputs, negative_model_inputs


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
    """Gather scheduler sigmas for the requested timesteps and output rank."""
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


def forward(
    module: ModelMixin,
    model_config: DiffusionModelConfig,
    model_inputs: dict,
    negative_model_inputs: Optional[dict],
) -> torch.Tensor:
    """Forward the model for single-pass prediction-space objectives."""
    return DiffusionModelBase.get_class(model_config).forward(module, model_config, model_inputs, negative_model_inputs)
