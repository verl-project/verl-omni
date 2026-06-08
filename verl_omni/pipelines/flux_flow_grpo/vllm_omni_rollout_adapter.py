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

import inspect
import ast
import json
import logging
import os
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Any, Literal

import torch
from vllm.transformers_utils.config import get_hf_file_to_dict
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux import FluxPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import batched_position_ids, coalesce_not_none, getattr_not_none, maybe_to_cpu

__all__ = ["FluxPipelineWithLogProb"]

logger = logging.getLogger(__name__)


def _tf_config_to_dict(tf_model_config: Any) -> dict[str, Any]:
    if tf_model_config is None:
        return {}
    if hasattr(tf_model_config, "to_dict"):
        return dict(tf_model_config.to_dict())
    if isinstance(tf_model_config, Mapping):
        return dict(tf_model_config)
    return {}


def _load_flux_transformer_config(model: str | None) -> dict[str, Any] | None:
    if not model:
        return None

    if os.path.isdir(model):
        config_path = os.path.join(model, "transformer", "config.json")
        if os.path.isfile(config_path):
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)

    try:
        return get_hf_file_to_dict("transformer/config.json", model)
    except (OSError, ValueError) as exc:
        logger.warning("Could not load FLUX transformer config for %s: %s", model, exc)
        return None


def _ensure_flux_transformer_config(od_config: OmniDiffusionConfig) -> None:
    """Fill missing structural FLUX transformer kwargs before vLLM-Omni builds it."""
    current_config = _tf_config_to_dict(getattr(od_config, "tf_model_config", None))
    checkpoint_config = _load_flux_transformer_config(getattr(od_config, "model", None))
    if not checkpoint_config:
        return

    current_overrides = {key: value for key, value in current_config.items() if value is not None}
    merged_config = {**checkpoint_config, **current_overrides}
    tf_model_config = TransformerConfig.from_dict(merged_config)
    if hasattr(od_config, "set_tf_model_config"):
        od_config.set_tf_model_config(tf_model_config)
    else:
        od_config.tf_model_config = tf_model_config


def _get_flux_transformer_config_kwargs(transformer_cls: type, tf_model_config: Any) -> dict[str, Any]:
    tf_config = _tf_config_to_dict(tf_model_config)
    if not tf_config:
        return {}

    try:
        parameters = inspect.signature(transformer_cls.__init__).parameters
    except (TypeError, ValueError):
        return {}

    return {
        name: tf_config[name]
        for name in parameters
        if name not in {"self", "od_config"} and tf_config.get(name) is not None
    }


def _normalize_sde_window(sde_window: tuple[int, int], num_timesteps: int) -> tuple[int, int]:
    if num_timesteps <= 0:
        raise ValueError("FLUX rollout requires at least one denoising timestep.")

    start, end = sde_window
    start = max(0, min(int(start), num_timesteps - 1))
    end = max(start + 1, min(int(end), num_timesteps))
    return start, end


def _module_parameter_dtype(module: torch.nn.Module, default: torch.dtype) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return parameter.dtype if parameter is not None else default


def _has_guidance_embeds(module: torch.nn.Module) -> bool:
    config = getattr(module, "config", None)
    if config is not None and hasattr(config, "guidance_embeds"):
        return bool(getattr(config, "guidance_embeds"))
    return bool(getattr(module, "guidance_embeds", False))


def _extract_prompt_batch(
    prompts: list[Any],
    prompt: str | list[str] | None = None,
    prompt_2: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    negative_prompt_2: str | list[str] | None = None,
) -> tuple[str | list[str] | None, str | list[str] | None, str | list[str] | None, str | list[str] | None]:
    if prompt is None and prompts:
        prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in prompts]
    if prompt_2 is None and prompts and all(isinstance(p, dict) for p in prompts):
        prompt_2 = [p.get("prompt_2") for p in prompts]
    if negative_prompt is None and prompts:
        negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in prompts]
        if all(not item for item in negative_prompt):
            negative_prompt = None
    if negative_prompt_2 is None and prompts and all(isinstance(p, dict) for p in prompts):
        negative_prompt_2 = [p.get("negative_prompt_2") for p in prompts]
        if all(item is None for item in negative_prompt_2):
            negative_prompt_2 = None
    return prompt, prompt_2, negative_prompt, negative_prompt_2


def _first_generator(generator: torch.Generator | list[torch.Generator] | None) -> torch.Generator | None:
    if isinstance(generator, list):
        return generator[0] if generator else None
    return generator


def _normalize_sde_window_args(
    sde_window_size: int | str | None,
    sde_window_range: tuple[int, int] | list[int] | str,
) -> tuple[int | None, tuple[int, int]]:
    if sde_window_size is not None:
        sde_window_size = int(sde_window_size)
    if isinstance(sde_window_range, str):
        sde_window_range = ast.literal_eval(sde_window_range)
    if len(sde_window_range) != 2:
        raise ValueError("FLUX rollout sde_window_range must contain exactly two values.")
    return sde_window_size, (int(sde_window_range[0]), int(sde_window_range[1]))


@contextmanager
def _patched_flux_transformer_constructor(od_config: OmniDiffusionConfig):
    """Forward checkpoint transformer kwargs while upstream vLLM-Omni does not.

    vLLM-Omni's FLUX pipeline constructs ``FluxTransformer2DModel`` inside its
    own ``__init__`` and currently only reads part of ``tf_model_config`` there.
    Patch that constructor only for the duration of ``super().__init__`` instead
    of copying the upstream pipeline initialization.
    """
    init_globals = FluxPipeline.__init__.__globals__
    transformer_cls = init_globals.get("FluxTransformer2DModel")
    if transformer_cls is None:
        yield
        return

    def configured_flux_transformer(*args: Any, **kwargs: Any):
        call_od_config = kwargs.get("od_config")
        if call_od_config is None and args:
            call_od_config = args[0]
        if call_od_config is None:
            call_od_config = od_config

        config_kwargs = _get_flux_transformer_config_kwargs(
            transformer_cls,
            getattr(call_od_config, "tf_model_config", None),
        )
        for name, value in config_kwargs.items():
            kwargs.setdefault(name, value)
        return transformer_cls(*args, **kwargs)

    init_globals["FluxTransformer2DModel"] = configured_flux_transformer
    try:
        yield
    finally:
        init_globals["FluxTransformer2DModel"] = transformer_cls


@VllmOmniPipelineBase.register("FluxPipeline", algorithm="flow_grpo")
class FluxPipelineWithLogProb(FluxPipeline):
    """vLLM-Omni FLUX rollout pipeline that returns FlowGRPO trajectory data."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        _ensure_flux_transformer_config(od_config)
        with _patched_flux_transformer_constructor(od_config):
            super().__init__(od_config=od_config, prefix=prefix)
        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        self.scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            model,
            subfolder="scheduler",
            local_files_only=local_files_only,
        )

    def diffuse(
        self,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        negative_pooled_prompt_embeds: torch.Tensor | None,
        latents: torch.Tensor,
        latent_image_ids: torch.Tensor,
        text_ids: torch.Tensor,
        negative_text_ids: torch.Tensor | None,
        timesteps: torch.Tensor,
        do_true_cfg: bool,
        guidance: torch.Tensor | None,
        true_cfg_scale: float,
        noise_level: float,
        sde_window: tuple[int, int],
        sde_type: str,
        generator: torch.Generator | list[torch.Generator] | None,
        logprobs: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        all_latents = []
        all_log_probs = []
        all_timesteps = []
        sde_window = _normalize_sde_window(sde_window, len(timesteps))
        self.scheduler.set_begin_index(0)
        self.transformer.do_true_cfg = do_true_cfg

        for i, timestep_value in enumerate(timesteps):
            if self.interrupt:
                continue

            if i < sde_window[0]:
                cur_noise_level = 0.0
            elif i == sde_window[0]:
                cur_noise_level = noise_level
                all_latents.append(latents.float())
            elif i > sde_window[0] and i < sde_window[1]:
                cur_noise_level = noise_level
            else:
                cur_noise_level = 0.0

            self._current_timestep = timestep_value
            timestep = timestep_value.expand(latents.shape[0]).to(device=latents.device, dtype=latents.dtype)

            positive_kwargs = {
                "hidden_states": latents,
                "timestep": timestep / 1000,
                "guidance": guidance,
                "pooled_projections": pooled_prompt_embeds,
                "encoder_hidden_states": prompt_embeds,
                "txt_ids": text_ids,
                "img_ids": latent_image_ids,
                "joint_attention_kwargs": self.joint_attention_kwargs,
                "return_dict": False,
            }

            negative_kwargs = None
            if do_true_cfg:
                negative_kwargs = {
                    "hidden_states": latents,
                    "timestep": timestep / 1000,
                    "guidance": guidance,
                    "pooled_projections": negative_pooled_prompt_embeds,
                    "encoder_hidden_states": negative_prompt_embeds,
                    "txt_ids": negative_text_ids,
                    "img_ids": latent_image_ids,
                    "joint_attention_kwargs": self.joint_attention_kwargs,
                    "return_dict": False,
                }

            noise_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg,
                true_cfg_scale,
                positive_kwargs,
                negative_kwargs,
                cfg_normalize=False,
            )

            latents, log_prob, _, _ = self.scheduler.step(
                noise_pred.float(),
                timestep_value,
                latents,
                generator=_first_generator(generator),
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents.float())
                all_log_probs.append(log_prob)
                all_timesteps.append(timestep_value)

        all_latents = torch.stack(all_latents, dim=1)
        all_log_probs = torch.stack(all_log_probs, dim=1) if all_log_probs and all_log_probs[0] is not None else None
        all_timesteps = torch.stack(all_timesteps).unsqueeze(0).expand(latents.shape[0], -1)
        return latents, all_latents, all_log_probs, all_timesteps

    def forward(
        self,
        req: OmniDiffusionRequest,
        prompt: str | list[str] | None = None,
        prompt_2: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        negative_prompt_2: str | list[str] | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        sigmas: list[float] | None = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end_tensor_inputs: tuple[str, ...] = ("latents",),
        max_sequence_length: int = 512,
        noise_level: float = 0.7,
        sde_window_size: int | None = None,
        sde_window_range: tuple[int, int] = (0, 5),
        sde_type: Literal["sde", "cps"] = "sde",
        logprobs: bool = True,
    ) -> DiffusionOutput:
        prompt, prompt_2, negative_prompt, negative_prompt_2 = _extract_prompt_batch(
            req.prompts,
            prompt,
            prompt_2,
            negative_prompt,
            negative_prompt_2,
        )

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        guidance_scale = getattr_not_none(sampling_params, "guidance_scale", guidance_scale)
        generator = sampling_params.generator or generator
        if generator is None and sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(sampling_params.seed)
        true_cfg_scale = coalesce_not_none(sampling_params.true_cfg_scale, true_cfg_scale)
        num_images_per_prompt = (
            sampling_params.num_outputs_per_prompt
            if sampling_params.num_outputs_per_prompt > 0
            else num_images_per_prompt
        )
        max_sequence_length = sampling_params.max_sequence_length or max_sequence_length

        noise_level = coalesce_not_none(sampling_params.extra_args.get("noise_level", None), noise_level)
        sde_window_size = coalesce_not_none(sampling_params.extra_args.get("sde_window_size", None), sde_window_size)
        sde_window_range = coalesce_not_none(sampling_params.extra_args.get("sde_window_range", None), sde_window_range)
        sde_window_size, sde_window_range = _normalize_sde_window_args(sde_window_size, sde_window_range)
        sde_type = coalesce_not_none(sampling_params.extra_args.get("sde_type", None), sde_type)
        logprobs = coalesce_not_none(sampling_params.extra_args.get("logprobs", None), logprobs)

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        self.check_cfg_parallel_validity(true_cfg_scale, has_neg_prompt)

        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )

        negative_text_ids = None
        if do_true_cfg:
            negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        num_channels_latents = self.transformer.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        timesteps, _ = self.prepare_timesteps(num_inference_steps, sigmas, latents.shape[1])
        self._num_timesteps = len(timesteps)

        if _has_guidance_embeds(self.transformer):
            guidance = torch.full([1], guidance_scale, dtype=torch.float32, device=self.device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if sde_window_size is not None:
            max_sde_window_start = max(
                sde_window_range[0],
                min(sde_window_range[1], len(timesteps)) - sde_window_size,
            )
            sde_generator = _first_generator(generator)
            start = torch.randint(
                sde_window_range[0],
                max_sde_window_start + 1,
                (1,),
                generator=sde_generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps))

        latents, all_latents, all_log_probs, all_timesteps = self.diffuse(
            prompt_embeds,
            pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            latents,
            latent_image_ids,
            text_ids,
            negative_text_ids,
            timesteps,
            do_true_cfg,
            guidance,
            true_cfg_scale,
            noise_level,
            sde_window,
            sde_type,
            generator,
            logprobs,
        )

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.to(dtype=_module_parameter_dtype(self.vae, latents.dtype))
            image = self.vae.decode(latents, return_dict=False)[0]

        return DiffusionOutput(
            output=maybe_to_cpu(image),
            custom_output={
                "all_latents": maybe_to_cpu(all_latents),
                "all_log_probs": maybe_to_cpu(all_log_probs),
                "all_timesteps": maybe_to_cpu(all_timesteps),
                "prompt_embeds": maybe_to_cpu(prompt_embeds),
                "pooled_prompt_embeds": maybe_to_cpu(pooled_prompt_embeds),
                "text_ids": maybe_to_cpu(batched_position_ids(text_ids, latents.shape[0])),
                "negative_prompt_embeds": maybe_to_cpu(negative_prompt_embeds),
                "negative_pooled_prompt_embeds": maybe_to_cpu(negative_pooled_prompt_embeds),
                "negative_text_ids": maybe_to_cpu(
                    batched_position_ids(negative_text_ids, latents.shape[0]) if negative_text_ids is not None else None
                ),
                "latent_image_ids": maybe_to_cpu(batched_position_ids(latent_image_ids, latents.shape[0])),
            },
        )
