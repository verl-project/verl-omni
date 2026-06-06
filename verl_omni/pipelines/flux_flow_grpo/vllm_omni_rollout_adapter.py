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

import os
from typing import Any, Literal

import torch
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.flux import FluxPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest

from verl_omni.pipelines.model_base import VllmOmniPipelineBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler

from .common import batched_position_ids, coalesce_not_none, maybe_to_cpu

__all__ = ["FluxPipelineWithLogProb"]


@VllmOmniPipelineBase.register("FluxPipeline", algorithm="flow_grpo")
class FluxPipelineWithLogProb(FluxPipeline):
    """vLLM-Omni FLUX rollout pipeline that returns FlowGRPO trajectory data."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
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
                generator=generator,
                noise_level=cur_noise_level,
                sde_type=sde_type,
                return_logprobs=logprobs,
                return_dict=False,
            )

            if i >= sde_window[0] and i < sde_window[1]:
                all_latents.append(latents)
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
        custom_prompt = req.prompts[0] if req.prompts else {}
        if isinstance(custom_prompt, dict):
            prompt = custom_prompt.get("prompt", prompt)
            prompt_2 = custom_prompt.get("prompt_2", prompt_2)
            negative_prompt = custom_prompt.get("negative_prompt", negative_prompt)
            negative_prompt_2 = custom_prompt.get("negative_prompt_2", negative_prompt_2)

        if prompt is None and req.prompts:
            prompt = [p if isinstance(p, str) else (p.get("prompt") or "") for p in req.prompts]
        if negative_prompt is None and req.prompts:
            negative_prompt = ["" if isinstance(p, str) else (p.get("negative_prompt") or "") for p in req.prompts]
            if all(not item for item in negative_prompt):
                negative_prompt = None

        sampling_params = req.sampling_params
        height = sampling_params.height or self.default_sample_size * self.vae_scale_factor
        width = sampling_params.width or self.default_sample_size * self.vae_scale_factor
        num_inference_steps = sampling_params.num_inference_steps or num_inference_steps
        sigmas = sampling_params.sigmas or sigmas
        if getattr(sampling_params, "guidance_scale_provided", False):
            guidance_scale = sampling_params.guidance_scale
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

        if self.transformer.guidance_embeds:
            guidance = torch.full([1], guidance_scale, dtype=torch.float32, device=self.device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        if sde_window_size is not None:
            start = torch.randint(
                sde_window_range[0],
                sde_window_range[1] - sde_window_size + 1,
                (1,),
                generator=generator,
                device=self.device,
            ).item()
            end = start + sde_window_size
            sde_window = (start, end)
        else:
            sde_window = (0, len(timesteps) - 1)

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
