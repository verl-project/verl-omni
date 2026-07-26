from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from tensordict import TensorDict
from verl.utils.device import get_device_name

from verl_omni.pipelines.model_base import DiffusionI2IModelBase, DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .common import (
    BOOGU_VAE_SCALE_FACTOR,
    apply_boogu_text_cfg,
    build_boogu_freqs_cis,
    build_boogu_sigmas,
    normalize_ref_image_hidden_states,
)

logger = logging.getLogger(__name__)

__all__ = ["BooguImageFlowGRPO"]


@DiffusionModelBase.register("BooguImageTurboPipeline", algorithm="flow_grpo")
@DiffusionModelBase.register("BooguImagePipeline", algorithm="flow_grpo")
class BooguImageFlowGRPO(DiffusionI2IModelBase):
    allow_missing_condition = True

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading Boogu transformer from %s via DiffusionPipeline", model_config.local_path)
        pipeline = DiffusionPipeline.from_pretrained(
            model_config.local_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        return pipeline.transformer

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler.from_pretrained(
            pretrained_model_name_or_path=model_config.local_path,
            subfolder="scheduler",
        )
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        height = model_config.pipeline.height
        width = model_config.pipeline.width
        latent_height = height // BOOGU_VAE_SCALE_FACTOR
        latent_width = width // BOOGU_VAE_SCALE_FACTOR
        num_tokens = latent_height * latent_width
        sigmas = build_boogu_sigmas(
            num_inference_steps=model_config.pipeline.num_inference_steps,
            num_tokens=num_tokens,
            scheduler_config=scheduler.config,
        )
        scheduler.set_timesteps(
            num_inference_steps=model_config.pipeline.num_inference_steps,
            device=device,
            sigmas=sigmas,
        )

    @classmethod
    def prepare_model_inputs(
        cls,
        module,
        model_config: DiffusionModelConfig,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds_mask: Optional[torch.Tensor],
        micro_batch: TensorDict,
        step: int,
    ) -> tuple[dict, Optional[dict]]:
        if prompt_embeds_mask is None:
            raise ValueError("prompt_embeds_mask is required for BooguImageFlowGRPO.prepare_model_inputs")

        hidden_states = latents[:, step]
        timestep = timesteps[:, step] / 1000.0
        freqs_cis = build_boogu_freqs_cis(
            tuple(module.config.axes_dim_rope),
            tuple(module.config.axes_lens),
            theta=10000,
        )

        model_inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "instruction_hidden_states": prompt_embeds,
            "freqs_cis": freqs_cis,
            "instruction_attention_mask": prompt_embeds_mask,
            "ref_image_hidden_states": None,
            "return_dict": False,
        }

        negative_model_inputs = None
        if model_config.pipeline.true_cfg_scale > 1.0:
            if negative_prompt_embeds is None or negative_prompt_embeds_mask is None:
                raise ValueError("Boogu true_cfg_scale > 1 requires negative prompt inputs.")
            negative_model_inputs = {
                **model_inputs,
                "instruction_hidden_states": negative_prompt_embeds,
                "instruction_attention_mask": negative_prompt_embeds_mask,
            }

        return model_inputs, negative_model_inputs

    @classmethod
    def prepare_condition(
        cls,
        micro_batch: TensorDict,
        latents: torch.Tensor,
        step: int,
    ) -> Optional[dict]:
        del latents, step
        condition_image_latents = micro_batch.get("condition_image_latents", None)
        if condition_image_latents is None:
            return None
        return {"ref_image_hidden_states": normalize_ref_image_hidden_states(condition_image_latents)}

    @classmethod
    def inject_condition(
        cls,
        model_inputs: dict,
        negative_model_inputs: Optional[dict],
        condition: Optional[dict],
    ) -> tuple[dict, Optional[dict]]:
        if not condition:
            return model_inputs, negative_model_inputs

        ref_image_hidden_states = condition.get("ref_image_hidden_states")
        if ref_image_hidden_states is None:
            return model_inputs, negative_model_inputs

        model_inputs = dict(model_inputs)
        model_inputs["ref_image_hidden_states"] = ref_image_hidden_states

        if negative_model_inputs is not None:
            negative_model_inputs = dict(negative_model_inputs)
            negative_model_inputs["ref_image_hidden_states"] = ref_image_hidden_states

        return model_inputs, negative_model_inputs

    @classmethod
    def forward(
        cls,
        module,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        noise_pred = module(**model_inputs)
        if model_config.pipeline.true_cfg_scale > 1.0:
            if negative_model_inputs is None:
                raise ValueError("Boogu true_cfg_scale > 1 requires negative prompt inputs.")
            negative_noise_pred = module(**negative_model_inputs)
            noise_pred = apply_boogu_text_cfg(noise_pred, negative_noise_pred, model_config.pipeline.true_cfg_scale)
        return noise_pred

    @classmethod
    def forward_and_sample_previous_step(
        cls,
        module,
        scheduler: FlowMatchSDEDiscreteScheduler,
        model_config: DiffusionModelConfig,
        model_inputs: dict[str, torch.Tensor],
        negative_model_inputs: Optional[dict[str, torch.Tensor]],
        scheduler_inputs: Optional[TensorDict | dict[str, torch.Tensor]],
        step: int,
    ):
        assert scheduler_inputs is not None
        latents = scheduler_inputs["all_latents"]
        timesteps = scheduler_inputs["all_timesteps"]

        noise_pred = cls.forward(module, model_config, model_inputs, negative_model_inputs)

        _, log_prob, prev_sample_mean, std_dev_t, sqrt_dt = scheduler.sample_previous_step(
            sample=latents[:, step].float(),
            model_output=noise_pred.float(),
            timestep=timesteps[:, step],
            noise_level=model_config.algo.noise_level,
            prev_sample=latents[:, step + 1].float(),
            sde_type=model_config.algo.sde_type,
            return_logprobs=True,
            return_sqrt_dt=True,
        )
        return log_prob, prev_sample_mean, std_dev_t, sqrt_dt
