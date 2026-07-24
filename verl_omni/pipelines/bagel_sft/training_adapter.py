# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Training adapter for BAGEL Uni-COT SFT."""

from __future__ import annotations

import logging

import torch
from verl.utils.device import get_device_name

from verl_omni.pipelines.bagel_flow_grpo.common import setup_bagel_sigmas
from verl_omni.pipelines.model_base import DiffusionModelBase
from verl_omni.pipelines.schedulers import FlowMatchSDEDiscreteScheduler
from verl_omni.workers.config import DiffusionModelConfig

from .bagel_sft_model import BagelForSFT

logger = logging.getLogger(__name__)


@DiffusionModelBase.register("OmniBagelForConditionalGeneration", algorithm="bagel_sft")
class BagelSFTDiffusion(DiffusionModelBase):
    """Diffusion registry hook that builds ``BagelForSFT`` for FSDP."""

    @classmethod
    def build_module(cls, model_config: DiffusionModelConfig, torch_dtype: torch.dtype):
        logger.info("Loading BagelForSFT from %s", model_config.local_path)
        return BagelForSFT.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def configure_train_mode(cls, module):
        inner = module.module if hasattr(module, "module") else module
        if not hasattr(inner, "layers"):
            return
        inner.training = False
        for layer in inner.layers:
            layer_inner = layer.module if hasattr(layer, "module") else layer
            layer_inner.training = False
            if hasattr(layer_inner, "self_attn"):
                layer_inner.self_attn.training = False

    @classmethod
    def configure_trainable_params(cls, module, model_config):
        # LoRA runs rely on PEFT to mark adapter parameters trainable.  For
        # full-weight SFT, keep the generation path and text head trainable.
        if getattr(model_config, "lora_rank", 0) > 0 or getattr(model_config, "lora_adapter_path", None) is not None:
            return
        for name, param in module.named_parameters():
            param.requires_grad = "moe_gen" in name or name.startswith("lm_head")
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

    @classmethod
    def build_scheduler(cls, model_config: DiffusionModelConfig):
        scheduler = FlowMatchSDEDiscreteScheduler()
        cls.set_timesteps(scheduler, model_config, get_device_name())
        return scheduler

    @classmethod
    def set_timesteps(cls, scheduler: FlowMatchSDEDiscreteScheduler, model_config: DiffusionModelConfig, device: str):
        setup_bagel_sigmas(scheduler, model_config.pipeline.num_inference_steps, device=device)

    @classmethod
    def prepare_model_inputs(cls, *args, **kwargs):
        raise NotImplementedError("BagelSFTDiffusion is driven by BagelSFTDiffusersFSDPEngine.")

    @classmethod
    def forward_and_sample_previous_step(cls, *args, **kwargs):
        raise NotImplementedError("BAGEL SFT is a supervised one-shot objective, not reverse-trajectory sampling.")
