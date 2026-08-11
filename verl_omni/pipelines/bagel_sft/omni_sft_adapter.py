# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""BAGEL adapter for the omni SFT engine."""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import torch
from verl.utils import hf_tokenizer

from verl_omni.pipelines.model_base import OmniModelBase

from .bagel_sft_model import BagelForSFT

logger = logging.getLogger(__name__)


@OmniModelBase.register("OmniBagelForConditionalGeneration", stage="thinker")
class BagelOmniSFTAdapter(OmniModelBase):
    """Adapter that plugs BAGEL SFT into the generic omni SFT path."""

    @classmethod
    def build_hf_config(cls, model_config):
        with open(os.path.join(model_config.local_hf_config_path, "config.json")) as f:
            raw_config = json.load(f)
        return SimpleNamespace(**raw_config)

    @classmethod
    def build_module(cls, model_config, torch_dtype: torch.dtype):
        logger.info("Loading BagelForSFT from %s", model_config.local_path)
        return BagelForSFT.from_pretrained(model_config.local_path, torch_dtype=torch_dtype)

    @classmethod
    def get_strip_modules(cls, model_config) -> list[str]:
        return []

    @classmethod
    def configure_tokenizer(cls, model_path: str, model_config):
        return hf_tokenizer(model_path, trust_remote_code=model_config.trust_remote_code, use_fast=True)

    @classmethod
    def configure_processor(cls, model_path: str, model_config):
        return None

    @classmethod
    def configure_train_mode(cls, module) -> None:
        inner = module.module if hasattr(module, "module") else module
        if not hasattr(inner, "layers"):
            return
        inner.training = False
        for layer in inner.layers:
            layer_inner = layer.module if hasattr(layer, "module") else layer
            layer_inner.training = False
            if hasattr(layer_inner, "self_attn"):
                layer_inner.self_attn.training = False
