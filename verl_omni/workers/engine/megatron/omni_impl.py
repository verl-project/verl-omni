# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni input adapter over verl's unchanged Megatron LM engine."""

from copy import copy, deepcopy

from verl.utils.model import extract_multi_modal_inputs
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from verl_omni.pipelines.qwen3_omni.megatron_inputs import qwen3_omni_megatron_inputs


@EngineRegistry.register(model_type="omni_model", backend="megatron")
class OmniMegatronEngine(MegatronEngineWithLMHead):
    """Train the Qwen3-Omni Thinker using BSHD and the upstream V1 engine."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        if model_config.hf_config.model_type != "qwen3_omni_moe" or model_config.model_stage != "thinker":
            raise ValueError("The Omni Megatron engine currently supports Qwen3-Omni Thinker only.")
        if engine_config.use_remove_padding or engine_config.use_fused_kernels:
            raise ValueError("Qwen3-Omni Megatron requires use_remove_padding=false and use_fused_kernels=false.")
        if engine_config.pipeline_model_parallel_size != 1 or engine_config.context_parallel_size != 1:
            raise ValueError("The Qwen3-Omni Megatron input adapter currently requires PP=CP=1.")
        # Upstream module construction reads text_config.hidden_size even for
        # policy models. Keep this compatibility view private to the engine;
        # the worker/rollout retain the original nested Omni configuration.
        model_config = copy(model_config)
        model_config.hf_config = deepcopy(model_config.hf_config)
        model_config.hf_config.text_config = model_config.hf_config.thinker_config.text_config
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        batch = next(batch_iter)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))
        with qwen3_omni_megatron_inputs(model, multi_modal_inputs):
            return super().forward_step(iter([batch]), model, logits_processor_func, postprocess_micro_batch_func)
