# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni input adapter over verl's unchanged Megatron LM engine."""

from contextlib import ExitStack
from copy import copy, deepcopy

from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

from verl_omni.pipelines.qwen3_omni.megatron_inputs import qwen3_omni_megatron_inputs


@EngineRegistry.register(model_type="omni_model", backend="megatron")
class OmniMegatronEngine(MegatronEngineWithLMHead):
    """Train the Qwen3-Omni Thinker using BSHD and the upstream V1 engine."""

    def __init__(self, model_config, engine_config, optimizer_config, checkpoint_config):
        if model_config.hf_config.model_type != "qwen3_omni_moe" or model_config.model_stage != "thinker":
            raise ValueError("The Omni Megatron engine currently supports Qwen3-Omni Thinker only.")
        if getattr(getattr(model_config, "mtp", None), "enable", False):
            raise ValueError("Qwen3-Omni Megatron does not support MTP because the Thinker constructs M-RoPE.")
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
        self._forward_model = None
        self._input_adapters = None
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        with ExitStack() as input_adapters:
            self._forward_model = model
            self._input_adapters = input_adapters
            try:
                return super().forward_step(batch_iter, model, logits_processor_func, postprocess_micro_batch_func)
            finally:
                self._input_adapters = None
                self._forward_model = None

    def prepare_model_inputs(self, batch):
        model_inputs = super().prepare_model_inputs(batch)
        if self._forward_model is None or self._input_adapters is None:
            raise RuntimeError("Qwen3-Omni model inputs must be prepared inside forward_step.")
        self._input_adapters.enter_context(
            qwen3_omni_megatron_inputs(self._forward_model, model_inputs["multi_modal_inputs"])
        )
        return model_inputs
