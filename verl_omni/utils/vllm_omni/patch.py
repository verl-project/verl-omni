# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Copyright contributors to the vLLM-Omni project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
"""Temporary CUDA Graph and LoRA compatibility for the pinned vLLM-Omni stack."""

from collections.abc import Sequence

import torch
from vllm.config.utils import config
from vllm.model_executor.models.utils import WeightsMapper
from vllm.sequence import IntermediateTensors
from vllm_omni.config import omni_config
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import Qwen3MoeLLMModel

_ORIGINAL_THINKER_FORWARD = Qwen3MoeLLMModel.forward


# TODO: Remove when the pin includes the LLM LoRA owner from
# https://github.com/vllm-project/vllm-omni/pull/8574.
# Use the existing typed loading projection for verl's generated LoRA flags;
# vLLM EngineArgs still constructs and validates LoRAConfig in the engine process.
@config(kw_only=True)
class _OmniLoRALoadConfig(omni_config.OmniStageLoadConfig):
    enable_lora: bool | None = None
    max_lora_rank: int | None = None
    max_loras: int | None = None
    fully_sharded_loras: bool | None = None


def patch_vllm_omni_lora_config() -> None:
    """Give verl's LoRA engine flags a typed LLM-only config and projection."""
    llm_fields = omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE[StageExecutionType.LLM_AR]
    if "enable_lora" in llm_fields:
        return
    lora_fields = frozenset(_OmniLoRALoadConfig.__annotations__)
    omni_config.OmniStageLoadConfig = _OmniLoRALoadConfig
    omni_config._LLM_LOAD_ENGINE_FIELDS |= lora_fields
    omni_config._LLM_STAGE_ENGINE_FIELDS |= lora_fields
    omni_config._STAGE_ENGINE_FIELDS |= lora_fields
    for execution_type in (StageExecutionType.LLM_AR, StageExecutionType.LLM_GENERATION):
        omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE[execution_type] |= lora_fields
    # stage_init_utils imports this mapping by reference; extend it in place.
    omni_config._LOAD_STAGE_ENGINE_FIELD_MAP.update({name: name for name in lora_fields})


def patch_vllm_lora_weights_mapper() -> None:
    """Keep verl's tensor LoRA loader using vLLM 0.30's rename-only mapper."""
    # TODO: Remove when the verl pin calls get_rename_mapper instead of the
    # get_unstacked_mapper name removed in vLLM 0.30. Never skip HF name mapping.
    if not hasattr(WeightsMapper, "get_unstacked_mapper"):
        WeightsMapper.get_unstacked_mapper = WeightsMapper.get_rename_mapper


def patch_qwen3_omni_thinker_forward() -> None:
    """Wrap the pinned Thinker forward before model construction in each worker."""
    if "update" in Qwen3MoeLLMModel.forward.__code__.co_names:
        Qwen3MoeLLMModel.forward = _qwen3_omni_thinker_forward


# Dynamo inlines the upstream call, consuming its local dict.update during tracing.
# Real buffer mutations still emit update in transformed bytecode and retain vLLM's guard.
# TODO: Remove when the pin fixes the local dict.update introduced by
# https://github.com/vllm-project/vllm-omni/pull/7345 and the unwrapped forward compiles.
def _qwen3_omni_thinker_forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
    capture_layer_indices: Sequence[int] | None = None,
    return_hidden_states: bool = False,
    deepstack_input_embeds: IntermediateTensors | None = None,
) -> torch.Tensor | IntermediateTensors:
    return _ORIGINAL_THINKER_FORWARD(
        self,
        input_ids=input_ids,
        positions=positions,
        intermediate_tensors=intermediate_tensors,
        inputs_embeds=inputs_embeds,
        capture_layer_indices=capture_layer_indices,
        return_hidden_states=return_hidden_states,
        deepstack_input_embeds=deepstack_input_embeds,
    )
