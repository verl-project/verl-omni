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
"""Temporary LoRA compatibility for the pinned vLLM-Omni stack."""

from vllm.config.utils import config
from vllm.model_executor.models.utils import WeightsMapper
from vllm_omni.config import omni_config
from vllm_omni.config.stage_config import StageExecutionType


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
    """Keep verl's tensor LoRA loader on the pinned vLLM's rename-only mapper."""
    # TODO: Remove when the verl pin calls get_rename_mapper instead of the
    # get_unstacked_mapper name removed in vLLM 0.30. Never skip HF name mapping.
    if not hasattr(WeightsMapper, "get_unstacked_mapper"):
        WeightsMapper.get_unstacked_mapper = WeightsMapper.get_rename_mapper
