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
"""Temporary LoRA and rope-read compatibility for the pinned vLLM-Omni stack."""

import functools

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
    """Keep verl's tensor LoRA loader using vLLM 0.30's rename-only mapper."""
    # TODO: Remove when the verl pin calls get_rename_mapper instead of the
    # get_unstacked_mapper name removed in vLLM 0.30. Never skip HF name mapping.
    if not hasattr(WeightsMapper, "get_unstacked_mapper"):
        WeightsMapper.get_unstacked_mapper = WeightsMapper.get_rename_mapper


class _RopeFallbackRegistry(dict):
    """Wrap registered default-rope initializers so a bare rope_theta read falls
    back to rope_parameters, where Transformers 5.16+ moved the value."""

    def __setitem__(self, key, value):
        if key == "default":
            value = _rope_theta_from_parameters_on_miss(value)
        super().__setitem__(key, value)

    def setdefault(self, key, default=None):
        # dict.setdefault bypasses __setitem__; qwen-tts registers through it.
        if key not in self:
            self[key] = default
        return dict.__getitem__(self, key)


def _rope_theta_from_parameters_on_miss(rope_fn):
    if rope_fn is None or getattr(rope_fn, "_rope_fallback_wrapped", False):
        return rope_fn

    @functools.wraps(rope_fn)
    def wrapper(config, *args, **kwargs):
        if not hasattr(config, "rope_theta"):
            theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta")
            if theta is not None:
                config.rope_theta = theta
        return rope_fn(config, *args, **kwargs)

    wrapper._rope_fallback_wrapped = True
    return wrapper


def patch_qwen3_tts_default_rope_read() -> None:
    """Intercept default-rope registrations in Transformers' rope registry.

    The pinned qwen-tts installs a process-global default-rope initializer that
    reads the bare rope_theta attribute; Transformers 5.16+ moved that value
    into rope_parameters on several config classes, so an unwrapped read kills
    every default-rope model build in a process that imported qwen_tts.
    """
    import sys

    from transformers import modeling_rope_utils

    registry = modeling_rope_utils.ROPE_INIT_FUNCTIONS
    if isinstance(registry, _RopeFallbackRegistry):
        return
    wrapped = _RopeFallbackRegistry()
    for key, value in registry.items():
        wrapped[key] = value
    modeling_rope_utils.ROPE_INIT_FUNCTIONS = wrapped
    # Modules that bound the original dict by reference keep reading it;
    # rebind every holder so later registrations wrap too.
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        for name, value in vars(module).items():
            if value is registry:
                setattr(module, name, wrapped)
