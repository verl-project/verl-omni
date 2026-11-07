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
"""Verify the real typed LoRA resolver-to-EngineArgs path without model weights."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
from vllm_omni.config import omni_config
from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, StageExecutionType
from vllm_omni.engine.stage_init_utils import _project_omni_stage_engine_args
from vllm_omni.model_executor.models.qwen3_omni.pipeline import QWEN3_OMNI_THINKER_ONLY_PIPELINE

from verl_omni.utils.vllm_omni import patch
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy


@pytest.fixture
def lora_config_patch(monkeypatch):
    # Keep the process-global compatibility changes local to each test.
    for name in ("OmniStageLoadConfig", "_LLM_LOAD_ENGINE_FIELDS", "_LLM_STAGE_ENGINE_FIELDS", "_STAGE_ENGINE_FIELDS"):
        monkeypatch.setattr(omni_config, name, getattr(omni_config, name))
    for execution in (StageExecutionType.LLM_AR, StageExecutionType.LLM_GENERATION):
        monkeypatch.setitem(
            omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE,
            execution,
            omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE[execution],
        )
    field_map = omni_config._LOAD_STAGE_ENGINE_FIELD_MAP
    original = dict(field_map)
    yield
    field_map.clear()
    field_map.update(original)


@pytest.mark.parametrize("source", ["global", "stage", "deploy"])
def test_lora_flags_reach_real_engine_args(lora_config_patch, source):
    flags = {"enable_lora": True, "max_lora_rank": 8, "max_loras": 1, "fully_sharded_loras": True}
    engine_args = dict(flags)
    ARStrategy(SimpleNamespace(config=SimpleNamespace())).prepare_engine_args(engine_args, Namespace())
    assert all(engine_args[key] == value for key, value in flags.items())
    cli = flags if source == "global" else {f"stage_0_{key}": value for key, value in flags.items()}
    deploy = None
    if source == "deploy":
        cli = {}
        deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, engine_extras=flags)])
    config = omni_config.VllmOmniConfig.from_pipeline_config(
        QWEN3_OMNI_THINKER_ONLY_PIPELINE, user_deploy_config=deploy, cli_overrides=cli
    )
    projected = _project_omni_stage_engine_args(config.stage_by_id(0))
    assert {key: projected[key] for key in flags} == flags
    patch.patch_vllm_omni_lora_config()
    assert omni_config.OmniStageLoadConfig is patch._OmniLoRALoadConfig


def test_lora_patch_does_not_disable_unknown_field_validation(lora_config_patch):
    patch.patch_vllm_omni_lora_config()
    with pytest.raises(ValueError, match="no structured config owner: not_an_engine_field"):
        omni_config.VllmOmniConfig.from_pipeline_config(
            QWEN3_OMNI_THINKER_ONLY_PIPELINE, cli_overrides={"stage_0_not_an_engine_field": True}
        )


def test_lora_patch_does_not_give_diffusion_stages_llm_lora_fields(lora_config_patch):
    before = omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE[StageExecutionType.DIFFUSION]
    patch.patch_vllm_omni_lora_config()
    assert omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE[StageExecutionType.DIFFUSION] == before
    assert "enable_lora" not in before


def test_lora_patch_is_not_installed_for_merged_lora(monkeypatch):
    monkeypatch.setattr(patch, "patch_vllm_omni_lora_config", lambda: pytest.fail("unexpected LoRA patch"))
    ARStrategy(SimpleNamespace(config=SimpleNamespace())).prepare_engine_args({}, Namespace())


def test_mapper_compatibility_preserves_unfused_module_names(monkeypatch):
    from vllm.model_executor.models.utils import WeightsMapper

    monkeypatch.setattr(WeightsMapper, "get_unstacked_mapper", None, raising=False)
    monkeypatch.delattr(WeightsMapper, "get_unstacked_mapper")
    mapper = WeightsMapper(
        orig_to_new_prefix={"thinker.model.": "language_model.model."},
        orig_to_new_stacked={"q_proj": ("qkv_proj", "q")},
    )
    patch.patch_vllm_lora_weights_mapper()
    renamed = mapper.get_unstacked_mapper()
    assert renamed._map_name("thinker.model.layers.0.q_proj.weight") == "language_model.model.layers.0.q_proj.weight"
    assert mapper._map_name("thinker.model.layers.0.q_proj.weight") == "language_model.model.layers.0.qkv_proj.weight"
    patch.patch_vllm_lora_weights_mapper()
    assert WeightsMapper.get_unstacked_mapper is WeightsMapper.get_rename_mapper


def test_mapper_compatibility_keeps_existing_upstream_method(monkeypatch):
    from vllm.model_executor.models.utils import WeightsMapper

    sentinel = lambda self: self
    monkeypatch.setattr(WeightsMapper, "get_unstacked_mapper", sentinel, raising=False)
    patch.patch_vllm_lora_weights_mapper()
    assert WeightsMapper.get_unstacked_mapper is sentinel


def test_real_verl_tensor_lora_loader_preserves_qwen_prefixes(monkeypatch):
    import torch
    from verl.utils.vllm.utils import TensorLoRARequest, VLLMHijack
    from vllm.lora import lora_model
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.model_executor.models.utils import WeightsMapper
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    monkeypatch.setattr(WeightsMapper, "get_unstacked_mapper", None, raising=False)
    monkeypatch.delattr(WeightsMapper, "get_unstacked_mapper")
    patch.patch_vllm_lora_weights_mapper()
    monkeypatch.setattr(LRUCacheWorkerLoRAManager, "_load_adapter", LRUCacheWorkerLoRAManager._load_adapter)
    # The GPU wheel enables pinning at import; CPU tests must not allocate pinned memory.
    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)
    VLLMHijack.hijack()
    modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    tensors = {}
    for index, module in enumerate(modules):
        prefix = f"base_model.model.thinker.model.layers.0.self_attn.{module}"
        tensors[f"{prefix}.lora_A.weight"] = torch.full((2, 4), float(index + 1))
        tensors[f"{prefix}.lora_B.weight"] = torch.full((4, 2), float(index + 2))
    request = TensorLoRARequest(
        lora_name="tiny",
        lora_int_id=1,
        lora_path="memory://tiny",
        peft_config={"r": 2, "lora_alpha": 4, "target_modules": modules},
        lora_tensors=tensors,
    )
    manager = SimpleNamespace(
        _adapter_manager=SimpleNamespace(
            supported_lora_modules=modules,
            packed_modules_mapping={},
            model=SimpleNamespace(hf_to_vllm_mapper=Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper),
        ),
        lora_config=SimpleNamespace(max_lora_rank=8, lora_dtype=torch.float32),
        _lora_model_cls=LoRAModel,
        vocab_size=512,
    )
    loaded = LRUCacheWorkerLoRAManager._load_adapter(manager, request)
    assert len(loaded.loras) == 4
    for index, module in enumerate(modules):
        lora = loaded.loras[f"language_model.model.layers.0.self_attn.{module}"]
        torch.testing.assert_close(lora.lora_a, torch.full((2, 4), float(index + 1)))
        torch.testing.assert_close(lora.lora_b, torch.full((4, 2), float(index + 2)))
