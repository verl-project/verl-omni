# Copyright 2026 Bytedance Ltd. and/or its affiliates.
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
"""Verify the typed LoRA resolver-to-EngineArgs path without model weights."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
from vllm_omni.config import omni_config
from vllm_omni.config.stage_config import DeployConfig, StageDeployConfig, StageExecutionType
from vllm_omni.engine.stage_init_utils import _project_omni_stage_engine_args
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_PIPELINE,
    QWEN3_OMNI_THINKER_ONLY_PIPELINE,
)

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy, lora_engine_extras


def test_lora_engine_extras_mirror_verl_rollout_flags():
    model_config = SimpleNamespace(lora={"rank": 66}, lora_rank=0)
    assert lora_engine_extras(model_config) == {"enable_lora": True, "max_loras": 1, "max_lora_rank": 128}

    model_config = SimpleNamespace(lora={}, lora_rank=8)
    assert lora_engine_extras(model_config) == {"enable_lora": True, "max_loras": 1, "max_lora_rank": 8}

    model_config = SimpleNamespace(lora={"rank": 66, "fully_sharded_loras": True}, lora_rank=0)
    assert lora_engine_extras(model_config)["fully_sharded_loras"] is True


def test_lora_engine_extras_empty_without_trainable_lora():
    # Merged-LoRA configs keep the rollout engine LoRA-free: verl injects no
    # flags for them, so the deploy config must not either.
    assert lora_engine_extras(SimpleNamespace(lora={"rank": 66, "merge": True}, lora_rank=0)) == {}
    assert lora_engine_extras(SimpleNamespace(lora={}, lora_rank=0)) == {}


@pytest.mark.parametrize("source", ["stage", "deploy"])
def test_lora_flags_reach_real_engine_args(source):
    flags = {"enable_lora": True, "max_lora_rank": 8, "max_loras": 1, "fully_sharded_loras": True}
    cli = {f"stage_0_{key}": value for key, value in flags.items()}
    deploy = None
    if source == "deploy":
        cli = {}
        deploy = DeployConfig(stages=[StageDeployConfig(stage_id=0, engine_extras=flags)])
    config = omni_config.VllmOmniConfig.from_pipeline_config(
        QWEN3_OMNI_THINKER_ONLY_PIPELINE, user_deploy_config=deploy, cli_overrides=cli
    )
    projected = _project_omni_stage_engine_args(config.stage_by_id(0))
    assert {key: projected[key] for key in flags} == flags


def test_prepare_engine_args_drops_global_lora_flags():
    engine_args = {"enable_lora": True, "max_loras": 1, "max_lora_rank": 128, "fully_sharded_loras": True}
    ARStrategy(SimpleNamespace(config=SimpleNamespace())).prepare_engine_args(engine_args, Namespace())
    assert engine_args.get("enable_lora") is None
    assert engine_args.get("max_loras") is None
    assert engine_args.get("max_lora_rank") is None
    assert engine_args.get("fully_sharded_loras") is None


def test_global_lora_flags_rejected_on_multi_stage_pipelines():
    # Why ``prepare_engine_args`` drops the globals: upstream vLLM-Omni owns
    # vLLM LoRA per LLM stage and rejects the broadcast flags outright.
    with pytest.raises(ValueError, match="set vLLM LoRA argument"):
        omni_config.VllmOmniConfig.from_pipeline_config(
            QWEN3_OMNI_PIPELINE, cli_overrides={"enable_lora": True, "max_loras": 1, "max_lora_rank": 8}
        )


def test_unknown_stage_engine_fields_still_rejected():
    with pytest.raises(ValueError, match="no structured config owner: not_an_engine_field"):
        omni_config.VllmOmniConfig.from_pipeline_config(
            QWEN3_OMNI_THINKER_ONLY_PIPELINE, cli_overrides={"stage_0_not_an_engine_field": True}
        )


def test_diffusion_stages_own_no_llm_lora_fields():
    diffusion_fields = omni_config._STAGE_ENGINE_FIELDS_BY_EXECUTION_TYPE[StageExecutionType.DIFFUSION]
    assert "enable_lora" not in diffusion_fields


def test_real_verl_tensor_lora_loader_preserves_qwen_prefixes(monkeypatch):
    import torch
    from verl.utils.vllm.utils import TensorLoRARequest, VLLMHijack
    from vllm.lora import lora_model
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

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
