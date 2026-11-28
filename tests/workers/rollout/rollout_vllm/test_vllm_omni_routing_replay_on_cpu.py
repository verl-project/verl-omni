# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the R3 rollout routing-replay wiring in :class:`ARStrategy`."""

import types

import numpy as np
import pytest
import yaml
from verl.utils.device import get_visible_devices_keyword

import verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy as ar_strategy
from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy


class _ExecutionType:
    """Stand-in for ``vllm_omni``'s enum so stubbed imports compare by identity."""

    LLM_AR = "llm_ar"
    LLM_GENERATION = "llm_generation"
    DIFFUSION = "diffusion"


def _stage(stage_id, *, final_output=True, execution_type=_ExecutionType.LLM_AR):
    return types.SimpleNamespace(
        stage_id=stage_id,
        execution_type=execution_type,
        final_output=final_output,
        final_output_type="text" if final_output else None,
        sampling_constraints={},
    )


def _adapter(stages):
    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="thinker_only"):
            return stages

        @classmethod
        def get_pipeline_id(cls, pipeline_mode="thinker_only"):
            return "qwen3_omni_moe"

    return Adapter


def _write_deploy_config(monkeypatch, *, routing_replay=True, stages=None):
    monkeypatch.setattr(ar_strategy, "StageExecutionType", _ExecutionType)
    monkeypatch.setenv(get_visible_devices_keyword(), "0,1")
    fake_self = types.SimpleNamespace(
        config=types.SimpleNamespace(
            tensor_model_parallel_size=2,
            enable_rollout_routing_replay=routing_replay,
            enable_prefix_caching=True,
            pipeline_model_parallel_size=1,
        ),
        model_config=types.SimpleNamespace(lora={}, lora_rank=0),
    )
    engine_kwargs: dict = {}
    strategy = ARStrategy(fake_self)
    strategy._write_deploy_config(engine_kwargs, "qwen3_omni_moe", _adapter(stages or [_stage(0)]), "thinker_only")
    try:
        with open(engine_kwargs["deploy_config"]) as f:
            return yaml.safe_load(f)
    finally:
        strategy.server._temp_deploy_ctx.cleanup()


def test_deploy_config_selects_model_runner_v2_with_routing_replay(monkeypatch):
    deploy = _write_deploy_config(monkeypatch, routing_replay=True)
    assert deploy["stages"][0]["model_runner"] == "v2"


def test_deploy_config_omits_model_runner_without_routing_replay(monkeypatch):
    deploy = _write_deploy_config(monkeypatch, routing_replay=False)
    assert "model_runner" not in deploy["stages"][0]


def test_deploy_config_rejects_multi_stage_routing_replay(monkeypatch):
    with pytest.raises(NotImplementedError, match="single-stage"):
        _write_deploy_config(monkeypatch, stages=[_stage(0), _stage(1, final_output=False)])


def test_deploy_config_rejects_non_final_policy_stage(monkeypatch):
    with pytest.raises(NotImplementedError, match="final autoregressive"):
        _write_deploy_config(monkeypatch, stages=[_stage(0, final_output=False)])


def _validation_server(**overrides):
    config_kwargs = dict(
        max_model_len=None,
        prompt_length=8,
        response_length=8,
        expert_parallel_size=1,
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        engine_kwargs={},
        enable_rollout_routing_replay=True,
        enable_prefix_caching=True,
        pipeline_model_parallel_size=1,
    )
    config_kwargs.update(overrides)
    return types.SimpleNamespace(config=types.SimpleNamespace(**config_kwargs), global_steps=0)


def test_validate_rejects_routing_replay_without_prefix_caching():
    strategy = ARStrategy(_validation_server(enable_prefix_caching=False))
    with pytest.raises(ValueError, match="enable_prefix_caching"):
        strategy.validate_configs()


def test_validate_rejects_routing_replay_with_pipeline_parallelism():
    strategy = ARStrategy(_validation_server(pipeline_model_parallel_size=2))
    with pytest.raises(ValueError, match="pipeline_model_parallel_size"):
        strategy.validate_configs()


def test_validate_rejects_adapters_that_remap_policy_tokens(monkeypatch):
    class RemapAdapter(OmniRolloutPipelineBase):
        @classmethod
        def postprocess_agent_loop_output(cls, output, *, tokenizer, response_length):
            return output

    monkeypatch.setattr(
        OmniRolloutPipelineBase,
        "_registry",
        {**OmniRolloutPipelineBase._registry, "r3_remap_test": RemapAdapter},
    )
    strategy = ARStrategy(
        _validation_server(engine_kwargs={"vllm_omni": {"pipeline_name": "r3_remap_test"}}),
    )
    with pytest.raises(NotImplementedError, match="overrides postprocess_agent_loop_output"):
        strategy.validate_configs()


def _completion(routed_experts, finish_reason="stop"):
    return types.SimpleNamespace(
        outputs=[
            types.SimpleNamespace(
                token_ids=[11, 12],
                logprobs=None,
                finish_reason=finish_reason,
                routed_experts=routed_experts,
            )
        ],
        prompt_token_ids=[1, 2],
    )


def _policy_params():
    return types.SimpleNamespace(prompt_logprobs=None, logprobs=None)


def test_process_output_forwards_routed_experts():
    strategy = ARStrategy(_validation_server())
    routes = np.zeros((4, 2, 1), dtype=np.int16)
    output = strategy.process_output(_completion(routes), _policy_params(), {})
    assert output.routed_experts is routes


def test_process_output_fails_loudly_when_routed_experts_are_missing():
    strategy = ARStrategy(_validation_server())
    with pytest.raises(RuntimeError, match="routed_experts"):
        strategy.process_output(_completion(None), _policy_params(), {})


def test_process_output_tolerates_missing_routed_experts_on_abort():
    strategy = ARStrategy(_validation_server())
    output = strategy.process_output(_completion(None, finish_reason="abort"), _policy_params(), {})
    assert output.routed_experts is None


def test_process_output_ignores_routed_experts_when_disabled():
    strategy = ARStrategy(_validation_server(enable_rollout_routing_replay=False))
    routes = np.zeros((4, 2, 1), dtype=np.int16)
    output = strategy.process_output(_completion(routes), _policy_params(), {})
    assert output.routed_experts is None
