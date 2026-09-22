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
"""Teacher replica and prompt-logprob contracts through the real AR strategy."""

from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from vllm import SamplingParams

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniReplica


def _output():
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=[8], finish_reason="length", logprobs=[{8: SimpleNamespace(logprob=-0.5)}])],
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=[None, {2: SimpleNamespace(logprob=-0.2, rank=1)}, {3: SimpleNamespace(logprob=-0.3, rank=1)}],
    )


@pytest.mark.parametrize("response_ids", [[], [198, 198, 271]])
def test_typed_request_preserves_minicpm_tokens_media_and_processor_options(monkeypatch, response_ids):
    from verl_omni.pipelines.minicpm import omni_rollout_adapter as adapter
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    monkeypatch.setattr(adapter, "_install_token_native_multimodal_replay", lambda: None)
    strategy = ARStrategy(
        SimpleNamespace(
            config=SimpleNamespace(max_model_len=64, prompt_length=32, response_length=8),
            model_config=SimpleNamespace(processor=None),
        )
    )
    strategy._rollout_adapter = adapter.MiniCPMRolloutAdapter
    replay = {"source_ids": [1, 8, 3], "expanded_ids": [1, 4, 4, 3]}
    kwargs = {adapter.MINICPM_PROMPT_KEY: replay, "max_slice_nums": 1, "use_image_id": False}
    image = object()
    request = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=replay["expanded_ids"] + response_ids,
        image_data=[image],
        mm_processor_kwargs=kwargs,
    )

    prompt, params = strategy.preprocess_input(request, {"max_tokens": 1, "prompt_logprobs": 0}, None)

    assert prompt["prompt_token_ids"] == replay["expanded_ids"] + response_ids
    assert prompt["multi_modal_data"] == {"image": [image]}
    assert prompt["mm_processor_kwargs"] == {
        adapter._MINICPM_PROCESSED_PROMPT_KEY: replay,
        "max_slice_nums": 1,
        "use_image_id": False,
    }
    assert adapter.MINICPM_PROMPT_KEY in kwargs
    assert params.prompt_logprobs == 0


@pytest.mark.parametrize("stage_index", [0, 1])
def test_teacher_scores_follow_selected_stage_and_next_token_alignment(stage_index, monkeypatch):
    import verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy as module

    extract = MagicMock(wraps=module.extract_prompt_logprobs)
    monkeypatch.setattr(module, "extract_prompt_logprobs", extract)
    strategy = ARStrategy(SimpleNamespace(global_steps=7))
    strategy._policy_stage_index = stage_index
    teacher_params = SamplingParams(max_tokens=1, prompt_logprobs=0)
    params = teacher_params if stage_index == 0 else [SamplingParams(), teacher_params]
    result = strategy.process_output(SimpleNamespace(request_output=_output()), params, {})
    assert result.extra_fields["prompt_ids"] == [[2], [3], [0]]
    assert result.extra_fields["prompt_logprobs"] == [[-0.2], [-0.3], [0.0]]
    assert result.extra_fields["rollout_prompt_ids"] == [1, 2, 3]
    assert result.log_probs is None
    extract.assert_called_once()


def test_teacher_fails_if_engine_omits_prompt_scores():
    strategy = ARStrategy(SimpleNamespace(global_steps=0))
    output = _output()
    output.prompt_logprobs = None
    with pytest.raises(RuntimeError, match="teacher did not return"):
        strategy.process_output(output, SamplingParams(max_tokens=1, prompt_logprobs=0), {})


def test_student_keeps_sampled_logprobs_without_teacher_fields():
    strategy = ARStrategy(SimpleNamespace(global_steps=0))
    result = strategy.process_output(_output(), SamplingParams(logprobs=0), {})
    assert result.log_probs == [-0.5]
    assert "prompt_logprobs" not in result.extra_fields


@pytest.mark.parametrize("explicit,expected", [(None, True), (False, False), (True, True)])
def test_teacher_model_loading_inherits_engine_trust_without_overwriting_explicit_value(
    monkeypatch, explicit, expected
):
    import verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server as module

    init = MagicMock(return_value=None)
    monkeypatch.setattr(module.vLLMReplica, "__init__", init)
    monkeypatch.setattr(module.ray, "remote", lambda cls: cls)
    rollout = SimpleNamespace(engine_kwargs={"vllm_omni": {"trust_remote_code": True, "output_mode": "ar"}})
    config = OmegaConf.create({"_target_": "verl.workers.config.HFModelConfig", "path": "teacher"})
    if explicit is not None:
        config.trust_remote_code = explicit
    vLLMOmniReplica(0, rollout, config, is_teacher_model=True)
    result = init.call_args.args[2]
    assert result.trust_remote_code is expected
    assert result._target_ == "verl_omni.workers.config.OmniModelConfig"
    assert config._target_ == "verl.workers.config.HFModelConfig"
    if explicit is None:
        assert "trust_remote_code" not in config


@pytest.mark.parametrize("registered", [False, True])
def test_registered_topology_owns_stage_names_for_single_stage_pipelines(registered):
    from verl_omni.pipelines.minicpm import MiniCPMRolloutAdapter

    strategy = ARStrategy(SimpleNamespace(config=SimpleNamespace(logprobs_mode="processed_logprobs")))
    if registered:
        strategy._rollout_adapter = MiniCPMRolloutAdapter
        strategy._single_model_stage = MiniCPMRolloutAdapter.build_stage_configs()[0].model_stage
    args = {"model_stage": "thinker"}
    strategy.prepare_engine_args(args, Namespace())
    assert args["model_stage"] == ("llm" if registered else "thinker")
    if registered:
        assert strategy._rollout_adapter.build_stage_configs()[0].model_stage == "llm"


def test_native_stage_factory_retains_minicpm_llm_stage(monkeypatch):
    from vllm_omni.config.config_factory import StageConfigFactory

    import verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy as module

    monkeypatch.setattr(
        StageConfigFactory,
        "get_hf_config",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="minicpmo",
            architectures=["MiniCPMO"],
            version="4.5",
        ),
    )
    monkeypatch.setattr(module, "get_visible_devices_keyword", lambda: "MINICPM_TEST_GPUS")
    monkeypatch.setenv("MINICPM_TEST_GPUS", "0,1")
    server = SimpleNamespace(config=SimpleNamespace(tensor_model_parallel_size=2, logprobs_mode="processed_logprobs"))
    strategy = ARStrategy(server)
    kwargs = {"pipeline_name": "minicpmo_4_5", "async_chunk": False}
    strategy.preprocess_engine_kwargs(kwargs)
    try:
        assert "model_stage" not in kwargs
        overrides = {"model_stage": "thinker"}
        strategy.prepare_engine_args(overrides, Namespace())
        stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(
            "minicpm-stage-test",
            trust_remote_code=True,
            cli_overrides=overrides,
            deploy_config_path=kwargs["deploy-config"],
        )
        assert len(stages) == 1
        assert stages[0].to_omegaconf().engine_args.model_stage == "llm"
    finally:
        server._temp_deploy_ctx.cleanup()


def test_teacher_replica_passes_identity_to_upstream(monkeypatch):
    import verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server as module

    init = MagicMock(return_value=None)
    monkeypatch.setattr(module.vLLMReplica, "__init__", init)
    monkeypatch.setattr(module.ray, "remote", lambda cls: cls)
    replica = vLLMOmniReplica(0, "rollout", "model", 2, is_teacher_model=True, name_suffix="teacher_a")
    init.assert_called_once_with(0, "rollout", "model", 2, False, True, "teacher_a")
    assert replica.server_class is module.vLLMOmniHttpServer
