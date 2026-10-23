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
    assert result.log_probs is None
    extract.assert_called_once()


def test_teacher_fails_if_engine_omits_prompt_scores():
    strategy = ARStrategy(SimpleNamespace(global_steps=0))
    output = _output()
    output.prompt_logprobs = None
    with pytest.raises(RuntimeError, match="teacher did not return"):
        strategy.process_output(output, SamplingParams(max_tokens=1, prompt_logprobs=0), {})


def test_teacher_fails_on_non_finite_prompt_scores():
    strategy = ARStrategy(SimpleNamespace(global_steps=0))
    output = _output()
    output.prompt_logprobs[2] = {3: SimpleNamespace(logprob=float("-inf"), rank=1)}
    with pytest.raises(RuntimeError, match="non-finite"):
        strategy.process_output(output, SamplingParams(max_tokens=1, prompt_logprobs=0), {})


def _engine_reporting(prompt_token_ids):
    class Engine:
        generate_kwargs = None

        async def generate(self, **kwargs):
            Engine.generate_kwargs = kwargs
            yield SimpleNamespace(request_output=SimpleNamespace(prompt_token_ids=prompt_token_ids))

    return Engine()


@pytest.mark.asyncio
@pytest.mark.parametrize("scored_ids", [[5, 6, 7], [5, 9, 7], [5, 6]])
async def test_teacher_must_score_exactly_the_requested_ids(scored_ids):
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    server = SimpleNamespace(
        config=SimpleNamespace(max_model_len=64, prompt_length=32, response_length=8),
        model_config=SimpleNamespace(processor=None),
        engine=_engine_reporting(scored_ids),
    )
    strategy = ARStrategy(server)
    request = OmniRolloutRequest.from_generate_kwargs(prompt_ids=[5, 6, 7])
    prompt, params = strategy.preprocess_input(request, {"max_tokens": 1, "prompt_logprobs": 0}, None)

    if scored_ids == [5, 6, 7]:
        await strategy.run_generation(prompt, params, "request-0", None, 0)
    else:
        with pytest.raises(RuntimeError, match="different token sequence"):
            await strategy.run_generation(prompt, params, "request-0", None, 0)
    assert server.engine.generate_kwargs["prompt"] == {"prompt_token_ids": [5, 6, 7]}


@pytest.mark.asyncio
async def test_student_rollout_does_not_check_engine_prompt_ids():
    from verl_omni.pipelines.rollout_request import OmniRolloutRequest

    server = SimpleNamespace(
        config=SimpleNamespace(max_model_len=64, prompt_length=32, response_length=8),
        model_config=SimpleNamespace(processor=None),
        engine=_engine_reporting([1]),
    )
    strategy = ARStrategy(server)
    prompt, params = strategy.preprocess_input(
        OmniRolloutRequest.from_generate_kwargs(prompt_ids=[5, 6, 7]), {"max_tokens": 2}, None
    )

    await strategy.run_generation(prompt, params, "request-0", None, 0)

    assert server.engine.generate_kwargs["prompt"] == {"prompt_token_ids": [5, 6, 7]}


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

    def fake_init(self, *args):
        init(*args)
        # The real base __init__ sets self.config, which the SP footprint check reads.
        self.config = SimpleNamespace()

    monkeypatch.setattr(module.vLLMReplica, "__init__", fake_init)
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


def test_teacher_replica_passes_identity_to_upstream(monkeypatch):
    import verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server as module

    init = MagicMock(return_value=None)

    def fake_init(self, *args):
        init(*args)
        # The real base __init__ sets self.config, which the SP footprint check reads.
        self.config = SimpleNamespace()

    monkeypatch.setattr(module.vLLMReplica, "__init__", fake_init)
    monkeypatch.setattr(module.ray, "remote", lambda cls: cls)
    replica = vLLMOmniReplica(0, "rollout", "model", 2, is_teacher_model=True, name_suffix="teacher_a")
    init.assert_called_once_with(0, "rollout", "model", 2, False, True, "teacher_a")
    assert replica.server_class is module.vLLMOmniHttpServer
