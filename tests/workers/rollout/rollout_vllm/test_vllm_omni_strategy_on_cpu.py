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

import asyncio
import inspect
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import yaml

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_omni.omni_rollout_adapter import Qwen3OmniRolloutAdapter
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
from verl_omni.pipelines.rollout_request import OmniRolloutRequest
from verl_omni.workers.config import DiffusionRolloutConfig
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_ar_strategy as ar_strategy_module
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as server_module
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_diffusion_strategy as diffusion_strategy_module
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy


@pytest.mark.parametrize(
    ("output_mode", "expected_kind"),
    [("ar", "ar"), ("diffusion", "diffusion"), (None, "diffusion")],
)
def test_server_selects_one_generation_strategy(monkeypatch, output_mode, expected_kind):
    class _Strategy:
        def __init__(self, server, kind):
            self.server = server
            self.kind = kind

        def init_config(self, config):
            return SimpleNamespace(kind=self.kind, seed=config.seed)

        def init_model_config(self, model_config):
            return (self.kind, model_config)

    monkeypatch.setattr(server_module, "ARStrategy", lambda server: _Strategy(server, "ar"))
    monkeypatch.setattr(server_module, "DiffusionStrategy", lambda server: _Strategy(server, "diffusion"))

    omni_kwargs = {} if output_mode is None else {"output_mode": output_mode}
    server = object.__new__(server_module.vLLMOmniHttpServer)
    rollout_config = server._init_config(SimpleNamespace(engine_kwargs={"vllm_omni": omni_kwargs}, seed=None))
    model_config = server._init_model_config("model-config")

    assert server._generate_strategy.kind == expected_kind
    assert rollout_config.kind == expected_kind
    assert rollout_config.seed == 42
    assert model_config == (expected_kind, "model-config")


@pytest.mark.asyncio
async def test_server_generate_delegates_without_changing_rpc_arguments():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server._generate_strategy = SimpleNamespace(generate=AsyncMock(return_value="result"))
    prompt_mask = torch.tensor([True, True])

    result = await server.generate(
        prompt_ids=[1, 2],
        sampling_params={"temperature": 0.7},
        request_id="request-1",
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
        mm_processor_kwargs={"fps": 8},
        negative_prompt_ids=[3, 4],
        prompt_mask=prompt_mask,
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
        priority=2,
    )

    assert result == "result"
    server._generate_strategy.generate.assert_awaited_once_with(
        prompt_ids=[1, 2],
        sampling_params={"temperature": 0.7},
        request_id="request-1",
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
        mm_processor_kwargs={"fps": 8},
        negative_prompt_ids=[3, 4],
        prompt_mask=prompt_mask,
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
        priority=2,
    )


def test_strategies_preserve_mode_specific_quantization(monkeypatch):
    monkeypatch.setattr(
        ar_strategy_module.vLLMHttpServer,
        "_apply_quantization",
        lambda server: ("fp8", {"source": server}),
    )
    server = SimpleNamespace()

    assert ARStrategy(server).apply_quantization() == ("fp8", {"source": server})
    assert DiffusionStrategy(server).apply_quantization() == (None, {})


def test_strategies_preserve_platform_worker_extensions():
    server = SimpleNamespace()

    assert ARStrategy(server).worker_extension_cls("npu").endswith("vLLMOmniColocateWorkerExtension")
    assert DiffusionStrategy(server).worker_extension_cls("cuda").endswith("vLLMOmniColocateWorkerExtension")
    assert DiffusionStrategy(server).worker_extension_cls("npu").endswith("vLLMOmniNPUColocateWorkerExtension")


@pytest.mark.parametrize("adapter_cls", [OmniRolloutPipelineBase, Qwen3OmniRolloutAdapter])
def test_optional_rollout_hooks_preserve_existing_ar_defaults(adapter_cls):
    first, final = object(), object()

    assert adapter_cls.supports_async_chunk is True
    assert adapter_cls.weight_sync_stage_ids("full") is None
    assert adapter_cls.policy_stage_id("full") == 0
    assert adapter_cls.prepare_engine_prompt([], None, {}) is None
    assert adapter_cls.combine_engine_outputs([final], {}) == (final, {})
    with pytest.raises(NotImplementedError, match="multiple final outputs"):
        adapter_cls.combine_engine_outputs([first, final], {})
    with pytest.raises(RuntimeError, match="no outputs"):
        adapter_cls.combine_engine_outputs([], {})


def test_ar_strategy_preserves_prompt_and_sampling_preprocessing():
    processor = SimpleNamespace(dedup_pad_tokens=lambda token_ids: token_ids[:2])
    server = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=8,
            prompt_length=4,
            response_length=4,
            repetition_penalty=1.1,
        ),
        model_config=SimpleNamespace(processor=processor),
    )
    strategy = ARStrategy(server)
    sampling_params = {"max_new_tokens": 10, "logprobs": True}

    request = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=[1, 2, 3],
        image_data=["image"],
        mm_processor_kwargs={"size": 224},
    )
    prompt, params = strategy.preprocess_input(request, sampling_params, None)

    assert prompt == {
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": ["image"]},
        "mm_processor_kwargs": {"size": 224},
    }
    assert params.max_tokens == 6
    assert params.logprobs == 0
    assert params.repetition_penalty == 1.1


@pytest.mark.parametrize("output_modalities", [None, ["latent", "audio"]])
def test_ar_strategy_rejects_unsupported_prompt_fields(output_modalities):
    strategy = ARStrategy(SimpleNamespace())
    strategy._rollout_output_modalities = output_modalities
    request = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=[1],
        prompt_mask=torch.tensor([True]),
        negative_prompt_ids=[2],
        extra_prompt_ids={"encoder": [3]},
        negative_extra_prompt_ids={"encoder": [4]},
    )

    with pytest.raises(ValueError, match="ARStrategy does not support request fields") as exc_info:
        strategy.preprocess_input(request, {}, None)

    message = str(exc_info.value)
    assert "prompt_mask" in message
    assert "negative_prompt_ids" in message
    assert "extra_prompt_ids" in message
    assert "negative_extra_prompt_ids" in message


def test_ar_strategy_preserves_output_conversion():
    server = SimpleNamespace(global_steps=12)
    strategy = ARStrategy(server)
    completion = SimpleNamespace(
        token_ids=[7, 8],
        logprobs=[
            {7: SimpleNamespace(logprob=-0.25)},
            {8: SimpleNamespace(logprob=-0.5)},
        ],
        finish_reason="length",
        num_preempted=2,
    )
    final_res = SimpleNamespace(request_output=SimpleNamespace(outputs=[completion]))

    output = strategy.process_output(
        final_res,
        params=SimpleNamespace(logprobs=0),
        sampling_params={},
    )

    assert output.token_ids == [7, 8]
    assert output.log_probs == [-0.25, -0.5]
    assert output.stop_reason == "completed"
    assert output.num_preempted == 2
    assert output.extra_fields == {"global_steps": 12}


def test_ar_strategy_preserves_engine_kwarg_normalization(monkeypatch):
    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: None)
    strategy = ARStrategy(SimpleNamespace())
    engine_kwargs = {
        "output_mode": "ar",
        "custom_pipeline": "unused",
        "pipeline_name": "missing",
        "stage_init_timeout": 45,
        "stage_overrides": {"stage": {}},
    }

    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert engine_kwargs == {
        "stage-init-timeout": 45,
        "stage-overrides": {"stage": {}},
        "init-timeout": 600,
    }


@pytest.mark.parametrize(
    ("timeout_kwargs", "expected"),
    [
        ({"stage-init-timeout": 45}, {"stage-init-timeout": 45, "init-timeout": 600}),
        (
            {"stage_init_timeout": 45, "init-timeout": 90},
            {"stage-init-timeout": 45, "init-timeout": 90},
        ),
    ],
)
def test_ar_strategy_preserves_hyphenated_timeout_kwargs(monkeypatch, timeout_kwargs, expected):
    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: None)
    strategy = ARStrategy(SimpleNamespace())
    engine_kwargs = {"pipeline_name": "missing", **timeout_kwargs}

    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert engine_kwargs == expected


def test_ar_strategy_takes_weight_sync_stages_from_engine_kwargs_without_an_adapter(monkeypatch):
    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: None)
    strategy = ARStrategy(SimpleNamespace())
    engine_kwargs = {"output_mode": "ar", "weight_sync_stage_ids": ["0"]}

    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert strategy._weight_sync_stage_ids == [0]
    assert "weight_sync_stage_ids" not in engine_kwargs, "not a vLLM engine arg"
    for method in ("update_weights_from_ipc", "policy_param_checksum"):
        assert strategy.collective_rpc_stage_ids(method) == [0]
    assert strategy.collective_rpc_stage_ids("sleep") is None


def test_ar_strategy_rejects_weight_sync_stages_that_an_adapter_already_declares(monkeypatch):
    class Adapter:
        pass

    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: Adapter)
    strategy = ARStrategy(SimpleNamespace(_rollout_flags={}))
    with pytest.raises(ValueError, match="declared by Adapter"):
        strategy.preprocess_engine_kwargs({"pipeline_name": "adapter", "weight_sync_stage_ids": [0]})


def test_ar_strategy_honors_adapter_chunking_and_weight_sync_contracts(monkeypatch):
    class Adapter:
        supports_async_chunk = False

        @staticmethod
        def rollout_flags(pipeline_mode):
            return {0: {"mode": pipeline_mode}}

        @staticmethod
        def weight_sync_stage_ids(pipeline_mode):
            assert pipeline_mode == "full"
            return [1]

        @staticmethod
        def get_engine_hf_overrides(pipeline_mode):
            assert pipeline_mode == "full"
            return {}

    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: Adapter)
    strategy = ARStrategy(SimpleNamespace(_rollout_flags={}))

    def write_deploy_config(*args):
        strategy._weight_sync_stage_ids = Adapter.weight_sync_stage_ids("full")

    monkeypatch.setattr(strategy, "_write_deploy_config", write_deploy_config)

    with pytest.raises(ValueError, match="requires async_chunk=false"):
        strategy.preprocess_engine_kwargs({"pipeline_name": "adapter", "pipeline_mode": "full"})

    engine_kwargs = {"pipeline_name": "adapter", "pipeline_mode": "full", "async_chunk": False}
    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert strategy._rollout_adapter is Adapter
    assert strategy._weight_sync_stage_ids == [1]
    assert strategy.server._rollout_flags == {0: {"mode": "full"}}
    assert engine_kwargs == {"async-chunk": False}


def test_ar_strategy_resolves_nonzero_policy_and_weight_sync_stages(monkeypatch):
    stages = [
        SimpleNamespace(stage_id=0, final_output=False, final_output_type=None, sampling_constraints={}),
        SimpleNamespace(stage_id=1, final_output=True, final_output_type="latent", sampling_constraints={}),
    ]

    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="thinker_only"):
            return stages

        @classmethod
        def get_pipeline_id(cls, pipeline_mode="thinker_only"):
            return "test_pipeline"

        @classmethod
        def policy_stage_id(cls, pipeline_mode="thinker_only"):
            return 1

        @classmethod
        def weight_sync_stage_ids(cls, pipeline_mode="thinker_only"):
            return [1]

    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: Adapter)
    monkeypatch.setattr(ar_strategy_module, "get_visible_devices_keyword", lambda: "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    server = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=1,
            text_encoder_tp_size=1,
            max_model_len=64,
            max_num_batched_tokens=64,
        ),
        _rollout_flags={},
    )
    strategy = ARStrategy(server)
    engine_kwargs = {"pipeline_name": "adapter"}

    strategy.preprocess_engine_kwargs(engine_kwargs)

    assert strategy._policy_stage_index == 1
    assert strategy._policy_sampling_constraints == {}
    assert strategy._weight_sync_stage_ids == [1]
    server._temp_deploy_ctx.cleanup()


def test_ar_strategy_moves_capacity_overrides_to_each_stage(monkeypatch):
    stages = [
        SimpleNamespace(stage_id=0, final_output=False, final_output_type=None, sampling_constraints={}),
        SimpleNamespace(stage_id=1, final_output=True, final_output_type="latent", sampling_constraints={}),
    ]

    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="thinker_only"):
            return stages

        @classmethod
        def get_pipeline_id(cls, pipeline_mode="thinker_only"):
            return "test_pipeline"

        @classmethod
        def get_stage_engine_extras(cls, stage_id, pipeline_mode="thinker_only"):
            return {"max_model_len": 65536} if stage_id == 1 else {}

    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: Adapter)
    monkeypatch.setattr(ar_strategy_module, "get_visible_devices_keyword", lambda: "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    server = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=1,
            text_encoder_tp_size=1,
            max_model_len=4096,
            max_num_batched_tokens=8192,
        ),
        _rollout_flags={},
    )
    strategy = ARStrategy(server)
    engine_kwargs = {"pipeline_name": "adapter"}

    strategy.preprocess_engine_kwargs(engine_kwargs)

    deploy = yaml.safe_load(Path(engine_kwargs["deploy-config"]).read_text(encoding="utf-8"))
    assert engine_kwargs["max_model_len"] is None
    assert engine_kwargs["max_num_batched_tokens"] is None
    assert deploy["stages"][0]["engine_extras"] == {
        "max_model_len": 4096,
        "max_num_batched_tokens": 8192,
    }
    assert deploy["stages"][1]["engine_extras"] == {
        "max_model_len": 65536,
        "max_num_batched_tokens": 8192,
    }
    server._temp_deploy_ctx.cleanup()


@pytest.mark.parametrize(
    ("policy_stage_id", "weight_sync_stage_ids", "message"),
    [
        (2, [1], "unknown stage 2"),
        (0, [2], "unknown stages"),
    ],
)
def test_ar_strategy_rejects_unknown_adapter_stages(monkeypatch, policy_stage_id, weight_sync_stage_ids, message):
    class Adapter(OmniRolloutPipelineBase):
        @classmethod
        def build_stage_configs(cls, pipeline_mode="thinker_only"):
            return [
                SimpleNamespace(stage_id=0, final_output=False, final_output_type=None, sampling_constraints={}),
                SimpleNamespace(stage_id=1, final_output=True, final_output_type="latent", sampling_constraints={}),
            ]

        @classmethod
        def policy_stage_id(cls, pipeline_mode="thinker_only"):
            return policy_stage_id

        @classmethod
        def weight_sync_stage_ids(cls, pipeline_mode="thinker_only"):
            return weight_sync_stage_ids

    monkeypatch.setattr(ar_strategy_module.OmniRolloutPipelineBase, "get_class", lambda pipeline_name: Adapter)
    strategy = ARStrategy(SimpleNamespace(_rollout_flags={}))

    with pytest.raises(ValueError, match=message):
        strategy.preprocess_engine_kwargs({"pipeline_name": "adapter"})


def test_ar_strategy_preserves_engine_argument_normalization():
    server = SimpleNamespace(config=SimpleNamespace(logprobs_mode="raw_logprobs"))
    strategy = ARStrategy(server)
    engine_args = {
        "compilation_config": {
            "keep": 1,
            "drop": None,
            "nested": {"keep": 2, "drop": None},
        }
    }

    strategy.prepare_engine_args(
        engine_args,
        Namespace(stage_init_timeout="45", init_timeout=None),
    )

    assert engine_args == {
        "stage_init_timeout": 45,
        "logprobs_mode": "raw_logprobs",
        "compilation_config": {"keep": 1, "nested": {"keep": 2}},
    }


def test_ar_strategy_prepares_sampling_params_for_nonzero_policy_stage():
    class Adapter:
        @staticmethod
        def prepare_engine_prompt(**kwargs):
            return {
                "prompt_token_ids": [1, 1, 1, 1],
                "additional_information": {"text": ["hello"]},
            }

    server = SimpleNamespace(
        model_config=SimpleNamespace(),
        global_steps=0,
        config=SimpleNamespace(
            max_model_len=64,
            prompt_length=16,
            response_length=8,
            repetition_penalty=1.0,
        ),
        engine=SimpleNamespace(
            default_sampling_params_list=[SimpleNamespace(stage="thinker"), ar_strategy_module.SamplingParams()]
        ),
    )
    strategy = ARStrategy(server)
    strategy._rollout_adapter = Adapter
    strategy._rollout_output_modalities = ["latent", "audio"]
    strategy._policy_stage_index = 1
    strategy._policy_sampling_constraints = {}

    request = OmniRolloutRequest.from_generate_kwargs(prompt_ids=[5, 6])
    prompt, params = strategy.preprocess_input(
        request,
        {"temperature": 0.8, "logprobs": True},
        None,
    )

    assert prompt["additional_information"]["max_new_tokens"] == [8]
    assert len(params) == 2
    assert params[0].stage == "thinker"
    assert params[1].max_tokens == 8
    assert params[1].temperature == pytest.approx(0.8)
    assert params[1].logprobs == 0

    _, next_params = strategy.preprocess_input(
        request,
        {"temperature": 0.2, "logprobs": True},
        None,
    )
    assert params[0] is next_params[0]
    assert params[1] is not next_params[1]
    assert params[1].temperature == pytest.approx(0.8)
    assert next_params[1].temperature == pytest.approx(0.2)

    completion = SimpleNamespace(
        token_ids=[7],
        logprobs=[{7: SimpleNamespace(logprob=-0.25)}],
        finish_reason="stop",
        num_preempted=0,
    )
    final_res = SimpleNamespace(
        request_id="request-0",
        request_output=SimpleNamespace(outputs=[completion]),
    )
    strategy._rollout_fields_by_request_id["request-0"] = {}
    assert strategy.process_output(final_res, params, {}).log_probs == [-0.25]
    assert strategy._rollout_fields_by_request_id == {}


@pytest.mark.parametrize(
    ("adapter_prompt", "message"),
    [
        ({"additional_information": {"text": ["hello"]}}, "must contain prompt_token_ids"),
        ([1, 2], "must return a dict or None"),
    ],
)
def test_ar_strategy_rejects_invalid_adapter_prompt(adapter_prompt, message):
    class Adapter:
        @staticmethod
        def prepare_engine_prompt(**kwargs):
            return adapter_prompt

    server = SimpleNamespace(
        model_config=SimpleNamespace(),
        config=SimpleNamespace(max_model_len=64, prompt_length=16, response_length=8),
    )
    strategy = ARStrategy(server)
    strategy._rollout_adapter = Adapter

    with pytest.raises((RuntimeError, TypeError), match=message):
        strategy.preprocess_input(OmniRolloutRequest.from_generate_kwargs(prompt_ids=[5, 6]), {}, None)


@pytest.mark.asyncio
async def test_ar_strategy_retains_requested_stage_outputs_and_targets_weight_sync():
    completion = SimpleNamespace(token_ids=[7], logprobs=None, finish_reason="stop", num_preempted=0)
    policy = SimpleNamespace(request_id="request-0", outputs=[completion])

    class Engine:
        def __init__(self):
            self.generate_kwargs = None
            self.rpc_kwargs = None
            self.num_stages = 1

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield policy

        async def collective_rpc(self, **kwargs):
            self.rpc_kwargs = kwargs
            return [None]

    class Adapter(OmniRolloutPipelineBase):
        @staticmethod
        def combine_engine_outputs(outputs, prompt):
            assert outputs == [policy]
            return policy, {"audio_sample_rate": 24_000}

    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = Engine()
    server.global_steps = 3
    server.replica_rank = 0
    strategy = ARStrategy(server)
    strategy._rollout_output_modalities = ["latent", "audio"]
    strategy._rollout_adapter = Adapter
    strategy._weight_sync_stage_ids = [0]
    server._generate_strategy = strategy

    result = await strategy.run_generation(
        {"prompt_token_ids": [1]}, ar_strategy_module.SamplingParams(), "request-0", None, 0
    )
    rpc_result = await server.collective_rpc("update_weights_from_ipc", kwargs={"base_sync_done": True})

    assert result is policy
    assert not hasattr(result, "_verl_omni_rollout_fields")
    assert strategy._rollout_fields_by_request_id == {"request-0": {"audio_sample_rate": 24_000}}
    assert server.engine.generate_kwargs["output_modalities"] == ["latent", "audio"]
    assert server.engine.rpc_kwargs["stage_ids"] == [0]
    assert rpc_result == [None]

    output = strategy.process_output(result, ar_strategy_module.SamplingParams(), {})
    assert output.extra_fields == {"global_steps": 3, "audio_sample_rate": 24_000}
    assert strategy._rollout_fields_by_request_id == {}

    server.config = SimpleNamespace(max_model_len=8, prompt_length=4, response_length=4)
    server.model_config = SimpleNamespace(processor=None)
    server._resolve_lora_request = AsyncMock(return_value=None)
    server.engine.default_sampling_params_list = [ar_strategy_module.SamplingParams(), SimpleNamespace(stage="audio")]
    output = await strategy.generate(
        prompt_ids=[1],
        sampling_params={"max_new_tokens": 2},
        request_id="request-0",
        image_data=["image"],
        audio_data=["audio"],
        mm_processor_kwargs={"size": 224},
    )

    assert server.engine.generate_kwargs["prompt"] == {
        "prompt_token_ids": [1],
        "multi_modal_data": {"image": ["image"], "audio": ["audio"]},
        "mm_processor_kwargs": {"size": 224},
    }
    params = server.engine.generate_kwargs["sampling_params_list"]
    assert params[0].max_tokens == 2
    assert params[1] is server.engine.default_sampling_params_list[1]
    assert server.engine.generate_kwargs["output_modalities"] == ["latent", "audio"]
    assert output.token_ids == [7]
    assert output.extra_fields == {"global_steps": 3, "audio_sample_rate": 24_000}
    assert strategy._rollout_fields_by_request_id == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        [{"supported": False, "error": "unsupported LoRA weight"}],
        [[None], [{"error": "second replica failed"}]],
        [{"todo": True}],
        [{"supported": False, "error": ""}],
        None,
        [],
        [[]],
    ],
)
async def test_weight_sync_rejects_best_effort_rpc_failures(reply):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=1, collective_rpc=AsyncMock(return_value=reply))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0])
    with pytest.raises(RuntimeError, match="update_weights_from_ipc failed"):
        await server.collective_rpc("update_weights_from_ipc", kwargs={"base_sync_done": True}, require_ack=True)
    server.engine.collective_rpc.assert_awaited_once_with(
        method="update_weights_from_ipc", timeout=None, args=(), kwargs={"base_sync_done": True}, stage_ids=[0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [[None], [[True], [None]], [{"supported": True}], [(None,)]])
async def test_weight_sync_accepts_successful_rpc_results(reply):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=1, collective_rpc=AsyncMock(return_value=reply))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: None)
    assert await server.collective_rpc("update_weights_from_ipc", require_ack=True) == list(reply)
    assert server.engine.collective_rpc.await_args.kwargs["stage_ids"] == [0]


@pytest.mark.asyncio
async def test_checked_collective_rpc_is_method_independent_and_preserves_reply_order():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    first = [None, {"supported": True}]
    second = [("second",), None]
    server.engine = SimpleNamespace(num_stages=2, collective_rpc=AsyncMock(side_effect=[first, second]))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [1, 1, 0])

    assert await server.collective_rpc("refresh_metadata", require_ack=True) == first + second
    assert [call.kwargs["stage_ids"] for call in server.engine.collective_rpc.await_args_list] == [[1], [0]]


@pytest.mark.asyncio
async def test_checked_collective_rpc_rejects_other_method_error_envelope():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=1, collective_rpc=AsyncMock(return_value=[{"todo": True}]))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0])

    with pytest.raises(RuntimeError, match="refresh_metadata failed"):
        await server.collective_rpc("refresh_metadata", require_ack=True)


@pytest.mark.asyncio
async def test_unchecked_collective_rpc_returns_raw_reply_with_unknown_topology():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    reply = [{"supported": False, "error": "best effort"}]
    server.engine = SimpleNamespace(collective_rpc=AsyncMock(return_value=reply))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: None)

    assert await server.collective_rpc("update_weights_from_ipc") is reply
    server.engine.collective_rpc.assert_awaited_once_with(
        method="update_weights_from_ipc", timeout=None, args=(), kwargs=None, stage_ids=None
    )


@pytest.mark.asyncio
async def test_omni_adapter_opts_ipc_into_ack_and_preserves_transport_flags():
    adapter = object.__new__(server_module.vLLMOmniServerAdapter)
    adapter._ensure_server_handle = MagicMock(return_value=True)
    pending = asyncio.get_running_loop().create_future()
    remote = MagicMock(return_value=pending)
    adapter.server_handle = SimpleNamespace(collective_rpc=SimpleNamespace(remote=remote))

    result = await adapter._execute_method(
        "update_weights_from_ipc", non_block=True, timeout=3, args=("weights",), kwargs={"ready": True}
    )
    assert result is pending
    remote.assert_called_once_with(
        "update_weights_from_ipc", timeout=3, args=("weights",), kwargs={"ready": True}, require_ack=True
    )
    pending.set_result([None])
    assert await result == [None]

    completed = asyncio.get_running_loop().create_future()
    completed.set_result(["ack"])
    remote.return_value = completed
    assert await adapter._execute_method("update_weights_from_ipc") == ["ack"]

    adapter._ensure_server_handle.return_value = False
    remote.reset_mock()
    assert await adapter._execute_method("update_weights_from_ipc", non_block=True) is None
    remote.assert_not_called()

    with patch.object(server_module.ServerAdapter, "_execute_method", new_callable=AsyncMock) as inherited:
        inherited.return_value = "base"
        assert await adapter._execute_method("optional_control", non_block=True, timeout=2) == "base"
        inherited.assert_awaited_once_with("optional_control", non_block=True, timeout=2, args=(), kwargs=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("receiver failed"), asyncio.CancelledError()])
async def test_omni_adapter_propagates_ipc_failure_and_cancellation(error):
    adapter = object.__new__(server_module.vLLMOmniServerAdapter)
    adapter._ensure_server_handle = MagicMock(return_value=True)
    pending = asyncio.get_running_loop().create_future()
    pending.set_exception(error)
    adapter.server_handle = SimpleNamespace(collective_rpc=SimpleNamespace(remote=MagicMock(return_value=pending)))

    with pytest.raises(type(error)) as raised:
        await adapter._execute_method("update_weights_from_ipc")
    assert raised.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", [[None], [{"error": "receiver failed"}]])
async def test_inherited_full_weight_update_requires_ack_before_publishing(reply):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=1, collective_rpc=AsyncMock(return_value=reply))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0])

    adapter = object.__new__(server_module.vLLMOmniServerAdapter)
    adapter._ensure_server_handle = MagicMock(return_value=True)
    adapter.zmq_handle = "ipc:///tmp/test-full-weight-sync.sock"
    adapter.use_shm = False
    adapter._has_server = True
    adapter.replica_rank = 0
    adapter.rollout_rank = 0
    adapter.config = SimpleNamespace(checkpoint_engine=SimpleNamespace(update_weights_bucket_megabytes=16))
    rpc_calls = []

    def remote(method, **kwargs):
        rpc_calls.append((method, kwargs))
        return asyncio.create_task(server.collective_rpc(method, **kwargs))

    clear_cache = AsyncMock()
    set_steps = AsyncMock()
    adapter.server_handle = SimpleNamespace(
        collective_rpc=SimpleNamespace(remote=remote),
        clear_kv_cache=SimpleNamespace(remote=clear_cache),
        set_global_steps=SimpleNamespace(remote=set_steps),
    )
    sender = SimpleNamespace(async_send_weights=AsyncMock())
    upstream_module = inspect.getmodule(server_module.ServerAdapter)
    with patch.object(upstream_module, "BucketedWeightSender", return_value=sender):
        if reply == [None]:
            await adapter.update_weights([("w", torch.zeros(1))], global_steps=7)
        else:
            with pytest.raises(RuntimeError, match="update_weights_from_ipc failed"):
                await adapter.update_weights([("w", torch.zeros(1))], global_steps=7)

    sender.async_send_weights.assert_awaited_once()
    assert len(rpc_calls) == 1
    assert rpc_calls[0][0] == "update_weights_from_ipc"
    assert rpc_calls[0][1]["require_ack"] is True
    if reply == [None]:
        clear_cache.assert_awaited_once()
        set_steps.assert_awaited_once_with(7)
    else:
        clear_cache.assert_not_awaited()
        set_steps.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("worker failed"), TimeoutError("control timed out")])
async def test_weight_sync_preserves_raised_error_and_other_controls(error):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=1, collective_rpc=AsyncMock(side_effect=error))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: None)
    with pytest.raises(type(error)) as raised:
        await server.collective_rpc("update_weights_from_ipc", require_ack=True)
    assert raised.value is error
    server.engine.collective_rpc = AsyncMock(return_value=[{"supported": False, "error": "optional control"}])
    assert await server.collective_rpc("optional_control") == [{"supported": False, "error": "optional control"}]
    server.engine.collective_rpc.assert_awaited_once_with(
        method="optional_control", timeout=None, args=(), kwargs=None, stage_ids=None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [[0, 1], None])
async def test_weight_sync_rejects_missing_second_stage_despite_two_first_stage_replies(selected):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=2, collective_rpc=AsyncMock(side_effect=[[None, None], []]))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: selected)
    with pytest.raises(RuntimeError, match="missing replica results for stage 1"):
        await server.collective_rpc("update_weights_from_ipc", require_ack=True)
    assert [call.kwargs["stage_ids"] for call in server.engine.collective_rpc.await_args_list] == [[0], [1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [[0, 1], None])
async def test_weight_sync_rejects_missing_stage_from_pinned_orchestrator(selected):
    from vllm_omni.engine.messages import CollectiveRPCRequestMessage
    from vllm_omni.engine.orchestrator import Orchestrator
    from vllm_omni.entrypoints import AsyncOmni

    pools = [
        SimpleNamespace(
            stage_id=stage_id,
            stage_type="diffusion",
            live_replica_ids=lambda replica_ids=replica_ids: replica_ids,
            collective_rpc=AsyncMock(return_value=None),
        )
        for stage_id, replica_ids in enumerate(([0, 1], []))
    ]
    orchestrator = object.__new__(Orchestrator)
    orchestrator.num_stages = len(pools)
    orchestrator.stage_pools = pools
    orchestrator.rpc_async_queue = asyncio.Queue()
    producer_messages = []
    requested_stages = []

    async def dispatch(**kwargs):
        requested_stages.append(kwargs["stage_ids"])
        request = CollectiveRPCRequestMessage(rpc_id="weight-sync", **kwargs)
        await orchestrator._handle_collective_rpc(request)
        result = await orchestrator.rpc_async_queue.get()
        producer_messages.append(result)
        # The pinned engine client discards result.stage_ids at this boundary.
        return result.results

    omni = SimpleNamespace(engine=SimpleNamespace(collective_rpc_async=dispatch))

    async def collective_rpc(**kwargs):
        return await AsyncOmni.collective_rpc(omni, **kwargs)

    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=2, collective_rpc=collective_rpc)
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: selected)

    with pytest.raises(RuntimeError, match="missing replica results for stage 1"):
        await server.collective_rpc("update_weights_from_ipc", require_ack=True)
    assert requested_stages == [[0], [1]]
    assert [message.stage_ids for message in producer_messages] == [[0, 0], []]
    assert [message.results for message in producer_messages] == [[None, None], []]
    assert pools[0].collective_rpc.await_count == 2
    pools[1].collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
async def test_weight_sync_dispatches_each_selected_stage_once():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=2, collective_rpc=AsyncMock(side_effect=[[None, None], [None]]))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0, 0, 1])

    assert await server.collective_rpc("update_weights_from_ipc", require_ack=True) == [None, None, None]
    assert [call.kwargs["stage_ids"] for call in server.engine.collective_rpc.await_args_list] == [[0], [1]]


@pytest.mark.asyncio
async def test_weight_sync_stops_before_next_stage_on_nested_worker_error():
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(
        num_stages=2,
        collective_rpc=AsyncMock(side_effect=[[[None], [{"error": "receiver failed"}]], [None]]),
    )
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0, 1])

    with pytest.raises(RuntimeError, match="receiver failed"):
        await server.collective_rpc("update_weights_from_ipc", require_ack=True)
    server.engine.collective_rpc.assert_awaited_once()
    assert server.engine.collective_rpc.await_args.kwargs["stage_ids"] == [0]


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [[0, 1], None])
async def test_weight_sync_checks_each_stage_with_unequal_replica_counts(selected):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=2, collective_rpc=AsyncMock(side_effect=[[None, None], [None]]))
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: selected)

    assert await server.collective_rpc(
        "update_weights_from_ipc", timeout=3, args=("weights",), kwargs={"ready": True}, require_ack=True
    ) == [None, None, None]
    assert [call.kwargs for call in server.engine.collective_rpc.await_args_list] == [
        {
            "method": "update_weights_from_ipc",
            "timeout": 3,
            "args": ("weights",),
            "kwargs": {"ready": True},
            "stage_ids": [stage_id],
        }
        for stage_id in (0, 1)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("num_stages", "selected"), [(2, []), (0, None), (None, [0]), (2, [2])])
async def test_weight_sync_rejects_empty_or_unknown_stages_before_dispatch(num_stages, selected):
    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=num_stages, collective_rpc=AsyncMock())
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: selected)

    with pytest.raises(RuntimeError, match="update_weights_from_ipc failed"):
        await server.collective_rpc("update_weights_from_ipc", require_ack=True)
    server.engine.collective_rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("receiver failed"), TimeoutError("receiver timed out")])
async def test_weight_sync_rejects_pinned_stage_pool_error_envelope(error):
    from vllm_omni.engine.stage_pool import StagePool
    from vllm_omni.entrypoints import AsyncOmni

    # Exercise the pinned producer's exception-to-data conversion, not a
    # fabricated error dictionary. No model or transport process is needed.
    pool = object.__new__(StagePool)
    pool.stage_id = 0
    pool.clients = [SimpleNamespace(stage_type="diffusion", collective_rpc_async=AsyncMock(side_effect=error))]

    async def dispatch(**kwargs):
        kwargs.pop("stage_ids")
        return [await pool.collective_rpc(replica_id=0, **kwargs)]

    omni = SimpleNamespace(engine=SimpleNamespace(collective_rpc_async=dispatch))

    async def collective_rpc(**kwargs):
        return await AsyncOmni.collective_rpc(omni, **kwargs)

    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = SimpleNamespace(num_stages=1, collective_rpc=collective_rpc)
    server._generate_strategy = SimpleNamespace(collective_rpc_stage_ids=lambda method: [0])
    with pytest.raises(RuntimeError, match="timed out" if isinstance(error, TimeoutError) else str(error)):
        await server.collective_rpc("update_weights_from_ipc", require_ack=True)


def test_ar_strategy_preserves_qwen3_omni_thinker_only_contract():
    server = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=8,
            prompt_length=4,
            response_length=4,
            repetition_penalty=1.0,
            logprobs_mode="processed_logprobs",
        ),
        model_config=SimpleNamespace(processor=None),
        global_steps=12,
    )
    strategy = ARStrategy(server)
    strategy._rollout_adapter = Qwen3OmniRolloutAdapter
    strategy._rollout_output_modalities = None

    engine_args = {"model_stage": "thinker"}
    strategy.prepare_engine_args(engine_args, Namespace(stage_init_timeout=None, init_timeout=None))
    assert engine_args["model_stage"] == "thinker"

    prompt, params = strategy.preprocess_input(
        request=OmniRolloutRequest.from_generate_kwargs(prompt_ids=[1, 2]),
        sampling_params={"max_new_tokens": 2, "logprobs": True},
        lora_request=None,
    )
    assert prompt == {"prompt_token_ids": [1, 2]}
    assert isinstance(params, ar_strategy_module.SamplingParams)

    completion = SimpleNamespace(
        token_ids=[7],
        logprobs=[{7: SimpleNamespace(logprob=-0.25)}],
        finish_reason="stop",
        num_preempted=0,
    )
    output = strategy.process_output(
        SimpleNamespace(request_output=SimpleNamespace(outputs=[completion])),
        params=params,
        sampling_params={},
    )
    assert output.extra_fields == {"global_steps": 12}


def test_ar_strategy_writes_qwen3_omni_thinker_only_deploy_config(monkeypatch):
    monkeypatch.setattr(ar_strategy_module, "get_visible_devices_keyword", lambda: "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    server = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=1,
            text_encoder_tp_size=1,
            max_model_len=8,
            max_num_batched_tokens=8,
        ),
        _rollout_flags={},
    )
    strategy = ARStrategy(server)
    engine_kwargs = {
        "pipeline_name": "qwen3_omni_moe",
        "pipeline_mode": "thinker_only",
    }

    strategy.preprocess_engine_kwargs(engine_kwargs)

    deploy_path = engine_kwargs["deploy-config"]
    deploy = yaml.safe_load(Path(deploy_path).read_text(encoding="utf-8"))
    assert deploy["pipeline"] == Qwen3OmniRolloutAdapter.get_pipeline_id("thinker_only")
    assert [stage["stage_id"] for stage in deploy["stages"]] == [0]
    assert strategy._rollout_output_modalities is None
    server._temp_deploy_ctx.cleanup()


def test_ar_strategy_preserves_qwen3_omni_full_without_combiner(monkeypatch):
    monkeypatch.setattr(ar_strategy_module, "get_visible_devices_keyword", lambda: "CUDA_VISIBLE_DEVICES")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    server = SimpleNamespace(
        config=SimpleNamespace(
            tensor_model_parallel_size=1,
            text_encoder_tp_size=1,
            max_model_len=8,
            max_num_batched_tokens=8,
        ),
        _rollout_flags={},
    )
    strategy = ARStrategy(server)

    engine_kwargs = {
        "pipeline_name": "qwen3_omni_moe",
        "pipeline_mode": "full",
    }
    strategy.preprocess_engine_kwargs(engine_kwargs)

    deploy_path = engine_kwargs["deploy-config"]
    deploy = yaml.safe_load(Path(deploy_path).read_text(encoding="utf-8"))
    assert [stage["stage_id"] for stage in deploy["stages"]] == [0, 1, 2]
    assert strategy._rollout_output_modalities is None
    server._temp_deploy_ctx.cleanup()


def test_diffusion_strategy_preserves_engine_argument_preparation(monkeypatch):
    imported = []
    monkeypatch.setattr(diffusion_strategy_module, "import_external_libs", imported.append)
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_pipeline_path",
        staticmethod(lambda **kwargs: "package.Adapter"),
    )
    pipeline_cls = SimpleNamespace(__name__="Adapter", supports_request_batch=False)
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_class",
        staticmethod(lambda **kwargs: pipeline_cls),
    )
    server = SimpleNamespace(
        config=DiffusionRolloutConfig(
            external_lib=["extension"],
            tensor_model_parallel_size=4,
            text_encoder_tp_size=1,
            step_execution=False,
            enable_prompt_embed_cache=True,
            prompt_embed_cache_size=16,
        ),
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    strategy = DiffusionStrategy(server)
    engine_args = {
        "max_num_seqs": 4,
        # vLLM LLM-only fields must not reach diffusion ingress (upstream
        # rejects unowned fields); diffusion-owned keys survive.
        "block_size": 16,
        "enable_lora": True,
        "max_loras": 2,
        "dtype": "bfloat16",
    }

    strategy.prepare_engine_args(engine_args, Namespace())

    assert imported == [["extension"]]
    assert engine_args == {
        "max_num_seqs": 1,
        # text_encoder_tp_size is diffusion-owned so it survives the ingress
        # allowlist; enable_dummy_pipeline is rejected upstream and dropped.
        "tensor_parallel_size": 4,
        "ulysses_degree": 1,
        "ring_degree": 1,
        "sequence_parallel_size": 1,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "vae_patch_parallel_size": 1,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": False,
        "text_encoder_tp_size": 1,
        "custom_pipeline_args": {"pipeline_class": "package.Adapter"},
        "enable_prompt_embed_cache": True,
        "prompt_embed_cache_size": 16,
        "dtype": "bfloat16",
    }


@pytest.mark.parametrize("num_stages", [1, 2])
@pytest.mark.parametrize("first_stage_type", ["llm", "diffusion"])
def test_diffusion_strategy_emits_canonical_prompt(num_stages, first_stage_type):
    defaults = [object() for _ in range(num_stages)]
    server = SimpleNamespace(
        engine=SimpleNamespace(
            default_sampling_params_list=defaults,
            engine=SimpleNamespace(get_stage_metadata=lambda stage_id: SimpleNamespace(stage_type=first_stage_type)),
        )
    )
    strategy = DiffusionStrategy(server)
    prompt_mask = torch.tensor([True, False])

    request = OmniRolloutRequest.from_generate_kwargs(
        prompt_ids=[1, 2],
        prompt_mask=prompt_mask,
        negative_prompt_ids=[3, 4],
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
        mm_processor_kwargs={"video_fps": 24, "audio_sample_rate": 32_000},
        image_data=["image"],
    )
    prompt, params = strategy.preprocess_input(request, {"pipeline_private_arg": 7}, None)

    ar_entrance = first_stage_type != "diffusion"
    key = "prompt_token_ids" if ar_entrance else "prompt_ids"
    assert prompt[key] == [1, 2]
    assert ("prompt_ids" if ar_entrance else "prompt_token_ids") not in prompt
    assert prompt["prompt_mask"] is prompt_mask
    if ar_entrance:
        assert prompt["modalities"] == ["image"]
    else:
        assert "modalities" not in prompt
    assert params[:-1] == defaults[:-1]
    assert prompt["negative_prompt_ids"] == [3, 4]
    assert "extra_prompt_ids" not in prompt
    assert "negative_extra_prompt_ids" not in prompt
    assert prompt["multi_modal_data"] == {"image": ["image"]}
    assert prompt["extra_args"] == {
        "extra_prompt_ids": {"encoder": [5]},
        "negative_extra_prompt_ids": {"encoder": [6]},
    }
    assert prompt["mm_processor_kwargs"] == {"video_fps": 24, "audio_sample_rate": 32_000}
    assert params[-1].extra_args == {"pipeline_private_arg": 7}


@pytest.mark.asyncio
async def test_diffusion_strategy_rejects_nonzero_priority():
    strategy = DiffusionStrategy(SimpleNamespace())

    with pytest.raises(ValueError, match="does not support nonzero request priority"):
        await strategy.run_generation(None, None, "request-id", None, priority=1)


def _joint_spec(sample_rate=48000):
    return DiffusionIOSpec(
        artifacts={
            "video_preview": MediaSpec("video", "decoded", "TCHW", fps=24),
            "audio": MediaSpec("audio", "decoded", "CT", sample_rate=sample_rate),
        }
    )


def _joint_video_audio_final_res(sample_rate=48000):
    from dataclasses import asdict

    from vllm_omni.outputs import OmniRequestOutput

    video = torch.zeros(2, 3, 8, 8, dtype=torch.uint8)
    audio = torch.zeros(1, 16)
    final_res = OmniRequestOutput.from_diffusion(
        request_id="joint-output",
        images=[{"video_preview": video, "audio": audio}],
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        multimodal_output={
            "metadata": {
                "media_artifacts": {
                    "primary": "video_preview",
                    "preview": "video_preview",
                    "audio": "audio",
                    "specs": {name: asdict(spec) for name, spec in _joint_spec(sample_rate).artifacts.items()},
                }
            }
        },
    )
    return final_res, audio


def test_diffusion_strategy_uses_declared_audio_sample_rate(monkeypatch):
    pipeline_cls = SimpleNamespace(diffusion_io_spec=_joint_spec())
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_class",
        staticmethod(lambda **kwargs: pipeline_cls),
    )
    server = SimpleNamespace(
        global_steps=1,
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    final_res, audio = _joint_video_audio_final_res()

    processed = DiffusionStrategy(server).process_output(final_res, None, {"output_type": "pt", "logprobs": False})

    # The audio sample rate comes from the adapter-declared DiffusionIOSpec,
    # not the strategy's legacy 32 kHz fallback.
    assert processed.extra_fields["audio_sample_rate"] == 48000
    # Audio is already a declared per-sample CT artifact, not a positional batch.
    torch.testing.assert_close(processed.extra_fields["audio"], audio)
    # The declared primary modality rides to downstream consumers so they read
    # the media kind instead of inferring it from the tensor rank.
    assert processed.extra_fields["media_kind"] == "video"


@pytest.mark.parametrize("runtime_kind", ["image", "depth"])
def test_diffusion_strategy_rejects_conflicting_media_kind(monkeypatch, runtime_kind):
    strategy = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None))
    monkeypatch.setattr(
        strategy,
        "_diffusion_io_spec",
        _joint_spec,
    )
    final_res, _ = _joint_video_audio_final_res()
    final_res.request_id = "request-conflict"
    final_res.multimodal_output["metadata"]["rl"] = {"media_kind": runtime_kind}

    with pytest.raises(ValueError, match="request-conflict.*media_kind"):
        strategy.process_output(final_res, None, {"output_type": "pt"})


def test_diffusion_strategy_accepts_matching_kind_and_runtime_audio_rate(monkeypatch):
    strategy = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None))
    monkeypatch.setattr(
        strategy,
        "_diffusion_io_spec",
        lambda: _joint_spec(sample_rate=None),
    )
    final_res, _ = _joint_video_audio_final_res(sample_rate=24000)
    final_res.multimodal_output["metadata"]["rl"] = {"media_kind": "video", "audio_sample_rate": 24000}

    result = strategy.process_output(final_res, None, {"output_type": "pt"})
    assert result.extra_fields["media_kind"] == "video"
    assert result.extra_fields["audio_sample_rate"] == 24000


@pytest.mark.parametrize("declared", [False, True])
def test_diffusion_strategy_rejects_extra_tuple_streams(monkeypatch, declared):
    strategy = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None))
    spec = _joint_spec() if declared else None
    monkeypatch.setattr(strategy, "_diffusion_io_spec", lambda: spec)
    final_res, audio = _joint_video_audio_final_res()
    final_res.images = [(final_res.images[0]["video_preview"], audio, audio.clone())]
    final_res.multimodal_output.clear()
    with pytest.raises(ValueError, match="named media_artifacts declaration required"):
        strategy.process_output(final_res, None, {"output_type": "pt"})


def test_diffusion_strategy_rejects_undeclared_named_audio(monkeypatch):
    strategy = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None))
    monkeypatch.setattr(
        strategy,
        "_diffusion_io_spec",
        lambda: DiffusionIOSpec(
            artifacts={
                "video_preview": MediaSpec("video", "decoded", "TCHW"),
            }
        ),
    )
    final_res, _ = _joint_video_audio_final_res()
    with pytest.raises(ValueError, match="undeclared artifact='audio'"):
        strategy.process_output(final_res, None, {"output_type": "pt"})


def test_diffusion_strategy_uses_runtime_header_without_static_spec(monkeypatch):
    # The actual artifact carries its rate; the strategy supplies no default.
    monkeypatch.setattr(
        diffusion_strategy_module.VllmOmniPipelineBase,
        "get_class",
        staticmethod(lambda **kwargs: SimpleNamespace()),
    )
    server = SimpleNamespace(
        global_steps=1,
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    final_res, audio = _joint_video_audio_final_res()

    processed = DiffusionStrategy(server).process_output(final_res, None, {"output_type": "pt", "logprobs": False})

    torch.testing.assert_close(processed.extra_fields["audio"], audio)
    assert processed.extra_fields["audio_sample_rate"] == 48000
    assert processed.extra_fields["media_kind"] == "video"


@pytest.mark.parametrize(
    ("strategy_cls", "priority", "expected_extra_keys", "params_key"),
    [
        # A single-output AR stage takes one ``SamplingParams`` object under the
        # singular ``sampling_params`` name, alongside the pinned egress modality.
        (ARStrategy, 3, {"lora_request", "priority", "output_modalities"}, "sampling_params"),
        # Diffusion stages always receive the per-stage sequence.
        (DiffusionStrategy, 0, set(), "sampling_params_list"),
    ],
)
@pytest.mark.asyncio
async def test_strategy_preserves_mode_specific_engine_call(strategy_cls, priority, expected_extra_keys, params_key):
    class _Engine:
        def generate(self, **kwargs):
            self.kwargs = kwargs

            async def _outputs():
                yield "first"
                yield "last"

            return _outputs()

    engine = _Engine()
    strategy = strategy_cls(SimpleNamespace(engine=engine))

    result = await strategy.run_generation(
        prompt={"prompt_token_ids": [1]},
        params=["params"],
        request_id="request-1",
        lora_request="lora",
        priority=priority,
    )

    assert result == "last"
    assert engine.kwargs["request_id"] == "request-1"
    assert engine.kwargs[params_key] == ["params"]
    assert set(engine.kwargs) - {"prompt", "request_id", params_key} == expected_extra_keys


@pytest.mark.asyncio
async def test_ar_strategy_switches_to_stage_list_for_multi_output_pipelines():
    """``_rollout_output_modalities`` flips the AR call back to the per-stage form.

    Set only when a pipeline declares more than one egress, and the singular
    ``sampling_params`` form raises ``ValueError: Expected <num_stages> sampling
    params, got a single sampling params object`` on such a server.
    """

    class _Engine:
        def generate(self, **kwargs):
            self.kwargs = kwargs

            async def _outputs():
                yield "first"
                yield "last"

            return _outputs()

    engine = _Engine()
    strategy = ARStrategy(SimpleNamespace(engine=engine))
    strategy._rollout_output_modalities = ["text"]
    # The multi-output path retains each stage's raw output for the adapter to combine,
    # so a stub combiner stands in for the registered pipeline adapter.
    strategy._rollout_adapter = SimpleNamespace(
        combine_engine_outputs=lambda outputs, prompt: ("combined", list(outputs))
    )

    result = await strategy.run_generation(
        prompt={"prompt_token_ids": [1]},
        params=["params"],
        request_id="request-1",
        lora_request="lora",
        priority=0,
    )

    assert result == "combined"
    assert strategy._rollout_fields_by_request_id["request-1"] == ["first", "last"]
    assert engine.kwargs["sampling_params_list"] == ["params"]
    assert engine.kwargs["output_modalities"] == ["text"]
    assert "sampling_params" not in engine.kwargs


@pytest.mark.parametrize("finish_reason,expected", [("stop", "completed"), ("abort", "aborted"), (None, "aborted")])
def test_diffusion_empty_output_preserves_completion_state(finish_reason, expected):
    from vllm_omni.outputs import OmniRequestOutput

    final = OmniRequestOutput.from_diffusion(request_id="empty", images=[])
    final.outputs = [SimpleNamespace(finish_reason=finish_reason)]
    result = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(final, None, {})
    assert result.diffusion_output.numel() == 0
    assert result.stop_reason == expected


@pytest.mark.parametrize("wrapped", [False, True])
def test_diffusion_named_output_preserves_completion_and_preemption_state(wrapped):
    from vllm_omni.outputs import OmniRequestOutput

    final, _ = _joint_video_audio_final_res()
    completion = SimpleNamespace(finish_reason="length", num_preempted=3)
    if wrapped:
        final.request_output = OmniRequestOutput.from_diffusion(request_id="wrapped-output", images=[])
        final.request_output.outputs = [completion]
    else:
        final.outputs = [completion]
    result = DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(final, None, {})
    assert result.stop_reason == "completed"
    assert result.num_preempted == 3
    assert result.artifacts["video_preview"].data.dtype == torch.uint8


@pytest.mark.parametrize("field", ["primary", "specs", "preview", "audio"])
def test_diffusion_rejects_incomplete_header_with_request_context(field):
    final, _ = _joint_video_audio_final_res()
    final.multimodal_output["metadata"]["media_artifacts"].pop(field)
    with pytest.raises(ValueError, match="joint-output.*invalid media_artifacts header"):
        DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(final, None, {})


def test_diffusion_rejects_non_tensor_named_payload_with_context():
    final, _ = _joint_video_audio_final_res()
    final.request_id = "bad-tensor"
    final.images[0]["audio"] = [0.0, 1.0]
    with pytest.raises(TypeError, match="bad-tensor.*artifact='audio'.*expected tensor"):
        DiffusionStrategy(SimpleNamespace(global_steps=1, model_config=None)).process_output(final, None, {})


def test_drop_defaulted_engine_args_keeps_only_explicit_overrides():
    """Only explicitly-set engine args may reach AsyncOmni (12e9280+)."""
    from dataclasses import asdict

    from vllm_omni.engine.arg_utils import OmniEngineArgs

    defaults = asdict(OmniEngineArgs(model=""))
    engine_args = dict(defaults)
    engine_args["model"] = "some-model"
    engine_args["tensor_parallel_size"] = 2

    filtered = server_module._drop_defaulted_engine_args(engine_args)

    # Defaulted values (including private runtime internals) are dropped ...
    assert "_api_process_count" not in filtered
    assert "_api_process_rank" not in filtered
    assert "convert" not in filtered
    # ... including OrchestratorArgs parser defaults (the ownership check
    # flags every non-None value, even False).
    engine_args["ulysses_mode"] = "strict"
    engine_args["cache_backend"] = "none"
    engine_args["step_execution"] = False
    filtered = server_module._drop_defaulted_engine_args(engine_args)
    assert "ulysses_mode" not in filtered
    assert "cache_backend" not in filtered
    assert "step_execution" not in filtered
    # ... while identity and real overrides survive.
    assert filtered["model"] == "some-model"
    assert filtered["tensor_parallel_size"] == 2
    # Keys unknown to OmniEngineArgs (orchestrator extras) are preserved.
    engine_args["custom_orchestrator_flag"] = True
    assert server_module._drop_defaulted_engine_args(engine_args)["custom_orchestrator_flag"] is True

    # Strict type checks prevent bool/int cross-equality (e.g. 0 == False)
    assert not server_module._is_defaulted_value("max_num_seqs", False, 0)
    assert not server_module._is_defaulted_value("enforce_eager", 0, False)
    assert server_module._is_defaulted_value("max_num_seqs", 0, 0)
    assert server_module._is_defaulted_value("enforce_eager", False, False)

    # Parser-level normalizations count as defaults: argparse applies
    # type=json.loads to string defaults (dict {} vs "{}"), and some string
    # flags default to "" in the CLI while the dataclass default is None.
    engine_args["diffusers_load_kwargs"] = {}
    engine_args["reasoning_parser_plugin"] = ""
    filtered = server_module._drop_defaulted_engine_args(engine_args)
    assert "diffusers_load_kwargs" not in filtered
    assert "reasoning_parser_plugin" not in filtered
    # ... but non-empty values still travel.
    engine_args["diffusers_load_kwargs"] = {"use_safetensors": True}
    engine_args["reasoning_parser_plugin"] = "openai"
    filtered = server_module._drop_defaulted_engine_args(engine_args)
    assert filtered["diffusers_load_kwargs"] == {"use_safetensors": True}
    assert filtered["reasoning_parser_plugin"] == "openai"


def test_drop_defaulted_engine_args_treats_cli_none_as_unset():
    """CLI ``None`` is the parser's *unset* sentinel, never an explicit value.

    Reproduces the 2026-10-14 ``bagel_think`` init failure: the vLLM-Omni CLI
    declares ``enable_lora``/``mm_device_do_normalize`` as ``bool | None``, so a
    parsed namespace carries ``None`` for both while the resolved dataclass
    defaults are ``False``/``True``. Equality-only filtering forwards them, and
    vllm-omni's diffusion ingress validates key *presence*, so the UND AR
    replica's stage 1 aborts with "Unknown diffusion config field(s) for stage
    1: 'enable_lora', 'mm_device_do_normalize'" before any decode runs.
    """
    from dataclasses import asdict

    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm_omni.config.omni_config import normalize_and_validate_diffusion_engine_ingress_kwargs
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    from vllm_omni.entrypoints.cli import serve

    parser = FlexibleArgumentParser(description="vLLM-Omni CLI")
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    for command in serve.cmd_init():
        command.subparser_init(subparsers).set_defaults(dispatch_function=command.cmd)
    parsed = parser.parse_args(args=["serve", "some-model"])

    engine_args = asdict(OmniEngineArgs.from_cli_args(parsed))
    # The parser really does hand over ``None`` for these two flags ...
    assert engine_args["enable_lora"] is None
    assert engine_args["mm_device_do_normalize"] is None
    # ... while the resolved dataclass defaults are ``False``/``True``, never
    # ``None``, so an equality-only comparison saw them as explicit overrides.
    assert server_module._is_defaulted_value("enable_lora", None, False)
    assert server_module._is_defaulted_value("mm_device_do_normalize", None, True)

    filtered = server_module._drop_defaulted_engine_args(engine_args)
    assert "enable_lora" not in filtered
    assert "mm_device_do_normalize" not in filtered
    # The surviving mapping must pass *every* stage's ingress, not just stage 0's.
    normalize_and_validate_diffusion_engine_ingress_kwargs(filtered, stage_id=1)

    # An explicitly requested value is still forwarded, so a non-merged LoRA
    # recipe keeps reaching the engine that consumes it.
    engine_args["enable_lora"] = True
    engine_args["mm_device_do_normalize"] = False
    filtered = server_module._drop_defaulted_engine_args(engine_args)
    assert filtered["enable_lora"] is True
    assert filtered["mm_device_do_normalize"] is False


def _write_deploy_config(tmp_path, stage_blocks):
    import yaml

    path = tmp_path / "deploy.yaml"
    path.write_text(yaml.safe_dump({"pipeline": "bagel_think", "stages": stage_blocks}), encoding="utf-8")
    return path


def test_drop_defaulted_engine_args_yields_to_deploy_stage_pins(tmp_path):
    """A RolloutConfig default must not outrank a deploy config's per-stage pin.

    Reproduces the 2026-10-14 ``bagel_corl_prfix_t07`` init failure. ``RolloutConfig``
    defaults ``max_num_seqs`` to 1024 while ``OmniEngineArgs`` defaults it to ``None``,
    so ``_is_defaulted_value`` never saw it as unset and it travelled as a plain global
    engine arg -- and a plain global arg beats a per-stage value. Every stage of
    ``bagel_corl_deploy_ar.yaml`` pins ``max_num_seqs: 1``, so the DiT stage received
    1024 and vLLM-Omni refused it:

        ValueError: 'BagelPipeline' does not support request-level batching.
        Use max_num_seqs=1 for serial request execution

    which surfaced as ``StageDiffusionProc`` -> "Orchestrator initialization failed"
    on four of the six attempts.
    """
    from verl.workers.config import RolloutConfig

    source_default = RolloutConfig().max_num_seqs
    assert source_default != 1, "the leak only exists while the source default is batched"

    deploy_config = _write_deploy_config(
        tmp_path,
        [
            {"stage_id": 0, "max_num_seqs": 1, "devices": "0"},
            {"stage_id": 1, "max_num_seqs": 1, "devices": "1"},
        ],
    )

    engine_args = {"model": "some-model", "max_num_seqs": source_default, "tensor_parallel_size": 2}
    filtered = server_module._drop_defaulted_engine_args(engine_args, deploy_config=deploy_config)
    # The unasked-for default is gone, so the stage's own ``max_num_seqs: 1`` wins ...
    assert "max_num_seqs" not in filtered
    # ... while identity and real overrides still travel.
    assert filtered["model"] == "some-model"
    assert filtered["tensor_parallel_size"] == 2

    # An *explicit* request still rides in and overrides the pin, as documented for
    # every plain engine arg; only a value indistinguishable from "unset" is dropped.
    engine_args["max_num_seqs"] = 4
    assert server_module._drop_defaulted_engine_args(engine_args, deploy_config=deploy_config)["max_num_seqs"] == 4


def test_drop_defaulted_engine_args_against_the_real_ar_deploy_config():
    """The shipped AR deploy config is what the failure hinged on, so pin it down."""
    from pathlib import Path

    deploy_config = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "agenticllmgrpo_trainer"
        / "bagel"
        / "bagel_corl_deploy_ar.yaml"
    )
    assert deploy_config.is_file(), deploy_config

    pinned = server_module._deploy_stage_pinned_keys(str(deploy_config))
    # Pinned by both stages => fully determined by the deploy config.
    assert "max_num_seqs" in pinned
    assert "enable_prefix_caching" in pinned
    # ``gpu_memory_utilization`` is pinned on the Thinker stage only, deliberately left
    # out: dropping it would hand the DiT stage vLLM-Omni's own default (0.92) instead.
    assert "gpu_memory_utilization" not in pinned

    from verl.workers.config import RolloutConfig

    engine_args = {"model": "m", "max_num_seqs": RolloutConfig().max_num_seqs}
    filtered = server_module._drop_defaulted_engine_args(engine_args, deploy_config=str(deploy_config))
    assert "max_num_seqs" not in filtered


def test_drop_defaulted_engine_args_keeps_source_defaults_no_stage_pins(tmp_path):
    """Only keys every stage pins may lose a source default; the rest degrade safely.

    ``gpu_memory_utilization`` defaults to ``0.5`` in ``RolloutConfig`` and to ``0.92``
    in ``OmniEngineArgs``. Dropping an unset 0.5 because *one* stage pins a budget would
    silently re-budget every other stage upwards, so a partially-pinned key keeps
    travelling exactly as before.
    """
    deploy_config = _write_deploy_config(
        tmp_path,
        [
            {"stage_id": 0, "gpu_memory_utilization": 0.40, "devices": "0"},
            {"stage_id": 1, "devices": "1"},
        ],
    )
    assert "gpu_memory_utilization" not in server_module._deploy_stage_pinned_keys(deploy_config)

    engine_args = {"model": "m", "gpu_memory_utilization": 0.5}
    assert (
        server_module._drop_defaulted_engine_args(engine_args, deploy_config=str(deploy_config))[
            "gpu_memory_utilization"
        ]
        == 0.5
    )


def test_deploy_stage_pinned_keys_missing_or_malformed_config_degrades(tmp_path):
    """An absent or unparsable deploy config must not change how args are filtered."""
    assert server_module._deploy_stage_pinned_keys(None) == frozenset()
    assert server_module._deploy_stage_pinned_keys(str(tmp_path / "nope.yaml")) == frozenset()

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("stages: [", encoding="utf-8")
    assert server_module._deploy_stage_pinned_keys(str(malformed)) == frozenset()

    engine_args = {"model": "m", "max_num_seqs": 1024}
    absent_deploy_config = str(tmp_path / "nope.yaml")
    assert server_module._drop_defaulted_engine_args(engine_args) == engine_args
    assert server_module._drop_defaulted_engine_args(engine_args, deploy_config=absent_deploy_config) == engine_args


def test_ar_deploy_config_omits_text_encoder_tp_size_on_ar_stages(tmp_path, monkeypatch):
    """text_encoder_tp_size is diffusion-only; upstream rejects it on AR stages."""
    import yaml
    from vllm_omni.config.stage_config import StageExecutionType

    from verl_omni.pipelines.model_base import OmniRolloutPipelineBase

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")

    def _stage(stage_id, execution_type):
        return SimpleNamespace(
            stage_id=stage_id,
            execution_type=execution_type,
            final_output=False,
            final_output_type=None,
            sampling_constraints={},
        )

    class _Adapter:
        combine_engine_outputs = OmniRolloutPipelineBase.combine_engine_outputs

        @staticmethod
        def ensure_pipeline_registered(pipeline_mode):
            return None

        @staticmethod
        def build_stage_configs(pipeline_mode=None):
            return [
                _stage(0, StageExecutionType.LLM_AR),
                _stage(1, StageExecutionType.DIFFUSION),
            ]

        @staticmethod
        def policy_stage_id(pipeline_mode=None):
            return 0

        @staticmethod
        def weight_sync_stage_ids(pipeline_mode=None):
            return None

        @staticmethod
        def get_pipeline_id(pipeline_mode=None):
            return "test-pipeline"

        @staticmethod
        def get_stage_engine_extras(stage_id, pipeline_mode=None):
            return {}

    server = SimpleNamespace(
        config=SimpleNamespace(tensor_model_parallel_size=2, text_encoder_tp_size=1),
    )
    strategy = ARStrategy(server)
    engine_kwargs: dict = {}
    strategy._write_deploy_config(engine_kwargs, "test-pipeline", _Adapter, "thinker_only")
    try:
        with open(engine_kwargs["deploy_config"]) as file:
            deploy = yaml.safe_load(file)
    finally:
        server._temp_deploy_ctx.cleanup()

    stages = {stage["stage_id"]: stage for stage in deploy["stages"]}
    assert "text_encoder_tp_size" not in stages[0]
    assert stages[1]["text_encoder_tp_size"] == 1


def test_restore_raw_compilation_config_keeps_user_mapping():
    """Only the raw user mapping may travel; expanded objects are rejected."""
    from dataclasses import asdict

    from vllm.config import CompilationConfig

    # A JSON string (as parsed from the CLI) stays a minimal mapping.
    engine_args = {"compilation_config": asdict(CompilationConfig())}
    server_module._restore_raw_compilation_config(
        engine_args, SimpleNamespace(compilation_config='{"cudagraph_mode": "FULL_AND_PIECEWISE"}')
    )
    assert engine_args["compilation_config"] == {"cudagraph_mode": "FULL_AND_PIECEWISE"}

    # An absent user value drops the expanded object entirely.
    engine_args = {"compilation_config": asdict(CompilationConfig())}
    server_module._restore_raw_compilation_config(engine_args, SimpleNamespace(compilation_config=None))
    assert "compilation_config" not in engine_args


def test_diffusion_ingress_allowed_fields_includes_pinned_frontend_fields():
    allowed = diffusion_strategy_module._diffusion_ingress_allowed_fields()

    assert isinstance(allowed, frozenset)
    assert "enable_flash_late_interaction" in allowed
    assert "height" in allowed or "num_inference_steps" in allowed or "dtype" in allowed


def test_diffusion_ingress_allowed_fields_supports_npu_frontend_path(monkeypatch):
    import sys
    from dataclasses import dataclass
    from types import ModuleType

    @dataclass
    class NpuFrontendArgs:
        host: str | None = None

    monkeypatch.setitem(sys.modules, "vllm.entrypoints.launchers", None)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.launchers.cli_args", None)
    npu_cli_args = ModuleType("vllm.entrypoints.openai.cli_args")
    npu_cli_args.FrontendArgs = NpuFrontendArgs
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.cli_args", npu_cli_args)

    allowed = diffusion_strategy_module._diffusion_ingress_allowed_fields()

    assert isinstance(allowed, frozenset)
    assert "host" in allowed
