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

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch
import yaml

from verl_omni.pipelines.model_base import OmniRolloutPipelineBase
from verl_omni.pipelines.qwen3_omni.omni_rollout_adapter import Qwen3OmniRolloutAdapter
from verl_omni.pipelines.rollout_media import DiffusionIOSpec, MediaSpec
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


def test_optional_rollout_hooks_preserve_existing_ar_defaults():
    first, final = object(), object()

    assert OmniRolloutPipelineBase.supports_async_chunk is True
    assert OmniRolloutPipelineBase.weight_sync_stage_ids() is None
    assert OmniRolloutPipelineBase.policy_stage_id() == 0
    assert OmniRolloutPipelineBase.prepare_engine_prompt([], None, {}) is None
    assert OmniRolloutPipelineBase.combine_engine_outputs([final], {}) == (final, {})
    with pytest.raises(NotImplementedError, match="multiple final outputs"):
        OmniRolloutPipelineBase.combine_engine_outputs([first, final], {})
    with pytest.raises(RuntimeError, match="no outputs"):
        OmniRolloutPipelineBase.combine_engine_outputs([], {})


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

    prompt, params = strategy.preprocess_input(
        prompt_ids=[1, 2, 3],
        sampling_params=sampling_params,
        multi_modal_data={"image": ["image"]},
        lora_request=None,
        negative_prompt_ids=None,
        mm_processor_kwargs={"size": 224},
    )

    assert prompt == {
        "prompt_token_ids": [1, 2],
        "multi_modal_data": {"image": ["image"]},
        "mm_processor_kwargs": {"size": 224},
    }
    assert params.max_tokens == 6
    assert params.logprobs == 0
    assert params.repetition_penalty == 1.1


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

    assert strategy._policy_stage_id == 1
    assert strategy._policy_stage_index == 1
    assert strategy._weight_sync_stage_ids == [1]
    server._temp_deploy_ctx.cleanup()


@pytest.mark.parametrize(
    ("policy_stage_id", "weight_sync_stage_ids", "error", "message"),
    [
        (True, [1], TypeError, "integer stage ID"),
        (2, [1], ValueError, "unknown stage 2"),
        (0, [], ValueError, "empty list"),
        (0, [1, 1], ValueError, "unique stage IDs"),
        (0, [2], ValueError, "unknown stages"),
        (0, [True], TypeError, "only integer stage IDs"),
    ],
)
def test_ar_strategy_rejects_invalid_adapter_stage_contracts(
    monkeypatch, policy_stage_id, weight_sync_stage_ids, error, message
):
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

    with pytest.raises(error, match=message):
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
    strategy._policy_stage_id = 1
    strategy._policy_stage_index = 1
    strategy._stage_sampling_constraints = {1: {}}

    prompt, params = strategy.preprocess_input(
        [5, 6],
        {"temperature": 0.8, "logprobs": True},
        {},
        None,
        None,
    )

    assert prompt["additional_information"]["max_new_tokens"] == [8]
    assert len(params) == 2
    assert params[0].stage == "thinker"
    assert params[1].max_tokens == 8
    assert params[1].temperature == pytest.approx(0.8)
    assert params[1].logprobs == 0

    completion = SimpleNamespace(
        token_ids=[7],
        logprobs=[{7: SimpleNamespace(logprob=-0.25)}],
        finish_reason="stop",
        num_preempted=0,
    )
    final_res = SimpleNamespace(
        request_output=SimpleNamespace(outputs=[completion]),
        _verl_omni_rollout_fields={},
    )
    assert strategy.process_output(final_res, params, {}).log_probs == [-0.25]


@pytest.mark.parametrize(
    ("adapter_prompt", "message"),
    [
        ({"additional_information": {"text": ["hello"]}}, "must contain prompt_token_ids"),
        ({"prompt_token_ids": "1,2"}, "list of integers"),
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
        strategy.preprocess_input([5, 6], {}, {}, None, None)


@pytest.mark.asyncio
async def test_ar_strategy_retains_requested_stage_outputs_and_targets_weight_sync():
    policy = SimpleNamespace(outputs=[])

    class Engine:
        def __init__(self):
            self.generate_kwargs = None
            self.rpc_kwargs = None

        async def generate(self, **kwargs):
            self.generate_kwargs = kwargs
            yield policy

        async def collective_rpc(self, **kwargs):
            self.rpc_kwargs = kwargs
            return "rpc-result"

    class Adapter:
        @staticmethod
        def combine_engine_outputs(outputs, prompt):
            assert outputs == [policy]
            return policy, {"audio_sample_rate": 24_000}

    server = object.__new__(server_module.vLLMOmniHttpServer)
    server.engine = Engine()
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
    assert result._verl_omni_rollout_fields == {"audio_sample_rate": 24_000}
    assert server.engine.generate_kwargs["output_modalities"] == ["latent", "audio"]
    assert server.engine.rpc_kwargs["stage_ids"] == [0]
    assert rpc_result == "rpc-result"


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
        prompt_ids=[1, 2],
        sampling_params={"max_new_tokens": 2, "logprobs": True},
        multi_modal_data={},
        lora_request=None,
        negative_prompt_ids=None,
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


def test_ar_strategy_rejects_qwen3_omni_full_multi_output_without_combiner():
    server = SimpleNamespace(_rollout_flags={})
    strategy = ARStrategy(server)

    with pytest.raises(ValueError, match="multiple final pipeline outputs"):
        strategy.preprocess_engine_kwargs(
            {
                "pipeline_name": "qwen3_omni_moe",
                "pipeline_mode": "full",
            }
        )


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
        config=SimpleNamespace(
            external_lib=["extension"],
            step_execution=False,
            enable_prompt_embed_cache=True,
            prompt_embed_cache_size=16,
        ),
        model_config=SimpleNamespace(architecture="Architecture", algorithm="Algorithm"),
    )
    strategy = DiffusionStrategy(server)
    engine_args = {"max_num_seqs": 4}

    strategy.prepare_engine_args(engine_args, Namespace())

    assert imported == [["extension"]]
    assert engine_args == {
        "max_num_seqs": 1,
        "enable_dummy_pipeline": True,
        "custom_pipeline_args": {"pipeline_class": "package.Adapter"},
        "enable_prompt_embed_cache": True,
        "prompt_embed_cache_size": 16,
    }


def test_diffusion_strategy_preserves_multistage_prompt_shape():
    server = SimpleNamespace(engine=SimpleNamespace(default_sampling_params_list=["ar-stage", "diffusion-stage"]))
    strategy = DiffusionStrategy(server)
    prompt_mask = torch.tensor([True, False])

    prompt, params = strategy.preprocess_input(
        prompt_ids=[1, 2],
        sampling_params={"pipeline_private_arg": 7},
        multi_modal_data={"image": ["image"]},
        lora_request=None,
        negative_prompt_ids=[3, 4],
        prompt_mask=prompt_mask,
        extra_prompt_ids={"encoder": [5]},
        negative_extra_prompt_ids={"encoder": [6]},
    )

    assert prompt["prompt_token_ids"] == [1, 2]
    assert prompt["prompt_mask"] is prompt_mask
    assert prompt["modalities"] == ["image"]
    assert prompt["negative_prompt_ids"] == [3, 4]
    assert prompt["extra_prompt_ids"] == {"encoder": [5]}
    assert prompt["negative_extra_prompt_ids"] == {"encoder": [6]}
    assert prompt["multi_modal_data"] == {"image": ["image"]}
    assert prompt["extra_args"] == {"multi_modal_data": {"image": ["image"]}}
    assert params[0] == "ar-stage"
    assert params[-1].extra_args == {"pipeline_private_arg": 7}


def _joint_video_audio_final_res():
    video = torch.zeros(1, 3, 2, 8, 8, dtype=torch.uint8)
    audio = torch.zeros(1, 16)
    final_res = SimpleNamespace(
        images=[(video, audio)],
        trajectory_latents=None,
        trajectory_timesteps=None,
        trajectory_log_probs=None,
        multimodal_output=None,
        request_output=None,
    )
    return final_res, audio


def test_diffusion_strategy_uses_declared_audio_sample_rate(monkeypatch):
    pipeline_cls = SimpleNamespace(
        diffusion_io_spec=DiffusionIOSpec(
            primary=MediaSpec("video"),
            auxiliary=(MediaSpec("audio", sample_rate=48000),),
        )
    )
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
    # process_output unbatches the leading dimension of auxiliary media.
    torch.testing.assert_close(processed.extra_fields["audio"], audio[0])


def test_diffusion_strategy_omits_audio_sample_rate_without_declared_spec(monkeypatch):
    # Without an adapter-declared DiffusionIOSpec the strategy still surfaces the
    # auxiliary audio stream but attaches no model-specific sample-rate default.
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

    torch.testing.assert_close(processed.extra_fields["audio"], audio[0])
    assert "audio_sample_rate" not in processed.extra_fields


@pytest.mark.parametrize(
    ("strategy_cls", "expected_extra_keys"),
    [
        (ARStrategy, {"lora_request", "priority"}),
        (DiffusionStrategy, set()),
    ],
)
@pytest.mark.asyncio
async def test_strategy_preserves_mode_specific_engine_call(strategy_cls, expected_extra_keys):
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
        priority=3,
    )

    assert result == "last"
    assert engine.kwargs["request_id"] == "request-1"
    assert engine.kwargs["sampling_params_list"] == ["params"]
    assert set(engine.kwargs) - {"prompt", "request_id", "sampling_params_list"} == expected_extra_keys
