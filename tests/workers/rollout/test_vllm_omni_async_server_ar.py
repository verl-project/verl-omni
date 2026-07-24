# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import asyncio
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu

_MISSING = object()
_ORIGINAL_MODULES = {}
_LIGHTWEIGHT_MODULE_NAMES = []


def _remember_module(name: str) -> None:
    if name not in _ORIGINAL_MODULES:
        _ORIGINAL_MODULES[name] = sys.modules.get(name, _MISSING)
        _LIGHTWEIGHT_MODULE_NAMES.append(name)


def _install_module(name: str, **attrs):
    _remember_module(name)
    module = sys.modules.get(name, types.ModuleType(name))
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        _remember_module(parent_name)
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, child_name, sys.modules[name])
    return sys.modules[name]


def _restore_lightweight_imports():
    for name in reversed(_LIGHTWEIGHT_MODULE_NAMES):
        original = _ORIGINAL_MODULES[name]
        if original is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


def _install_lightweight_imports():
    repo_root = Path(__file__).resolve().parents[3]

    for package_name, package_path in {
        "verl_omni": repo_root / "verl_omni",
        "verl_omni.pipelines": repo_root / "verl_omni" / "pipelines",
        "verl_omni.utils": repo_root / "verl_omni" / "utils",
        "verl_omni.workers": repo_root / "verl_omni" / "workers",
        "verl_omni.workers.config": repo_root / "verl_omni" / "workers" / "config",
        "verl_omni.workers.rollout": repo_root / "verl_omni" / "workers" / "rollout",
        "verl_omni.workers.rollout.vllm_rollout": repo_root / "verl_omni" / "workers" / "rollout" / "vllm_rollout",
    }.items():
        package = _install_module(package_name)
        package.__path__ = [str(package_path)]

    class FakeHijack:
        @staticmethod
        def hijack():
            pass

    class FakeDiffusionModelConfig:
        pass

    class FakeDiffusionRolloutConfig:
        pass

    class FakeVllmOmniPipelineBase:
        @classmethod
        def get_pipeline_path(cls, *_args, **_kwargs):
            return None

    _install_module("verl_omni.pipelines.model_base", VllmOmniPipelineBase=FakeVllmOmniPipelineBase)
    _install_module("verl_omni.utils.vllm_omni", VLLMOmniHijack=FakeHijack)
    _install_module(
        "verl_omni.workers.config",
        DiffusionModelConfig=FakeDiffusionModelConfig,
        DiffusionRolloutConfig=FakeDiffusionRolloutConfig,
    )
    _install_module("verl_omni.workers.rollout.replica", DiffusionOutput=SimpleNamespace)

    _install_module("vllm_omni")
    _install_module("vllm_omni.entrypoints")
    _install_module("vllm_omni.entrypoints.cli")
    _install_module("vllm_omni.entrypoints.cli.serve", run_headless=lambda *_args, **_kwargs: None)
    _install_module("vllm_omni.entrypoints.openai")
    _install_module("vllm_omni.entrypoints.openai.api_server", omni_init_app_state=lambda *_args, **_kwargs: None)
    _install_module("vllm_omni.engine.arg_utils", OmniEngineArgs=SimpleNamespace)
    _install_module("vllm_omni.entrypoints", AsyncOmni=object)
    _install_module("vllm_omni.inputs.data", OmniCustomPrompt=dict, OmniDiffusionSamplingParams=SimpleNamespace)
    _install_module("vllm_omni.lora.request", LoRARequest=SimpleNamespace)
    _install_module("vllm_omni.outputs", OmniRequestOutput=SimpleNamespace)
    _install_module("vllm_omni.utils")
    _install_module("vllm_omni.utils.tracking_parser", TrackingNamespace=SimpleNamespace)


_install_lightweight_imports()

# These imports must happen after the lightweight dependency modules are installed.
# isort: off
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as server_mod  # noqa: E402
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer  # noqa: E402
# isort: on

_restore_lightweight_imports()


class _Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def test_ar_mode_uses_hf_model_config_and_rollout_generation_defaults(monkeypatch):
    def fake_omega_conf_to_dataclass(_config, dataclass_type=None):
        return SimpleNamespace(dataclass_type=dataclass_type)

    monkeypatch.setattr(server_mod, "omega_conf_to_dataclass", fake_omega_conf_to_dataclass)

    server = object.__new__(vLLMOmniHttpServer)
    server.config = _Config(
        engine_kwargs={"vllm_omni": {"output_mode": "ar"}},
        max_model_len=None,
        prompt_length=8,
        response_length=4,
        temperature=1.0,
        top_k=-1,
        top_p=1.0,
    )

    model_config = server._init_model_config({"path": "dummy", "trust_remote_code": True})
    server._validate_configs()

    assert server._ar_mode is True
    assert model_config.dataclass_type.__name__ == "HFModelConfig"
    assert server.config.max_model_len == 12
    assert server._get_override_generation_config()["max_new_tokens"] == 4


def test_preprocess_engine_kwargs_removes_internal_ar_mode_key():
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    engine_kwargs = {
        "output_mode": "ar",
        "custom_pipeline": "ignored",
        "stage_configs_path": "/tmp/stage.yaml",
        "async_chunk": True,
    }

    server._preprocess_engine_kwargs(engine_kwargs)

    assert "output_mode" not in engine_kwargs
    assert "custom_pipeline" not in engine_kwargs
    assert engine_kwargs["stage-configs-path"] == "/tmp/stage.yaml"
    assert engine_kwargs["async-chunk"] is True


def test_preprocess_engine_kwargs_propagates_ar_startup_timeouts(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_STARTUP_HANDSHAKE_TIMEOUT", raising=False)

    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    engine_kwargs = {
        "output_mode": "ar",
        "stage_init_timeout": "1800",
    }

    server._preprocess_engine_kwargs(engine_kwargs)

    assert engine_kwargs["stage-init-timeout"] == "1800"
    assert engine_kwargs["init-timeout"] == 1800
    assert os.environ["VLLM_OMNI_STARTUP_HANDSHAKE_TIMEOUT"] == "1800"


def _write_stage_config(tmp_path, body: str) -> Path:
    path = tmp_path / "stage.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _preflight_server(*, gpus_per_node: int = 1) -> vLLMOmniHttpServer:
    server = object.__new__(vLLMOmniHttpServer)
    server.config = _Config(
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    server.nnodes = 1
    server.gpus_per_node = gpus_per_node
    return server


def test_ar_placement_preflight_rejects_outer_and_inner_rollout_dp(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    stage_config = _write_stage_config(
        tmp_path,
        """
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0,1"
      num_replicas: 2
    engine_args:
      tensor_parallel_size: 1
""",
    )
    server = _preflight_server(gpus_per_node=2)

    with pytest.raises(ValueError, match="Use exactly one DP owner"):
        server._run_ar_placement_preflight(SimpleNamespace(stage_configs_path=str(stage_config)))


def test_ar_placement_preflight_accepts_outer_dp_with_single_stage_replica(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    stage_config = _write_stage_config(
        tmp_path,
        """
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "0"
    engine_args:
      tensor_parallel_size: 1
""",
    )
    server = _preflight_server(gpus_per_node=2)

    preflight = server._run_ar_placement_preflight(SimpleNamespace(stage_configs_path=str(stage_config)))

    assert preflight.outer_replicas == 2
    assert preflight.max_stage_replicas == 1
    assert preflight.stages[0].devices == "0"


def test_ar_placement_preflight_rejects_nonlocal_stage_devices(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    stage_config = _write_stage_config(
        tmp_path,
        """
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime:
      devices: "1"
    engine_args:
      tensor_parallel_size: 1
""",
    )
    server = _preflight_server(gpus_per_node=1)

    with pytest.raises(ValueError, match="actor-local CUDA ids"):
        server._run_ar_placement_preflight(SimpleNamespace(stage_configs_path=str(stage_config)))


def test_run_server_preserves_ar_startup_timeouts_after_engine_arg_parse(monkeypatch):
    captured_engine_kwargs = {}

    @dataclass
    class _FakeEngineArgs:
        model: str = "fake-model"
        compilation_config: dict = field(default_factory=lambda: {"drop": None, "keep": "x"})

    class _FakeOmniEngineArgs:
        @staticmethod
        def from_cli_args(_args):
            return _FakeEngineArgs()

    class _FakeAsyncOmni:
        def __init__(self, **kwargs):
            captured_engine_kwargs.update(kwargs)

    class _FakeSocket:
        def close(self):
            pass

    async def _fake_init_app_state(*_args, **_kwargs):
        return None

    async def _fake_run_uvicorn(*_args, **_kwargs):
        return 8000, object()

    monkeypatch.setattr(server_mod, "OmniEngineArgs", _FakeOmniEngineArgs)
    monkeypatch.setattr(server_mod, "AsyncOmni", _FakeAsyncOmni)
    monkeypatch.setattr(server_mod, "build_app", lambda _args: SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(server_mod, "omni_init_app_state", _fake_init_app_state)
    monkeypatch.setattr(server_mod, "run_uvicorn", _fake_run_uvicorn)

    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    server.nnodes = 1
    server.config = _Config(logprobs_mode="raw_logprobs", step_execution=False)
    server._server_address = "127.0.0.1"
    server._ensure_tracking_namespace = lambda args: args
    server._configure_omni_distributed_args = lambda args, headless=False: None
    server._run_ar_placement_preflight = lambda _args: None
    monkeypatch.setattr(server_mod, "get_free_port", lambda *_args, **_kwargs: (29500, _FakeSocket()))

    args = SimpleNamespace(stage_init_timeout="1800", init_timeout="1800")

    asyncio.run(server.run_server(args))

    assert captured_engine_kwargs["stage_init_timeout"] == 1800
    assert captured_engine_kwargs["init_timeout"] == 1800
    assert captured_engine_kwargs["compilation_config"] == {"keep": "x"}
    assert captured_engine_kwargs["logprobs_mode"] == "raw_logprobs"


def test_configure_omni_distributed_args_uses_controlled_ports(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_MASTER_ZMQ_PORT", "60392")
    monkeypatch.setenv("VLLM_OMNI_DIST_MASTER_PORT", "65322")

    server = object.__new__(vLLMOmniHttpServer)
    server.nnodes = 4
    server._ar_mode = True
    server._master_address = "10.66.154.11"
    server._master_port = 29911
    args = SimpleNamespace(master_addr="10.66.154.11", master_port=29911)

    server._configure_omni_distributed_args(args, headless=False)

    assert args.omni_master_address == "10.66.154.11"
    assert args.omni_master_port == 60392
    # Addr ownership stays with verl's rollout server[0]; the env only controls
    # the port so Ray head and rollout DP coordinator cannot diverge.
    assert args.master_addr == "10.66.154.11"
    assert args.master_port == 65322
    assert args.omni_dp_size_local == 1
    assert args.worker_backend == "multi_process"
    assert args.headless is False


def test_configure_omni_distributed_args_preserves_legacy_port_without_env(monkeypatch):
    monkeypatch.delenv("VERL_OMNI_MASTER_ZMQ_PORT", raising=False)
    monkeypatch.delenv("VLLM_OMNI_DIST_MASTER_PORT", raising=False)

    server = object.__new__(vLLMOmniHttpServer)
    server.nnodes = 4
    server._ar_mode = True
    server._master_address = "10.66.154.11"
    server._master_port = 29911
    args = SimpleNamespace(master_addr="10.66.154.11", master_port=29911)

    server._configure_omni_distributed_args(args, headless=True)

    assert args.omni_master_port == 29911
    assert args.master_port == 29911
    assert args.headless is True


def test_ar_text_rollout_rejects_multimodal_inputs():
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True

    with pytest.raises(NotImplementedError, match="AR text rollout does not support: image_data"):
        server._validate_generate_multimodal_args(
            image_data=["image"],
            video_data=None,
            audio_data=None,
            mm_processor_kwargs=None,
        )

    with pytest.raises(NotImplementedError, match="audio_data, mm_processor_kwargs"):
        server._validate_generate_multimodal_args(
            image_data=None,
            video_data=None,
            audio_data=["audio"],
            mm_processor_kwargs={"fps": 1},
        )


def test_diffusion_rollout_keeps_image_video_but_rejects_unwired_audio_args():
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = False

    server._validate_generate_multimodal_args(
        image_data=["image"],
        video_data=["video"],
        audio_data=None,
        mm_processor_kwargs=None,
    )

    with pytest.raises(NotImplementedError, match="audio_data, mm_processor_kwargs"):
        server._validate_generate_multimodal_args(
            image_data=None,
            video_data=None,
            audio_data=["audio"],
            mm_processor_kwargs={"sample_rate": 16000},
        )


def test_ar_preprocess_input_normalizes_true_logprobs_to_sampled_token_only():
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    server.config = _Config(
        max_model_len=16,
        prompt_length=8,
        response_length=4,
        repetition_penalty=1.0,
    )

    _prompt, params = server._preprocess_input(
        prompt_ids=[1, 2, 3],
        sampling_params={"logprobs": True},
        multi_modal_data={},
        lora_request=None,
        negative_prompt_ids=None,
    )

    assert params.logprobs == 0


class _FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


def test_ar_process_output_extracts_sampled_token_logprobs(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_LOGPROB_DEBUG_LIMIT", "2")

    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    server.global_steps = 3
    server.config = _Config(logprobs_mode="processed_logprobs")

    final_res = SimpleNamespace(
        request_output=SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    token_ids=[11, 22],
                    logprobs=[
                        {11: _FakeLogprob(-1.5), 17: _FakeLogprob(-3.0)},
                        {22: _FakeLogprob(-2.5), 19: _FakeLogprob(-4.0)},
                    ],
                    finish_reason="length",
                    num_preempted=0,
                )
            ]
        )
    )

    result = server._process_output(final_res, SimpleNamespace(logprobs=0), {})

    assert result.token_ids == [11, 22]
    assert result.log_probs == [-1.5, -2.5]
    assert result.stop_reason == "completed"
    assert result.extra_fields["global_steps"] == 3


class _FakeAsyncOmniEngine:
    def __init__(self):
        self.request_states = {
            "internal-0": SimpleNamespace(external_request_id="external-a"),
            "internal-1": SimpleNamespace(external_request_id="external-b"),
        }
        self.aborted_internal_batches = []
        self.aborted_external = []
        self.reset_prefix_cache_calls = 0

    async def _abort_internal_requests(self, request_ids):
        self.aborted_internal_batches.append(list(request_ids))
        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    async def abort(self, request_id):
        self.aborted_external.append(request_id)
        for rid, state in list(self.request_states.items()):
            if state.external_request_id == request_id:
                self.request_states.pop(rid, None)

    async def reset_prefix_cache(self):
        self.reset_prefix_cache_calls += 1
        return True


def test_abort_all_requests_uses_async_omni_request_states():
    server = object.__new__(vLLMOmniHttpServer)
    server.node_rank = 0
    server.engine = _FakeAsyncOmniEngine()

    result = asyncio.run(server.abort_all_requests())

    assert result == {"aborted_count": 2, "request_ids": ["internal-0", "internal-1"]}
    assert server.engine.aborted_internal_batches == [["internal-0", "internal-1"]]
    assert server.engine.reset_prefix_cache_calls == 1
    assert server.engine.request_states == {}


def test_abort_request_supports_internal_and_external_async_omni_ids():
    server = object.__new__(vLLMOmniHttpServer)
    server.node_rank = 0
    server.engine = _FakeAsyncOmniEngine()

    internal_result = asyncio.run(server.abort_request("internal-0", reset_prefix_cache=False))
    external_result = asyncio.run(server.abort_request("external-b"))

    assert internal_result == {"aborted": True, "request_id": "internal-0"}
    assert external_result == {"aborted": True, "request_id": "external-b"}
    assert server.engine.aborted_internal_batches == [["internal-0"]]
    assert server.engine.aborted_external == ["external-b"]
    assert server.engine.reset_prefix_cache_calls == 1
    assert server.engine.request_states == {}
