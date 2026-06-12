# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu


def _install_module(name: str, **attrs):
    module = sys.modules.get(name, types.ModuleType(name))
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    if "." in name:
        parent_name, child_name = name.rsplit(".", 1)
        parent = sys.modules.setdefault(parent_name, types.ModuleType(parent_name))
        setattr(parent, child_name, sys.modules[name])
    return sys.modules[name]


def _install_lightweight_imports():
    repo_root = Path(__file__).resolve().parents[3]

    for package_name, package_path in {
        "verl_omni": repo_root / "verl_omni",
        "verl_omni.pipelines": repo_root / "verl_omni" / "pipelines",
        "verl_omni.utils": repo_root / "verl_omni" / "utils",
        "verl_omni.workers": repo_root / "verl_omni" / "workers",
        "verl_omni.workers.config": repo_root / "verl_omni" / "workers" / "config",
        "verl_omni.workers.rollout": repo_root / "verl_omni" / "workers" / "rollout",
        "verl_omni.workers.rollout.vllm_rollout": repo_root
        / "verl_omni"
        / "workers"
        / "rollout"
        / "vllm_rollout",
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
    _install_module("vllm_omni.entrypoints.cli.serve")
    _install_module("vllm_omni.entrypoints.openai")
    _install_module("vllm_omni.entrypoints.openai.api_server", omni_init_app_state=lambda *_args, **_kwargs: None)
    _install_module("vllm_omni.engine.arg_utils", OmniEngineArgs=SimpleNamespace)
    _install_module("vllm_omni.entrypoints", AsyncOmni=object)
    _install_module("vllm_omni.inputs.data", OmniCustomPrompt=dict, OmniDiffusionSamplingParams=SimpleNamespace)
    _install_module("vllm_omni.lora.request", LoRARequest=SimpleNamespace)
    _install_module("vllm_omni.outputs", OmniRequestOutput=SimpleNamespace)


_install_lightweight_imports()

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server as server_mod


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
