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

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from hydra import compose, initialize_config_dir

from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.rollout.vllm_rollout.utils import vLLMOmniColocateWorkerExtension
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
    _drop_none_mapping_values,
    vLLMOmniHttpServer,
)

pytestmark = pytest.mark.cpu


class Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def _ar_server(**config_overrides) -> vLLMOmniHttpServer:
    server = object.__new__(vLLMOmniHttpServer)
    server._ar_mode = True
    server.model_config = SimpleNamespace(processor=None)
    config = {
        "max_model_len": 32,
        "prompt_length": 16,
        "response_length": 16,
        "data_parallel_size": 1,
        "calculate_log_probs": True,
        "logprobs_mode": "raw_logprobs",
        "repetition_penalty": 1.0,
        "tensor_model_parallel_size": 2,
    }
    config.update(config_overrides)
    server.config = Config(config)
    return server


def test_ar_timeout_args_are_normalized_before_cli_parse():
    server = _ar_server()
    engine_kwargs = {
        "output_mode": "ar",
        "deploy_config": "/tmp/deploy.yaml",
        "stage_init_timeout": 900,
    }

    server._preprocess_engine_kwargs(engine_kwargs)

    assert engine_kwargs["deploy-config"] == "/tmp/deploy.yaml"
    assert engine_kwargs["stage-init-timeout"] == 900
    assert engine_kwargs["init-timeout"] == 900


def test_ar_adapter_generates_thinker_only_deploy_config(monkeypatch):
    server = _ar_server()
    server._rollout_flags = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    engine_kwargs = {
        "output_mode": "ar",
        "pipeline_name": "qwen3_omni_moe",
        "pipeline_mode": "thinker_only",
    }

    server._preprocess_engine_kwargs(engine_kwargs)

    with open(engine_kwargs["deploy-config"]) as f:
        deploy_config = yaml.safe_load(f)
    assert len(deploy_config["stages"]) == 1
    assert deploy_config["stages"][0]["devices"] == "0,1"
    assert deploy_config["stages"][0]["tensor_parallel_size"] == 2
    assert engine_kwargs["hf_overrides"]["enable_audio_output"] is False


def test_ar_model_config_always_uses_omni_contract(monkeypatch):
    server = object.__new__(vLLMOmniHttpServer)
    server.config = SimpleNamespace(engine_kwargs={"vllm_omni": {"output_mode": "ar"}})
    sentinel = object()
    calls = []

    def fake_omega_conf_to_dataclass(model_config, dataclass_type):
        calls.append((model_config, dataclass_type))
        return sentinel

    monkeypatch.setattr(
        "verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server.omega_conf_to_dataclass",
        fake_omega_conf_to_dataclass,
    )

    model_config = {"model_type": "language_model"}
    assert server._init_model_config(model_config) is sentinel
    assert calls == [(model_config, OmniModelConfig)]


def test_default_omni_megatron_config_uses_fully_async_omni_contract():
    config_dir = Path(__file__).resolve().parents[4] / "verl_omni" / "trainer" / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="omni_megatron_trainer")

    assert config.actor_rollout_ref.model._target_ == "verl_omni.workers.config.omni.OmniModelConfig"
    assert config.actor_rollout_ref.model.model_type == "language_model"
    assert config.actor_rollout_ref.actor.strategy == "megatron"
    assert config.actor_rollout_ref.ref.strategy == "megatron"
    assert config.actor_rollout_ref.hybrid_engine is False
    assert config.actor_rollout_ref.rollout.name == "vllm_omni"
    assert config.actor_rollout_ref.rollout.mode == "async"
    assert config.actor_rollout_ref.rollout.data_parallel_size == 1
    assert config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.output_mode == "ar"
    assert config.async_training.require_batches == 1
    assert config.data.continuous_token.enable is False
    assert config.reward.reward_manager.source == "register"
    assert config.reward.reward_manager.name == "naive"


def test_compilation_config_drops_nested_none_values():
    config = {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "pass_config": {
            "enable_qk_norm_rope_fusion": None,
            "fuse_allreduce_rms": False,
        },
        "compile_sizes": [],
    }

    assert _drop_none_mapping_values(config) == {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "pass_config": {"fuse_allreduce_rms": False},
        "compile_sizes": [],
    }


def test_ar_prompt_forwards_all_supported_multimodal_inputs():
    server = _ar_server()
    multimodal = server._build_multi_modal_data(
        image_data=["image"],
        video_data=["video"],
        audio_data=["audio"],
    )

    prompt, _params = server._preprocess_input(
        prompt_ids=[1, 2, 3],
        sampling_params={"logprobs": True},
        multi_modal_data=multimodal,
        lora_request=None,
        negative_prompt_ids=None,
        mm_processor_kwargs={"fps": 1},
    )

    assert prompt["multi_modal_data"] == {
        "image": ["image"],
        "video": ["video"],
        "audio": ["audio"],
    }
    assert prompt["mm_processor_kwargs"] == {"fps": 1}


def test_ar_server_selects_omni_worker_extension():
    server = _ar_server()

    assert server._get_worker_extension_cls().endswith(".vLLMOmniColocateWorkerExtension")


def test_ar_full_weight_update_uses_omni_bucketed_loader(monkeypatch):
    from verl.utils.vllm import patch as vllm_patch
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer
    from vllm.model_executor.model_loader import utils as model_loader_utils

    events = []

    class FakeModel:
        def load_weights(self, weights):
            events.append(("load", [name for name, _tensor in weights]))

    model = FakeModel()
    model_config = object()

    class FakeModelRunner:
        def get_model(self):
            return model

    model_runner = FakeModelRunner()
    model_runner.model_config = model_config

    class FakeReceiver:
        def __init__(self, **kwargs):
            assert kwargs["zmq_handle"] == "ipc:///test-ar-full-weight"
            assert kwargs["device"] == torch.device("cpu")
            assert kwargs["use_shm"] is True

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("layer.a", torch.ones(1))])
            on_bucket_received([("layer.b", torch.zeros(1))])

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    monkeypatch.setattr(
        vllm_patch,
        "patch_vllm_moe_model_weight_loader",
        lambda patched_model: events.append(("patch", patched_model)),
    )
    monkeypatch.setattr(
        model_loader_utils,
        "process_weights_after_loading",
        lambda processed_model, config, device: events.append(("postprocess", processed_model, config, device)),
    )

    worker = object.__new__(vLLMOmniColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.model_runner = model_runner
    worker._pending_lora_peft_config = None

    worker.update_weights_from_ipc(
        peft_config=None,
        base_sync_done=False,
        use_shm=True,
        zmq_handle="ipc:///test-ar-full-weight",
    )

    assert events == [
        ("patch", model),
        ("load", ["layer.a"]),
        ("load", ["layer.b"]),
        ("postprocess", model, model_config, torch.device("cpu")),
    ]


def test_diffusion_lora_update_accumulates_buckets_before_add(monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer

    class FakeReceiver:
        def __init__(self, **_kwargs):
            pass

        def receive_weights(self, on_bucket_received):
            on_bucket_received([("layer.a", torch.ones(1))])
            on_bucket_received([("layer.b", torch.zeros(1))])

    monkeypatch.setattr(bucketed_weight_transfer, "BucketedWeightReceiver", FakeReceiver)
    worker = object.__new__(vLLMOmniColocateWorkerExtension)
    worker.device = torch.device("cpu")
    worker.local_rank = 0
    worker.model_runner = None
    removed = []
    added = []
    worker.remove_lora = removed.append
    worker.add_lora = added.append

    worker.update_weights_from_ipc(peft_config={"r": 8}, base_sync_done=True)

    assert len(removed) == 1
    assert len(added) == 1
    assert set(added[0].lora_tensors) == {"layer.a", "layer.b"}


class FakeAsyncOmni:
    def __init__(self):
        self.request_states = {
            "internal-a": SimpleNamespace(request_id="internal-a", external_request_id="external-a"),
            "internal-b": SimpleNamespace(request_id="internal-b", external_request_id="external-b"),
        }
        self.output_processor = None
        self.calls = []

    async def pause_generation(self, **kwargs):
        self.calls.append(("pause", kwargs))

    async def abort(self, request_id):
        request_ids = [request_id] if isinstance(request_id, str) else list(request_id)
        self.calls.append(("abort_external", request_ids))
        for internal_id, state in list(self.request_states.items()):
            if state.external_request_id in request_ids:
                self.request_states.pop(internal_id)

    async def reset_prefix_cache(self, **kwargs):
        self.calls.append(("reset_prefix", kwargs))

    async def reset_mm_cache(self):
        self.calls.append(("reset_mm", None))

    async def reset_encoder_cache(self):
        self.calls.append(("reset_encoder", None))

    async def resume_generation(self):
        self.calls.append(("resume", None))


def test_abort_all_uses_async_omni_request_state_contract(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_ABORT_DRAIN_TIMEOUT_S", "0")
    server = _ar_server()
    server.node_rank = 0
    server.engine = FakeAsyncOmni()
    enqueued = []
    cache_clears = []

    server._enqueue_abort_output = lambda internal_id, state: enqueued.append((internal_id, state.external_request_id))

    async def clear_kv_cache():
        cache_clears.append(True)

    server.clear_kv_cache = clear_kv_cache

    result = asyncio.run(server.abort_all_requests())

    assert result == {
        "aborted_count": 2,
        "request_ids": ["external-a", "external-b"],
    }
    assert server.engine.calls == [
        ("abort_external", ["external-a", "external-b"]),
    ]
    assert enqueued == [("internal-a", "external-a"), ("internal-b", "external-b")]
    assert cache_clears == [True]


def test_abort_request_accepts_external_id_and_resume_uses_engine_api():
    server = _ar_server()
    server.node_rank = 0
    server.engine = FakeAsyncOmni()
    enqueued = []

    server._enqueue_abort_output = lambda internal_id, state: enqueued.append((internal_id, state.external_request_id))

    async def clear_kv_cache():
        return None

    server.clear_kv_cache = clear_kv_cache

    result = asyncio.run(server.abort_request("external-b"))
    asyncio.run(server.resume_generation())

    assert result == {"aborted": True, "request_id": "external-b"}
    assert ("abort_external", ["external-b"]) in server.engine.calls
    assert server.engine.calls[-1] == ("resume", None)
    assert enqueued == [("internal-b", "external-b")]
