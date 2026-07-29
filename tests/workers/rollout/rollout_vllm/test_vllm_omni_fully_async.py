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
import yaml
from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

from verl_omni.workers.config import OmniModelConfig
from verl_omni.workers.rollout.vllm_rollout.placement_guard import (
    load_stage_placements,
    validate_vllm_omni_rollout_placement,
)
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


@pytest.mark.parametrize(
    ("body", "expected_devices"),
    [
        (
            """
stage_args:
  - stage_id: 0
    runtime:
      devices: "0,1"
    engine_args:
      tensor_parallel_size: 2
""",
            "0,1",
        ),
        (
            """
stages:
  - stage_id: 0
    devices: "0,1"
    tensor_parallel_size: 2
""",
            "0,1",
        ),
    ],
)
def test_load_stage_placements_supports_legacy_and_deploy_configs(tmp_path, body, expected_devices):
    path = tmp_path / "stage.yaml"
    path.write_text(body)

    stages = load_stage_placements(path)

    assert len(stages) == 1
    assert stages[0].devices == expected_devices
    assert stages[0].tensor_parallel_size == 2
    assert stages[0].num_replicas == 1


def test_placement_rejects_inner_replica_parallelism(tmp_path):
    path = tmp_path / "deploy.yaml"
    path.write_text(
        """
stages:
  - stage_id: 0
    devices: "0,1"
    num_replicas: 2
    tensor_parallel_size: 1
"""
    )

    with pytest.raises(ValueError, match="exactly one inner stage replica"):
        validate_vllm_omni_rollout_placement(
            config_path=path,
            visible_device_count=2,
        )


def test_placement_rejects_physical_device_ids(tmp_path):
    path = tmp_path / "stage.yaml"
    path.write_text(
        """
stage_args:
  - stage_id: 0
    runtime:
      devices: "2,3"
"""
    )

    with pytest.raises(ValueError, match="actor-local CUDA ids"):
        validate_vllm_omni_rollout_placement(
            config_path=path,
            visible_device_count=2,
        )


def test_ar_config_requires_single_dp_owner():
    server = _ar_server(data_parallel_size=2)

    with pytest.raises(ValueError, match="data_parallel_size=1"):
        server._validate_configs()


def test_ar_config_requires_explicit_supported_logprob_semantics():
    server = _ar_server(logprobs_mode="unsupported")

    with pytest.raises(ValueError, match="raw_logprobs or processed_logprobs"):
        server._validate_configs()


def test_ar_preflight_rejects_cross_node_replica():
    server = _ar_server()
    server.nnodes = 2

    with pytest.raises(ValueError, match="must fit on one node"):
        server._run_ar_placement_preflight(SimpleNamespace(deploy_config=None, stage_configs_path=None))


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


def test_megatron_fully_async_recipe_uses_omni_model_config():
    recipe_path = (
        Path(__file__).resolve().parents[4]
        / "examples"
        / "gspo_trainer"
        / "qwen3_omni"
        / "config"
        / "qwen3_omni_thinker_gspo_megatron_fully_async.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text())

    assert recipe["actor_rollout_ref"]["model"]["_target_"] == ("verl_omni.workers.config.omni.OmniModelConfig")
    assert recipe["actor_rollout_ref"]["actor"]["strategy"] == "megatron"
    assert recipe["actor_rollout_ref"]["ref"]["strategy"] == "megatron"
    assert recipe["actor_rollout_ref"]["rollout"]["name"] == "vllm_omni"
    assert recipe["actor_rollout_ref"]["rollout"]["mode"] == "async"


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


def test_ar_weight_update_reuses_verl_vllm_bounded_bucket_path(monkeypatch):
    worker = object.__new__(vLLMOmniColocateWorkerExtension)
    model = SimpleNamespace(load_weights=lambda _weights: None)
    worker.model_runner = SimpleNamespace(get_model=lambda: model, model_config=object())
    calls = []

    def fake_update(self, peft_config=None, base_sync_done=False, use_shm=False):
        calls.append((self, peft_config, base_sync_done, use_shm))
        return "updated"

    monkeypatch.setattr(vLLMColocateWorkerExtension, "update_weights_from_ipc", fake_update)

    result = worker.update_weights_from_ipc(peft_config=None, base_sync_done=False, use_shm=True)

    assert result == "updated"
    assert calls == [(worker, None, False, True)]


def test_ar_weight_update_disables_layerwise_reload():
    worker = object.__new__(vLLMOmniColocateWorkerExtension)

    assert worker._maybe_reload_standard_weights_from_ipc(receiver=object()) is False


class FakeAsyncOmni:
    def __init__(self):
        self.request_states = {
            "internal-a": SimpleNamespace(external_request_id="external-a"),
            "internal-b": SimpleNamespace(external_request_id="external-b"),
        }
        self.calls = []

    async def pause_generation(self, **kwargs):
        self.calls.append(("pause", kwargs))

    async def _abort_internal_requests(self, request_ids):
        ids = [request_ids] if isinstance(request_ids, str) else list(request_ids)
        self.calls.append(("abort_internal", ids))
        for request_id in ids:
            self.request_states.pop(request_id, None)

    async def abort(self, request_id):
        self.calls.append(("abort_external", request_id))
        for internal_id, state in list(self.request_states.items()):
            if state.external_request_id == request_id:
                self.request_states.pop(internal_id)

    async def reset_prefix_cache(self, **kwargs):
        self.calls.append(("reset_prefix", kwargs))

    async def reset_mm_cache(self):
        self.calls.append(("reset_mm", None))

    async def reset_encoder_cache(self):
        self.calls.append(("reset_encoder", None))

    async def resume_generation(self):
        self.calls.append(("resume", None))


def test_abort_all_pauses_then_aborts_and_clears_caches():
    server = _ar_server()
    server.node_rank = 0
    server.engine = FakeAsyncOmni()

    result = asyncio.run(server.abort_all_requests())

    assert result == {
        "aborted_count": 2,
        "request_ids": ["internal-a", "internal-b"],
    }
    assert server.engine.calls == [
        ("pause", {"wait_for_inflight_requests": False, "clear_cache": False}),
        ("abort_internal", ["internal-a", "internal-b"]),
        ("reset_prefix", {"reset_running_requests": True, "reset_connector": True}),
        ("reset_mm", None),
        ("reset_encoder", None),
    ]


def test_abort_request_accepts_external_id_and_resume_uses_engine_api():
    server = _ar_server()
    server.node_rank = 0
    server.engine = FakeAsyncOmni()

    result = asyncio.run(server.abort_request("external-b"))
    asyncio.run(server.resume_generation())

    assert result == {"aborted": True, "request_id": "external-b"}
    assert ("abort_external", "external-b") in server.engine.calls
    assert server.engine.calls[-1] == ("resume", None)
