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
import os
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from verl.workers.rollout.replica import RolloutMode

from verl_omni.utils.rl_insight import enable_rl_insight
from verl_omni.workers import engine_workers
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server, vllm_omni_strategy_base


class _TraceStrategy(vllm_omni_strategy_base.OmniStrategyBase):
    rollout_config_cls = object
    model_config_cls = object

    def worker_extension_cls(self, device_type: str) -> str:
        return ""

    def prepare_engine_args(self, engine_args: dict, args) -> None:
        pass

    def preprocess_input(self, *args, **kwargs):
        return "prompt", "params"

    async def run_generation(self, *args, **kwargs):
        return "final"

    def process_output(self, final_res, params, sampling_params):
        return final_res


@contextmanager
def _record_trace(events, state_name, *, state_lane_id):
    events.append(("enter", state_name, state_lane_id))
    yield
    events.append(("exit", state_name, state_lane_id))


@pytest.mark.parametrize("logger", ["rl_insight", ["console", "rl_insight"]])
def test_enable_rl_insight_accepts_string_or_list_logger(monkeypatch, logger):
    monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    config = SimpleNamespace(trainer={"logger": logger})

    enable_rl_insight(config)

    assert os.environ["VERL_RL_INSIGHT_ENABLE"] == "1"


@pytest.mark.parametrize("logger", [None, "console", ["console"]])
def test_enable_rl_insight_leaves_environment_unchanged_when_not_selected(monkeypatch, logger):
    monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    config = SimpleNamespace(trainer={"logger": logger})

    enable_rl_insight(config)

    assert "VERL_RL_INSIGHT_ENABLE" not in os.environ


@pytest.mark.parametrize("already_enabled", [False, True])
def test_enable_rl_insight_warns_when_ray_is_already_initialized(monkeypatch, caplog, already_enabled):
    if already_enabled:
        monkeypatch.setenv("VERL_RL_INSIGHT_ENABLE", "1")
    else:
        monkeypatch.delenv("VERL_RL_INSIGHT_ENABLE", raising=False)
    monkeypatch.setattr("verl_omni.utils.rl_insight.ray.is_initialized", lambda: True)
    config = SimpleNamespace(trainer={"logger": ["rl_insight"]})

    with caplog.at_level("WARNING"):
        enable_rl_insight(config)

    assert "enabled after Ray initialization" in caplog.text


def test_generation_trace_uses_replica_lane(monkeypatch):
    events = []

    async def resolve_lora_request():
        return None

    monkeypatch.setattr(
        vllm_omni_strategy_base.RLInsightLogger,
        "trace_state",
        lambda *args, **kwargs: _record_trace(events, *args, **kwargs),
    )
    strategy = _TraceStrategy(SimpleNamespace(replica_rank=3, _resolve_lora_request=resolve_lora_request))

    result = asyncio.run(strategy.generate(prompt_ids=[1, 2], sampling_params={}, request_id="request-id"))

    assert result == "final"
    assert events == [("enter", "vllm_generate", "replica_3"), ("exit", "vllm_generate", "replica_3")]


@pytest.mark.parametrize(("disable_log_stats", "expected"), [(False, True), (True, False)])
def test_run_server_forwards_log_stats_to_async_omni(monkeypatch, disable_log_stats, expected):
    @dataclass
    class EngineArgs:
        log_stats: bool = False
        enable_fault_tolerance: bool = False
        fault_tolerance_config: object | None = None
        seed: int | None = None

    class StopAfterEngineArgs(RuntimeError):
        pass

    captured = {}

    def capture_engine_args(**kwargs):
        captured.update(kwargs)
        raise StopAfterEngineArgs

    monkeypatch.setattr(vllm_omni_async_server.OmniEngineArgs, "from_cli_args", lambda args: EngineArgs())
    monkeypatch.setattr(vllm_omni_async_server, "orchestrator_field_names", lambda: set())
    monkeypatch.setattr(
        vllm_omni_async_server, "get_free_port", lambda *args, **kwargs: (12345, SimpleNamespace(close=lambda: None))
    )
    monkeypatch.setattr(vllm_omni_async_server, "AsyncOmni", capture_engine_args)
    server = object.__new__(vllm_omni_async_server.vLLMOmniHttpServer)
    server.config = SimpleNamespace(
        disable_log_stats=disable_log_stats,
        step_execution=False,
        rollout_attn_backend=None,
    )
    server._generate_strategy = SimpleNamespace(prepare_engine_args=lambda engine_args, args: None)

    with pytest.raises(StopAfterEngineArgs):
        asyncio.run(server.run_server(SimpleNamespace(deploy_config=None)))

    assert captured["log_stats"] is expected


@pytest.mark.parametrize(
    ("method_name", "state_name"),
    [
        ("sleep", "vllm_sleep"),
        ("wake_up", "vllm_wake_up"),
        ("release_kv_cache", "vllm_release_kv_cache"),
        ("resume_kv_cache", "vllm_resume_kv_cache"),
    ],
)
def test_lifecycle_trace_uses_replica_lane(monkeypatch, method_name, state_name):
    events = []

    async def succeed(*args, **kwargs):
        return [SimpleNamespace(status="SUCCESS")]

    async def clear_cache():
        pass

    engine = SimpleNamespace(
        sleep=succeed,
        wake_up=succeed,
        resume_generation=succeed,
        renderer=SimpleNamespace(clear_mm_cache_async=clear_cache),
    )
    server = object.__new__(vllm_omni_async_server.vLLMOmniHttpServer)
    server.engine = engine
    server.replica_rank = 3
    server.node_rank = 0
    server.rollout_mode = RolloutMode.HYBRID
    server.config = SimpleNamespace(free_cache_engine=True)
    server._lora_request_cache = None
    monkeypatch.setattr(
        vllm_omni_async_server.RLInsightLogger,
        "trace_state",
        lambda *args, **kwargs: _record_trace(events, *args, **kwargs),
    )

    asyncio.run(getattr(server, method_name)())

    assert events == [("enter", state_name, "replica_3"), ("exit", state_name, "replica_3")]


def test_weight_sync_trace_uses_actor_rank(monkeypatch):
    events = []

    class Engine:
        module = None

        def get_per_tensor_param(self, **kwargs):
            return [], None

    async def send_weights(weights):
        assert events == [("enter", "update_weights", "rank_5")]

    monkeypatch.setattr(
        engine_workers.RLInsightLogger,
        "trace_state",
        lambda *args, **kwargs: _record_trace(events, *args, **kwargs),
    )
    worker = object.__new__(engine_workers.ActorRolloutRefWorker)
    worker.rank = 5
    worker.config = SimpleNamespace(
        rollout=SimpleNamespace(
            checkpoint_engine=SimpleNamespace(backend="remote"),
            rollout_adapter="default",
        )
    )
    worker.actor = SimpleNamespace(engine=Engine())
    worker.peft_merge = False
    worker.checkpoint_engine = SimpleNamespace(send_weights=send_weights)

    asyncio.run(worker.update_weights(mode="remote"))

    assert events == [
        ("enter", "update_weights", "rank_5"),
        ("exit", "update_weights", "rank_5"),
    ]
