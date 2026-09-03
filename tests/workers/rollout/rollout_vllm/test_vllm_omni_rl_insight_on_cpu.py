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
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from verl_omni.workers import engine_workers
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_async_server, vllm_omni_strategy_base


class TraceStrategy(vllm_omni_strategy_base.OmniStrategyBase):
    rollout_config_cls = object
    model_config_cls = object

    def worker_extension_cls(self, device_type: str) -> str:
        return ""

    def prepare_engine_args(self, engine_args: dict, args) -> None:
        pass

    def preprocess_input(
        self,
        prompt_ids,
        sampling_params,
        multi_modal_data,
        lora_request,
        negative_prompt_ids,
        prompt_mask=None,
        mm_processor_kwargs=None,
        extra_prompt_ids=None,
        negative_extra_prompt_ids=None,
    ):
        return "prompt", "params"

    async def run_generation(self, prompt, params, request_id, lora_request, priority):
        assert self.events == [("enter", "vllm_generate", "replica_3")]
        await asyncio.sleep(0)
        return "final"

    def process_output(self, final_res, params, sampling_params):
        return final_res


def test_generation_trace_covers_engine_generation(monkeypatch):
    events = []

    @contextmanager
    def trace_state(state_name, *, state_lane_id):
        events.append(("enter", state_name, state_lane_id))
        yield
        events.append(("exit", state_name, state_lane_id))

    async def resolve_lora_request():
        return None

    monkeypatch.setattr(vllm_omni_strategy_base.RLInsightLogger, "trace_state", trace_state)
    server = SimpleNamespace(replica_rank=3, _resolve_lora_request=resolve_lora_request)
    strategy = TraceStrategy(server)
    strategy.events = events

    result = asyncio.run(
        strategy.generate(
            prompt_ids=[1, 2],
            sampling_params={},
            request_id="request-id",
        )
    )

    assert result == "final"
    assert events == [
        ("enter", "vllm_generate", "replica_3"),
        ("exit", "vllm_generate", "replica_3"),
    ]


@pytest.mark.parametrize(
    ("disable_log_stats", "expected_log_stats"),
    [(False, True), (True, False)],
)
def test_run_server_forwards_log_stats_to_async_omni(monkeypatch, disable_log_stats, expected_log_stats):
    @dataclass
    class EngineArgs:
        log_stats: bool = False
        enable_fault_tolerance: bool = False
        fault_tolerance_config: object | None = None
        seed: int | None = None

    class StopAfterEngineArgs(RuntimeError):
        pass

    class Socket:
        def close(self):
            pass

    captured = {}

    def capture_engine_args(**kwargs):
        captured.update(kwargs)
        raise StopAfterEngineArgs

    monkeypatch.setattr(vllm_omni_async_server.OmniEngineArgs, "from_cli_args", lambda args: EngineArgs())
    monkeypatch.setattr(vllm_omni_async_server, "orchestrator_field_names", lambda: set())
    monkeypatch.setattr(vllm_omni_async_server, "get_free_port", lambda *args, **kwargs: (12345, Socket()))
    monkeypatch.setattr(vllm_omni_async_server, "AsyncOmni", capture_engine_args)

    server = object.__new__(vllm_omni_async_server.vLLMOmniHttpServer)
    server.config = SimpleNamespace(
        disable_log_stats=disable_log_stats,
        step_execution=False,
        rollout_attn_backend=None,
    )
    server._generate_strategy = SimpleNamespace(prepare_engine_args=lambda engine_args, args: None)
    args = SimpleNamespace(deploy_config=None)

    with pytest.raises(StopAfterEngineArgs):
        asyncio.run(server.run_server(args))

    assert captured["log_stats"] is expected_log_stats


def test_async_worker_state_trace_covers_weight_sync(monkeypatch):
    events = []

    @contextmanager
    def trace_state(state_name, *, state_lane_id):
        events.append(("enter", state_name, state_lane_id))
        yield
        events.append(("exit", state_name, state_lane_id))

    class Engine:
        module = None

        def get_per_tensor_param(self, **kwargs):
            return [], None

    class CheckpointEngine:
        async def send_weights(self, weights):
            assert events == [("enter", "update_weights", "rank_5")]

    monkeypatch.setattr(engine_workers.RLInsightLogger, "trace_state", trace_state)
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
    worker.checkpoint_engine = CheckpointEngine()

    asyncio.run(worker.update_weights(mode="remote"))

    assert events == [
        ("enter", "update_weights", "rank_5"),
        ("exit", "update_weights", "rank_5"),
    ]
