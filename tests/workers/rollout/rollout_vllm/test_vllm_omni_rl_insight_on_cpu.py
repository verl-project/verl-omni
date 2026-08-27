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
from types import SimpleNamespace

from verl_omni.workers.rollout.vllm_rollout import vllm_omni_strategy_base


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
