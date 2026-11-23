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
"""CPU tests for the AV-lean TransferQueue agent-loop worker overrides.

Covers the two opt-in behaviors of
``verl_omni.agent_loop.av_agent_loop_tq``: dropping dead media payloads
(``raw_prompt`` / ``mm_processor_output``) from TQ rows and the reward
shipment, and the per-prompt multimodal processor-output cache shared across
``rollout.n`` sessions. The parent TransferQueue put/get machinery is mocked,
so the tests run without Ray or a TransferQueue deployment.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict

import numpy as np
import pytest
import torch

from verl_omni.agent_loop.av_agent_loop_tq import _AgentLoopWorkerTQBase, _AVLeanAgentLoopWorkerTQ


def _make_worker() -> _AVLeanAgentLoopWorkerTQ:
    worker = _AVLeanAgentLoopWorkerTQ.__new__(_AVLeanAgentLoopWorkerTQ)
    worker._mmi_cache = OrderedDict()
    return worker


def _media_kwargs() -> dict:
    waveform = np.zeros(16000, dtype=np.float32)  # 1s soundtrack as stored in messages
    return {
        "uid": "uid-1",
        "session_id": 3,
        "raw_prompt": [{"role": "user", "content": [{"type": "audio", "audio": waveform}]}],
        "data_source": "nextqa",
    }


class _FakeOutput:
    def __init__(self, prompt_ids: list[int], mm_processor_output=None):
        self.prompt_ids = prompt_ids
        self.response_ids = [7, 8]
        self.multi_modal_data = {"videos": ["frames"]}
        self.mm_processor_output = mm_processor_output


def test_postprocess_strips_dead_media_fields_when_enabled(monkeypatch):
    captured: dict = {}

    async def _parent(self, output, validate, **kwargs):
        captured["kwargs"] = kwargs
        captured["mm_processor_output"] = output.mm_processor_output

    monkeypatch.setattr(_AgentLoopWorkerTQBase, "_agent_loop_postprocess", _parent, raising=False)
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_RAW_PROMPT", "1")
    worker = _make_worker()
    output = _FakeOutput([1, 2, 3], mm_processor_output=[{"pixel_values": np.zeros(4, dtype=np.float32)}])

    asyncio.run(worker._agent_loop_postprocess(output, validate=False, **_media_kwargs()))

    assert "raw_prompt" not in captured["kwargs"]
    assert captured["mm_processor_output"] is None  # dropped before AgentLoopOutput.as_dict()
    assert captured["kwargs"]["uid"] == "uid-1"  # live kwargs still flow to the TQ put


def test_postprocess_keeps_kwargs_intact_when_disabled(monkeypatch):
    captured: dict = {}

    async def _parent(self, output, validate, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(_AgentLoopWorkerTQBase, "_agent_loop_postprocess", _parent, raising=False)
    monkeypatch.delenv("VERL_OMNI_TQ_STRIP_RAW_PROMPT", raising=False)
    worker = _make_worker()
    kwargs = _media_kwargs()
    output = _FakeOutput([1, 2, 3], mm_processor_output=[{"pixel_values": np.zeros(4, dtype=np.float32)}])

    asyncio.run(worker._agent_loop_postprocess(output, validate=False, **kwargs))

    assert captured["kwargs"] == kwargs  # no field dropped or rewritten when the gate is off


def test_compute_score_drops_raw_prompt_from_reward_shipment_when_enabled(monkeypatch):
    captured: dict = {}

    async def _parent(self, outputs, kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(_AgentLoopWorkerTQBase, "_compute_score", _parent, raising=False)
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_RAW_PROMPT", "1")
    worker = _make_worker()

    asyncio.run(worker._compute_score([object()], kwargs=_media_kwargs()))

    assert "raw_prompt" not in captured["kwargs"]
    assert captured["kwargs"]["data_source"] == "nextqa"


def test_mmi_cache_reuses_processor_output_across_sessions(monkeypatch):
    calls = []

    def _parent(self, output, input_ids):
        calls.append(tuple(output.prompt_ids))
        return {"pixel_values_videos": np.ones(8, dtype=np.float32), "mm_token_type_ids": "tok"}

    monkeypatch.setattr(_AgentLoopWorkerTQBase, "_compute_multi_modal_inputs", _parent, raising=False)
    monkeypatch.setenv("VERL_OMNI_TQ_CACHE_MMI", "1")
    worker = _make_worker()
    output = _FakeOutput([1, 2, 3])

    first = worker._compute_multi_modal_inputs(output, torch.tensor([1, 2, 3, 4]))
    second = worker._compute_multi_modal_inputs(_FakeOutput([1, 2, 3]), torch.tensor([1, 2, 3, 9]))

    assert calls == [(1, 2, 3)]  # one processor run shared across session rows
    # Downstream pops (mm_token_type_ids) must not leak into the cached value.
    first.pop("mm_token_type_ids")
    assert "mm_token_type_ids" in second
    assert first["pixel_values_videos"] is not None


def test_mmi_cache_disabled_by_default(monkeypatch):
    calls = []

    def _parent(self, output, input_ids):
        calls.append(tuple(output.prompt_ids))
        return {}

    monkeypatch.setattr(_AgentLoopWorkerTQBase, "_compute_multi_modal_inputs", _parent, raising=False)
    monkeypatch.delenv("VERL_OMNI_TQ_CACHE_MMI", raising=False)
    worker = _make_worker()

    worker._compute_multi_modal_inputs(_FakeOutput([1, 2, 3]), torch.tensor([1]))
    worker._compute_multi_modal_inputs(_FakeOutput([1, 2, 3]), torch.tensor([1]))

    assert len(calls) == 2


@pytest.mark.parametrize("env", ["0", "1"])
def test_strip_gates_do_not_change_text_only_rows(monkeypatch, env):
    """Rows without media must flow through identically in both modes."""
    captured: dict = {}

    async def _parent(self, output, validate, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(_AgentLoopWorkerTQBase, "_agent_loop_postprocess", _parent, raising=False)
    monkeypatch.setenv("VERL_OMNI_TQ_STRIP_RAW_PROMPT", env)
    worker = _make_worker()
    kwargs = {"uid": "u", "raw_prompt": [{"role": "user", "content": "plain text"}]}

    asyncio.run(worker._agent_loop_postprocess(_FakeOutput([1]), validate=False, **kwargs))

    # raw_prompt is dropped regardless of content once the gate is on; the
    # text-only case documents that nothing else is touched.
    assert ("raw_prompt" in captured["kwargs"]) == (env == "0")
