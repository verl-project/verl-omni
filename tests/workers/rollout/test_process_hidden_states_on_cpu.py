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
"""CPU tests for the vllm-omni hidden-states monkey-patches.

Exercises the runner-side payload collection and the client-side extraction
against duck-typed stand-ins, without a real vllm-omni engine. The flag
transport (SamplingParams.extra_args) and the multimodal_output channel
semantics are covered by assertion on the shapes flowing through
``_collect_hidden_states_payloads`` / ``_merge_into_multimodal_outputs`` /
``extract_hidden_states``.
"""

from types import SimpleNamespace

import pytest
import torch

from verl_omni.workers.rollout.vllm_rollout.process_hidden_states import (
    RETURN_FLAG_KEY,
    TEACHER_HIDDEN_STATES_KEY,
    _collect_hidden_states_payloads,
    _merge_into_multimodal_outputs,
    extract_hidden_states,
    request_wants_hidden_states,
)


def _make_request(extra_args=None, prompt_len=5, num_computed_tokens=0):
    return SimpleNamespace(
        prompt_token_ids=list(range(prompt_len)),
        num_computed_tokens=num_computed_tokens,
        sampling_params=SimpleNamespace(extra_args=extra_args),
    )


class _FakeRunner:
    def __init__(self, requests, query_start_loc, req_id_to_index):
        self.requests = requests
        self._query_start_loc = query_start_loc
        self.input_batch = SimpleNamespace(req_id_to_index=req_id_to_index)

    def _snapshot_query_start_loc_cpu(self):
        return self._query_start_loc


def _make_runner(hidden_len=6, d=4):
    reqs = {"r0": _make_request(extra_args={RETURN_FLAG_KEY: True})}
    qsl = torch.tensor([0, 5])  # r0 occupies rows [0, 5)
    runner = _FakeRunner(reqs, qsl, {"r0": 0})
    hidden = torch.randn(hidden_len, d)
    sched = SimpleNamespace(num_scheduled_tokens={"r0": 5})
    return runner, sched, hidden


def test_request_wants_hidden_states_flag():
    runner, _, _ = _make_runner()
    assert request_wants_hidden_states(runner, "r0") is True
    assert request_wants_hidden_states(runner, "missing") is False


def test_collect_slices_full_prefill_hidden():
    runner, sched, hidden = _make_runner()
    payloads = _collect_hidden_states_payloads(runner, sched, hidden)
    assert set(payloads) == {"r0"}
    h = payloads["r0"][TEACHER_HIDDEN_STATES_KEY]
    assert torch.equal(h, hidden[:5])
    assert h.device.type == "cpu"


def test_collect_skips_requests_without_flag():
    runner, sched, hidden = _make_runner()
    runner.requests["r0"] = _make_request(extra_args=None)
    assert _collect_hidden_states_payloads(runner, sched, hidden) == {}


def test_collect_raises_on_chunked_prefill():
    runner, sched, hidden = _make_runner()
    sched.num_scheduled_tokens = {"r0": 3}  # 2 prompt tokens left unscheduled
    with pytest.raises(RuntimeError, match="chunked prefill"):
        _collect_hidden_states_payloads(runner, sched, hidden)


def test_merge_into_empty_multimodal_outputs():
    output = SimpleNamespace(req_ids=["r0", "r1"], req_id_to_index={"r0": 0, "r1": 1}, multimodal_outputs=None)
    payloads = {"r0": {TEACHER_HIDDEN_STATES_KEY: torch.zeros(5, 4)}}
    _merge_into_multimodal_outputs(output, payloads)
    assert len(output.multimodal_outputs) == 2
    assert output.multimodal_outputs[0][TEACHER_HIDDEN_STATES_KEY].shape == (5, 4)
    assert output.multimodal_outputs[1] is None


def test_merge_preserves_existing_entry():
    existing = {"audio": torch.zeros(3)}
    output = SimpleNamespace(req_ids=["r0"], req_id_to_index={"r0": 0}, multimodal_outputs=[dict(existing)])
    payloads = {"r0": {TEACHER_HIDDEN_STATES_KEY: torch.ones(5, 4)}}
    _merge_into_multimodal_outputs(output, payloads)
    entry = output.multimodal_outputs[0]
    assert "audio" in entry and TEACHER_HIDDEN_STATES_KEY in entry
    # The original dict passed in was not mutated in place.
    assert TEACHER_HIDDEN_STATES_KEY not in existing


def test_extract_hidden_states_from_completion():
    hidden = torch.randn(5, 4)

    class _MM(dict):
        pass

    class _Completion(SimpleNamespace):
        pass

    completion = _Completion(multimodal_output=_MM({TEACHER_HIDDEN_STATES_KEY: hidden}))
    req_output = SimpleNamespace(outputs=[completion])
    assert torch.equal(extract_hidden_states(req_output), hidden)


def test_extract_hidden_states_none_cases():
    assert extract_hidden_states(SimpleNamespace(outputs=[])) is None
    assert extract_hidden_states(SimpleNamespace(outputs=[SimpleNamespace()])) is None
    assert extract_hidden_states(SimpleNamespace(outputs=[SimpleNamespace(multimodal_output=None)])) is None
    assert extract_hidden_states(SimpleNamespace(outputs=[SimpleNamespace(multimodal_output={})])) is None
