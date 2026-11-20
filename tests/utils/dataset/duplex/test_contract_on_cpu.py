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
"""Contract test: the trainer-side wire mirror vs the installed vLLM-Omni.

Runs only when the installed vLLM-Omni provides the duplex action-capture
surface (vllm-project/vllm-omni#10627); at the verl-omni pin it skips, and
the pin-compat story is issue #919's third PR. When it runs, it pins the
mirror to the real implementation: vocabulary, wire round-trip, and the
capture path's own validation semantics.
"""

import pytest

from verl_omni.utils.dataset.duplex import records as mirror

pytest.importorskip(
    "vllm_omni.protocol.duplex.action_records",
    reason="installed vLLM-Omni predates the action-capture surface (see verl-omni #919)",
)

from vllm_omni.protocol.duplex import action_records as upstream  # noqa: E402


def test_vocabulary_matches_upstream():
    assert mirror.ACTION_CAPTURE_RECORD_VERSION == upstream.ACTION_CAPTURE_RECORD_VERSION
    assert mirror.ACTION_KINDS == upstream.ACTION_KINDS
    assert mirror.RECORD_STATUSES == upstream.RECORD_STATUSES
    assert mirror.STATUS_IN_PROGRESS == upstream.STATUS_IN_PROGRESS
    assert mirror.INPUT_ORIGINS == upstream.INPUT_ORIGINS


def test_mirror_decodes_upstream_wire_payloads():
    wire = upstream.DuplexActionRecordSet(
        session_id="duplex-x",
        response_id="resp-1",
        epoch=2,
        turn_id=1,
        status="truncated_interruption",
        token_ids=(5, 6),
        logprobs=(-0.5, None),
        action_kinds=("sampled", "reemit"),
        reason_codes=("", "terminator_reemit"),
        stream_names=("content", "boundary"),
        unit_offsets=(0, 1, 2),
        stage_ids=(0, 0),
        input_modalities=("audio", "audio"),
        input_origins=("client", "engine"),
        emission_ns=(1, 2),
        input_commit_ns=(None, None),
        sampling_params=({"temperature": 0.7}, {"temperature": 0.7}),
        effective_seed=99,
        model={"minicpmo_duplex": {"stage0_prob_chunk_eos": 0.25}},
    ).to_wire()
    decoded = mirror.DuplexActionRecordSet.from_wire(wire)
    assert decoded.status == "truncated_interruption"
    assert decoded.token_ids == (5, 6)
    assert decoded.logprobs == (-0.5, None)
    assert [record.input_chunk.origin for record in decoded.records()] == ["client", "engine"]
    assert decoded.model["minicpmo_duplex"]["stage0_prob_chunk_eos"] == pytest.approx(0.25)
    # And back: the mirror's wire form must be upstream-decodable.
    roundtrip = upstream.DuplexActionRecordSet.from_wire(decoded.to_wire())
    assert roundtrip.token_ids == (5, 6)
    assert roundtrip.status == decoded.status


def test_mirror_rejects_what_upstream_rejects():
    wire = upstream.DuplexActionRecordSet(
        session_id="s",
        response_id=None,
        epoch=0,
        turn_id=0,
        status="completed",
        token_ids=(1,),
        logprobs=(float("nan"),),
        action_kinds=("sampled",),
        reason_codes=("",),
        stream_names=("content",),
        unit_offsets=(0, 1),
        stage_ids=(0,),
        input_modalities=("audio",),
        input_origins=("client",),
        input_commit_ns=(None,),
        emission_ns=(1,),
        sampling_params=({"temperature": 0.7},),
    ).to_wire()
    with pytest.raises(mirror.RecordSetError):
        mirror.DuplexActionRecordSet.from_wire(wire)


def test_duplex_engine_client_guard_accepts_this_build():
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

    assert require_duplex_vllm_omni() is DuplexOmni
