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
"""CPU tests for the trainer-side record-set wire mirror."""

import math

import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    CHUNK_EOS_TOKEN_ID,
    LISTEN_TOKEN_ID,
    record_set_wire,
    unit_row,
)
from verl_omni.utils.dataset.duplex.records import (
    ACTION_FORCED,
    ACTION_REEMIT,
    ACTION_SAMPLED,
    STATUS_IN_PROGRESS,
    DuplexActionRecordSet,
    RecordSetError,
    validate_record_sets,
)


def test_from_wire_roundtrip_preserves_columns():
    wire = record_set_wire()
    record_set = DuplexActionRecordSet.from_wire(wire)
    assert record_set.session_id == "duplex-abc"
    assert record_set.response_id == "resp-1"
    assert record_set.status == "completed"
    assert record_set.sealed and not record_set.truncated
    assert record_set.num_units() == 0 and record_set.num_tokens() == 0
    assert record_set.to_wire() == wire


def test_forced_kinds_and_origins_pass_validation():
    wire = record_set_wire(
        units=[
            {"action": (LISTEN_TOKEN_ID, ACTION_FORCED, None), "reason": "silent_unit_listen", "origin": "engine"},
            {"action": (CHUNK_EOS_TOKEN_ID, ACTION_FORCED, None), "reason": "chunk_token_cap"},
            {"action": (CHUNK_EOS_TOKEN_ID, ACTION_REEMIT, None), "reason": "terminator_reemit"},
        ]
    )
    record_set = DuplexActionRecordSet.from_wire(wire)
    kinds = [action.action_kind for record in record_set.records() for action in record.actions]
    assert kinds == [ACTION_FORCED, ACTION_FORCED, ACTION_REEMIT]
    origins = [record.input_chunk.origin for record in record_set.records()]
    assert origins == ["engine", "client", "client"]


def test_records_view_splits_units_at_offsets():
    wire = record_set_wire(
        units=[
            {"action": (LISTEN_TOKEN_ID, ACTION_SAMPLED, -0.1)},
            {"action": (CHUNK_EOS_TOKEN_ID, ACTION_SAMPLED, -0.2)},
        ]
    )
    record_set = DuplexActionRecordSet.from_wire(wire)
    records = list(record_set.records())
    assert [record.unit_index for record in records] == [0, 1]
    assert all(len(record.actions) == 1 for record in records)
    assert records[0].actions[0].token_id == LISTEN_TOKEN_ID
    assert records[1].actions[0].token_id == CHUNK_EOS_TOKEN_ID
    assert records[0].sampling == {"temperature": 0.7, "top_k": 100, "top_p": 0.8}


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda w: w.update(schema_version=2), id="schema-version"),
        pytest.param(lambda w: w.update(status="sealed"), id="bad-status"),
        pytest.param(lambda w: w.update(action_kinds=["guessed"]), id="bad-kind"),
        pytest.param(lambda w: w.update(input_origins=["server"]), id="bad-origin"),
        pytest.param(lambda w: w.update(logprobs=[0.5, 0.6]), id="column-mismatch"),
        pytest.param(lambda w: w.update(unit_offsets=[0, 2]), id="offsets-past-end"),
        pytest.param(lambda w: w.update(unit_offsets=[1, 1]), id="offsets-not-from-zero"),
        pytest.param(lambda w: w.update(stage_ids=[0, 0]), id="unit-columns-mismatch"),
        pytest.param(lambda w: w.update(token_ids=[True]), id="bool-token-id"),
        pytest.param(lambda w: w.pop("session_id"), id="missing-session"),
        pytest.param(lambda w: w.update(effective_seed="seven"), id="bad-seed"),
        pytest.param(lambda w: w.update(model=["minicpmo"]), id="bad-model-extension"),
    ],
)
def test_contract_violations_are_rejected(mutate):
    wire = record_set_wire(units=[{"action": (LISTEN_TOKEN_ID, ACTION_SAMPLED, -0.1)}])
    mutate(wire)
    with pytest.raises(RecordSetError):
        DuplexActionRecordSet.from_wire(wire)


def test_sampled_rows_require_finite_logprobs():
    wire = record_set_wire(
        units=[
            {"action": (LISTEN_TOKEN_ID, ACTION_SAMPLED, None)},
        ]
    )
    with pytest.raises(RecordSetError, match="non-finite logprob"):
        DuplexActionRecordSet.from_wire(wire)
    wire = record_set_wire(units=[{"action": (LISTEN_TOKEN_ID, ACTION_SAMPLED, float("nan"))}])
    with pytest.raises(RecordSetError, match="non-finite logprob"):
        DuplexActionRecordSet.from_wire(wire)


def test_forced_rows_may_carry_a_draw_logprob():
    # The mid-turn listen rewrite is forced but sampled under the policy
    # distribution; the draw's logprob must survive validation.
    wire = record_set_wire(units=[{"action": (LISTEN_TOKEN_ID, ACTION_FORCED, -0.9)}])
    record_set = DuplexActionRecordSet.from_wire(wire)
    assert math.isfinite(record_set.logprobs[0])


def test_in_progress_snapshot_is_decodable_but_flagged():
    wire = record_set_wire(status=STATUS_IN_PROGRESS, units=[{"action": (LISTEN_TOKEN_ID, ACTION_SAMPLED, -0.1)}])
    record_set = DuplexActionRecordSet.from_wire(wire)
    assert not record_set.sealed


def test_cross_set_validation_rejects_mixed_sessions_and_live_snapshots():
    listening = record_set_wire(response_id=None, session_id="duplex-one", status="truncated_close")
    speaking = record_set_wire(response_id="resp-1", session_id="duplex-two", status="completed")
    with pytest.raises(RecordSetError, match="session"):
        validate_record_sets(
            [DuplexActionRecordSet.from_wire(listening), DuplexActionRecordSet.from_wire(speaking)],
            session_id="duplex-one",
        )
    live = record_set_wire(status=STATUS_IN_PROGRESS)
    with pytest.raises(RecordSetError, match="in_progress"):
        validate_record_sets([DuplexActionRecordSet.from_wire(live)], session_id="duplex-abc")


def test_unit_row_helper_is_contract_shaped():
    # Guards the fixture helper itself: origins and emission clocks must be
    # the shapes the engine emits, or every downstream test is testing fiction.
    row = unit_row(origin="engine")
    assert row["origin"] == "engine"
    assert row["emission_ns"] == 1_000_000_000
    wire = record_set_wire(units=[{"action": (LISTEN_TOKEN_ID, ACTION_SAMPLED, -0.1), **row}])
    record_set = DuplexActionRecordSet.from_wire(wire)
    assert list(record_set.records())[0].input_chunk.commit_ns == 990_000_000
