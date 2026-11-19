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
"""CPU tests for DuplexTrajectory assembly over synthetic engine scopes."""

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    CHUNK_EOS_TOKEN_ID,
    LISTEN_TOKEN_ID,
    SPEAK_CONTENT_TOKEN_ID,
    listening_then_speaking_wires,
    record_set_wire,
    reemit_variant_wires,
)
from verl_omni.utils.dataset.duplex.records import RecordSetError
from verl_omni.utils.dataset.duplex.trajectory import (
    AXIS_INTERRUPTION,
    AXIS_PAUSE,
    AXIS_TURN,
    DuplexEpisodeEvent,
    DuplexTrajectory,
)


def _chunks(count: int, samples: int = 16000) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.standard_normal(samples).astype(np.float32) for _ in range(count)]


def _pause_event() -> DuplexEpisodeEvent:
    return DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(1.0, 3.0))


def _turn_event() -> DuplexEpisodeEvent:
    return DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=2.0)


def _interruption_event() -> DuplexEpisodeEvent:
    return DuplexEpisodeEvent(axis=AXIS_INTERRUPTION, interrupt_speech_end_s=4.0)


def test_assemble_flattens_scopes_in_order_and_masks_forced_tokens():
    wires = listening_then_speaking_wires(engine_silence_units=1)
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_pause_event(),
        record_set_wires=wires,
        client_audio_chunks=_chunks(5),  # 2 listen + 2 speak + terminator are client-fed; the engine unit is not
    )
    # Listening scope: listen, listen, forced engine listen; response: speak, speak, chunk_eos.
    assert trajectory.token_ids == (
        LISTEN_TOKEN_ID,
        LISTEN_TOKEN_ID,
        LISTEN_TOKEN_ID,
        SPEAK_CONTENT_TOKEN_ID,
        SPEAK_CONTENT_TOKEN_ID,
        CHUNK_EOS_TOKEN_ID,
    )
    assert trajectory.action_mask == (1, 1, 0, 1, 1, 1)
    assert trajectory.action_kinds.count("forced") == 1
    assert trajectory.token_scope_indices == (0, 0, 0, 1, 1, 1)
    # Cumulative unit indices continue across the scope boundary.
    assert trajectory.token_unit_indices == (0, 1, 2, 3, 4, 5)
    assert trajectory.record_sets[0].num_units() == 3
    assert trajectory.status == "completed"
    assert not trajectory.barge_in


def test_unit_decisions_timeline_uses_emission_clock():
    wires = listening_then_speaking_wires()
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_turn_event(),
        record_set_wires=wires,
        client_audio_chunks=_chunks(5),
    )
    decisions = trajectory.unit_decisions(listen_token_id=LISTEN_TOKEN_ID)
    assert [decision.speaking for decision in decisions] == [False, False, True, True, True]
    assert [decision.origin for decision in decisions] == ["client"] * 5
    # Emission timestamps start at the first unit and advance at the unit cadence.
    assert decisions[0].time_s == pytest.approx(0.0)
    assert decisions[3].time_s == pytest.approx(3.0)
    # The speech onset the turn-latency reward measures.
    first_speak = next(decision for decision in decisions if decision.speaking)
    assert first_speak.time_s == pytest.approx(2.0)


def test_barge_in_episode_flags_truncated_interruption():
    wires = listening_then_speaking_wires(barge_in=True)
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_interruption_event(),
        record_set_wires=wires,
        client_audio_chunks=_chunks(5),
    )
    assert trajectory.barge_in
    assert trajectory.status == "truncated_interruption"


def test_client_chunk_misalignment_is_a_rollout_failure():
    wires = listening_then_speaking_wires()
    with pytest.raises(RecordSetError, match="client chunks"):
        DuplexTrajectory.assemble(
            session_id="duplex-abc",
            seed=7,
            event=_pause_event(),
            record_set_wires=wires,
            client_audio_chunks=_chunks(3),
        )


def test_engine_silent_units_replay_but_do_not_consume_client_chunks():
    wires = listening_then_speaking_wires(engine_silence_units=2)
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_pause_event(),
        record_set_wires=wires,
        client_audio_chunks=_chunks(5),
    )
    decisions = trajectory.unit_decisions(listen_token_id=LISTEN_TOKEN_ID)
    engine_units = [decision for decision in decisions if decision.origin == "engine"]
    assert len(engine_units) == 2
    assert all(not decision.speaking for decision in engine_units)


def test_multi_action_units_are_rejected_loudly():
    wire = record_set_wire(
        units=[{"action": (LISTEN_TOKEN_ID, "sampled", -0.1)}],
    )
    wire["token_ids"] = [LISTEN_TOKEN_ID, SPEAK_CONTENT_TOKEN_ID]
    wire["logprobs"] = [-0.1, -0.2]
    wire["action_kinds"] = ["sampled", "sampled"]
    wire["reason_codes"] = ["", ""]
    wire["stream_names"] = ["content", "content"]
    wire["unit_offsets"] = [0, 2]
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_pause_event(),
        record_set_wires=[wire],
        client_audio_chunks=_chunks(1),
    )
    with pytest.raises(RecordSetError, match="one-decision-per-unit"):
        trajectory.unit_decisions(listen_token_id=LISTEN_TOKEN_ID)


@pytest.mark.parametrize(
    ("axis", "kwargs"),
    [
        (AXIS_PAUSE, {}),
        (AXIS_TURN, {}),
        (AXIS_INTERRUPTION, {}),
    ],
)
def test_event_annotation_requires_its_axis_window(axis, kwargs):
    with pytest.raises(ValueError, match="require"):
        DuplexEpisodeEvent(axis=axis, **kwargs)


def test_conflicting_seeds_across_scopes_are_rejected():
    listening, speaking = listening_then_speaking_wires()
    speaking["effective_seed"] = 8
    with pytest.raises(RecordSetError, match="effective seeds"):
        DuplexTrajectory.assemble(
            session_id="duplex-abc",
            seed=7,
            event=_turn_event(),
            record_set_wires=[listening, speaking],
            client_audio_chunks=_chunks(4),
        )


def test_reemit_rows_are_restatements_not_decisions():
    wires = reemit_variant_wires()
    # Six flat tokens (two of them reemit restatements), four client chunks:
    # the reemit rows consumed no fed audio.
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_pause_event(),
        record_set_wires=wires,
        client_audio_chunks=_chunks(4),
    )
    assert len(trajectory.token_ids) == 6
    assert trajectory.action_mask == (1, 1, 0, 1, 1, 0)
    decisions = trajectory.unit_decisions(listen_token_id=LISTEN_TOKEN_ID)
    assert len(decisions) == 4  # only the decision units open unit windows
    assert all(decision.action_token_id != CHUNK_EOS_TOKEN_ID or decision.speaking for decision in decisions)
