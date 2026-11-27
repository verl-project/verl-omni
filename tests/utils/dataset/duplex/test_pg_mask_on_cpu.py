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
"""CPU tests for the replay's policy-gradient gate (``pg_mask``)."""

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    CHUNK_EOS_TOKEN_ID,
    SPEAK_CONTENT_TOKEN_ID,
    listen_unit,
    record_set_wire,
    spoken_unit,
)
from tests.utils.dataset.duplex.test_replay_on_cpu import FakeTokenizer
from verl_omni.utils.dataset.duplex.replay import (
    DuplexReplay,
    build_replay,
    resolve_duplex_vocabulary,
)
from verl_omni.utils.dataset.duplex.telemetry import pause_window_overlaps_unit
from verl_omni.utils.dataset.duplex.trajectory import (
    AXIS_INTERRUPTION,
    AXIS_PAUSE,
    AXIS_TURN,
    DuplexEpisodeEvent,
    DuplexTrajectory,
)

_UNIT_NS = 1_000_000_000


def _replay(
    event: DuplexEpisodeEvent,
    *,
    engine_silence_at: int | None = None,
    unit_times: tuple[float, ...] = (0, 1, 2, 3, 4),
) -> DuplexReplay:
    """Replay units listening at the first three ``unit_times`` and speaking at the last two (one scope).

    ``engine_silence_at`` appends an engine-injected forced listen unit at that time.
    """
    units = [listen_unit(emission_ns=int((time_s + 1) * _UNIT_NS)) for time_s in unit_times[:3]]
    units += [spoken_unit(emission_ns=int((time_s + 1) * _UNIT_NS)) for time_s in unit_times[3:]]
    if engine_silence_at is not None:
        units.append(
            listen_unit(
                origin="engine",
                kind="forced",
                reason="force_listen",
                emission_ns=(engine_silence_at + 1) * _UNIT_NS,
            )
        )
    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=event,
        record_set_wires=[record_set_wire(units=units)],
        client_audio_chunks=[np.zeros(16000, dtype=np.float32) for _ in range(5)],
    )
    return build_replay(trajectory, resolve_duplex_vocabulary(FakeTokenizer()))


def _unit_positions(replay: DuplexReplay, time_s: float) -> list[int]:
    return [
        position for layout in replay.unit_layouts if layout.time_s == time_s for position in layout.action_positions
    ]


def test_pause_episode_keeps_only_the_units_overlapping_the_window():
    replay = _replay(DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(2.0, 3.0)))
    assert [layout.time_s for layout in replay.unit_layouts] == [0.0, 1.0, 2.0, 3.0, 4.0]
    # The two listen units that ended before the window opened and the two speak
    # units that start after it closed are gated; the unit covering the window keeps its token.
    gated = [position for time_s in (0.0, 1.0, 3.0, 4.0) for position in _unit_positions(replay, time_s)]
    kept = _unit_positions(replay, 2.0)
    assert all(replay.response_mask[position] == 1 and replay.pg_mask[position] == 0 for position in gated)
    assert [replay.pg_mask[position] for position in kept] == [1]
    assert sum(replay.response_mask) == 9 and sum(replay.pg_mask) == 1
    # A window straddling two units keeps both.
    later = _replay(DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(2.5, 3.5)))
    assert sum(later.pg_mask) == 1 + 3  # the listen unit at 2 and the first spoken unit
    assert sum(_replay(DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(3.5, 4.5))).pg_mask) == 6


def test_pause_window_inside_a_unit_keeps_the_covering_unit_gradient():
    # No unit starts inside [2.2, 2.8), but the unit at t = 2 covers it: it decides the reward.
    replay = _replay(DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(2.2, 2.8)))
    assert [replay.pg_mask[position] for position in _unit_positions(replay, 2.0)] == [1]
    assert sum(replay.pg_mask) == 1


@pytest.mark.parametrize(
    "window",
    [(0.0, 1.0), (0.4, 0.6), (1.5, 2.5), (2.0, 3.0), (2.9, 4.2), (3.2, 3.8), (4.0, 6.0)],
)
def test_pause_gate_keeps_exactly_the_units_the_reward_measures(window):
    event = DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=window)
    replay = _replay(event)
    for layout in replay.unit_layouts:
        kept = any(replay.pg_mask[position] for position in layout.action_positions)
        assert kept == pause_window_overlaps_unit(event, unit_times=[layout.time_s])


def test_pause_window_in_a_unit_gap_gates_every_unit():
    # Units cover [0, 1), [1, 2), [2, 3), [4, 5), [5, 6); the window [3.2, 3.8) falls between them.
    event = DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(3.2, 3.8))
    replay = _replay(event, unit_times=(0, 1, 2, 4, 5))
    assert sum(replay.pg_mask) == 0
    assert sum(replay.response_mask) > 0


def test_interruption_episode_gates_units_before_the_speech_onset():
    replay = _replay(
        DuplexEpisodeEvent(axis=AXIS_INTERRUPTION, interrupt_speech_onset_s=3.0, interrupt_speech_end_s=4.0)
    )
    gated = [position for time_s in (0.0, 1.0, 2.0) for position in _unit_positions(replay, time_s)]
    kept = [position for layout in replay.unit_layouts[3:] for position in layout.action_positions]
    assert all(replay.response_mask[position] == 1 and replay.pg_mask[position] == 0 for position in gated)
    # From the onset on, the full mask: the yield decision is what the episode scores.
    assert [replay.pg_mask[position] for position in kept] == [1, 1, 1, 1, 1, 1]
    assert [replay.response_ids[position] for position in kept[:3]] == [SPEAK_CONTENT_TOKEN_ID] * 2 + [
        CHUNK_EOS_TOKEN_ID
    ]


def test_interruption_onset_inside_a_unit_keeps_the_covering_unit():
    replay = _replay(
        DuplexEpisodeEvent(axis=AXIS_INTERRUPTION, interrupt_speech_onset_s=2.5, interrupt_speech_end_s=4.0)
    )
    assert [replay.pg_mask[position] for position in _unit_positions(replay, 2.0)] == [1]
    assert all(replay.pg_mask[position] == 0 for time_s in (0.0, 1.0) for position in _unit_positions(replay, time_s))
    assert sum(replay.pg_mask) == 1 + 6


def test_turn_episode_keeps_the_full_mask():
    replay = _replay(DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=3.0))
    # An early floor-take is scored on this axis and sits before the turn end, so those units stay in the mask.
    assert replay.pg_mask == replay.response_mask


def test_framing_positions_and_forced_tokens_stay_zero_on_every_axis():
    events = [
        DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(0.0, 10.0)),
        DuplexEpisodeEvent(axis=AXIS_INTERRUPTION, interrupt_speech_onset_s=0.0, interrupt_speech_end_s=1.0),
        DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=0.0),
    ]
    for event in events:
        replay = _replay(event, engine_silence_at=5)
        assert [layout.time_s for layout in replay.unit_layouts] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        assert len(replay.pg_mask) == len(replay.response_mask) == len(replay.response_ids)
        # <unit>, its placeholders and </unit> are never loss positions, and the
        # engine-injected forced listen at t = 5 is masked out even ungated.
        assert all(pg <= mask for pg, mask in zip(replay.pg_mask, replay.response_mask, strict=True))
        assert replay.response_mask[_unit_positions(replay, 5.0)[0]] == 0
        assert replay.pg_mask[_unit_positions(replay, 5.0)[0]] == 0
        assert sum(replay.pg_mask) == 9  # nothing outside the scored window on any of the three events


def test_pg_mask_rides_the_payload_beside_response_mask():
    replay = _replay(DuplexEpisodeEvent(axis=AXIS_PAUSE, pause_window_s=(3.0, 4.0)))
    payload = replay.to_extra_fields(include_pcm=False)
    assert list(payload).index("pg_mask") == list(payload).index("response_mask") + 1
    assert payload["pg_mask"] == replay.pg_mask
    restored = DuplexReplay.from_extra_fields(payload)
    assert restored.pg_mask == replay.pg_mask
    assert restored.pg_mask != restored.response_mask
    del payload["pg_mask"]
    with pytest.raises(KeyError):
        DuplexReplay.from_extra_fields(payload)
