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
"""CPU tests for the duplex turn-taking rule reward scorer."""

import pytest

from tests.utils.dataset.duplex._duplex_fixtures import interruption_event
from verl_omni.utils.reward_score.duplex_turn_taking import compute_score_duplex_turn_taking

LISTEN_TOKEN_ID = 0


def _replay(axis: str, speaking, times=None, content_ids=None, origins=None, event=None):
    times = times or [float(index) for index in range(len(speaking))]
    event = (
        event
        or {
            "pause": {"axis": "pause", "pause_window_s": [1.0, 3.0]},
            "turn": {"axis": "turn", "user_turn_end_s": 2.0},
            "interruption": interruption_event(onset_s=2.0, end_s=5.0),
        }[axis]
    )
    # Mirror the real replay payload: every unit's action token rides
    # response_ids at the layout's action_position.
    tokens = [
        (content_ids[index] if content_ids else 700 + 2 * index) if speaks else LISTEN_TOKEN_ID
        for index, speaks in enumerate(speaking)
    ]
    return {
        "event": event,
        "response_ids": tokens,
        "unit_layouts": [
            {
                "time_s": time_s,
                "speaking": speaks,
                "action_position": index,
                "origin": origins[index] if origins else "client",
            }
            for index, (time_s, speaks) in enumerate(zip(times, speaking, strict=True))
        ],
    }


def _score(axis: str, speaking, times=None, event=None, origins=None, **kwargs):
    replay = _replay(axis, speaking, times, event=event, origins=origins)
    return compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={"duplex_replay": replay}, **kwargs)


def test_pause_axis_penalizes_speaking_in_the_window():
    violated = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("pause", [False, True, False])}
    )
    assert violated["score"] == -1.0
    assert violated["duplex_pause_violation"] == 1.0
    clean = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("pause", [False, False, False, True])}
    )
    assert clean["score"] == 0.0  # speaking starts at 3.0 s, past the [1, 3) window
    assert clean["duplex_pause_violation"] == 0.0


def test_turn_axis_full_credit_band_and_deadline():
    fast = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, False, True])}
    )
    assert fast["score"] == 0.0  # onset at 2.0 s, turn end 2.0 s -> inside the band
    mid = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, False, False, False, True])}
    )
    assert mid["score"] == pytest.approx(-0.5)  # onset 2 s after the turn end: half the deadline span
    never = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, False])}
    )
    assert never["score"] == -1.0
    assert never["duplex_turn_latency_s"] == float("inf")


def test_turn_axis_grades_speaking_before_the_turn_ends():
    # One second early uses the same 1 s band as a one-second-late take.
    early = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, True, False, True])}
    )
    assert early["score"] == 0.0
    assert early["duplex_turn_latency_s"] == pytest.approx(-1.0)
    assert early["duplex_turn_took_floor"] == 1.0
    # Two seconds early is halfway from the band to the 3 s deadline.
    earlier = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [True, False, False])}
    )
    assert earlier["score"] == pytest.approx(-0.5)
    assert earlier["duplex_turn_latency_s"] == pytest.approx(-2.0)


def test_interruption_axis_anchors_the_yield_latency_to_the_speech_onset():
    # The interrupting speech starts at 2.0 s; the policy speaks from t=0
    # (it is the one being interrupted).
    prompt = _score("interruption", [True, True, True, False])  # silent at 3.0: onset + 1.0
    assert prompt["score"] == 0.0  # inside the 1 s full-credit band
    assert prompt["duplex_yield_latency_s"] == pytest.approx(1.0)
    slow = _score("interruption", [True, True, True, True, False])  # silent at 4.0: onset + 2.0
    assert slow["score"] == pytest.approx(-0.5)  # halfway from the band to the 3 s deadline
    assert slow["duplex_yield_latency_s"] == pytest.approx(2.0)
    never = _score("interruption", [True] * 5)
    assert never["score"] == -1.0
    assert never["duplex_yield_latency_s"] == float("inf")
    # Talking the user down until their speech ends (4.5 s) and falling silent
    # right after: the onset anchor charges the whole 3.5 s overlap, past the
    # deadline.
    talks_over = _score("interruption", [True] * 5 + [False], event=interruption_event(onset_s=1.5, end_s=4.5))
    assert talks_over["score"] == -1.0
    assert talks_over["duplex_yield_latency_s"] == pytest.approx(3.5)


def test_interruption_axis_floors_a_policy_that_was_not_speaking_at_the_onset():
    # Never spoke, or first speaks strictly after the onset: not interrupted.
    silent = _score("interruption", [False, False, False, False])
    assert silent["score"] == -1.0
    assert silent["duplex_yield_latency_s"] == float("inf")
    late = _score("interruption", [False, False, False, True])  # first speaks at 3.0, onset is 2.0
    assert late["score"] == -1.0
    assert late["duplex_yield_latency_s"] == float("inf")
    # The onset unit itself is the injection point: speaking there is interrupted.
    on_onset = _score("interruption", [False, False, True, False])
    assert on_onset["score"] == 0.0
    assert on_onset["duplex_yield_latency_s"] == pytest.approx(1.0)


def test_pacing_corrupted_column_follows_the_engine_units():
    # The turn window is [2.0, 5.0): an engine-injected unit at t=2 corrupts
    # the episode, one at t=1 does not; the flag reports, the score stands.
    inside = _score("turn", [False, False, False, True], origins=["client", "client", "engine", "client"])
    assert inside["duplex_pacing_corrupted"] == 1.0
    assert inside["score"] == 0.0
    outside = _score("turn", [False, False, False, True], origins=["client", "engine", "client", "client"])
    assert outside["duplex_pacing_corrupted"] == 0.0
    assert outside["score"] == 0.0


def test_deadlines_forwarded_as_kwargs():
    result = compute_score_duplex_turn_taking(
        "dailytalk",
        "",
        "",
        extra_info={"duplex_replay": _replay("turn", [False, False, False, True])},
        turn_full_credit_band_s=0.0,
        turn_deadline_s=2.0,
    )
    assert result["score"] == pytest.approx(-0.5)
    result = _score("interruption", [True, True, True, True, False], yield_full_credit_band_s=0.0, yield_deadline_s=4.0)
    assert result["score"] == pytest.approx(-0.5)  # a 2 s latency over a 4 s deadline


def test_missing_replay_is_a_contract_error():
    with pytest.raises(ValueError, match="duplex_replay"):
        compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={})
    with pytest.raises(ValueError, match="no unit timeline"):
        compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={"duplex_replay": {"event": {"axis": "turn"}}})
    # An interruption replay without its onset anchor cannot be scored.
    with pytest.raises(ValueError, match="require interrupt_speech_onset_s"):
        compute_score_duplex_turn_taking(
            "dailytalk",
            "",
            "",
            extra_info={"duplex_replay": _replay("interruption", [True, False], event={"axis": "interruption"})},
        )


def test_trip_wire_columns_and_episode_failed_flag():
    repetitive = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("pause", [True] * 5, content_ids=[777] * 5)}
    )
    assert repetitive["duplex_speak_ratio"] == pytest.approx(1.0)
    assert repetitive["duplex_silence_share"] == pytest.approx(0.0)
    assert repetitive["duplex_duplicate_ngram_share"] == pytest.approx(1.0)
    assert repetitive["duplex_episode_failed"] == 1.0  # every wire breached
    clean = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, False, True])}
    )
    assert clean["duplex_speak_ratio"] == pytest.approx(1 / 3)
    assert clean["duplex_silence_share"] == pytest.approx(2 / 3)
    assert clean["duplex_duplicate_ngram_share"] == 0.0
    assert clean["duplex_episode_failed"] == 0.0
    # Only the speak content stream feeds the repetition check; listen tokens
    # (every unit carries one) must not manufacture repeated n-grams.
    listen_heavy = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, False, False, False])}
    )
    assert listen_heavy["duplex_duplicate_ngram_share"] == 0.0


def test_speaking_unit_without_a_resolvable_action_position_is_a_contract_error():
    broken = {
        "event": {"axis": "turn", "user_turn_end_s": 2.0},
        "unit_layouts": [{"time_s": 3.0, "speaking": True}],  # no response_ids / action_position
    }
    with pytest.raises(ValueError, match="response_ids"):
        compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={"duplex_replay": broken})


def test_unknown_axis_is_refused_not_scored():
    replay = _replay("turn", [False, False, True], event={"axis": "sideways"})
    with pytest.raises(ValueError, match="axis"):
        compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={"duplex_replay": replay})


def test_pacing_window_follows_the_overridden_deadlines():
    # An engine unit 2.5 s past the turn end (2.0 s) sits inside the default
    # 3 s window but outside an overridden 1 s one.
    late = _replay("turn", [False, False, True], times=[0.0, 4.5, 4.6], origins=["client", "engine", "client"])
    default = compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={"duplex_replay": late})
    shortened = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": late}, turn_deadline_s=1.0
    )
    assert default["duplex_pacing_corrupted"] == 1.0
    assert shortened["duplex_pacing_corrupted"] == 0.0
