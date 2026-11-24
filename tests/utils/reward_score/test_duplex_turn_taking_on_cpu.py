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

from verl_omni.utils.reward_score.duplex_turn_taking import compute_score_duplex_turn_taking

LISTEN_TOKEN_ID = 0


def _replay(axis: str, speaking, times=None, content_ids=None):
    times = times or [float(index) for index in range(len(speaking))]
    event = {
        "pause": {"axis": "pause", "pause_window_s": [1.0, 3.0]},
        "turn": {"axis": "turn", "user_turn_end_s": 2.0},
        "interruption": {"axis": "interruption", "interrupt_speech_end_s": 2.0},
    }[axis]
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
            {"time_s": time_s, "speaking": speaks, "action_position": index}
            for index, (time_s, speaks) in enumerate(zip(times, speaking, strict=True))
        ],
    }


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


def test_turn_axis_penalizes_speaking_before_the_turn_ends():
    # Onset at 1.0 s while the user's utterance ends at 2.0 s: a barge-in on
    # the user, not a floor take — saturates like never speaking.
    early = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("turn", [False, True, False, True])}
    )
    assert early["score"] == -1.0
    assert early["duplex_turn_latency_s"] == float("inf")
    assert early["duplex_turn_took_floor"] == 1.0  # it did speak; the timing is what fails


def test_interruption_axis_rewards_prompt_yielding():
    prompt = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("interruption", [True, True, False])}
    )
    assert prompt["score"] == 0.0  # yielded 1 s after speech end -> inside the band
    slow = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("interruption", [True] * 5)}
    )
    assert slow["score"] == -1.0
    silent = compute_score_duplex_turn_taking(
        "dailytalk", "", "", extra_info={"duplex_replay": _replay("interruption", [False, False])}
    )
    assert silent["score"] == 0.0  # never spoke: nothing to yield


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


def test_missing_replay_is_a_contract_error():
    with pytest.raises(ValueError, match="duplex_replay"):
        compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={})
    with pytest.raises(ValueError, match="no unit timeline"):
        compute_score_duplex_turn_taking("dailytalk", "", "", extra_info={"duplex_replay": {"event": {"axis": "turn"}}})


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
