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


def _replay(axis: str, speaking, times=None):
    times = times or [float(index) for index in range(len(speaking))]
    event = {
        "pause": {"axis": "pause", "pause_window_s": [1.0, 3.0]},
        "turn": {"axis": "turn", "user_turn_end_s": 2.0},
        "interruption": {"axis": "interruption", "interrupt_speech_end_s": 2.0},
    }[axis]
    return {
        "event": event,
        "unit_layouts": [
            {"time_s": time_s, "speaking": speaks} for time_s, speaks in zip(times, speaking, strict=True)
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
