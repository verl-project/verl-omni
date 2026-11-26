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
"""CPU tests for event mining, telemetry, calibration, and FDB metrics."""

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    LISTEN_TOKEN_ID,
    interruption_event,
    listening_then_speaking_wires,
)
from verl_omni.utils.dataset.duplex.calibration import audit_calibration
from verl_omni.utils.dataset.duplex.fdb_metrics import fdb_rule_metrics
from verl_omni.utils.dataset.duplex.mining import (
    DuplexEpisodeSpec,
    WordTimestamp,
    mine_episodes,
    mine_interruption_candidates,
    mine_pause_candidates,
)
from verl_omni.utils.dataset.duplex.telemetry import (
    episode_rule_metrics,
    interruption_yield_latency,
    pacing_corrupted,
)
from verl_omni.utils.dataset.duplex.trajectory import (
    AXIS_INTERRUPTION,
    AXIS_PAUSE,
    AXIS_TURN,
    DuplexEpisodeEvent,
    DuplexTrajectory,
)


def _words(speaker: str) -> list[WordTimestamp]:
    if speaker == "user":
        # One utterance 0.5-3.0, a 1.2 s pause, then 4.2-5.0; a second utterance 9.0-12.0.
        spans = [(0.5, 1.0), (1.2, 2.0), (2.2, 3.0), (3.9, 4.6), (4.8, 5.2), (9.0, 9.5), (9.7, 12.0)]
    else:
        spans = [(2.3, 3.1), (4.0, 4.8)]
    return [WordTimestamp(text=f"w{i}", start_s=start, end_s=end) for i, (start, end) in enumerate(spans)]


def test_mine_episodes_produces_all_three_axes_with_relative_windows():
    specs = mine_episodes(
        _words("user"),
        _words("model"),
        user_audio_ref="dailytalk/0001/user.wav",
        dialogue_id="dd-0001",
    )
    axes = {spec.axis for spec in specs}
    assert axes == {AXIS_PAUSE, AXIS_TURN, AXIS_INTERRUPTION}
    pause = next(spec for spec in specs if spec.axis == AXIS_PAUSE)
    # The pause window is episode-relative: the 3.0->3.9 intra-utterance silence.
    assert pause.start_s == pytest.approx(1.0)  # 3.0 - 2.0 lead
    assert pause.event.pause_window_s == (pytest.approx(2.0), pytest.approx(2.9))
    turn = next(spec for spec in specs if spec.axis == AXIS_TURN)
    # The turn window leads the utterance end by exactly its lead parameter.
    assert turn.event.user_turn_end_s == pytest.approx(3.0)
    interruption = next(spec for spec in specs if spec.axis == AXIS_INTERRUPTION)
    assert interruption.inject_delay_s is not None
    assert interruption.interrupt_speech_s is not None
    # The interrupting clip's onset and end are episode-relative like the other anchors.
    assert interruption.event.interrupt_speech_onset_s == pytest.approx(0.0)
    assert interruption.event.interrupt_speech_end_s == pytest.approx(4.7)  # the 0.5-5.2 utterance
    # Every window is bounded by the mining cap.
    assert all(spec.end_s - spec.start_s <= 8.0 for spec in specs)


def test_mined_specs_roundtrip_through_the_dataset_row():
    specs = mine_episodes(_words("user"), _words("model"), user_audio_ref="a.wav", dialogue_id="dd")
    spec = specs[0]
    restored = DuplexEpisodeSpec.from_dict(spec.to_dict())
    assert restored.episode_id == spec.episode_id
    assert restored.event.axis == spec.event.axis
    assert restored.start_s == spec.start_s
    assert restored.inject_delay_s == spec.inject_delay_s
    interruption = next(spec for spec in specs if spec.axis == AXIS_INTERRUPTION)
    assert DuplexEpisodeSpec.from_dict(interruption.to_dict()).event == interruption.event


def test_negative_lead_clamps_to_the_track_start():
    # A 2 s lead before a pause at 0.62 s would start at -1.38 s.
    words = [
        WordTimestamp(text="a", start_s=0.0, end_s=0.62),
        WordTimestamp(text="b", start_s=1.24, end_s=2.0),
    ]
    specs = mine_episodes(words, [], user_audio_ref="a.wav", dialogue_id="d", axes=(AXIS_PAUSE,))
    pause = specs[0]
    assert pause.start_s == 0.0
    assert pause.event.pause_window_s == (pytest.approx(0.62), pytest.approx(1.24))


def test_pause_mining_skips_short_silences():
    words = [WordTimestamp("a", 0.0, 0.5), WordTimestamp("b", 1.0, 1.5)]  # 0.5 s gap only
    specs = mine_episodes(words, [], user_audio_ref="a.wav", dialogue_id="dd", axes=[AXIS_PAUSE])
    assert specs == []


def test_pause_mining_keeps_same_speaker_silences_inside_the_band():
    # Gaps of 0.6 s, 1.8 s, 0.5 s and 1.9 s: the band's two edges are pauses
    # (the speaker resumes), the 0.5 s one is too short and the 1.9 s one
    # ends the utterance.
    spans = [(2.5, 3.0), (3.6, 4.0), (5.8, 6.0), (6.5, 7.0), (8.9, 9.5)]
    words = [WordTimestamp(text=f"w{index}", start_s=start, end_s=end) for index, (start, end) in enumerate(spans)]
    specs = mine_episodes(words, [], user_audio_ref="a.wav", dialogue_id="dd", axes=[AXIS_PAUSE])
    assert [spec.event.pause_window_s for spec in specs] == [
        (pytest.approx(2.0), pytest.approx(2.6)),
        (pytest.approx(2.0), pytest.approx(3.8)),
    ]
    # The explicit upper bound binds once the utterance gap is wider than the band.
    wide_gap = mine_pause_candidates(words, min_pause_s=0.6, max_pause_s=1.0, utterance_gap_s=2.0)
    assert [candidate.pause_start_s for candidate in wide_gap] == [pytest.approx(3.0)]


def test_interruption_mining_requires_the_min_clip_length():
    model_words = [WordTimestamp("m", 0.0, 6.0)]
    short = [WordTimestamp("u", 1.0, 2.4)]  # 1.4 s of user speech
    assert mine_interruption_candidates(short, model_words) == []
    long = [WordTimestamp("u", 1.0, 2.5), WordTimestamp("v", 2.6, 2.9)]  # 1.9 s
    assert len(mine_interruption_candidates(long, model_words)) == 1


def _event(axis: str) -> DuplexEpisodeEvent:
    return DuplexEpisodeEvent(
        axis=axis,
        pause_window_s=(1.0, 3.0) if axis == AXIS_PAUSE else None,
        user_turn_end_s=2.0 if axis == AXIS_TURN else None,
        interrupt_speech_onset_s=1.0 if axis == AXIS_INTERRUPTION else None,
        interrupt_speech_end_s=4.0 if axis == AXIS_INTERRUPTION else None,
    )


def _trajectory(axis: str, speak_pattern: list[bool], emission_step: float = 1.0, content_ids=None):
    wires = listening_then_speaking_wires()
    # Rewrite the two scopes into one decision stream matching speak_pattern.
    units = []
    for index, speaking in enumerate(speak_pattern):
        token_id = content_ids[index] if content_ids else 1000 + index
        units.append(
            {
                "action": (token_id if speaking else LISTEN_TOKEN_ID, "sampled", -0.3),
                "emission_ns": 1_000_000_000 + int(index * emission_step * 1e9),
            }
        )
    wire = wires[0]
    wire["units"] = units
    wire["token_ids"] = [unit["action"][0] for unit in units]
    wire["logprobs"] = [unit["action"][2] for unit in units]
    wire["action_kinds"] = ["sampled"] * len(units)
    wire["reason_codes"] = [""] * len(units)
    wire["stream_names"] = ["content"] * len(units)
    wire["unit_offsets"] = list(range(0, len(units) + 1))
    wire["stage_ids"] = [0] * len(units)
    wire["input_modalities"] = ["audio"] * len(units)
    wire["input_origins"] = ["client"] * len(units)
    wire["input_commit_ns"] = [None] * len(units)
    wire["emission_ns"] = [unit["emission_ns"] for unit in units]
    wire["sampling_params"] = [{"temperature": 0.7, "top_k": 100, "top_p": 0.8}] * len(units)
    wire["terminators"] = [[] for _ in units]
    return DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=_event(axis),
        record_set_wires=[wire],
        client_audio_chunks=[np.zeros(16000, dtype=np.float32) for _ in speak_pattern],
    )


def test_telemetry_measures_each_axis():
    pause = episode_rule_metrics(_trajectory(AXIS_PAUSE, [False, True, True, False]), listen_token_id=LISTEN_TOKEN_ID)
    assert pause["pause_violations"] == 2.0
    assert pause["silence_share"] == pytest.approx(0.5)
    turn = episode_rule_metrics(_trajectory(AXIS_TURN, [False, False, True, True]), listen_token_id=LISTEN_TOKEN_ID)
    assert turn["turn_latency_s"] == pytest.approx(0.0)  # speak onset exactly at turn end
    never = episode_rule_metrics(_trajectory(AXIS_TURN, [False, False, False]), listen_token_id=LISTEN_TOKEN_ID)
    assert never["turn_latency_s"] == float("inf")
    yield_now = episode_rule_metrics(
        _trajectory(AXIS_INTERRUPTION, [True, True, False, False]), listen_token_id=LISTEN_TOKEN_ID
    )
    # Speaking before the 1.0 s onset, first silent unit at t=2.0: a 1 s yield latency.
    assert yield_now["yield_latency_s"] == pytest.approx(1.0)


def test_interruption_latency_floors_a_policy_silent_through_the_onset():
    # The yield latency shares the reward's silence floor: a policy that never
    # spoke, or whose first speak falls strictly after the onset, has no yield
    # to measure.
    never = episode_rule_metrics(_trajectory(AXIS_INTERRUPTION, [False] * 4), listen_token_id=LISTEN_TOKEN_ID)
    assert never["yield_latency_s"] == float("inf")
    late = episode_rule_metrics(
        _trajectory(AXIS_INTERRUPTION, [False, False, True, True]), listen_token_id=LISTEN_TOKEN_ID
    )
    assert late["yield_latency_s"] == float("inf")
    talks_on = episode_rule_metrics(_trajectory(AXIS_INTERRUPTION, [True] * 4), listen_token_id=LISTEN_TOKEN_ID)
    assert talks_on["yield_latency_s"] == float("inf")


def test_interruption_yield_latency_is_onset_anchored_with_a_silence_floor():
    event = DuplexEpisodeEvent(axis=AXIS_INTERRUPTION, interrupt_speech_onset_s=2.0, interrupt_speech_end_s=5.0)
    # Yielding at onset + 1 s, even while the user is still talking.
    assert interruption_yield_latency(event, speaking_times=[0.0, 1.0, 2.0], silent_times=[3.0, 4.0]) == pytest.approx(
        1.0
    )
    # Talking on through the episode.
    assert interruption_yield_latency(event, speaking_times=[0.0, 1.0, 2.0, 3.0, 4.0], silent_times=[]) == float("inf")
    # Never spoke, or first spoke strictly after the onset: the silence floor.
    assert interruption_yield_latency(event, speaking_times=[], silent_times=[0.0, 1.0]) is None
    assert interruption_yield_latency(event, speaking_times=[2.1, 3.0], silent_times=[0.0, 1.0, 4.0]) is None
    # Speaking on the onset itself is interrupted; silence at 4.0 s is a 2 s yield.
    assert interruption_yield_latency(event, speaking_times=[2.0, 3.0], silent_times=[0.0, 1.0, 4.0]) == pytest.approx(
        2.0
    )


@pytest.mark.parametrize(
    ("axis", "engine_times", "corrupted"),
    [
        # Pause window [1.0, 3.0).
        (AXIS_PAUSE, [1.0], True),
        (AXIS_PAUSE, [2.999], True),
        (AXIS_PAUSE, [0.999], False),
        (AXIS_PAUSE, [3.0], False),
        # Turn window [2.0, 2.0 + 3.0).
        (AXIS_TURN, [2.0], True),
        (AXIS_TURN, [4.999], True),
        (AXIS_TURN, [1.999], False),
        (AXIS_TURN, [5.0], False),
        # Interruption window [1.0, 1.0 + 3.0).
        (AXIS_INTERRUPTION, [1.0], True),
        (AXIS_INTERRUPTION, [3.999], True),
        (AXIS_INTERRUPTION, [0.999], False),
        (AXIS_INTERRUPTION, [4.0], False),
    ],
)
def test_pacing_corrupted_is_an_engine_unit_inside_the_scored_window(axis, engine_times, corrupted):
    assert pacing_corrupted(_event(axis), engine_times) is corrupted
    assert pacing_corrupted(_event(axis), []) is False


def test_pacing_corrupted_window_follows_the_deadline_kwargs():
    assert pacing_corrupted(_event(AXIS_TURN), [2.5]) is True
    assert pacing_corrupted(_event(AXIS_TURN), [2.5], turn_deadline_s=0.5) is False  # window [2.0, 2.5)
    assert pacing_corrupted(_event(AXIS_INTERRUPTION), [3.5]) is True
    assert pacing_corrupted(_event(AXIS_INTERRUPTION), [3.5], yield_deadline_s=2.0) is False  # window [1.0, 3.0)


def test_turn_axis_latency_saturates_for_pre_turn_end_speech():
    # Speaking before the turn end is a barge-in, the same miss as never speaking.
    early = episode_rule_metrics(_trajectory(AXIS_TURN, [True, False, False]), listen_token_id=LISTEN_TOKEN_ID)
    assert early["turn_latency_s"] == float("inf")
    # The boundary stays full credit: onset exactly at the turn end.
    on_time = episode_rule_metrics(_trajectory(AXIS_TURN, [False, False, True, True]), listen_token_id=LISTEN_TOKEN_ID)
    assert on_time["turn_latency_s"] == pytest.approx(0.0)


def test_duplicate_ngram_share_flags_repetition():
    repeated = episode_rule_metrics(
        _trajectory(AXIS_PAUSE, [True] * 8, content_ids=[777] * 8), listen_token_id=LISTEN_TOKEN_ID
    )
    assert repeated["duplicate_ngram_share"] == 1.0
    varied = episode_rule_metrics(
        _trajectory(AXIS_PAUSE, [True] * 8, content_ids=list(range(100, 108))), listen_token_id=LISTEN_TOKEN_ID
    )
    assert varied["duplicate_ngram_share"] == 0.0


def test_calibration_gate_requires_a_mid_band_reward_on_both_sides():
    # Rule rewards live in [-1, 0]: a floored base model floors an axis and a
    # saturated one leaves the group no variance to learn from.
    report = audit_calibration(axis_rewards={"pause": [0.0] * 10, "turn": [-1.0] * 10, "interruption": [-0.5] * 10})
    assert report.passed is False
    assert report["pause"]["passed"] is False  # normalized 1.0: saturated
    assert "saturates" in report["pause"]["reason"]
    assert report["turn"]["passed"] is False  # normalized 0.0: floored
    assert "< 0.30" in report["turn"]["reason"]
    assert report["interruption"]["passed"] is True  # normalized 0.5
    good = audit_calibration(axis_rewards={"pause": [-0.25] * 10, "turn": [-0.5] * 10, "interruption": [-0.65] * 10})
    assert good.passed is True
    assert audit_calibration({"turn": [-0.1] * 10})["turn"]["passed"] is False  # normalized 0.9
    assert audit_calibration({"turn": [-0.8] * 10})["turn"]["passed"] is False  # normalized 0.2
    too_few = audit_calibration({"pause": [0.0]})
    assert too_few.passed is False
    # An empty rollout measures nothing; the gate must fail closed.
    empty = audit_calibration({})
    assert empty["overall_pass"] is False
    assert empty.passed is False
    assert "no axes measured" in empty["reason"]


def _replay_payload(axis: str, speaking: list[bool], times: list[float] | None = None) -> dict:
    times = times or [float(index) for index in range(len(speaking))]
    event = {
        AXIS_PAUSE: {"axis": axis, "pause_window_s": [1.0, 3.0]},
        AXIS_TURN: {"axis": axis, "user_turn_end_s": 2.0},
        AXIS_INTERRUPTION: interruption_event(onset_s=2.0, end_s=5.0),
    }[axis]
    return {
        "event": event,
        "unit_layouts": [
            {"time_s": time_s, "speaking": speaks} for time_s, speaks in zip(times, speaking, strict=True)
        ],
    }


def test_fdb_rule_metrics_over_dumped_replays():
    payloads = [
        _replay_payload(AXIS_PAUSE, [False, True, False]),  # violation
        _replay_payload(AXIS_PAUSE, [False, False, False]),  # clean
        _replay_payload(AXIS_TURN, [False, False, True, True], times=[0.0, 1.0, 3.0, 4.0]),  # 1 s latency
        _replay_payload(AXIS_TURN, [False, False, False]),  # never took the floor
        _replay_payload(AXIS_INTERRUPTION, [True, True, False], times=[0.0, 1.0, 3.0]),  # 1 s after the onset
        _replay_payload(AXIS_INTERRUPTION, [True, True, True]),  # never yielded
        _replay_payload(AXIS_INTERRUPTION, [False, False, False, False]),  # never spoke: the silence floor
    ]
    metrics = fdb_rule_metrics(payloads)
    assert metrics["fdb/pause_violation_rate"] == pytest.approx(0.5)
    assert metrics["fdb/turn_taking_rate"] == pytest.approx(0.5)
    assert metrics["fdb/turn_latency_s_median"] == pytest.approx(1.0)
    assert metrics["fdb/yield_rate"] == pytest.approx(1 / 3)
    assert metrics["fdb/yield_latency_s_median"] == pytest.approx(1.0)
    assert metrics["fdb/pause_episodes"] == 2.0
