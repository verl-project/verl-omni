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

from tests.utils.dataset.duplex._duplex_fixtures import LISTEN_TOKEN_ID, listening_then_speaking_wires
from verl_omni.utils.dataset.duplex.calibration import audit_calibration
from verl_omni.utils.dataset.duplex.fdb_metrics import fdb_rule_metrics
from verl_omni.utils.dataset.duplex.mining import DuplexEpisodeSpec, WordTimestamp, mine_episodes
from verl_omni.utils.dataset.duplex.telemetry import (
    TRIP_WIRE_DEFAULTS,
    aggregate_duplex_metrics,
    episode_rule_metrics,
    evaluate_trip_wires,
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


def test_pause_mining_skips_short_silences():
    words = [WordTimestamp("a", 0.0, 0.5), WordTimestamp("b", 1.0, 1.5)]  # 0.5 s gap only
    specs = mine_episodes(words, [], user_audio_ref="a.wav", dialogue_id="dd", axes=[AXIS_PAUSE])
    assert specs == []


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
    event = DuplexEpisodeEvent(
        axis=axis,
        pause_window_s=(1.0, 3.0) if axis == AXIS_PAUSE else None,
        user_turn_end_s=2.0 if axis == AXIS_TURN else None,
        interrupt_speech_end_s=2.0 if axis == AXIS_INTERRUPTION else None,
    )
    return DuplexTrajectory.assemble(
        session_id="duplex-abc",
        seed=7,
        event=event,
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
    # First silent unit at t=2.0 is exactly at the deadline -> latency 0.
    assert yield_now["yield_latency_s"] == pytest.approx(0.0)


def test_duplicate_ngram_share_flags_repetition():
    repeated = episode_rule_metrics(
        _trajectory(AXIS_PAUSE, [True] * 8, content_ids=[777] * 8), listen_token_id=LISTEN_TOKEN_ID
    )
    assert repeated["duplicate_ngram_share"] == 1.0
    varied = episode_rule_metrics(
        _trajectory(AXIS_PAUSE, [True] * 8, content_ids=list(range(100, 108))), listen_token_id=LISTEN_TOKEN_ID
    )
    assert varied["duplicate_ngram_share"] == 0.0


def test_trip_wires_fire_on_degenerate_aggregates():
    episodes = [
        {"num_units": 5.0, "speak_ratio": 0.0, "silence_share": 1.0, "duplicate_ngram_share": 0.0},
    ] * 4
    aggregated = aggregate_duplex_metrics(episodes)
    violations = evaluate_trip_wires(aggregated, TRIP_WIRE_DEFAULTS)
    assert any("silence share" in violation for violation in violations)
    assert any("speak ratio" in violation for violation in violations)
    healthy = aggregate_duplex_metrics(
        [{"num_units": 5.0, "speak_ratio": 0.4, "silence_share": 0.6, "duplicate_ngram_share": 0.1}] * 4
    )
    assert evaluate_trip_wires(healthy, TRIP_WIRE_DEFAULTS) == []


def test_calibration_gate_requires_mid_band_rewards():
    # Rule rewards live in [-1, 0]; a floored base model floors an axis.
    report = audit_calibration(axis_rewards={"pause": [0.0] * 10, "turn": [-1.0] * 10, "interruption": [-0.2] * 10})
    assert report.passed is False
    assert report["pause"]["passed"] is True  # clean pauses: normalized 1.0
    assert report["turn"]["passed"] is False  # never takes the floor
    good = audit_calibration(axis_rewards={"pause": [-0.1] * 10, "turn": [-0.3] * 10, "interruption": [-0.4] * 10})
    assert good.passed is True
    too_few = audit_calibration({"pause": [0.0]})
    assert too_few.passed is False


def _replay_payload(axis: str, speaking: list[bool], times: list[float] | None = None) -> dict:
    times = times or [float(index) for index in range(len(speaking))]
    event = {
        AXIS_PAUSE: {"axis": axis, "pause_window_s": [1.0, 3.0]},
        AXIS_TURN: {"axis": axis, "user_turn_end_s": 2.0},
        AXIS_INTERRUPTION: {"axis": axis, "interrupt_speech_end_s": 2.0},
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
        _replay_payload(AXIS_INTERRUPTION, [True, True, False], times=[0.0, 1.0, 3.0]),  # 1 s yield
        _replay_payload(AXIS_INTERRUPTION, [True, True, True]),  # never yielded
    ]
    metrics = fdb_rule_metrics(payloads)
    assert metrics["fdb/pause_violation_rate"] == pytest.approx(0.5)
    assert metrics["fdb/turn_taking_rate"] == pytest.approx(0.5)
    assert metrics["fdb/turn_latency_s_median"] == pytest.approx(1.0)
    assert metrics["fdb/yield_rate"] == pytest.approx(0.5)
    assert metrics["fdb/yield_latency_s_median"] == pytest.approx(1.0)
    assert metrics["fdb/pause_episodes"] == 2.0
