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
"""Per-axis temporal rule rewards for streaming duplex GRPO (RFC #905 §3.5).

CPU rule rewards computed from timestamps only — no VAD model, no judge: the
pause axis penalizes speaking inside the tagged pause window; the turn axis
penalizes the delay from the user's turn end to the model's speech onset with
a tuned full-credit band; the interruption axis penalizes the delay from the
end of the interrupting speech to the model's return to silence with a tuned
deadline. Each episode carries exactly one axis; the score is that axis's
reward, and the raw measurements ride the returned dict as metrics.

Selected from the run config (see the reward rules):

    reward.custom_reward_function.path=verl_omni/utils/reward_score/duplex_turn_taking.py
    reward.custom_reward_function.name=compute_score_duplex_turn_taking
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY
from verl_omni.utils.dataset.duplex.telemetry import TRIP_WIRE_DEFAULTS, duplicate_ngram_share

__all__ = ["compute_score_duplex_turn_taking"]

#: Latency within this band earns full credit (seconds); beyond the deadline
#: the penalty saturates at -1 (RFC #905 §3.5, "tuned full-credit band").
TURN_FULL_CREDIT_BAND_S = 1.0
TURN_DEADLINE_S = 3.0
YIELD_FULL_CREDIT_BAND_S = 0.5
YIELD_DEADLINE_S = 2.0


def _linear_penalty(latency_s: float, full_credit_band_s: float, deadline_s: float) -> float:
    """0 within the credit band, -1 past the deadline, linear in between."""
    if latency_s <= full_credit_band_s:
        return 0.0
    if latency_s >= deadline_s or math.isinf(latency_s):
        return -1.0
    return -(latency_s - full_credit_band_s) / (deadline_s - full_credit_band_s)


def compute_score_duplex_turn_taking(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Score one duplex episode's turn-taking behavior from its replay.

    Args:
        data_source: The dataset source tag (unused; episodes self-describe).
        solution_str: The decoded action stream (unused; timing is the signal).
        ground_truth: The reference text (unused for rule timing rewards).
        extra_info: Must carry the episode's replay payload under
            ``extra_info["duplex_replay"]`` (the namespaced agent-loop
            ``extra_fields`` key), with the event annotation and the
            per-unit decision timeline.
        **kwargs: ``turn_full_credit_band_s`` / ``turn_deadline_s`` /
            ``yield_full_credit_band_s`` / ``yield_deadline_s`` override the
            defaults (forwarded config keys).

    Returns:
        ``{"score": float, ...raw measurements}`` — the raw keys become
        metrics verbatim. Alongside the per-axis measurements it always
        carries the trip-wire columns (``duplex_speak_ratio``,
        ``duplex_silence_share``, ``duplex_duplicate_ngram_share``; the
        telemetry definitions) and ``duplex_episode_failed`` (1.0 when any
        pre-registered threshold in ``telemetry.TRIP_WIRE_DEFAULTS`` is
        breached — the per-episode degeneracy trip-wire the trainer
        aggregates per step).
    """
    replay = extra_info.get(DUPLEX_REPLAY_KEY) if isinstance(extra_info, dict) else None
    if not isinstance(replay, dict):
        raise ValueError(
            "duplex turn-taking rewards require the episode replay under "
            f"extra_info[{DUPLEX_REPLAY_KEY!r}]; the duplex agent loop must stamp it"
        )
    event = replay.get("event") or {}
    axis = str(event.get("axis", ""))
    layouts = replay.get("unit_layouts") or []
    if not layouts:
        raise ValueError(f"{DUPLEX_REPLAY_KEY!r} carries no unit timeline; refusing to score a broken episode")
    turn_band = float(kwargs.get("turn_full_credit_band_s", TURN_FULL_CREDIT_BAND_S))
    turn_deadline = float(kwargs.get("turn_deadline_s", TURN_DEADLINE_S))
    yield_band = float(kwargs.get("yield_full_credit_band_s", YIELD_FULL_CREDIT_BAND_S))
    yield_deadline = float(kwargs.get("yield_deadline_s", YIELD_DEADLINE_S))

    metrics: dict[str, Any] = {"duplex_axis": axis, "duplex_num_units": len(layouts)}
    if axis == "pause":
        start, end = event["pause_window_s"]
        violated = any(layout["speaking"] and start <= layout["time_s"] < end for layout in layouts)
        metrics["duplex_pause_violation"] = float(violated)
        score = -1.0 if violated else 0.0
    elif axis == "turn":
        turn_end = float(event["user_turn_end_s"])
        onset = next((layout["time_s"] for layout in layouts if layout["speaking"]), None)
        if onset is None or onset < turn_end:
            # Speaking before the user's utterance ends is a barge-in on the
            # user, not a floor take: saturate like never speaking (the axis's
            # worst score, -1 past the deadline).
            latency = float("inf")
        else:
            latency = onset - turn_end
        score = _linear_penalty(latency, turn_band, turn_deadline)
        metrics["duplex_turn_latency_s"] = latency
        metrics["duplex_turn_took_floor"] = float(onset is not None)
    elif axis == "interruption":
        speech_end = float(event["interrupt_speech_end_s"])
        silent_after = [layout for layout in layouts if not layout["speaking"] and layout["time_s"] >= speech_end]
        ever_spoke = any(layout["speaking"] for layout in layouts)
        if not ever_spoke:
            latency = 0.0  # nothing to yield
        elif silent_after:
            latency = max(0.0, silent_after[0]["time_s"] - speech_end)
        else:
            latency = float("inf")
        score = _linear_penalty(latency, yield_band, yield_deadline)
        metrics["duplex_yield_latency_s"] = latency
    else:
        raise ValueError(f"duplex replay carries unknown axis {axis!r}")
    metrics.update(_trip_wire_metrics(replay, layouts))
    return {"score": score, **metrics}


def _trip_wire_metrics(replay: Mapping[str, Any], layouts: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """The per-episode trip-wire columns (telemetry's shared definitions).

    Args:
        replay: The replay payload; a speaking unit's action token is read at
            its ``action_position`` inside ``response_ids``.
        layouts: The unit layouts (must be non-empty).

    Returns:
        ``duplex_speak_ratio`` / ``duplex_silence_share`` /
        ``duplex_duplicate_ngram_share`` and ``duplex_episode_failed`` (1.0
        when any ``telemetry.TRIP_WIRE_DEFAULTS`` threshold is breached).
    """
    response_ids = replay.get("response_ids") or []
    content_stream: list[int] = []
    for layout in layouts:
        if not layout.get("speaking"):
            continue
        position = int(layout.get("action_position", -1))
        if not 0 <= position < len(response_ids):
            raise ValueError(
                "duplex turn-taking rewards need the replay's response_ids to measure repetition: "
                f"a speaking unit at time {layout.get('time_s')} has no resolvable action position"
            )
        content_stream.append(int(response_ids[position]))
    num_units = len(layouts)
    speaking = sum(1 for layout in layouts if layout.get("speaking"))
    speak_ratio = speaking / num_units
    silence_share = (num_units - speaking) / num_units
    duplicate_share = duplicate_ngram_share(content_stream)
    failed = float(
        duplicate_share > TRIP_WIRE_DEFAULTS.duplicate_ngram_share_max
        or not TRIP_WIRE_DEFAULTS.speak_ratio_band[0] <= speak_ratio <= TRIP_WIRE_DEFAULTS.speak_ratio_band[1]
        or not TRIP_WIRE_DEFAULTS.silence_share_band[0] <= silence_share <= TRIP_WIRE_DEFAULTS.silence_share_band[1]
    )
    return {
        "duplex_speak_ratio": speak_ratio,
        "duplex_silence_share": silence_share,
        "duplex_duplicate_ngram_share": duplicate_share,
        "duplex_episode_failed": failed,
    }
