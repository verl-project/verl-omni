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
scores the signed delay from the user's turn end (early speech is penalized
by its earliness on the deadline; late speech keeps the full-credit band);
the interruption axis penalizes the delay from the interrupting speech onset
to the return to silence. Speech on the onset unit counts as interrupted. A
policy that never spoke, or first speaks after that unit, takes the floor.
Each episode carries exactly one axis; the score is that axis's reward, and
the raw measurements ride the returned dict as metrics.

Selected from the run config (see the reward rules):

    reward.custom_reward_function.path=verl_omni/utils/reward_score/duplex_turn_taking.py
    reward.custom_reward_function.name=compute_score_duplex_turn_taking
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from verl_omni.utils.dataset.duplex.records import ORIGIN_ENGINE
from verl_omni.utils.dataset.duplex.replay import DUPLEX_REPLAY_KEY
from verl_omni.utils.dataset.duplex.telemetry import (
    TRIP_WIRE_DEFAULTS,
    TURN_DEADLINE_S,
    TURN_FULL_CREDIT_BAND_S,
    YIELD_DEADLINE_S,
    YIELD_FULL_CREDIT_BAND_S,
    duplicate_ngram_share,
    interruption_yield_latency,
    pacing_corrupted,
    turn_take_latency,
)
from verl_omni.utils.dataset.duplex.trajectory import DuplexEpisodeEvent

__all__ = ["compute_score_duplex_turn_taking"]


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
        telemetry definitions), ``duplex_episode_failed`` (1.0 when any
        pre-registered threshold in ``telemetry.TRIP_WIRE_DEFAULTS`` is
        breached — the per-episode degeneracy trip-wire the trainer
        aggregates per step) and ``duplex_pacing_corrupted`` (1.0 when an
        engine-injected silent unit fell inside the scored window, evaluated
        with these deadline overrides; the agent loop drops an episode on the
        same predicate at the default deadlines, so a lengthened deadline
        can flag here an episode the loop kept).
    """
    replay = extra_info.get(DUPLEX_REPLAY_KEY) if isinstance(extra_info, dict) else None
    if not isinstance(replay, dict):
        raise ValueError(
            "duplex turn-taking rewards require the episode replay under "
            f"extra_info[{DUPLEX_REPLAY_KEY!r}]; the duplex agent loop must stamp it"
        )
    event_payload = replay.get("event") or {}
    layouts = replay.get("unit_layouts") or []
    if not layouts:
        raise ValueError(f"{DUPLEX_REPLAY_KEY!r} carries no unit timeline; refusing to score a broken episode")
    event = DuplexEpisodeEvent.from_mapping(event_payload)
    turn_band = float(kwargs.get("turn_full_credit_band_s", TURN_FULL_CREDIT_BAND_S))
    turn_deadline = float(kwargs.get("turn_deadline_s", TURN_DEADLINE_S))
    yield_band = float(kwargs.get("yield_full_credit_band_s", YIELD_FULL_CREDIT_BAND_S))
    yield_deadline = float(kwargs.get("yield_deadline_s", YIELD_DEADLINE_S))

    metrics: dict[str, Any] = {"duplex_axis": event.axis, "duplex_num_units": len(layouts)}
    speaking_times = [layout["time_s"] for layout in layouts if layout["speaking"]]
    silent_times = [layout["time_s"] for layout in layouts if not layout["speaking"]]
    if event.axis == "pause":
        start, end = event.pause_window_s
        violated = any(start <= time_s < end for time_s in speaking_times)
        metrics["duplex_pause_violation"] = float(violated)
        score = -1.0 if violated else 0.0
    elif event.axis == "turn":
        turn_end = event.user_turn_end_s
        first_speak = speaking_times[0] if speaking_times else None
        latency = turn_take_latency(first_speak, turn_end)
        # Earliness uses the deadline with no credit band, so waiting longer scores higher.
        if latency < 0:
            score = _linear_penalty(-latency, 0.0, turn_deadline)
        else:
            score = _linear_penalty(latency, turn_band, turn_deadline)
        metrics["duplex_turn_latency_s"] = latency
        metrics["duplex_turn_took_floor"] = float(first_speak is not None)
    elif event.axis == "interruption":
        latency = interruption_yield_latency(event, speaking_times=speaking_times, silent_times=silent_times)
        score = -1.0 if latency is None else _linear_penalty(latency, yield_band, yield_deadline)
        metrics["duplex_yield_latency_s"] = float("inf") if latency is None else latency
    else:
        raise ValueError(f"duplex replay carries unknown axis {event.axis!r}")
    engine_unit_times = [layout["time_s"] for layout in layouts if layout.get("origin") == ORIGIN_ENGINE]
    metrics["duplex_pacing_corrupted"] = float(
        pacing_corrupted(event, engine_unit_times, turn_deadline_s=turn_deadline, yield_deadline_s=yield_deadline)
    )
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
