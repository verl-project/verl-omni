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
"""Duplex degeneracy trip-wire definitions (RFC verl-omni #905 §3.5).

Rule-reward degeneracy — silence collapse, repetitive speech — is the
documented failure mode of duplex RL, so the thresholds are defined once
here and applied per episode by the turn-taking reward scorer (its
``duplex_episode_failed`` column); the trainer aggregates the scorer's
numeric columns per step. The module also owns the reward bands and the two
timing definitions every consumer shares — the interruption yield latency
and the pacing-corruption predicate — so the scorer, the telemetry, the FDB
metrics and the agent loop cannot drift apart.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verl_omni.utils.dataset.duplex.trajectory import (
    AXIS_INTERRUPTION,
    AXIS_PAUSE,
    AXIS_TURN,
    DuplexEpisodeEvent,
    DuplexTrajectory,
)

__all__ = [
    "TRIP_WIRE_DEFAULTS",
    "TURN_DEADLINE_S",
    "TURN_FULL_CREDIT_BAND_S",
    "DuplexTripWires",
    "YIELD_DEADLINE_S",
    "YIELD_FULL_CREDIT_BAND_S",
    "duplicate_ngram_share",
    "episode_rule_metrics",
    "interruption_yield_latency",
    "pacing_corrupted",
    "turn_take_latency",
]

#: Latency within this band earns full credit (seconds); beyond the deadline
#: the penalty saturates at -1 past the deadline.
TURN_FULL_CREDIT_BAND_S = 1.0
TURN_DEADLINE_S = 3.0
#: The yield latency anchors to the interrupting speech *onset*. At the 1 Hz
#: decision cadence the first unit after the anchor decides a second later,
#: so a shorter band would be unreachable.
YIELD_FULL_CREDIT_BAND_S = 1.0
YIELD_DEADLINE_S = 3.0


@dataclass(frozen=True, slots=True)
class DuplexTripWires:
    """Pre-registered degeneracy thresholds (RFC #905 §3.6)."""

    #: Maximum share of repeated 3-grams in the content-token stream.
    duplicate_ngram_share_max: float = 0.4
    #: Acceptable band for the share of units spent silent.
    silence_share_band: tuple[float, float] = (0.2, 0.9)
    #: Acceptable band for the share of units spent speaking.
    speak_ratio_band: tuple[float, float] = (0.05, 0.8)

    def to_dict(self) -> dict[str, float | tuple[float, float]]:
        return {
            "duplicate_ngram_share_max": self.duplicate_ngram_share_max,
            "silence_share_band": self.silence_share_band,
            "speak_ratio_band": self.speak_ratio_band,
        }


#: The recipe's default trip-wires.
TRIP_WIRE_DEFAULTS = DuplexTripWires()


def episode_rule_metrics(
    trajectory: DuplexTrajectory,
    *,
    listen_token_id: int,
    ngram: int = 3,
) -> dict[str, float]:
    """One episode's rule-telemetry: ratios, overlap, and repetition.

    Args:
        trajectory: The assembled episode.
        listen_token_id: The ``<|listen|>`` id separating listen from speak.
        ngram: N-gram width for the duplicate-share metric.

    Returns:
        Flat metrics: ``num_units``, ``speak_ratio``, ``silence_share``,
        ``duplicate_ngram_share``, and the per-axis latency family the rewards
        share (``pause_violations``, ``turn_latency_s``, ``yield_latency_s``;
        ``nan`` when the episode's axis does not measure one, ``inf`` for a
        saturated latency).
    """
    decisions = trajectory.unit_decisions(listen_token_id=listen_token_id)
    num_units = len(decisions)
    if num_units == 0:
        raise ValueError(f"episode {trajectory.session_id} has no unit decisions to measure")
    speaking = sum(1 for decision in decisions if decision.speaking)
    content_stream = [decision.action_token_id for decision in decisions if decision.speaking]
    duplicate_share = duplicate_ngram_share(content_stream, ngram=ngram)
    metrics: dict[str, float] = {
        "num_units": float(num_units),
        "speak_ratio": speaking / num_units,
        "silence_share": (num_units - speaking) / num_units,
        "duplicate_ngram_share": duplicate_share,
    }
    metrics.update(_axis_latencies(trajectory, decisions))
    return metrics


def _axis_latencies(trajectory: DuplexTrajectory, decisions: Sequence[Any]) -> dict[str, float]:
    """Pause/turn/yield measurements from the decision timeline."""
    event = trajectory.event
    first_speak = next((decision for decision in decisions if decision.speaking), None)
    metrics: dict[str, float] = {
        "pause_violations": float("nan"),
        "turn_latency_s": float("nan"),
        "yield_latency_s": float("nan"),
    }
    if event.axis == AXIS_PAUSE and event.pause_window_s is not None:
        start, end = event.pause_window_s
        metrics["pause_violations"] = float(
            sum(1 for decision in decisions if decision.speaking and start <= decision.time_s < end)
        )
    elif event.axis == AXIS_TURN and event.user_turn_end_s is not None:
        latency = turn_take_latency(None if first_speak is None else first_speak.time_s, event.user_turn_end_s)
        metrics["turn_latency_s"] = float("inf") if latency < 0 else latency
    elif event.axis == AXIS_INTERRUPTION:
        latency = interruption_yield_latency(
            event,
            speaking_times=[decision.time_s for decision in decisions if decision.speaking],
            silent_times=[decision.time_s for decision in decisions if not decision.speaking],
        )
        metrics["yield_latency_s"] = float("inf") if latency is None else latency
    return metrics


def turn_take_latency(first_speak_s: float | None, turn_end_s: float) -> float:
    """Signed delay from the user's turn end to the first speak decision.

    Negative when the policy speaks before the turn ends. ``inf`` when it
    never speaks.
    """
    if first_speak_s is None:
        return float("inf")
    return first_speak_s - turn_end_s


def interruption_yield_latency(
    event: DuplexEpisodeEvent,
    *,
    speaking_times: Sequence[float],
    silent_times: Sequence[float],
) -> float | None:
    """The yield latency from the interrupting speech onset, or ``None``.

    The policy was interrupted when its first speak is at or before the onset,
    including the onset unit the driver uses as the injection point. A policy
    that never spoke, or whose first speak falls strictly after the onset,
    was not speaking when the interruption began: ``None``. Otherwise the
    latency runs from the onset to the first silent decision at or after it;
    ``inf`` when the policy talks on through the rest of the episode.

    Args:
        event: The interruption episode's annotation (anchors on the unit
            clock — the live re-anchored one, not the corpus-relative mining
            value).
        speaking_times: ``time_s`` of the speaking units, in unit order.
        silent_times: ``time_s`` of the non-speaking units, in unit order.

    Returns:
        The latency in seconds, ``inf`` for no yield, or ``None`` when the
        silence floor failed.
    """
    onset = event.interrupt_speech_onset_s
    if not speaking_times or speaking_times[0] > onset:
        return None
    yield_at = next((time_s for time_s in silent_times if time_s >= onset), None)
    return float("inf") if yield_at is None else yield_at - onset


def pacing_corrupted(
    event: DuplexEpisodeEvent,
    engine_unit_times: Sequence[float],
    *,
    turn_deadline_s: float = TURN_DEADLINE_S,
    yield_deadline_s: float = YIELD_DEADLINE_S,
) -> bool:
    """Whether an engine-injected silent unit landed inside the scored window.

    The driver submits one chunk per unit on the native 1 s cadence; when it
    misses the deadline the runtime's silence-continuation clock injects a
    unit of its own (``origin=engine``). The unit replays faithfully, but its
    timing is the runtime's, so a window it falls inside cannot be scored:
    the agent loop drops the episode (``duplex_pacing_corrupted`` = 1.0 on its
    reward extras — the trainer's per-step mean of that column,
    ``reward/duplex_pacing_corrupted``, is the drop rate) and the scorer
    classifies a replay the same way. An engine unit strictly outside the
    window leaves the episode scoreable.

    Args:
        event: The episode's annotation (anchors on the unit clock).
        engine_unit_times: ``time_s`` of the engine-origin units, seconds
            from the episode's first unit.
        turn_deadline_s: The turn window's end, offset from
            ``user_turn_end_s``.
        yield_deadline_s: The interruption window's end, offset from the
            interrupting speech onset.

    Returns:
        True when any engine unit's time lies in the scored window
        (``[pause_start, pause_end)``, ``[user_turn_end_s, +turn_deadline_s)``
        or ``[interrupt_speech_onset_s, +yield_deadline_s)``).
    """
    if event.axis == AXIS_PAUSE:
        window = event.pause_window_s
    elif event.axis == AXIS_TURN:
        window = (event.user_turn_end_s, event.user_turn_end_s + turn_deadline_s)
    elif event.axis == AXIS_INTERRUPTION:
        window = (event.interrupt_speech_onset_s, event.interrupt_speech_onset_s + yield_deadline_s)
    else:
        raise ValueError(f"unknown duplex event axis {event.axis!r}")
    start, end = window
    return any(start <= time_s < end for time_s in engine_unit_times)


def duplicate_ngram_share(token_ids: Sequence[int], *, ngram: int = 3) -> float:
    """Share of n-grams that repeat an earlier one (0 for short streams)."""
    if len(token_ids) < ngram:
        return 0.0
    grams = [tuple(token_ids[index : index + ngram]) for index in range(len(token_ids) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)
