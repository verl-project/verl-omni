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
numeric columns per step.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verl_omni.utils.dataset.duplex.trajectory import DuplexTrajectory

__all__ = [
    "TRIP_WIRE_DEFAULTS",
    "DuplexTripWires",
    "duplicate_ngram_share",
    "episode_rule_metrics",
]


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
        ``nan`` when the episode's axis does not measure one).
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
    if event.axis == "pause" and event.pause_window_s is not None:
        start, end = event.pause_window_s
        metrics["pause_violations"] = float(
            sum(1 for decision in decisions if decision.speaking and start <= decision.time_s < end)
        )
    elif event.axis == "turn" and event.user_turn_end_s is not None:
        if first_speak is not None and first_speak.time_s >= event.user_turn_end_s:
            metrics["turn_latency_s"] = first_speak.time_s - event.user_turn_end_s
        else:
            # Same semantics as the reward: speaking before the user's turn
            # ends is a barge-in on the user, not a floor take — saturate like
            # never speaking.
            metrics["turn_latency_s"] = float("inf")
    elif event.axis == "interruption" and event.interrupt_speech_end_s is not None:
        deadline = event.interrupt_speech_end_s
        after = [decision for decision in decisions if decision.time_s >= deadline]
        silent = next((decision for decision in after if not decision.speaking), None)
        if silent is not None:
            metrics["yield_latency_s"] = silent.time_s - deadline
        elif not any(decision.speaking for decision in decisions):
            metrics["yield_latency_s"] = 0.0  # never spoke: nothing to yield
        else:
            metrics["yield_latency_s"] = float("inf")
    return metrics


def duplicate_ngram_share(token_ids: Sequence[int], *, ngram: int = 3) -> float:
    """Share of n-grams that repeat an earlier one (0 for short streams)."""
    if len(token_ids) < ngram:
        return 0.0
    grams = [tuple(token_ids[index : index + ngram]) for index in range(len(token_ids) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count for count in counts.values() if count > 1)
    return repeated / len(grams)
