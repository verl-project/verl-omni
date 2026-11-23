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
"""Minimal Full-Duplex-Bench v1 harness for the duplex GRPO regression gate.

Implements verl-omni#905 §3.6's regression gate: rule-based timing metrics
from Full-Duplex-Bench v1's static protocol, restricted to the timing axes
(github.com/DanielLin94144/Full-Duplex-Bench).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from verl_omni.utils.reward_score.duplex.rule_rewards import DuplexTrajectory, _media_time, _spoke

__all__ = ["FDBResult", "evaluate_fdb_timing", "check_regression_gate"]


@dataclass(frozen=True)
class FDBResult:
    """Timing metrics from one FDB v1 episode."""

    event_type: str
    response_latency: float | None  # seconds from overlap/turn end to first model speech
    yield_latency: float | None  # seconds from overlap end to model silence
    spoke_during_pause: bool
    responded_after_overlap: bool


def evaluate_fdb_timing(trajectory: DuplexTrajectory) -> FDBResult:
    """Evaluate one FDB v1 episode's timing metrics from the chunk log."""
    chunks = trajectory.chunks
    preamble = trajectory.preamble

    if trajectory.event_type == "pause":
        spoke = any(_spoke(c) for c in chunks)
        return FDBResult(
            event_type="pause",
            response_latency=None,
            yield_latency=None,
            spoke_during_pause=spoke,
            responded_after_overlap=False,
        )

    if trajectory.event_type == "turn":
        turn_end = float(trajectory.event_metadata.get("turn_boundary", 0.0))
        first_speech = None
        for c in chunks:
            if _spoke(c) and _media_time(c, preamble) >= turn_end:
                first_speech = _media_time(c, preamble)
                break
        return FDBResult(
            event_type="turn",
            response_latency=first_speech - turn_end if first_speech is not None else None,
            yield_latency=None,
            spoke_during_pause=False,
            responded_after_overlap=first_speech is not None,
        )

    if trajectory.event_type == "interruption":
        overlap_end = float(trajectory.event_metadata.get("overlap_end", 0.0))
        first_speech = None
        first_silence = None
        for c in chunks:
            t = _media_time(c, preamble)
            if first_speech is None and _spoke(c) and t >= overlap_end:
                first_speech = t
            if first_silence is None and not _spoke(c) and t >= overlap_end:
                first_silence = t
        return FDBResult(
            event_type="interruption",
            response_latency=first_speech - overlap_end if first_speech is not None else None,
            yield_latency=first_silence - overlap_end if first_silence is not None else None,
            spoke_during_pause=False,
            responded_after_overlap=first_speech is not None,
        )

    raise ValueError(f"Unknown event type: {trajectory.event_type}")


@dataclass(frozen=True)
class RegressionGateConfig:
    """Pre-registered thresholds from verl-omni#905 §3.6."""

    pause_violation_rate_max: float = 0.20
    avg_turn_latency_max: float = 3.0
    avg_yield_latency_max: float = 2.0
    responded_after_overlap_min: float = 0.60


def check_regression_gate(results: Sequence[FDBResult], config: RegressionGateConfig | None = None) -> list[str]:
    """Return a list of violated regression thresholds (empty = pass)."""
    config = config or RegressionGateConfig()
    violations: list[str] = []

    pauses = [r for r in results if r.event_type == "pause"]
    if pauses:
        violation_rate = sum(1 for r in pauses if r.spoke_during_pause) / len(pauses)
        if violation_rate > config.pause_violation_rate_max:
            violations.append(f"pause_violation_rate {violation_rate:.2f} > {config.pause_violation_rate_max}")

    turns = [r for r in results if r.event_type == "turn" and r.response_latency is not None]
    if turns:
        avg_latency = sum(r.response_latency for r in turns) / len(turns)
        if avg_latency > config.avg_turn_latency_max:
            violations.append(f"avg_turn_latency {avg_latency:.2f}s > {config.avg_turn_latency_max}s")

    interr = [r for r in results if r.event_type == "interruption"]
    if interr:
        responded = sum(1 for r in interr if r.responded_after_overlap) / len(interr)
        if responded < config.responded_after_overlap_min:
            violations.append(f"responded_after_overlap {responded:.2f} < {config.responded_after_overlap_min}")
        yields = [r.yield_latency for r in interr if r.yield_latency is not None]
        if yields:
            avg_yield = sum(yields) / len(yields)
            if avg_yield > config.avg_yield_latency_max:
                violations.append(f"avg_yield_latency {avg_yield:.2f}s > {config.avg_yield_latency_max}s")

    return violations
