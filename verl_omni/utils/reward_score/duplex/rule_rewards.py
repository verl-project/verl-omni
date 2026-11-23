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
"""Three-axis rule rewards for MiniCPM-o 4.5 duplex GRPO.

Implements the reward design from verl-omni#905 §3.5: CPU rule rewards
computed from timestamps and VAD over the duplex trajectory's per-second
chunk log. Three axes, each normalized to [-1, 0]:

  pause_violation: −1 if the model speaks inside a tagged pause window.
  turn_latency:    −(delay from user turn end to model speech start),
                   with a full-credit band.
  interruption_yield: −(delay from end of user's interrupting speech to the
                       model's return to silence), with a deadline.

Plus training telemetry and degeneracy trip-wires.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

__all__ = [
    "DuplexTrajectory",
    "compute_pause_violation",
    "compute_turn_latency",
    "compute_interruption_yield",
    "compute_all_rewards",
    "compute_telemetry",
    "check_trip_wires",
    "TripWireConfig",
]

# ---------------------------------------------------------------------------
# Trajectory model (matches the duplex wrapper's JSON output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplexTrajectory:
    """Per-second chunk log from a duplex session (the trainer's view)."""

    chunks: list[dict[str, Any]]  # {"text", "is_listen", "current_time", ...}
    event_type: str  # "pause" | "turn" | "interruption"
    event_metadata: dict[str, Any]  # timing ground truth from the event miner
    preamble: float = 0.0  # user-speech audio duration before the live stream


def _media_time(chunk: dict[str, Any], preamble: float) -> float:
    """Map a chunk's session current_time onto the media timeline."""
    return max(0.0, float(chunk.get("current_time", 0) or 0) - preamble)


def _spoke(chunk: dict[str, Any]) -> bool:
    return bool(chunk.get("text", "").strip()) and not chunk.get("is_listen", True)


# ---------------------------------------------------------------------------
# Axis 1: Pause violation
# ---------------------------------------------------------------------------


def compute_pause_violation(trajectory: DuplexTrajectory) -> dict[str, float]:
    """−1 if the model speaks during a tagged pause window, else 0."""
    if trajectory.event_type != "pause":
        return {"pause_violation": 0.0, "pause_violation_detail": 0.0}
    pause_start = float(trajectory.event_metadata.get("pause_start", 0.0))
    pause_end = float(trajectory.event_metadata.get("pause_end", 0.0))
    spoke = any(
        _spoke(c) and pause_start <= _media_time(c, trajectory.preamble) <= pause_end
        for c in trajectory.chunks
    )
    return {"pause_violation": -1.0 if spoke else 0.0, "pause_violation_detail": float(spoke)}


# ---------------------------------------------------------------------------
# Axis 2: Turn latency
# ---------------------------------------------------------------------------


def compute_turn_latency(trajectory: DuplexTrajectory, full_credit_band: float = 1.0) -> dict[str, float]:
    """Negative delay from user turn end to model speech start.

    Within ``full_credit_band`` seconds of the turn boundary the reward is 0
    (no penalty). Beyond that, a linearly scaled negative down to −1 at the
    segment end.
    """
    if trajectory.event_type != "turn":
        return {"turn_latency": 0.0, "turn_latency_seconds": 0.0}
    turn_end = float(trajectory.event_metadata.get("turn_boundary", 0.0))
    segment_end = float(trajectory.event_metadata.get("segment_end", turn_end + 5.0))

    first_speech = None
    for c in trajectory.chunks:
        if _spoke(c):
            t = _media_time(c, trajectory.preamble)
            if t >= turn_end:
                first_speech = t
                break

    if first_speech is None:
        # Model never took the floor: full penalty
        return {"turn_latency": -1.0, "turn_latency_seconds": float(segment_end - turn_end)}

    delay = first_speech - turn_end
    if delay <= full_credit_band:
        reward = 0.0
    else:
        span = max(segment_end - turn_end - full_credit_band, 1e-6)
        reward = -min((delay - full_credit_band) / span, 1.0)
    return {"turn_latency": reward, "turn_latency_seconds": delay}


# ---------------------------------------------------------------------------
# Axis 3: Interruption yield
# ---------------------------------------------------------------------------


def compute_interruption_yield(trajectory: DuplexTrajectory, deadline: float = 2.0, full_credit_band: float = 0.5) -> dict[str, float]:
    """Negative delay from user's speech end to model's return to silence."""
    if trajectory.event_type != "interruption":
        return {"interruption_yield": 0.0, "interruption_yield_seconds": 0.0}
    overlap_end = float(trajectory.event_metadata.get("overlap_end", 0.0))

    first_silence = None
    for c in trajectory.chunks:
        t = _media_time(c, trajectory.preamble)
        if t >= overlap_end and not _spoke(c):
            first_silence = t
            break

    if first_silence is None:
        return {"interruption_yield": -1.0, "interruption_yield_seconds": float(deadline)}

    delay = first_silence - overlap_end
    if delay <= full_credit_band:
        reward = 0.0
    elif delay >= deadline:
        reward = -1.0
    else:
        span = max(deadline - full_credit_band, 1e-6)
        reward = -min((delay - full_credit_band) / span, 1.0)
    return {"interruption_yield": reward, "interruption_yield_seconds": delay}


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------


def compute_all_rewards(
    trajectory: DuplexTrajectory,
    full_credit_band: float = 1.0,
    yield_deadline: float = 2.0,
    yield_full_credit_band: float = 0.5,
) -> dict[str, float]:
    """Compute all three axes and return the aggregate."""
    results: dict[str, float] = {}
    results.update(compute_pause_violation(trajectory))
    results.update(compute_turn_latency(trajectory, full_credit_band))
    results.update(compute_interruption_yield(trajectory, yield_deadline, yield_full_credit_band))
    axes = [results.get("pause_violation", 0.0), results.get("turn_latency", 0.0), results.get("interruption_yield", 0.0)]
    results["reward"] = sum(axes)
    return results


# ---------------------------------------------------------------------------
# Training telemetry (verl-omni#905 §3.5)
# ---------------------------------------------------------------------------


def compute_telemetry(trajectories: Sequence[DuplexTrajectory]) -> dict[str, float]:
    """Per-batch telemetry for the trip-wires and dashboards."""
    if not trajectories:
        return {}
    total_chunks = sum(len(t.chunks) for t in trajectories)
    total_speak = sum(1 for t in trajectories for c in t.chunks if _spoke(c))
    total_listen = sum(1 for t in trajectories for c in t.chunks if not _spoke(c))

    # Duplicate n-gram share (n=3) over concatenated speech
    all_text = " ".join(c.get("text", "") for t in trajectories for c in t.chunks).strip()
    words = all_text.split()
    trigrams = Counter(tuple(words[i : i + 3]) for i in range(len(words) - 2))
    dup_trigrams = sum(c - 1 for c in trigrams.values() if c > 1)
    total_trigrams = max(len(words) - 2, 1)

    # Turn-taking / overlap rates
    turn_count = sum(1 for t in trajectories if t.event_type == "turn")
    overlap_count = sum(1 for t in trajectories if t.event_type == "interruption")

    return {
        "telemetry/total_chunks": float(total_chunks),
        "telemetry/speak_ratio": total_speak / max(total_chunks, 1),
        "telemetry/silence_ratio": total_listen / max(total_chunks, 1),
        "telemetry/duplicate_trigram_share": dup_trigrams / total_trigrams,
        "telemetry/turn_count": float(turn_count),
        "telemetry/overlap_count": float(overlap_count),
        "telemetry/avg_turn_latency": _avg_axis(trajectories, "turn_latency"),
        "telemetry/avg_interruption_yield": _avg_axis(trajectories, "interruption_yield"),
    }


def _avg_axis(trajectories: Sequence[DuplexTrajectory], axis: str) -> float:
    values = []
    for t in trajectories:
        if t.event_type in ("turn", "interruption"):
            r = compute_all_rewards(t)
            v = r.get(axis, 0.0)
            if v != 0.0:
                values.append(v)
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Trip-wires (verl-omni#905 §3.6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TripWireConfig:
    """Pre-registered thresholds that block a run when breached."""

    duplicate_trigram_share_max: float = 0.30
    silence_share_max: float = 0.85
    silence_share_min: float = 0.05
    speak_ratio_min: float = 0.10
    speak_ratio_max: float = 0.90


def check_trip_wires(
    telemetry: dict[str, float],
    config: TripWireConfig | None = None,
) -> list[str]:
    """Return a list of violated trip-wire descriptions (empty = all clear)."""
    config = config or TripWireConfig()
    violations: list[str] = []

    dup = telemetry.get("telemetry/duplicate_trigram_share", 0.0)
    if dup > config.duplicate_trigram_share_max:
        violations.append(f"duplicate_trigram_share {dup:.3f} > {config.duplicate_trigram_share_max}")

    silence = telemetry.get("telemetry/silence_ratio", 0.0)
    if silence > config.silence_share_max:
        violations.append(f"silence_ratio {silence:.3f} > {config.silence_share_max} (silence collapse)")
    if silence < config.silence_share_min:
        violations.append(f"silence_ratio {silence:.3f} < {config.silence_share_min} (never listens)")

    speak = telemetry.get("telemetry/speak_ratio", 0.0)
    if speak < config.speak_ratio_min:
        violations.append(f"speak_ratio {speak:.3f} < {config.speak_ratio_min}")
    if speak > config.speak_ratio_max:
        violations.append(f"speak_ratio {speak:.3f} > {config.speak_ratio_max}")

    return violations
