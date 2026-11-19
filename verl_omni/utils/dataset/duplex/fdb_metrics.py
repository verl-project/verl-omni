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
"""Minimal Full-Duplex-Bench v1 rule-metric harness (RFC #905 §3.7).

Restricted to FDB-v1's rule-based timing metrics — pause-violation rate, turn
latency distribution, and barge-in yield latency — computed over dumped
duplex replays rather than a live interactive judge: the static protocol the
RFC pins for the PoC. Consumes the JSONL dump the recipe writes (one
``DuplexReplay.to_extra_fields()`` payload per line) so evaluation runs after
the fact, offline.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from verl_omni.utils.dataset.duplex.trajectory import AXIS_INTERRUPTION, AXIS_PAUSE, AXIS_TURN

__all__ = ["fdb_rule_metrics", "main"]


def _episode_measurements(payload: Mapping[str, Any]) -> dict[str, float] | None:
    """Episode-relative timing measurements from one replay dump line."""
    event = payload.get("event") or {}
    axis = event.get("axis")
    layouts = payload.get("unit_layouts") or []
    if axis not in (AXIS_PAUSE, AXIS_TURN, AXIS_INTERRUPTION) or not layouts:
        return None
    speaking_times = [layout["time_s"] for layout in layouts if layout["speaking"]]
    silent_times = [layout["time_s"] for layout in layouts if not layout["speaking"]]
    measurements: dict[str, float] = {"num_units": float(len(layouts))}
    if axis == AXIS_PAUSE:
        start, end = event["pause_window_s"]
        measurements["pause_violation"] = float(any(start <= time_s < end for time_s in speaking_times))
    elif axis == AXIS_TURN:
        turn_end = float(event["user_turn_end_s"])
        onset = next((time_s for time_s in speaking_times if time_s >= turn_end), None)
        measurements["took_turn"] = float(onset is not None)
        measurements["turn_latency_s"] = onset - turn_end if onset is not None else float("inf")
    else:
        speech_end = float(event["interrupt_speech_end_s"])
        yield_at = next((time_s for time_s in silent_times if time_s >= speech_end), None)
        measurements["yield_latency_s"] = (
            yield_at - speech_end if yield_at is not None else (0.0 if not speaking_times else float("inf"))
        )
    return measurements


def fdb_rule_metrics(payloads: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Aggregate FDB-v1 rule metrics over dumped replays.

    Args:
        payloads: The replay dump entries (one per episode).

    Returns:
        ``pause_violation_rate``, ``turn_taking_rate``, ``turn_latency_s``
        (median over episodes that took the turn), ``yield_rate``, and
        ``yield_latency_s`` (median over episodes that yielded), plus counts.
    """
    pause_hits: list[float] = []
    turn_latencies: list[float] = []
    turn_taken: list[float] = []
    yield_latencies: list[float] = []
    yielded: list[float] = []
    counts = {"pause": 0, "turn": 0, "interruption": 0}
    for payload in payloads:
        measurements = _episode_measurements(payload)
        if measurements is None:
            continue
        if "pause_violation" in measurements:
            counts["pause"] += 1
            pause_hits.append(measurements["pause_violation"])
        if "took_turn" in measurements:
            counts["turn"] += 1
            turn_taken.append(measurements["took_turn"])
            if measurements["took_turn"] > 0:
                turn_latencies.append(measurements["turn_latency_s"])
        if "yield_latency_s" in measurements:
            counts["interruption"] += 1
            if measurements["yield_latency_s"] != float("inf"):
                yielded.append(1.0)
                yield_latencies.append(measurements["yield_latency_s"])
            else:
                yielded.append(0.0)
    metrics: dict[str, float] = {
        "fdb/pause_episodes": float(counts["pause"]),
        "fdb/turn_episodes": float(counts["turn"]),
        "fdb/interruption_episodes": float(counts["interruption"]),
    }
    if pause_hits:
        metrics["fdb/pause_violation_rate"] = sum(pause_hits) / len(pause_hits)
    if turn_taken:
        metrics["fdb/turn_taking_rate"] = sum(turn_taken) / len(turn_taken)
    if turn_latencies:
        metrics["fdb/turn_latency_s_median"] = sorted(turn_latencies)[len(turn_latencies) // 2]
    if yielded:
        metrics["fdb/yield_rate"] = sum(yielded) / len(yielded)
    if yield_latencies:
        metrics["fdb/yield_latency_s_median"] = sorted(yield_latencies)[len(yield_latencies) // 2]
    return metrics


def _read_dump(path: Path) -> Iterable[Mapping[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: ``python -m verl_omni.utils.dataset.duplex.fdb_metrics <dump.jsonl>...``."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(__doc__)
        return 2
    for argument in arguments:
        metrics = fdb_rule_metrics(_read_dump(Path(argument)))
        print(json.dumps({"dump": argument, **metrics}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
