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
"""Session adapter for the ``experiment-campaign`` skill.

A session is one closed optimisation loop::

    PRELUDE -> [ENABLEMENT] -> OPTIMIZATION <-> SWEEP -> CLOSE
                                  ^             |
                                  +-- reloop ---+

ENABLEMENT only runs when the baseline does not; a sweep followed by another
optimisation cycle increments the cycle counter.

`rounds.tsv` is an append-only log of what was measured and decided; every other
file in the session directory is derived from it. The adapter owns exactly that
log: it allocates round ids, refuses to rewrite a verdict, and re-derives
`state.json`, `session.md` and `session_breakdown.json` from the rows.

Formats (columns, state fields, round files) are specified in
``references/records.md``; the phase rules live in ``references/phases.md``.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# AUTHORITATIVE column order; every session's header must match it byte for byte.
COLUMNS = "ts round phase event change value noise verdict quality why".split()

EVENTS = ("start", "result", "verdict", "phase", "close")
PHASES = ("prelude", "enablement", "optimization", "sweep")
TERMINAL = "close"
VERDICTS = ("baseline", "keep", "revert", "crash")
QUALITIES = ("ok", "smoke", "degraded")
PHASE_CODE = {"prelude": "pre", "enablement": "ena", "optimization": "opt", "sweep": "swp"}
# phase -> phases a session may move to next
NEXT_PHASES = {
    "prelude": ("enablement", "optimization", "close"),
    "enablement": ("optimization", "close"),
    "optimization": ("sweep", "close"),
    "sweep": ("optimization", "close"),
    "close": (),
}

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_HEADER = 2
EXIT_INVALID = 3
EXIT_ORDER = 4

_DEFAULT_NOISE = 0.01


class SessionError(Exception):
    """A contract violation, carrying the exit code it maps to."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


def _now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _build_round(cycle: int) -> str:
    return f"c{cycle:02d}"


def _row(values: dict[str, Any]) -> str:
    return "\t".join(str(values.get(column, "")) for column in COLUMNS)


def _number(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SessionError(EXIT_INVALID, f"{what} must be a number, got {value!r}") from error
    if number != number or number in (float("inf"), float("-inf")):
        raise SessionError(EXIT_INVALID, f"{what} must be finite, got {value!r}")
    return number


def _ledger(directory: Path) -> Path:
    return directory / "rounds.tsv"


def read_rows(directory: Path) -> list[dict[str, str]]:
    """Read the log, verifying its header."""
    path = _ledger(directory)
    if not path.is_file():
        raise SessionError(EXIT_USAGE, f"no rounds.tsv in {directory}; run `init` first")
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "\t".join(COLUMNS):
        raise SessionError(
            EXIT_HEADER,
            f"rounds.tsv header does not match the schema ({len(COLUMNS)} columns: {', '.join(COLUMNS)})",
        )
    rows = []
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != len(COLUMNS):
            raise SessionError(EXIT_INVALID, f"line {number}: expected {len(COLUMNS)} columns, got {len(fields)}")
        rows.append(dict(zip(COLUMNS, fields, strict=True)))
    return rows


def append_row(directory: Path, values: dict[str, Any]) -> dict[str, str]:
    """Validate one row and append it atomically."""
    row = {column: str(values.get(column, "")) for column in COLUMNS}
    if any("\t" in value or "\n" in value for value in row.values()):
        raise SessionError(EXIT_INVALID, "a field must not contain tabs or line breaks")
    if row["event"] not in EVENTS:
        raise SessionError(EXIT_INVALID, f"unknown event {row['event']!r}; expected one of {EVENTS}")
    if row["verdict"] and row["verdict"] not in VERDICTS:
        raise SessionError(EXIT_INVALID, f"unknown verdict {row['verdict']!r}; expected one of {VERDICTS}")
    if row["quality"] and row["quality"] not in QUALITIES:
        raise SessionError(EXIT_INVALID, f"unknown quality {row['quality']!r}")
    if row["value"]:
        _number(row["value"], "value")
    if row["noise"]:
        _number(row["noise"], "noise")
    check_order(read_rows(directory), row)

    path = _ledger(directory)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(_row(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return row


def check_order(rows: list[dict[str, str]], row: dict[str, str]) -> None:
    """Refuse to rewrite history: one start, one result, one verdict per round."""
    if any(existing["event"] == "close" for existing in rows):
        raise SessionError(EXIT_ORDER, "the session is closed; a closed log is terminal — start a new session")
    event = row["event"]
    if event == "close":
        return
    if event == "phase":
        current = _current_phase(rows)
        target = row["phase"]
        if target != TERMINAL and target not in NEXT_PHASES.get(current, ()):
            raise SessionError(
                EXIT_ORDER,
                f"phase {current!r} cannot move to {target!r}; allowed: "
                f"{', '.join(NEXT_PHASES.get(current, ())) or 'nothing (closed)'}",
            )
        return
    round_id = row["round"]
    if not round_id:
        raise SessionError(EXIT_INVALID, f"{event} requires --round")
    mine = [existing for existing in rows if existing["round"] == round_id]
    started = [existing for existing in mine if existing["event"] == "start"]
    results = [existing for existing in mine if existing["event"] == "result"]
    verdicts = [existing for existing in mine if existing["event"] == "verdict"]
    if event == "start":
        open_rounds = {
            existing["round"]
            for existing in rows
            if existing["event"] == "start"
            and not any(other["event"] == "verdict" and other["round"] == existing["round"] for other in rows)
        }
        if open_rounds:
            raise SessionError(
                EXIT_ORDER,
                f"round {sorted(open_rounds)[0]} is still open; record its verdict before starting another",
            )
        if started:
            raise SessionError(EXIT_ORDER, f"round {round_id} already exists")
    elif event == "result":
        if not started:
            raise SessionError(EXIT_ORDER, f"round {round_id} has no start row")
        if results:
            raise SessionError(EXIT_ORDER, f"round {round_id} already has a result; a measurement is recorded once")
        if verdicts:
            raise SessionError(EXIT_ORDER, f"round {round_id} is already decided")
    elif event == "verdict":
        if not results:
            raise SessionError(EXIT_ORDER, f"round {round_id} has no result: measure before judging")
        if verdicts:
            raise SessionError(EXIT_ORDER, f"round {round_id} already has a verdict; verdicts are not rewritten")
        if row["verdict"] in ("baseline", "keep") and not results[0]["value"]:
            raise SessionError(EXIT_INVALID, f"a {row['verdict']} needs a measured value")


def _current_phase(rows: list[dict[str, str]]) -> str:
    for row in reversed(rows):
        if row["event"] in ("phase", "close"):
            return row["phase"]
    return "prelude"


def _rounds(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Fold start/result/verdict rows into one record per round."""
    folded: dict[str, dict[str, str]] = {}
    for row in rows:
        if not row["round"]:
            continue
        entry = folded.setdefault(row["round"], {"round": row["round"], "phase": row["phase"], "change": row["change"]})
        if row["event"] == "start":
            entry.update(phase=row["phase"], change=row["change"], ts=row["ts"])
        elif row["event"] == "result":
            entry.update(value=row["value"], noise=row["noise"], quality=row["quality"], why=row["why"])
        elif row["event"] == "verdict":
            entry.update(verdict=row["verdict"], why=row["why"] or entry.get("why", ""), decided=row["ts"])
    return folded


def _cycle(rows: list[dict[str, str]]) -> int:
    """A sweep followed by more optimisation is a new cycle."""
    cycle = 1
    previous = "prelude"
    for row in rows:
        if row["event"] not in ("phase", "close"):
            continue
        if previous == "sweep" and row["phase"] == "optimization":
            cycle += 1
        previous = row["phase"]
    return cycle


def derive(directory: Path) -> dict[str, Any]:
    """Recompute the whole session view from the log."""
    rows = read_rows(directory)
    state_file = directory / "state.json"
    previous = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    rounds = _rounds(rows)
    decided = {key: value for key, value in rounds.items() if value.get("verdict")}
    all_baselines = [value for value in decided.values() if value["verdict"] == "baseline"]
    # The anchor is what PRELUDE measured. A baseline re-measured later (a control for
    # drift) must not move the number the earlier verdicts were judged against.
    prelude = [value for value in all_baselines if value.get("phase") == "prelude"]
    anchor_rows = prelude or all_baselines
    baselines = [float(value["value"]) for value in anchor_rows if value["value"]]
    controls = len(all_baselines) - len(anchor_rows)
    keeps = [value for key, value in decided.items() if value["verdict"] == "keep"]
    crashs = [value for value in decided.values() if value["verdict"] == "crash"]
    phase = _current_phase(rows)
    closed = any(row["event"] == "close" for row in rows)
    open_round = next((key for key, value in rounds.items() if not value.get("verdict")), "")
    cycle = _cycle(rows)

    budget = previous.get("budget", {})
    max_rounds = int(budget.get("max_rounds", 0) or 0)
    mission = previous.get("mission", {})
    direction = mission.get("direction", "lower")
    anchor_value = statistics.median(baselines) if baselines else None
    anchor_sd = statistics.stdev(baselines) if len(baselines) > 1 else 0.0
    noise = max(mission.get("rel_floor", _DEFAULT_NOISE) * abs(anchor_value or 0.0), anchor_sd * 2)

    best = anchor_value
    for value in keeps:
        candidate = float(value["value"])
        if best is None:
            best = candidate
            continue
        better = candidate < best - noise if direction == "lower" else candidate > best + noise
        if better:
            best = candidate
    target = mission.get("target")
    reached = (
        target is not None
        and best is not None
        and (best <= float(target) if direction == "lower" else best >= float(target))
    )
    if closed:
        next_action = "done"
    elif open_round:
        next_action = "finish_round"
    elif phase == "prelude" and not baselines:
        next_action = "measure_baseline"
    elif max_rounds and len(decided) >= max_rounds:
        next_action = "close"
    elif reached:
        next_action = "close"
    else:
        next_action = "run_round"

    entered: list[str] = []
    for row in rows:
        if row["phase"] and row["phase"] not in entered:
            entered.append(row["phase"])

    return {
        "session_id": previous.get("session_id", directory.name),
        "phase": phase,
        "phases_entered": entered,
        "cycle": cycle,
        "mission": mission,
        "budget": budget,
        "anchor": {
            "value": anchor_value,
            "sd": anchor_sd,
            "measurements": baselines,
            "rounds": len(baselines),
            "controls": controls,
        },
        "noise_floor": round(noise, 4),
        "best": best,
        "target_reached": reached,
        "stack": [
            {"round": round_id, "change": value.get("change", ""), "value": value.get("value", "")}
            for round_id, value in decided.items()
            if value["verdict"] == "keep"
        ],
        "rounds": len(decided),
        "crashes": len(crashs),
        "current": open_round or "",
        "next_action": next_action,
        "stop": previous.get("stop", {"reason": None, "detail": None}),
    }


def _write(directory: Path, name: str, payload: str) -> None:
    path = directory / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _session_md(state: dict[str, Any], records: dict[str, dict[str, str]]) -> str:
    mission = state["mission"]
    anchor = state["anchor"]
    lines = [
        f"# {state['session_id']} — {mission.get('metric', '')} "
        f"({'lower' if mission.get('direction') == 'lower' else 'higher'} is better)"
        f"{'  CLOSED' if state['phase'] == TERMINAL else ''}",
        "",
        f"Updated: {_now()} · Phase: **{state['phase']}** · Cycle: {state['cycle']} · Next: `{state['next_action']}`",
        "",
        "## Mission",
        f"- metric `{mission.get('metric', '')}` ({mission.get('direction', '')}), "
        f"target {mission.get('target', 'none')}",
        f"- budget: {state['budget'].get('max_rounds', 'unbounded')} rounds, "
        f"{state['budget'].get('per_round_timeout', 'no')} per round",
        "",
        "## Where",
        f"- anchor **{anchor['value']}** over {anchor['rounds']} baseline round(s), "
        f"sd {round(anchor['sd'], 3)}, noise floor {state['noise_floor']}",
        f"- best kept **{state['best']}**, rounds decided {state['rounds']}, crashes {state['crashes']}",
        f"- target reached: {state['target_reached']}",
        "",
        "## Rounds",
        "| round | phase | change | value | verdict | why |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for round_id, record in records.items():
        lines.append(
            f"| {round_id} | {record.get('phase', '')} | {record.get('change', '')} | "
            f"{record.get('value', '')} | {record.get('verdict', 'pending')} | {record.get('why', '')} |"
        )
    if not records:
        lines.append("| — | — | — | — | — | — |")
    lines += ["", "## Next action", f"`{state['next_action']}`", ""]
    if state["phase"] == TERMINAL:
        lines += ["## Stop", f"{state['stop'].get('reason')}: {state['stop'].get('detail')}", ""]
    return "\n".join(lines)


def _breakdown(state: dict[str, Any], records: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": "verl-omni.session_breakdown.v1",
        "session_id": state["session_id"],
        "phases": {
            "order": ["prelude", "enablement", "optimization", "sweep", "close"],
            "entered": state["phases_entered"],
            "current": state["phase"],
            "cycles": state["cycle"],
        },
        "outcome": {
            "baseline": state["anchor"]["value"],
            "baseline_sd": state["anchor"]["sd"],
            "noise_floor": state["noise_floor"],
            "final": state["best"],
            "target_reached": state["target_reached"],
            "rounds": state["rounds"],
            "crashes": state["crashes"],
            "stop": state["stop"],
        },
        "rounds": list(records.values()),
    }


def _render(directory: Path) -> dict[str, Any]:
    state = derive(directory)
    records = _rounds(read_rows(directory))
    _write(directory, "state.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    _write(directory, "session.md", _session_md(state, records))
    breakdown = json.dumps(_breakdown(state, records), indent=2, ensure_ascii=False)
    _write(directory, "session_breakdown.json", breakdown + "\n")
    return state


def _round_dir(directory: Path, round_id: str) -> Path:
    path = directory / "rounds" / round_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "artifacts").mkdir(exist_ok=True)
    return path


def cmd_init(args: argparse.Namespace) -> int:
    """Create a session skeleton."""
    directory = Path(args.dir).resolve()
    if _ledger(directory).is_file() and _ledger(directory).read_text(encoding="utf-8").strip():
        raise SessionError(EXIT_USAGE, f"{directory} already holds a session")
    for name in ("rounds", "kb"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    _ledger(directory).write_text("\t".join(COLUMNS) + "\n", encoding="utf-8")
    mission = {}
    for item in args.mission.split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        mission[key.strip()] = float(value) if key.strip() == "target" and value.strip() else value.strip()
    if "metric" not in mission or "direction" not in mission:
        raise SessionError(EXIT_INVALID, "--mission needs at least metric=... and direction=lower|higher")
    state = {
        "session_id": args.id or directory.name,
        "phase": "prelude",
        "cycle": 1,
        "mission": mission,
        "budget": {"max_rounds": args.max_rounds, "per_round_timeout": args.per_round_timeout},
        "anchor": {"value": None, "sd": 0.0, "measurements": [], "rounds": 0},
        "noise_floor": 0.0,
        "best": None,
        "target_reached": False,
        "stack": [],
        "rounds": 0,
        "crashes": 0,
        "current": "",
        "next_action": "measure_baseline",
        "stop": {"reason": None, "detail": None},
    }
    _write(directory, "state.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "session_id": state["session_id"],
        "created_at": _now(),
        "host": os.uname().nodename,
        "cwd": str(Path.cwd()),
        "launcher": args.launcher or "",
        "extractor": args.extractor or "",
    }
    _write(directory, "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    _render(directory)
    print(json.dumps({"session": str(directory), "phase": "prelude", "next_action": "measure_baseline"}))
    return EXIT_OK


def cmd_start(args: argparse.Namespace) -> int:
    """Allocate the next round for a phase and record what it will try."""
    directory = Path(args.dir).resolve()
    state = derive(directory)
    if state["phase"] == TERMINAL:
        raise SessionError(EXIT_ORDER, "the session is closed; start a new session")
    phase = args.phase or state["phase"]
    if phase not in PHASES:
        raise SessionError(EXIT_INVALID, f"phase must be one of {PHASES}, got {phase!r}")
    if phase != state["phase"]:
        append_row(directory, {"ts": _now(), "phase": phase, "event": "phase"})
        state = derive(directory)
    existing = {
        row["round"] for row in read_rows(directory) if row["round"].startswith(_build_round(state["cycle"]) + "-")
    }
    round_id = f"{_build_round(state['cycle'])}-{PHASE_CODE[phase]}-{len(existing) + 1:03d}"
    append_row(
        directory,
        {"ts": _now(), "round": round_id, "phase": phase, "event": "start", "change": args.change or "none"},
    )
    round_dir = _round_dir(directory, round_id)
    (round_dir / "change.yaml").write_text(f"phase: {phase}\nchange: {args.change or 'none'}\n", encoding="utf-8")
    (round_dir / "command.txt").write_text((args.command or "") + "\n", encoding="utf-8")
    _render(directory)
    print(json.dumps({"round": round_id, "phase": phase, "dir": str(round_dir)}))
    return EXIT_OK


def cmd_result(args: argparse.Namespace) -> int:
    """Record the measurement of a round."""
    directory = Path(args.dir).resolve()
    value = args.value
    append_row(
        directory,
        {
            "ts": _now(),
            "round": args.round,
            "phase": _round_phase(directory, args.round),
            "event": "result",
            "value": "" if value is None else value,
            "noise": "" if args.noise is None else args.noise,
            "quality": args.quality,
            "why": args.why or "",
        },
    )
    round_dir = _round_dir(directory, args.round)
    payload = {
        "round": args.round,
        "value": None if value is None else _number(value, "value"),
        "noise": None if args.noise is None else _number(args.noise, "noise"),
        "quality": args.quality,
        "why": args.why or "",
        "evidence": args.evidence or f"rounds/{args.round}/artifacts",
        "recorded_at": _now(),
    }
    (round_dir / "result.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _render(directory)
    print(json.dumps({"round": args.round, "value": payload["value"], "next_action": derive(directory)["next_action"]}))
    return EXIT_OK


def cmd_verdict(args: argparse.Namespace) -> int:
    """Record the keep/revert decision for a round."""
    directory = Path(args.dir).resolve()
    append_row(
        directory,
        {
            "ts": _now(),
            "round": args.round,
            "phase": _round_phase(directory, args.round),
            "event": "verdict",
            "verdict": args.verdict,
            "why": args.why or "",
        },
    )
    round_dir = _round_dir(directory, args.round)
    payload = {"round": args.round, "verdict": args.verdict, "why": args.why or "", "decided_at": _now()}
    (round_dir / "verdict.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.notes:
        (round_dir / "notes.md").write_text(args.notes.rstrip() + "\n", encoding="utf-8")
    _render(directory)
    print(json.dumps({"round": args.round, "verdict": args.verdict, "next_action": derive(directory)["next_action"]}))
    return EXIT_OK


def _round_phase(directory: Path, round_id: str) -> str:
    for row in read_rows(directory):
        if row["round"] == round_id and row["event"] == "start":
            return row["phase"]
    raise SessionError(EXIT_INVALID, f"unknown round {round_id!r}")


def cmd_phase(args: argparse.Namespace) -> int:
    """Move the session to another phase."""
    directory = Path(args.dir).resolve()
    append_row(directory, {"ts": _now(), "phase": args.to, "event": "phase"})
    _render(directory)
    print(json.dumps({"phase": args.to}))
    return EXIT_OK


def cmd_close(args: argparse.Namespace) -> int:
    """Stop the session and record why."""
    directory = Path(args.dir).resolve()
    append_row(directory, {"ts": _now(), "phase": TERMINAL, "event": "close", "why": args.reason})
    state = derive(directory)
    state["stop"] = {"reason": args.reason, "detail": args.detail or ""}
    _write(directory, "state.json", json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    records = _rounds(read_rows(directory))
    close_md = directory / "close.md"
    if not close_md.is_file():
        lines = [
            f"# Session closed — {state['session_id']}",
            "",
            f"- reason: **{args.reason}**",
            f"- detail: {args.detail or 'none'}",
            f"- anchor {state['anchor']['value']} → best {state['best']} (noise floor {state['noise_floor']})",
            f"- target reached: {state['target_reached']}; rounds {state['rounds']}; crashes {state['crashes']}",
            "",
            "## What was kept",
            "",
        ]
        for round_id, record in records.items():
            if record.get("verdict") == "keep":
                lines.append(f"- {round_id}: {record.get('change')} → {record.get('value')}")
        lines += ["", "## What to remember", "", "- (add the lessons and the winning recipe to kb/)"]
        close_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _render(directory)
    print(json.dumps({"closed": True, "reason": args.reason}))
    return EXIT_OK


def cmd_render(args: argparse.Namespace) -> int:
    """Re-derive state, session.md and the breakdown from the log."""
    directory = Path(args.dir).resolve()
    state = _render(directory)
    print(json.dumps(state, ensure_ascii=False))
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    """Build the command line interface."""
    parser = argparse.ArgumentParser(description="Append-only session log for experiment-campaign.")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="create a session skeleton")
    init.add_argument("--dir", required=True)
    init.add_argument("--id", default="")
    init.add_argument("--mission", required=True, help="metric=...,direction=lower|higher[,target=...]")
    init.add_argument("--max-rounds", dest="max_rounds", type=int, default=0)
    init.add_argument("--round-timeout", dest="per_round_timeout", default="")
    init.add_argument("--launcher", default="")
    init.add_argument("--extractor", default="")
    init.set_defaults(handler=cmd_init)

    start = commands.add_parser("start", help="allocate the next round")
    start.add_argument("--dir", required=True)
    start.add_argument("--phase", default="", help="omit to keep the current phase")
    start.add_argument("--change", default="", help="the one change this round tries")
    start.add_argument("--command", default="", help="the exact command that will run")
    start.set_defaults(handler=cmd_start)

    result = commands.add_parser("result", help="record the measurement of a round")
    result.add_argument("--dir", required=True)
    result.add_argument("--round", required=True)
    result.add_argument("--value", default=None)
    result.add_argument("--noise", default=None)
    result.add_argument("--quality", default="ok", choices=QUALITIES)
    result.add_argument("--why", default="")
    result.add_argument("--evidence", default="")
    result.set_defaults(handler=cmd_result)

    verdict = commands.add_parser("verdict", help="record keep/revert")
    verdict.add_argument("--dir", required=True)
    verdict.add_argument("--round", required=True)
    verdict.add_argument("--verdict", required=True, choices=VERDICTS)
    verdict.add_argument("--why", default="")
    verdict.add_argument("--notes", default="")
    verdict.set_defaults(handler=cmd_verdict)

    phase = commands.add_parser("phase", help="move to another phase")
    phase.add_argument("--dir", required=True)
    phase.add_argument("--to", required=True, choices=PHASES + (TERMINAL,))
    phase.set_defaults(handler=cmd_phase)

    close = commands.add_parser("close", help="stop the session")
    close.add_argument("--dir", required=True)
    close.add_argument("--reason", required=True, help="budget | converged | target | plateau | stopped")
    close.add_argument("--detail", default="")
    close.set_defaults(handler=cmd_close)

    render = commands.add_parser("render", help="re-derive state and reports")
    render.add_argument("--dir", required=True)
    render.set_defaults(handler=cmd_render)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command and map contract violations to exit codes."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except SessionError as error:
        print(f"session: {error}", file=sys.stderr)
        return error.code


if __name__ == "__main__":
    sys.exit(main())
