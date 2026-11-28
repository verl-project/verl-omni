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
"""Contract tests for the experiment-campaign session adapter.

The adapter owns the append-only round log and derives every other file from it,
so these tests drive it through the CLI: round ordering, one verdict per round,
phase transitions, the close terminal, and the derived state.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER = REPO_ROOT / ".agents" / "skills" / "experiment-campaign" / "adapters" / "session.py"

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_HEADER = 2
EXIT_INVALID = 3
EXIT_ORDER = 4


def run(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Invoke the adapter the way the controller would."""
    completed = subprocess.run([sys.executable, str(ADAPTER), *args], capture_output=True, text=True, check=False)
    if check and completed.returncode != 0:
        raise AssertionError(f"adapter failed ({completed.returncode}): {completed.stderr}")
    return completed


@pytest.fixture()
def session(tmp_path: Path) -> Path:
    """A session with a mission and a two-round budget."""
    directory = tmp_path / "session"
    run(
        "init",
        "--dir",
        str(directory),
        "--id",
        "test",
        "--mission",
        "metric=timing_s/step,direction=lower,target=150",
        "--max-rounds",
        "2",
        check=True,
    )
    return directory


def start(directory: Path, phase: str = "", change: str = "none") -> str:
    """Start a round and return its id."""
    args = ["start", "--dir", str(directory), "--change", change]
    if phase:
        args += ["--phase", phase]
    payload = json.loads(run(*args, check=True).stdout)
    return payload["round"]


def measure(directory: Path, round_id: str, value: str, why: str = "") -> subprocess.CompletedProcess[str]:
    """Record a result."""
    return run("result", "--dir", str(directory), "--round", round_id, "--value", value, "--why", why)


def decide(directory: Path, round_id: str, verdict: str, why: str = "") -> subprocess.CompletedProcess[str]:
    """Record a verdict."""
    return run("verdict", "--dir", str(directory), "--round", round_id, "--verdict", verdict, "--why", why)


def state(directory: Path) -> dict:
    """Render and return the derived state."""
    return json.loads(run("render", "--dir", str(directory), check=True).stdout)


def test_init_creates_the_session_skeleton(session: Path) -> None:
    """init seeds the log, the state, the manifest and the folders."""
    header = (session / "rounds.tsv").read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert header[:4] == ["ts", "round", "phase", "event"]
    assert (session / "state.json").is_file()
    assert (session / "manifest.json").is_file()
    assert (session / "session.md").is_file()
    assert (session / "rounds").is_dir() and (session / "kb").is_dir()
    assert state(session)["next_action"] == "measure_baseline"
    assert run("init", "--dir", str(session), "--mission", "metric=m,direction=lower").returncode == EXIT_USAGE


def test_round_lifecycle_and_files(session: Path) -> None:
    """start → result → verdict, one folder per round with the evidence pointers."""
    round_id = start(session, "prelude", "none")
    assert round_id == "c01-pre-001"
    assert measure(session, round_id, "166.5", "measured").returncode == EXIT_OK
    assert decide(session, round_id, "baseline", "measured").returncode == EXIT_OK
    for name in ("change.yaml", "command.txt", "result.json", "verdict.json"):
        assert (session / "rounds" / round_id / name).is_file(), name
    derived = state(session)
    assert derived["anchor"]["value"] == 166.5
    assert derived["rounds"] == 1


def test_a_round_cannot_be_rewritten(session: Path) -> None:
    """Result and verdict are written once; a second write is refused."""
    round_id = start(session, "prelude")
    assert measure(session, round_id, "166.5").returncode == EXIT_OK
    assert measure(session, round_id, "165.0").returncode == EXIT_ORDER
    assert decide(session, round_id, "baseline").returncode == EXIT_OK
    assert decide(session, round_id, "keep").returncode == EXIT_ORDER
    # a decided round no longer blocks the next one
    assert run("start", "--dir", str(session), "--change", "other", "--phase", "optimization").returncode == EXIT_OK


def test_measure_before_judging(session: Path) -> None:
    """A verdict needs a result; a result needs a start."""
    round_id = start(session, "prelude")
    assert decide(session, round_id, "keep").returncode == EXIT_ORDER
    assert run("result", "--dir", str(session), "--round", "c01-opt-999", "--value", "1").returncode == EXIT_INVALID
    assert run("verdict", "--dir", str(session), "--round", "nope", "--verdict", "keep").returncode == EXIT_INVALID


def test_open_round_blocks_a_new_one(session: Path) -> None:
    """Only one round is in flight: finish it before starting another."""
    first = start(session, "prelude")
    blocked = run("start", "--dir", str(session), "--phase", "optimization", "--change", "x")
    assert blocked.returncode == EXIT_ORDER
    assert first in blocked.stderr


def test_keep_needs_a_measured_value(session: Path) -> None:
    """A keep with no number is not a keep."""
    round_id = start(session, "prelude")
    assert run("result", "--dir", str(session), "--round", round_id, "--quality", "degraded").returncode == EXIT_OK
    assert decide(session, round_id, "keep").returncode == EXIT_INVALID


def test_phase_order_and_cycles(session: Path) -> None:
    """prelude cannot jump to sweep, and sweep → optimization starts a new cycle."""
    assert run("phase", "--dir", str(session), "--to", "sweep").returncode == EXIT_ORDER
    assert run("phase", "--dir", str(session), "--to", "optimization").returncode == EXIT_OK
    round_id = start(session)
    assert round_id.startswith("c01-opt-")
    assert measure(session, round_id, "160.0").returncode == EXIT_OK
    assert decide(session, round_id, "keep", "improved").returncode == EXIT_OK
    assert run("phase", "--dir", str(session), "--to", "sweep").returncode == EXIT_OK
    assert run("phase", "--dir", str(session), "--to", "optimization").returncode == EXIT_OK
    assert state(session)["cycle"] == 2
    assert start(session).startswith("c02-opt-")


def test_noise_floor_from_the_baseline_repeats(session: Path) -> None:
    """The floor is max(2 sd, 1 % of the anchor), so one repeat cannot fake one."""
    for value in ("166.5", "168.1", "165.2"):
        round_id = start(session, "prelude")
        measure(session, round_id, value)
        decide(session, round_id, "baseline")
    derived = state(session)
    assert derived["anchor"]["value"] == 166.5
    assert derived["anchor"]["sd"] > 1.0
    assert derived["noise_floor"] == pytest.approx(2 * derived["anchor"]["sd"], rel=1e-3)


def test_only_keeps_enter_the_stack_and_the_best(session: Path) -> None:
    """A reverted round with a better number changes nothing."""
    baseline = start(session, "prelude")
    measure(session, baseline, "166.5")
    decide(session, baseline, "baseline")
    assert run("phase", "--dir", str(session), "--to", "optimization").returncode == EXIT_OK
    kept = start(session, "optimization", "max_num_seqs=128")
    measure(session, kept, "160.0")
    decide(session, kept, "keep", "improved")
    rejected = start(session, "optimization", "rollout.n=4")
    measure(session, rejected, "120.0")
    decide(session, rejected, "revert", "accuracy below floor")
    derived = state(session)
    assert derived["best"] == 160.0
    assert [entry["round"] for entry in derived["stack"]] == [kept]
    assert derived["target_reached"] is False


def test_anchor_is_the_prelude_and_later_baselines_are_controls(session: Path) -> None:
    """A control re-measure must not move the number earlier verdicts were judged against."""
    for value in ("160.0", "158.0"):
        round_id = start(session, "prelude")
        measure(session, round_id, value)
        decide(session, round_id, "baseline")
    assert state(session)["anchor"]["value"] == 159.0

    assert run("phase", "--dir", str(session), "--to", "optimization").returncode == EXIT_OK
    kept = start(session, "optimization", "max_num_seqs=128")
    measure(session, kept, "155.0")
    decide(session, kept, "keep", "improved")

    assert run("phase", "--dir", str(session), "--to", "sweep").returncode == EXIT_OK
    control = start(session, "sweep", "none")
    measure(session, control, "120.0")
    decide(session, control, "baseline", "control")

    derived = state(session)
    assert derived["anchor"]["value"] == 159.0, "a control moved the anchor"
    assert derived["anchor"]["controls"] == 1
    assert derived["best"] == 155.0


def test_close_is_terminal(session: Path) -> None:
    """Nothing is appended after close, and the close file records why."""
    round_id = start(session, "prelude")
    measure(session, round_id, "166.5")
    decide(session, round_id, "baseline")
    assert run("close", "--dir", str(session), "--reason", "budget", "--detail", "2 rounds").returncode == EXIT_OK
    assert (session / "close.md").is_file()
    derived = state(session)
    assert derived["phase"] == "close"
    assert derived["next_action"] == "done"
    assert derived["stop"]["reason"] == "budget"
    assert run("start", "--dir", str(session)).returncode == EXIT_ORDER
    assert run("close", "--dir", str(session), "--reason", "again").returncode == EXIT_ORDER


def test_budget_exhaustion_asks_to_close(session: Path) -> None:
    """With max_rounds reached the derived next action is close."""
    for value in ("166.5", "160.0"):
        round_id = start(session, "prelude")
        measure(session, round_id, value)
        decide(session, round_id, "baseline")
    assert state(session)["next_action"] == "close"


def test_fields_are_tsv_safe_and_the_header_is_checked(session: Path) -> None:
    """A tab would corrupt the log; a drifted header is refused rather than parsed."""
    assert run("start", "--dir", str(session), "--change", "a\tb").returncode == EXIT_INVALID
    ledger = session / "rounds.tsv"
    ledger.write_text(ledger.read_text(encoding="utf-8").replace("event", "kind"), encoding="utf-8")
    assert run("start", "--dir", str(session), "--change", "x").returncode == EXIT_HEADER
