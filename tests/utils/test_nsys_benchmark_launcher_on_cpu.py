# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Exercise the report collector with stub Nsight sessions and no Ray or GPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def launch(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python").symlink_to(sys.executable)
    nsys = bin_dir / "nsys"
    nsys.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['TEST_ROOT'])\n"
        "counter = root / 'calls'\n"
        "index = int(counter.read_text()) if counter.exists() else 0\n"
        "counter.write_text(str(index + 1))\n"
        "responses = json.loads((root / 'responses').read_text())\n"
        "response = responses[min(index, len(responses) - 1)]\n"
        "if response == 'query-error':\n"
        "    print('mock query failure', file=sys.stderr)\n"
        "    raise SystemExit(1)\n"
        "if response == 'slow-query':\n"
        "    time.sleep(2)\n"
        "    response = []\n"
        "print('invalid-json' if response == 'invalid-json' else json.dumps(response))\n"
    )
    nsys.chmod(0o700)
    recipe = tmp_path / "recipe.sh"
    recipe.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_ROOT/argv"\n'
        'mkdir -p "$TMPDIR/ray/session_latest/logs/nsight"\n'
        'printf report > "$TMPDIR/ray/session_latest/logs/nsight/worker_trial_pid123.nsys-rep"\n'
        'printf raw > "$TMPDIR/ray/session_latest/logs/nsight/metrics_trial_pid456.qdstrm"\n'
        'printf other > "$TMPDIR/ray/session_latest/logs/nsight/worker_other_pid789.nsys-rep"\n'
        'exit "${TRAINING_STATUS:-0}"\n'
    )

    def run(responses, version="v1", training_status=0, **overrides):
        (tmp_path / "responses").write_text(json.dumps(responses))
        wrapper = Path(__file__).resolve().parents[2] / "examples" / f"run_diffusion_benchmark_nsys_{version}.sh"
        env = dict(
            os.environ,
            PATH=f"{bin_dir}:{os.environ['PATH']}",
            TEST_ROOT=str(tmp_path),
            TMPDIR=str(tmp_path),
            RUN_DIR=str(tmp_path / "run"),
            PROFILE_RUN_ID="trial",
            TRAINING_STEPS="4",
            NUM_NODES="1",
            TRAINING_STATUS=str(training_status),
        )
        env.update(overrides)
        return subprocess.run(
            ["bash", str(wrapper), str(recipe)],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )

    return run


@pytest.mark.parametrize("version", ["v0", "v1"])
def test_collector_waits_for_new_sessions_and_ignores_existing_sessions(launch, tmp_path, version):
    old = {"name": "profile-1-python", "state": "Generating"}
    result = launch(
        [
            [old],
            [old, {"name": "profile-2-python", "state": "Unknown"}],
            [
                old,
                {"name": "profile-2-python", "state": "StartRange"},
                {"name": "profile-3-python", "state": "Launched"},
            ],
        ],
        version,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "calls").read_text() == "3"
    argv = (tmp_path / "argv").read_text().splitlines()
    assert not any("session-new" in arg for arg in argv)
    assert {p.name: p.read_text() for p in (tmp_path / "run/nsight").iterdir()} == {
        "worker_trial_pid123.nsys-rep": "report",
        "metrics_trial_pid456.qdstrm": "raw",
    }


@pytest.mark.parametrize(
    "response, diagnostic", [("query-error", "mock query failure"), ("invalid-json", "JSONDecodeError")]
)
@pytest.mark.parametrize("before_training, training_status, exit_code", [(True, 0, 1), (False, 0, 1), (False, 7, 7)])
def test_collector_reports_failures_and_preserves_training_status(
    launch, tmp_path, response, diagnostic, before_training, training_status, exit_code
):
    result = launch([response] if before_training else [[], response], training_status=training_status)

    assert result.returncode == exit_code
    assert "Nsight session" in result.stderr
    assert diagnostic in result.stderr
    assert "still pending" not in result.stderr
    assert (tmp_path / "argv").exists() is not before_training
    if not before_training:
        assert (tmp_path / "run/nsight/worker_trial_pid123.nsys-rep").read_text() == "report"
        assert (tmp_path / "run/nsight/metrics_trial_pid456.qdstrm").read_text() == "raw"


def test_collector_accepts_slow_query_that_finishes_after_wait_budget(launch):
    result = launch([[], "slow-query"], PROFILE_FINALIZE_TIMEOUT_S="1")

    assert result.returncode == 0, result.stderr
    query_log = next(
        line for line in result.stderr.splitlines() if line.startswith("NSIGHT_QUERY phase=after_training")
    )
    assert "exit_code=0" in query_log
    assert int(query_log.split("elapsed_ms=")[1]) >= 2000


def test_collector_reports_wait_budget_expiry_and_preserves_artifacts(launch, tmp_path):
    result = launch([[], [{"name": "profile-2", "state": "Generating"}]], PROFILE_FINALIZE_TIMEOUT_S="1")

    assert result.returncode == 1
    assert "still pending after 1s waiting budget" in result.stderr
    assert "profile-2 Generating" in result.stderr
    assert (tmp_path / "run/nsight/worker_trial_pid123.nsys-rep").read_text() == "report"
    assert (tmp_path / "run/nsight/metrics_trial_pid456.qdstrm").read_text() == "raw"
