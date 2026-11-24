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
        'nsight_dir="$TMPDIR/ray/session_latest/logs/nsight"\n'
        'mkdir -p "$nsight_dir"\n'
        'if [[ "${TEST_WRITE_REPORT:-1}" == 1 ]]; then\n'
        '    printf report > "$nsight_dir/worker.run-${PROFILE_RUN_ID}.rank0.host.pid123.1.nsys-rep"\n'
        "fi\n"
        'if [[ "${TEST_WRITE_RAW:-1}" == 1 ]]; then\n'
        '    printf raw > "$nsight_dir/gpu_metrics.run-${PROFILE_RUN_ID}.host.pid456.1.qdstrm"\n'
        "fi\n"
        'printf other > "$nsight_dir/worker.run-other.rank0.host.pid789.nsys-rep"\n'
        'exit "${TRAINING_STATUS:-0}"\n'
    )

    def run(responses, version="v1", training_status=0, hydra_overrides=(), **overrides):
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
            ["bash", str(wrapper), str(recipe), *hydra_overrides],
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
    assert (
        '+global_profiler.global_tool_config.nsys.controller_nsight_options.o="controller.run-trial.%h.pid%p"' in argv
    )
    assert (
        '+global_profiler.global_tool_config.nsys.worker_nsight_options.o="worker.run-trial.rank%q{RANK}.%h.pid%p"'
        in argv
    )
    if version == "v1":
        assert '++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.gpu-metrics-devices="0"' in argv
        assert "++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.gpu-metrics-frequency=10" in argv
        assert (
            '++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.o="gpu_metrics.run-trial.%h.pid%p"'
            in argv
        )
    assert {p.name: p.read_text() for p in (tmp_path / "run/nsight").iterdir()} == {
        "worker.run-trial.rank0.host.pid123.1.nsys-rep": "report",
        "gpu_metrics.run-trial.host.pid456.1.qdstrm": "raw",
    }


def test_collector_matches_complete_run_field_with_role_prefixes(launch, tmp_path):
    source = tmp_path / "ray/session_latest/logs/nsight"
    source.mkdir(parents=True)
    selected = {
        "controller.run-1.host.pid8.1.nsys-rep",
        "reward_loop.rank2.worker.run-1.rank2.host.pid9.2.qdstrm",
        "rollout_server_node_0.rank0.worker.run-1.rank0.host.pid10.1.nsys-rep",
    }
    unrelated = {
        "worker.run-other.rank0.host.pid123.nsys-rep",
        "worker.run-other.rank0.host.pid123.1.nsys-rep",
        "worker.run-12.rank0.host.pid8.nsys-rep",
        "controller.run-1-extra.host.pid9.qdstrm",
    }
    for name in selected | unrelated:
        (source / name).write_text(name)

    result = launch([[], []], PROFILE_RUN_ID="1")

    assert result.returncode == 0, result.stderr
    expected = selected | {"worker.run-1.rank0.host.pid123.1.nsys-rep", "gpu_metrics.run-1.host.pid456.1.qdstrm"}
    assert {p.name for p in (tmp_path / "run/nsight").iterdir()} == expected
    assert all((source / name).read_text() == name for name in selected | unrelated)


def test_collector_keeps_runs_separate_in_shared_source_directory(launch, tmp_path):
    first = launch([[], []], RUN_DIR=str(tmp_path / "first"))
    second = launch([[], []], PROFILE_RUN_ID="trial-next", RUN_DIR=str(tmp_path / "second"))

    assert first.returncode == second.returncode == 0
    for directory, run_id in [("first", "trial"), ("second", "trial-next")]:
        assert {p.name for p in (tmp_path / directory / "nsight").iterdir()} == {
            f"worker.run-{run_id}.rank0.host.pid123.1.nsys-rep",
            f"gpu_metrics.run-{run_id}.host.pid456.1.qdstrm",
        }


@pytest.mark.parametrize(
    "training_status, write_raw, expected_status",
    [(0, False, 1), (0, True, 1), (7, True, 7)],
    ids=["no-artifacts", "raw-only", "training-failure"],
)
def test_collector_fails_without_reports(launch, tmp_path, training_status, write_raw, expected_status):
    result = launch(
        [[], []],
        training_status=training_status,
        TEST_WRITE_REPORT="0",
        TEST_WRITE_RAW=str(int(write_raw)),
    )

    assert result.returncode == expected_status
    assert "no .nsys-rep reports" in result.stderr
    assert "Copied 0 report(s)" in result.stdout
    raw = tmp_path / "run/nsight/gpu_metrics.run-trial.host.pid456.1.qdstrm"
    assert raw.exists() is write_raw
    if write_raw:
        assert raw.read_text() == "raw"


def test_v1_metrics_defaults_preserve_caller_overrides(launch, tmp_path):
    overrides = [
        '++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.gpu-metrics-devices="0,4"',
        "++global_profiler.global_tool_config.nsys.gpu_metrics_nsight_options.gpu-metrics-frequency=20",
    ]
    result = launch([[], []], hydra_overrides=overrides)

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "argv").read_text().splitlines()[-2:] == overrides


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
        assert (tmp_path / "run/nsight/worker.run-trial.rank0.host.pid123.1.nsys-rep").read_text() == "report"
        assert (tmp_path / "run/nsight/gpu_metrics.run-trial.host.pid456.1.qdstrm").read_text() == "raw"


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
    assert (tmp_path / "run/nsight/worker.run-trial.rank0.host.pid123.1.nsys-rep").read_text() == "report"
    assert (tmp_path / "run/nsight/gpu_metrics.run-trial.host.pid456.1.qdstrm").read_text() == "raw"
