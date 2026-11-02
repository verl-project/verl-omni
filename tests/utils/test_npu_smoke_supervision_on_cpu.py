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
"""Check supervised serial dispatch and shell exits without starting accelerator workers."""

import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOCATION = "2,3,4,5,6,7,0,1"
_FLAG = "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"


@pytest.fixture
def parallel_smoke(monkeypatch):
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "tests/npu_smoke"))
    return importlib.import_module("run_npu_smoke_parallel")


@pytest.fixture
def shell_runner(tmp_path):
    scripts = tmp_path / "repo/tests/npu_smoke"
    scripts.mkdir(parents=True)
    source = (_REPO_ROOT / "tests/npu_smoke/lib_npu_smoke.sh").read_text()
    # macOS Bash 3 lacks only this display expansion; dispatch and cleanup are unchanged.
    (scripts / "lib_npu_smoke.sh").write_text(source.replace("${NPU_SMOKE_GROUP^^}", "${NPU_SMOKE_GROUP}"))
    binary = tmp_path / "bin"
    binary.mkdir()
    capture = tmp_path / "commands.jsonl"
    executable = binary / "python3"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\n"
        "with open(os.environ['CAPTURE'],'a') as stream:\n"
        " stream.write(json.dumps({'args':sys.argv[1:],'env':"
        "{'ASCEND_VISIBLE_DEVICES':os.environ.get('ASCEND_VISIBLE_DEVICES')}})+'\\n')\n"
        "raise SystemExit(int(os.environ.get('CLEANUP_STATUS','0')) if sys.argv[1:]=="
        "['tests/npu_smoke/run_npu_smoke_processes.py','cleanup'] else 0)\n"
    )
    executable.chmod(0o755)
    for name in ("ray", "pkill"):
        executable = binary / name
        executable.write_text(f'#!/bin/bash\necho {name} >> "$FORBIDDEN"\nexit 97\n')
        executable.chmod(0o755)

    def run(body, *, count=2, parallel="0", full="0", child="0", cleanup=0, group="core"):
        capture.unlink(missing_ok=True)
        env = {key: value for key, value in os.environ.items() if not key.startswith("NPU_SMOKE_")}
        env.update(
            PATH=str(binary) + os.pathsep + os.environ["PATH"],
            LD_PRELOAD="libjemalloc-cpu-boundary.so",
            LOG_DIR=str(tmp_path / "logs"),
            NPU_SMOKE_GROUP=group,
            NUM_NPUS=str(count),
            ASCEND_RT_VISIBLE_DEVICES=",".join(str(device) for device in range(count)),
            ASCEND_VISIBLE_DEVICES=_ALLOCATION,
            NPU_SMOKE_PARALLEL=parallel,
            NPU_SMOKE_FULL_TOPOLOGY=full,
            NPU_SMOKE_PARALLEL_CHILD=child,
            NPU_SMOKE_TEST_IDS="4,6" if group == "core" else "18,20",
            CLEANUP_STATUS=str(cleanup),
            CAPTURE=str(capture),
            FORBIDDEN=str(tmp_path / "forbidden.log"),
        )
        result = subprocess.run(
            [
                "/bin/bash",
                "-c",
                'source "$1" --num-npus "$NUM_NPUS"; ' + body,
                "scope-test",
                str(scripts / "lib_npu_smoke.sh"),
            ],
            env=env,
            text=True,
            capture_output=True,
        )
        records = [json.loads(line) for line in capture.read_text().splitlines()] if capture.exists() else []
        assert not (tmp_path / "forbidden.log").exists(), "Cleanup invoked a global process command."
        return result, records

    return run


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("group", ["core", "e2e"])
@pytest.mark.parametrize("full,parallel", [("0", "0"), ("1", "0"), ("1", "1")])
def test_serial_and_full_topology_dispatch_to_one_supervisor(shell_runner, count, group, full, parallel):
    result, records = shell_runner("dispatch_parallel", count=count, group=group, full=full, parallel=parallel)
    assert result.returncode == 0, result.stderr
    assert len(records) == 1
    assert records[0]["args"][:4] == ["tests/npu_smoke/run_npu_smoke_parallel.py", group, "--num-npus", str(count)]
    assert records[0]["args"][-1] == "--serial"
    assert records[0]["env"]["ASCEND_VISIBLE_DEVICES"] == _ALLOCATION


def test_parallel_dispatch_retains_parallel_layout(shell_runner):
    result, records = shell_runner("dispatch_parallel", count=8, parallel="1")
    assert result.returncode == 0, result.stderr
    assert len(records) == 1 and "--serial" not in records[0]["args"]


@pytest.mark.parametrize("original", [0, 1, 134])
@pytest.mark.parametrize("cleanup", [0, 7])
def test_exit_cleanup_is_scoped_and_preserves_original_failure(shell_runner, original, cleanup):
    result, records = shell_runner(f"exit {original}", child="1", cleanup=cleanup)
    assert result.returncode == (original or cleanup), result.stderr
    assert len(records) == 1
    assert records[0]["args"] == ["tests/npu_smoke/run_npu_smoke_processes.py", "cleanup"]


def test_failed_pre_workload_cleanup_never_runs_test(shell_runner):
    result, records = shell_runner("run_test 4 probe echo WORKLOAD_WAS_RUN", child="1", cleanup=7)
    assert result.returncode == 7
    assert "WORKLOAD_WAS_RUN" not in result.stdout
    assert len(records) == 2
    assert all(row["args"] == ["tests/npu_smoke/run_npu_smoke_processes.py", "cleanup"] for row in records)


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("status", [0, 134])
def test_serial_plan_runs_selected_group_once_and_preserves_raw_status(
    parallel_smoke, monkeypatch, tmp_path, count, status
):
    monkeypatch.setattr(parallel_smoke.signal, "signal", lambda *args: None)
    controller = object.__new__(parallel_smoke._Controller)
    controller.group = "core"
    controller.log_dir = tmp_path
    controller.devices = ([str(device) for device in range(count)], _ALLOCATION.split(","))
    starts = []

    def start(name, size, ids, offset, port):
        starts.append((name, size, ids, offset, port))
        folder = tmp_path / name
        folder.mkdir()
        (folder / "summary.log").write_text("4 PASS 1s fixture4\n6 PASS 1s fixture6\n")
        return {"name": name, "process": SimpleNamespace(returncode=status), "log_dir": folder, "ids": ids}

    monkeypatch.setattr(controller, "_start", start)
    monkeypatch.setattr(controller, "_wait", lambda children: None)
    assert controller.run({4, 6}, serial=True) == int(status != 0)
    assert starts == [("serial", count, (4, 6), 0, 32000)]
    assert json.loads((tmp_path / "parallel-results.json").read_text())["raw_exits"] == {"serial": status}


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("full", ["0", "1"])
@pytest.mark.parametrize("custom", [False, True])
def test_serial_environment_preserves_legacy_settings(parallel_smoke, monkeypatch, tmp_path, count, full, custom):
    class LaunchCaptured(Exception):
        pass

    captured = {}

    def capture(command, *, env, **kwargs):
        captured.update(env)
        raise LaunchCaptured

    for variable in ("NPU_SMOKE_FIXTURE_ROOT", "NPU_SMOKE_MASTER_PORT_RANGE"):
        monkeypatch.delenv(variable, raising=False)
    if custom:
        monkeypatch.setenv("NPU_SMOKE_FIXTURE_ROOT", "/caller-fixtures")
        monkeypatch.setenv("NPU_SMOKE_MASTER_PORT_RANGE", "[12000,13000]")
    monkeypatch.setenv(_FLAG, "1")
    monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "1")
    monkeypatch.setenv("NPU_SMOKE_FULL_TOPOLOGY", full)
    monkeypatch.setattr(parallel_smoke.subprocess, "Popen", capture)
    controller = object.__new__(parallel_smoke._Controller)
    controller.group = "core"
    controller.log_dir = tmp_path
    controller.devices = ([str(device) for device in range(count)], _ALLOCATION.split(","))
    controller.cancel_signal = 0
    controller.identity = SimpleNamespace(pid=4242, start=12345)
    (tmp_path / ".processes").mkdir()
    try:
        with pytest.raises(LaunchCaptured):
            controller._start("serial", count, (4, 6), 0, 32000)
    finally:
        runtime_file = tmp_path / ".processes/serial/runtime.json"
        if runtime_file.exists():
            shutil.rmtree(json.loads(runtime_file.read_text())["runtime"])
    assert captured["ASCEND_RT_VISIBLE_DEVICES"] == ",".join(str(device) for device in range(count))
    assert captured["ASCEND_VISIBLE_DEVICES"] == _ALLOCATION
    assert captured["NPU_SMOKE_FULL_TOPOLOGY"] == full
    assert captured["NPU_SMOKE_TEST_IDS"] == "4,6"
    assert captured[_FLAG] == captured["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] == "1"
    assert captured.get("NPU_SMOKE_FIXTURE_ROOT") == ("/caller-fixtures" if custom else None)
    assert captured.get("NPU_SMOKE_MASTER_PORT_RANGE") == ("[12000,13000]" if custom else None)


@pytest.mark.parametrize("count", [2, 4, 8])
@pytest.mark.parametrize("full", ["0", "1"])
def test_serial_main_accepts_requested_topology(parallel_smoke, monkeypatch, tmp_path, count, full):
    captured = {}

    class Controller:
        def __init__(self, group, log_dir, devices):
            captured.update(group=group, devices=devices)

        def run(self, selected, *, serial):
            captured.update(selected=selected, serial=serial)
            return 0

        def close(self):
            captured["closed"] = True

    monkeypatch.setattr(parallel_smoke, "enable_subreaper", lambda: None)
    monkeypatch.setattr(parallel_smoke, "_Controller", Controller)
    monkeypatch.setattr(
        sys, "argv", ["controller", "core", "--num-npus", str(count), "--log-dir", str(tmp_path), "--serial"]
    )
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("ASCEND_VISIBLE_DEVICES", _ALLOCATION)
    monkeypatch.setenv("NPU_SMOKE_TEST_IDS", "0,4")
    monkeypatch.setenv("NPU_SMOKE_PARALLEL_CHILD", "0")
    monkeypatch.setenv("NPU_SMOKE_FULL_TOPOLOGY", full)
    assert parallel_smoke.main() == 0
    assert captured["devices"] == ([str(device) for device in range(count)], _ALLOCATION.split(","))
    assert captured["selected"] == {0, 4}
    assert captured["serial"] is True and captured["closed"] is True
