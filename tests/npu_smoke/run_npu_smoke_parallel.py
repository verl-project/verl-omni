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
"""Run disjoint NPU smoke lanes with authenticated process-tree cleanup."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from run_npu_smoke_processes import DescendantScope, _read_state, enable_subreaper, process_identity

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAYOUTS = {
    "core": (("exclusive", 8, (0, 3)), ("lane0", 4, (7,)), ("lane1", 4, (4, 5, 6))),
    "e2e": (("exclusive", 8, (1,)), ("lane0", 2, ()), ("lane1", 2, ()), ("lane2", 4, (2,))),
}
_SUMMARY_ROW = re.compile(r"^(\d+)\s+(PASS|FAIL)\s+(\d+s)\s+(.+)$")


def _device_list(name: str, default: str, *, expected: int = 8) -> list[str]:
    value = os.environ.get(name) or default
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ValueError(f"{name} must contain comma-separated NPU IDs.")
    devices = value.split(",")
    allocation = name == "ASCEND_VISIBLE_DEVICES"
    valid_size = len(devices) >= expected if allocation else len(devices) == expected
    if not valid_size or len({int(device) for device in devices}) != len(devices):
        count = f"at least {expected}" if allocation else str(expected)
        raise ValueError(f"{name} must contain {count} distinct NPU IDs.")
    if not allocation and [int(device) for device in devices] != sorted(int(device) for device in devices):
        raise ValueError(f"{name} must use ascending runtime NPU IDs for parallel Ascend lanes.")
    return devices


def _selected_ids(group: str) -> set[int]:
    registered = {test_id for _, _, ids in _LAYOUTS[group] for test_id in ids}
    source = (_REPO_ROOT / f"tests/npu_smoke/run_npu_smoke_{group}.sh").read_text()
    actual = [int(value) for value in re.findall(r"^run_test\s+([0-9]+)\s", source, re.MULTILINE)]
    if set(actual) != registered or len(actual) != len(registered):
        raise ValueError(f"Parallel {group} plan must match the group registry exactly: {actual}")
    value = os.environ.get("NPU_SMOKE_TEST_IDS", "")
    if not value:
        return registered
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:,(?:0|[1-9][0-9]*))*", value):
        raise ValueError("NPU_SMOKE_TEST_IDS must contain comma-separated canonical test IDs.")
    ids = [int(test_id) for test_id in value.split(",")]
    if len(set(ids)) != len(ids) or not set(ids) <= registered:
        raise ValueError(f"NPU_SMOKE_TEST_IDS must be distinct registered {group} IDs: {sorted(registered)}")
    return set(ids)


def _discard_successful_runtime(child: dict) -> None:
    if not child.get("successful"):
        return
    runtime = child["runtime"]
    metadata = runtime.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o777 != 0o700:
        raise RuntimeError(f"Refusing to remove changed smoke runtime directory: {runtime}")
    shutil.rmtree(runtime)


class _Controller:
    def __init__(self, group: str, log_dir: Path, devices: tuple[list[str], list[str]]):
        self.group = group
        self.log_dir = log_dir
        self.devices = devices
        self.children: list[dict] = []
        self.cancel_signal = 0
        self.identity = process_identity(os.getpid())
        self.scope = DescendantScope(self.identity)
        (log_dir / ".processes").mkdir(mode=0o700)

    def _signal(self, signum: int, _frame) -> None:
        self.cancel_signal = signum

    def _command(self, state_path: Path) -> list[str]:
        return [
            sys.executable,
            str(Path(__file__).with_name("run_npu_smoke_processes.py")),
            "supervise",
            self.group,
            "--repo-root",
            str(_REPO_ROOT),
            "--state-path",
            str(state_path),
        ]

    def _metadata(self, child: dict, *, complete: bool = False) -> dict:
        state = _read_state(child["state_path"])
        if (state.get("pid"), state.get("start"), state.get("token")) != (
            child["process"].pid,
            child["identity"].start,
            child["token"],
        ):
            raise RuntimeError(f"{child['name']}: invalid supervisor identity metadata.")
        if complete and (
            state.get("phase") != "complete"
            or state.get("cleanup_complete") is not True
            or state.get("returncode") != child["process"].returncode
        ):
            raise RuntimeError(f"{child['name']}: supervisor exited without verified cleanup and raw status.")
        return state

    def _start(self, name: str, count: int, ids: tuple[int, ...], offset: int, port_start: int) -> dict:
        if self.cancel_signal:
            raise InterruptedError(f"Received signal {self.cancel_signal}")
        log_dir = self.log_dir / name
        log_dir.mkdir(mode=0o700)
        private = self.log_dir / ".processes" / name
        private.mkdir(mode=0o700)
        fixtures = private / "fixtures"
        fixtures.mkdir(mode=0o700)
        runtime = Path(tempfile.mkdtemp(prefix="vn-", dir="/tmp"))
        state_path = private / "state.json"
        token = secrets.token_hex(24)
        (private / "runtime.json").write_text(json.dumps({"runtime": str(runtime)}))
        env = dict(os.environ)
        env.pop("RAY_ADDRESS", None)
        env.update(
            NUM_NPUS=str(count),
            ASCEND_RT_VISIBLE_DEVICES=",".join(self.devices[0][offset : offset + count]),
            ASCEND_VISIBLE_DEVICES=",".join(self.devices[1]),
            NPU_SMOKE_PARALLEL_CHILD="1",
            NPU_SMOKE_TEST_IDS=",".join(str(test_id) for test_id in ids),
            NPU_SMOKE_FULL_TOPOLOGY=os.environ.get("NPU_SMOKE_FULL_TOPOLOGY", "0"),
            NPU_SMOKE_FIXTURE_ROOT=str(fixtures),
            NPU_SMOKE_MASTER_PORT_RANGE=f"[{port_start},{port_start + 4000}]",
            LOG_DIR=str(log_dir),
            TMPDIR=str(runtime),
            RAY_TMPDIR=str(runtime),
            VERL_ZMQ_DIR=str(runtime),
            NPU_SMOKE_CONTROLLER_PID=str(self.identity.pid),
            NPU_SMOKE_CONTROLLER_START=str(self.identity.start),
            NPU_SMOKE_SUPERVISOR_TOKEN=token,
        )
        if name == "serial":
            for variable in ("NPU_SMOKE_FIXTURE_ROOT", "NPU_SMOKE_MASTER_PORT_RANGE"):
                if variable in os.environ:
                    env[variable] = os.environ[variable]
                else:
                    env.pop(variable, None)
        if name not in ("exclusive", "serial"):
            # Ray returns physical IDs; resource workers must use Ray-managed local visibility.
            for variable in (
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
                "RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES",
                "RAY_EXPERIMENTAL_NOSET_HABANA_VISIBLE_MODULES",
                "RAY_EXPERIMENTAL_NOSET_NEURON_RT_VISIBLE_CORES",
                "RAY_EXPERIMENTAL_NOSET_TPU_VISIBLE_CHIPS",
                "RAY_EXPERIMENTAL_NOSET_ONEAPI_DEVICE_SELECTOR",
            ):
                env[variable] = ""
            env["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
        command = self._command(state_path)
        output = (log_dir / "runner.log").open("wb")
        try:
            process = subprocess.Popen(
                command, env=env, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
        except BaseException:
            output.close()
            raise
        try:
            identity = process_identity(process.pid)
            pidfd = os.pidfd_open(process.pid, 0)
            if process_identity(process.pid).start != identity.start:
                os.close(pidfd)
                raise RuntimeError(f"{name}: supervisor identity changed before acknowledgement.")
        except (FileNotFoundError, ProcessLookupError):
            process.wait(timeout=5)
            process.stdin.close()
            output.close()
            raise RuntimeError(f"{name}: supervisor disappeared before acknowledgement.") from None
        child = {
            "name": name,
            "identity": identity,
            "pidfd": pidfd,
            "token": token,
            "state_path": state_path,
            "runtime": runtime,
            "ids": ids,
            "process": process,
            "output": output,
            "reader": (log_dir / "runner.log").open("rb"),
            "decoder": codecs.getincrementaldecoder("utf-8")("replace"),
            "line_start": True,
            "stream_finished": False,
            "log_dir": log_dir,
        }
        self.children.append(child)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if self.cancel_signal:
                raise InterruptedError(f"Received signal {self.cancel_signal}")
            if process.poll() is not None:
                raise RuntimeError(f"{name}: supervisor exited before workload acknowledgement.")
            if state_path.exists():
                state = self._metadata(child)
                if state.get("phase") != "ready":
                    raise RuntimeError(f"{name}: workload started before controller acknowledgement.")
                process.stdin.write((token + "\n").encode())
                process.stdin.flush()
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"{name}: supervisor startup timed out; no workload was acknowledged.")
        print(f"[process lane] {name}: IDs {ids}; {env['ASCEND_RT_VISIBLE_DEVICES']}; log={log_dir}", flush=True)
        return child

    def _stream(self, child: dict, *, final: bool = False) -> int:
        if child["stream_finished"]:
            return 0
        data = child["reader"].read(65536)
        text = child["decoder"].decode(data, final=final and not data)
        child["stream_finished"] = final and not data
        for part in text.splitlines(keepends=True):
            prefix = f"[{child['name']}] " if child["line_start"] else ""
            print(prefix + part, end="", flush=True)
            child["line_start"] = part.endswith(("\n", "\r"))
        return len(data)

    def _drain(self, child: dict) -> None:
        while self._stream(child):
            pass
        self._stream(child, final=True)

    def _wait(self, children: list[dict]) -> None:
        while True:
            if self.cancel_signal:
                raise InterruptedError(f"Received signal {self.cancel_signal}")
            for child in children:
                self._stream(child)
            for child in children:
                if child["process"].poll() is not None:
                    self._metadata(child, complete=True)
            if all(child["process"].returncode is not None for child in children):
                for child in children:
                    child["process"].wait()
                    self._drain(child)
                return
            time.sleep(0.1)

    def close(self) -> None:
        errors = []
        # Closing the ACK pipe also asks a live supervisor to cancel on parent EOF.
        for child in self.children:
            if child["process"].stdin is not None:
                child["process"].stdin.close()
            if child["process"].poll() is None:
                try:
                    signal.pidfd_send_signal(child["pidfd"], signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 15
        while any(child["process"].poll() is None for child in self.children) and time.monotonic() < deadline:
            for child in self.children:
                self._stream(child)
            time.sleep(0.05)
        for child in self.children:
            if child["process"].poll() is None:
                try:
                    signal.pidfd_send_signal(child["pidfd"], signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for child in self.children:
            try:
                child["process"].wait(timeout=5)
                self._drain(child)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(error)
            finally:
                os.close(child["pidfd"])
                child["reader"].close()
                child["output"].close()

        # Subreaping at the controller catches detached orphans left by a dead supervisor.
        # Popen children have all been waited before waitpid(-1), preserving their raw exits.
        def reap_adopted() -> None:
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    return
                if pid == 0:
                    return

        try:
            self.scope.cleanup(set(), reap=reap_adopted)
            reap_adopted()
            while True:
                try:
                    if os.waitpid(-1, os.WNOHANG)[0] == 0:
                        errors.append(RuntimeError("Controller still has an unreaped child."))
                        break
                except ChildProcessError:
                    break
        finally:
            self.scope.close()
        if errors:
            raise RuntimeError(f"Unable to finish scoped process cleanup: {errors}") from errors[0]
        for child in self.children:
            _discard_successful_runtime(child)

    def run(self, selected: set[int], *, serial: bool = False) -> int:
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._signal)
        selected_children = []
        offset = 0
        layout = (("serial", len(self.devices[0]), tuple(sorted(selected))),) if serial else _LAYOUTS[self.group]
        for name, count, ids in layout:
            chosen = tuple(test_id for test_id in ids if test_id in selected)
            if chosen:
                child = self._start(
                    name,
                    count,
                    chosen,
                    0 if name in ("exclusive", "serial") else offset,
                    32000 if name in ("exclusive", "serial") else 20000 + offset // 2 * 4000,
                )
                selected_children.append(child)
                if name in ("exclusive", "serial"):
                    self._wait([child])
            if name != "exclusive":
                offset += count
        self._wait(selected_children)

        rows, errors, exits = {}, [], {}
        for child in selected_children:
            exits[child["name"]] = child["process"].returncode
            summary = child["log_dir"] / "summary.log"
            child_ids = []
            for line in summary.read_text().splitlines() if summary.is_file() else []:
                match = _SUMMARY_ROW.fullmatch(line.strip())
                if match:
                    test_id = int(match[1])
                    child_ids.append(test_id)
                    if test_id in rows:
                        errors.append(f"Duplicate result for test {test_id}")
                    rows[test_id] = match.groups()[1:]
            if set(child_ids) != set(child["ids"]) or len(child_ids) != len(child["ids"]):
                errors.append(f"{child['name']} summary IDs {child_ids} differ from {child['ids']}")
        if set(rows) != selected:
            errors.append(f"Merged summary IDs {sorted(rows)} differ from {sorted(selected)}")
        (self.log_dir / "parallel-results.json").write_text(
            json.dumps({"raw_exits": exits, "errors": errors}, indent=2)
        )
        if errors:
            message = "\n".join(f"[FAIL] {error}" for error in errors) + "\n"
            (self.log_dir / "summary.log").write_text(message)
            print(message, file=sys.stderr, end="")
            return 1
        for child in selected_children:
            child["successful"] = exits[child["name"]] == 0 and all(
                rows[test_id][0] == "PASS" for test_id in child["ids"]
            )

        lines = ["-" * 78, f"  {self.group.upper()} NPU SMOKE TEST SUMMARY", "-" * 78]
        for test_id in sorted(rows):
            status, duration, name = rows[test_id]
            lines.append(f"{test_id:<4}  {status:<7}  {duration:<8}  {name}")
        passed = sum(row[0] == "PASS" for row in rows.values())
        failed_runners = sum(code != 0 for code in exits.values())
        lines.extend(
            [
                "-" * 78,
                f"Passed: {passed}",
                f"Failed: {len(rows) - passed}",
                f"Failed runners: {failed_runners}; raw exits: {exits}",
                f"Logs  : {self.log_dir}",
            ]
        )
        summary = "\n".join(lines) + "\n"
        (self.log_dir / "summary.log").write_text(summary)
        print(summary, end="", flush=True)
        return int(passed != len(selected) or failed_runners != 0)


def main() -> int:
    """Validate devices, supervise serial or parallel groups, and reconcile selected cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=_LAYOUTS)
    parser.add_argument("--num-npus", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--serial", action="store_true", help="Supervise one serial group without splitting its devices."
    )
    args = parser.parse_args()
    controller = None
    try:
        full_topology = os.environ.get("NPU_SMOKE_FULL_TOPOLOGY", "0")
        if args.num_npus <= 0 or full_topology not in ("0", "1"):
            raise ValueError("Smoke requires a positive NPU count and NPU_SMOKE_FULL_TOPOLOGY=0 or 1.")
        if not args.serial and (args.num_npus != 8 or full_topology != "0"):
            raise ValueError("Parallel smoke requires eight NPUs; use the serial entry for full-topology runs.")
        if os.environ.get("NPU_SMOKE_PARALLEL_CHILD", "0") != "0":
            raise ValueError("A process-lane child cannot start another parallel controller.")
        selected = _selected_ids(args.group)
        default_devices = ",".join(str(device) for device in range(args.num_npus))
        rt_devices = _device_list("ASCEND_RT_VISIBLE_DEVICES", default_devices, expected=args.num_npus)
        devices = (
            rt_devices,
            _device_list("ASCEND_VISIBLE_DEVICES", ",".join(rt_devices), expected=args.num_npus),
        )
        if (
            args.group == "core"
            and 0 in selected
            and {int(device) for device in rt_devices} != set(range(args.num_npus))
        ):
            raise ValueError(
                f"Core test 0's existing fixture resets RT visibility to device IDs 0 through {args.num_npus - 1}."
            )
        log_dir = args.log_dir.expanduser().resolve()
        enable_subreaper()
        log_dir.mkdir(parents=True, exist_ok=True)
        if (log_dir / "summary.log").exists() or (log_dir / ".processes").exists():
            raise ValueError(f"Refusing to reuse parallel smoke results: {log_dir}")
        controller = _Controller(args.group, log_dir, devices)
        return controller.run(selected, serial=args.serial)
    except InterruptedError as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 128 + controller.cancel_signal
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            controller.close()
