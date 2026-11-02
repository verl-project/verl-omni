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
"""Run disjoint NPU smoke lanes in private mount and PID namespaces."""

from __future__ import annotations

import argparse
import codecs
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LAYOUTS = {
    "core": (("exclusive", 8, (0, 3, 13, 14, 28)), ("lane0", 4, (7, 11, 12)), ("lane1", 4, (4, 5, 6, 15, 17, 30))),
    "e2e": (
        ("exclusive", 8, (1,)),
        ("lane0", 2, (18, 20, 23)),
        ("lane1", 2, (19, 21, 25, 34)),
        ("lane2", 4, (2, 22, 24, 26)),
    ),
}
_SUMMARY_ROW = re.compile(r"^(\d+)\s+(PASS|FAIL)\s+(\d+s)\s+(.+)$")


def _device_list(name: str, default: str) -> list[str]:
    value = os.environ.get(name) or default
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ValueError(f"{name} must contain eight comma-separated NPU IDs.")
    devices = value.split(",")
    if len(devices) != 8 or len({int(device) for device in devices}) != 8:
        raise ValueError(f"{name} must contain eight distinct NPU IDs.")
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


def _outside_tmp(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path == Path("/tmp").resolve() or Path("/tmp").resolve() in path.parents:
        raise ValueError(f"Namespace input must remain accessible after /tmp is mounted: {path}")
    return path


class _Controller:
    def __init__(self, group: str, log_dir: Path, unshare: str, devices: tuple[list[str], list[str]]):
        self.group = group
        self.log_dir = log_dir
        self.unshare = unshare
        self.devices = devices
        self.children: list[dict] = []
        self.cancel_signal = 0
        (log_dir / ".namespace").mkdir(mode=0o700)

    def _signal(self, signum: int, _frame) -> None:
        self.cancel_signal = signum

    def _command(self, backing: Path, group: str) -> list[str]:
        return [
            self.unshare,
            "--mount",
            "--propagation",
            "private",
            "--pid",
            "--fork",
            "--kill-child",
            "--mount-proc",
            "bash",
            str(_REPO_ROOT / "tests/npu_smoke/run_npu_smoke_namespace.sh"),
            str(backing),
            str(_REPO_ROOT),
            group,
        ]

    def _start(self, name: str, count: int, ids: tuple[int, ...], offset: int, port_start: int) -> dict:
        if self.cancel_signal:
            raise InterruptedError(f"Received signal {self.cancel_signal}")
        log_dir = self.log_dir / name
        log_dir.mkdir(mode=0o700)
        private = self.log_dir / ".namespace" / name
        private.mkdir(mode=0o700)
        backing, fixtures = private / "tmp", private / "fixtures"
        backing.mkdir(mode=0o700)
        fixtures.mkdir(mode=0o700)
        env = dict(os.environ)
        env.pop("RAY_ADDRESS", None)
        env.update(
            NUM_NPUS=str(count),
            ASCEND_RT_VISIBLE_DEVICES=",".join(self.devices[0][offset : offset + count]),
            ASCEND_VISIBLE_DEVICES=",".join(self.devices[1][offset : offset + count]),
            NPU_SMOKE_PARALLEL_CHILD="1",
            NPU_SMOKE_TEST_IDS=",".join(str(test_id) for test_id in ids),
            NPU_SMOKE_FULL_TOPOLOGY="0",
            NPU_SMOKE_FIXTURE_ROOT=str(fixtures),
            NPU_SMOKE_MASTER_PORT_RANGE=f"[{port_start},{port_start + 4000}]",
            LOG_DIR=str(log_dir),
        )
        command = self._command(backing, "--probe" if name == "namespace-check" else self.group)
        output = (log_dir / "runner.log").open("wb")
        try:
            process = subprocess.Popen(
                command, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
            )
        except BaseException:
            output.close()
            raise
        child = {
            "name": name,
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
        print(f"[namespace] {name}: IDs {ids}; {env['ASCEND_RT_VISIBLE_DEVICES']}; log={log_dir}", flush=True)
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
            if all(child["process"].poll() is not None for child in children):
                for child in children:
                    child["process"].wait()
                    self._drain(child)
                return
            time.sleep(0.1)

    def close(self) -> None:
        live = [child for child in self.children if child["process"].poll() is None]
        for child in live:
            try:
                os.killpg(child["process"].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 5
        while any(child["process"].poll() is None for child in live) and time.monotonic() < deadline:
            time.sleep(0.1)
        for child in live:
            if child["process"].poll() is not None:
                continue
            try:
                os.killpg(child["process"].pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        errors = []
        for child in self.children:
            try:
                child["process"].wait(timeout=10)
                self._drain(child)
            except (OSError, subprocess.TimeoutExpired) as error:
                errors.append(error)
            finally:
                child["reader"].close()
                child["output"].close()
        if errors:
            raise RuntimeError(f"Unable to finish namespace cleanup: {errors}") from errors[0]

    def run(self, selected: set[int]) -> int:
        for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._signal)
        probe = self._start("namespace-check", 8, (), 0, 32000)
        self._wait([probe])
        if probe["process"].returncode:
            print(
                "[FAIL] Private mount/PID namespaces are unavailable; no smoke workload was started.", file=sys.stderr
            )
            return 2

        selected_children = []
        offset = 0
        for name, count, ids in _LAYOUTS[self.group]:
            chosen = tuple(test_id for test_id in ids if test_id in selected)
            if chosen:
                child = self._start(
                    name,
                    count,
                    chosen,
                    0 if name == "exclusive" else offset,
                    32000 if name == "exclusive" else 20000 + offset // 2 * 4000,
                )
                selected_children.append(child)
                if name == "exclusive":
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
    """Validate the eight-device allocation, then run and reconcile every selected case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=_LAYOUTS)
    parser.add_argument("--num-npus", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    controller = None
    try:
        if args.num_npus != 8 or os.environ.get("NPU_SMOKE_FULL_TOPOLOGY", "0") != "0":
            raise ValueError("Parallel smoke requires eight NPUs; use the serial entry for full-topology runs.")
        if os.environ.get("NPU_SMOKE_PARALLEL_CHILD", "0") != "0":
            raise ValueError("A namespace child cannot start another parallel controller.")
        selected = _selected_ids(args.group)
        rt_devices = _device_list("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
        devices = (rt_devices, _device_list("ASCEND_VISIBLE_DEVICES", ",".join(rt_devices)))
        if args.group == "core" and 0 in selected and {int(device) for device in rt_devices} != set(range(8)):
            raise ValueError("Core test 0's existing fixture resets RT visibility to device IDs 0 through 7.")
        log_dir = _outside_tmp(args.log_dir)
        _outside_tmp(_REPO_ROOT)
        if sys.platform != "linux" or not (unshare := shutil.which("unshare")):
            raise ValueError("Parallel smoke requires Linux unshare with mount and PID namespace permissions.")
        log_dir.mkdir(parents=True, exist_ok=True)
        if (log_dir / "summary.log").exists() or (log_dir / ".namespace").exists():
            raise ValueError(f"Refusing to reuse parallel smoke results: {log_dir}")
        controller = _Controller(args.group, log_dir, unshare, devices)
        return controller.run(selected)
    except InterruptedError as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 128 + controller.cancel_signal
    except (OSError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    finally:
        if controller is not None:
            controller.close()


if __name__ == "__main__":
    sys.exit(main())
