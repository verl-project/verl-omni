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
"""Own and reap one NPU smoke process tree without OS namespaces."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import select
import signal
import stat
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple


class _Process(NamedTuple):
    pid: int
    start: int
    parent: int
    state: str


def process_identity(pid: int) -> _Process:
    """Read the identity and current parent of a Linux process."""
    fields = Path(f"/proc/{pid}/stat").read_text().rpartition(") ")[2].split()
    return _Process(pid, int(fields[19]), int(fields[1]), fields[0])


def enable_subreaper() -> None:
    """Require pidfd support and adopt orphaned descendants before launching work."""
    if sys.platform != "linux" or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Parallel NPU smoke requires Linux subreapers and pidfd signaling.")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


def _ancestry(pid: int, root: _Process) -> list[_Process] | None:
    chain, seen = [], set()
    try:
        process = process_identity(pid)
        while process.pid not in seen:
            seen.add(process.pid)
            chain.append(process)
            if process.pid == root.pid:
                if process.start != root.start:
                    return None
                # Recheck every edge: a parent PID may have exited and been reused.
                return chain if all(process_identity(item.pid)[:3] == item[:3] for item in chain) else None
            if process.parent <= 0:
                return None
            process = process_identity(process.parent)
    except (FileNotFoundError, ProcessLookupError):
        pass
    return None


def _snapshot() -> dict[int, _Process]:
    result = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                process = process_identity(int(path.name))
                result[process.pid] = process
            except (FileNotFoundError, ProcessLookupError):
                pass
    return result


class DescendantScope:
    """Signal only processes verified beneath one still-live supervisor identity."""

    def __init__(self, root: _Process):
        self.root = root
        self.pidfd = os.pidfd_open(root.pid, 0)
        if process_identity(root.pid).start != root.start:
            self.close()
            raise RuntimeError("Smoke supervisor identity changed.")
        signal.pidfd_send_signal(self.pidfd, 0)

    def close(self) -> None:
        os.close(self.pidfd)

    def _alive(self) -> None:
        if select.select([self.pidfd], [], [], 0)[0] or process_identity(self.root.pid).start != self.root.start:
            raise RuntimeError("Smoke supervisor exited before process ownership could be verified.")

    def members(self, protected: set[tuple[int, int]]) -> list[_Process]:
        self._alive()
        snapshot = _snapshot()
        children = {self.root.pid}
        while True:
            enlarged = children | {item.pid for item in snapshot.values() if item.parent in children}
            if enlarged == children:
                break
            children = enlarged
        return [
            item
            for pid, item in snapshot.items()
            if pid in children and pid != self.root.pid and item.state != "Z" and (pid, item.start) not in protected
        ]

    def send(self, process: _Process, signum: int) -> None:
        self._alive()
        try:
            pidfd = os.pidfd_open(process.pid, 0)
        except ProcessLookupError:
            return
        try:
            current = process_identity(process.pid)
            if current.start == process.start and _ancestry(process.pid, self.root) is not None:
                signal.pidfd_send_signal(pidfd, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            os.close(pidfd)

    def cleanup(
        self,
        protected: set[tuple[int, int]],
        *,
        grace: float = 3,
        timeout: float = 10,
        reap: Callable[[], None] | None = None,
    ) -> None:
        """Terminate owned descendants, rescan forks, and require consecutive quiet scans."""
        started = time.monotonic()
        quiet = 0
        while time.monotonic() - started < timeout:
            if reap is not None:
                reap()
            members = self.members(protected)
            quiet = quiet + 1 if not members else 0
            if quiet >= 2:
                return
            for process in members:
                self.send(process, signal.SIGTERM if time.monotonic() - started < grace else signal.SIGKILL)
            time.sleep(0.05)
        raise RuntimeError(f"Timed out cleaning descendants of smoke supervisor {self.root.pid}.")


def _write_state(path: Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".new")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(state, stream)
    os.replace(temporary, path)


def _read_state(path: Path) -> dict:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor) as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise RuntimeError("Smoke supervisor metadata must be an owned private regular file.")
        return json.load(stream)


def cleanup_current() -> int:
    """Authenticate a lane child, then clean its siblings while protecting its ancestry."""
    pid = int(os.environ["NPU_SMOKE_SUPERVISOR_PID"])
    start = int(os.environ["NPU_SMOKE_SUPERVISOR_START"])
    state = _read_state(Path(os.environ["NPU_SMOKE_SUPERVISOR_STATE"]))
    if (state.get("pid"), state.get("start"), state.get("token")) != (
        pid,
        start,
        os.environ["NPU_SMOKE_SUPERVISOR_TOKEN"],
    ):
        raise RuntimeError("Smoke supervisor metadata does not match the lane identity.")
    if state.get("phase") != "running":
        raise RuntimeError("Smoke supervisor is not running its authenticated group runner.")
    scope = DescendantScope(_Process(pid, start, 0, ""))
    try:
        chain = _ancestry(os.getpid(), scope.root)
        if chain is None:
            raise RuntimeError("Cleanup caller is not a descendant of its declared smoke supervisor.")
        scope.cleanup({(item.pid, item.start) for item in chain})
    finally:
        scope.close()
    return 0


def supervise(group: str, repo: Path, state_path: Path) -> int:
    """Run one lane and reap every adopted child without losing its raw wait status."""
    enable_subreaper()
    identity = process_identity(os.getpid())
    expected_parent = (int(os.environ["NPU_SMOKE_CONTROLLER_PID"]), int(os.environ["NPU_SMOKE_CONTROLLER_START"]))
    token = os.environ["NPU_SMOKE_SUPERVISOR_TOKEN"]
    cancelled = 0

    def request_stop(signum, _frame):
        nonlocal cancelled
        cancelled = signum

    for signum in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    state = {"pid": identity.pid, "start": identity.start, "token": token, "phase": "ready"}
    _write_state(state_path, state)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        parent = process_identity(os.getppid())
        if cancelled or (parent.pid, parent.start) != expected_parent:
            raise RuntimeError("Smoke controller disappeared before lane acknowledgement.")
        if select.select([sys.stdin], [], [], 0.05)[0]:
            if sys.stdin.readline().strip() != token:
                raise RuntimeError("Missing or invalid smoke controller acknowledgement.")
            break
    else:
        raise RuntimeError("Timed out waiting for smoke controller acknowledgement.")

    if cancelled:
        raise RuntimeError("Smoke lane cancelled before workload startup.")
    state.update(phase="running")
    _write_state(state_path, state)
    scope = DescendantScope(identity)
    runner = os.fork()
    if runner == 0:
        os.environ.update(
            NPU_SMOKE_SUPERVISOR_PID=str(identity.pid),
            NPU_SMOKE_SUPERVISOR_START=str(identity.start),
            NPU_SMOKE_SUPERVISOR_STATE=str(state_path),
        )
        os.chdir(repo)
        descriptor = os.open(os.devnull, os.O_RDONLY)
        os.dup2(descriptor, 0)
        os.close(descriptor)
        os.execvp("bash", ["bash", f"tests/npu_smoke/run_npu_smoke_{group}.sh", "--num-npus", os.environ["NUM_NPUS"]])
    runner_identity = process_identity(runner)
    raw_status = None
    completed = False

    def reap() -> None:
        nonlocal raw_status
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            if pid == runner:
                raw_status = os.waitstatus_to_exitcode(status)

    try:
        state.update(phase="running", runner=runner)
        _write_state(state_path, state)
        while raw_status is None and not cancelled:
            reap()
            if raw_status is not None:
                break
            parent = process_identity(os.getppid())
            if (parent.pid, parent.start) != expected_parent:
                cancelled = signal.SIGTERM
            if select.select([sys.stdin], [], [], 0)[0]:
                # The controller holds this pipe open after ACK; EOF means it is gone.
                if os.read(sys.stdin.fileno(), 1) != b"":
                    raise RuntimeError("Unexpected data after smoke lane acknowledgement.")
                cancelled = signal.SIGTERM
            time.sleep(0.05)
        if cancelled:
            scope.send(runner_identity, signal.SIGSTOP)
        scope.cleanup(set(), reap=reap)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            reap()
            if not _snapshot().get(runner) and not any(item.parent == identity.pid for item in _snapshot().values()):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError("Smoke supervisor could not reap all children.")
        result = -cancelled if cancelled else raw_status
        state.update(phase="complete", cleanup_complete=True, returncode=result)
        _write_state(state_path, state)
        completed = True
        return result
    finally:
        try:
            if not completed:
                scope.send(runner_identity, signal.SIGSTOP)
                scope.cleanup(set(), reap=reap)
                reap()
        finally:
            scope.close()


def main() -> int:
    """Run a scoped cleanup or supervise one authenticated smoke lane."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("cleanup", "supervise"))
    parser.add_argument("group", nargs="?", choices=("core", "e2e"))
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--state-path", type=Path)
    args = parser.parse_args()
    try:
        if args.action == "cleanup":
            return cleanup_current()
        if args.group is None or args.repo_root is None or args.state_path is None:
            parser.error("supervise requires group, --repo-root and --state-path")
        return supervise(args.group, args.repo_root, args.state_path)
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    result = main()
    if result < 0:
        signal.signal(-result, signal.SIG_DFL)
        os.kill(os.getpid(), -result)
    sys.exit(result)
