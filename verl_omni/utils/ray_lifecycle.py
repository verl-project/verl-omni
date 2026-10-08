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
"""Confirm termination of an owned Ray actor and its subprocesses."""

import os
import time
from typing import Any

import psutil
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


def actor_process_identity(worker: Any) -> dict[str, Any]:
    """Capture identity before submitting work that may block this actor."""
    process = psutil.Process()
    return {
        "pid": os.getpid(),
        "created": process.create_time(),
        "node_id": ray.get_runtime_context().get_node_id(),
        "children": [{"pid": child.pid, "created": child.create_time()} for child in process.children(recursive=True)],
    }


@ray.remote(num_cpus=0, max_retries=0)
def _terminate_actor_process_tree(actor, identity, timeout):
    deadline = time.monotonic() + timeout
    processes = {}
    trackers = set()

    def alive(process, created):
        try:
            return (
                process.is_running()
                and process.create_time() == created
                and process.status()
                not in (
                    psutil.STATUS_ZOMBIE,
                    psutil.STATUS_DEAD,
                )
            )
        except psutil.NoSuchProcess:
            return False

    def capture(process, created):
        if not alive(process, created):
            return False
        process.suspend()
        processes[process.pid] = (process, created)
        command = process.cmdline()
        if "-c" in command[:-1] and command[command.index("-c") + 1].startswith(
            "from multiprocessing.resource_tracker import main;main("
        ):
            trackers.add(process.pid)
        return True

    if ray.get_runtime_context().get_node_id() != identity["node_id"]:
        raise RuntimeError("Actor termination must run on the owning node")
    root = psutil.Process(identity["pid"])
    if not alive(root, identity["created"]) or root.pid == os.getpid():
        raise RuntimeError("Cannot confirm the owned actor process identity")
    processes[root.pid] = (root, identity["created"])
    try:
        # Freeze parents before walking children so no new consumers can fork
        # between the ownership snapshot and Ray's asynchronous actor kill.
        root.suspend()
        # Engine shutdown may orphan children before this cleanup starts.
        for owned in identity.get("children", []):
            try:
                capture(psutil.Process(owned["pid"]), owned["created"])
            except psutil.NoSuchProcess:
                pass
        while True:
            added = False
            children = {
                child.pid: child
                for parent, created in list(processes.values())
                if alive(parent, created)
                for child in parent.children(recursive=True)
            }
            for process in children.values():
                if process.pid not in processes:
                    try:
                        added = capture(process, process.create_time()) or added
                    except psutil.NoSuchProcess:
                        pass
            if not added:
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out capturing the owned actor process tree")
        # Trackers must run before Ray signals the process group; they only reclaim
        # registered resources once the stopped consumers close their pipes.
        for pid in trackers:
            process, created = processes[pid]
            if alive(process, created):
                process.resume()
        ray.kill(actor, no_restart=True)
        # Ray does not guarantee engine subprocesses terminate with their actor.
        for process, created in reversed(list(processes.values())):
            if process.pid not in trackers and alive(process, created):
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
        # Let Python's tracker unlink its registry after the consumers close its pipe.
        # Killing the tracker skips that cleanup and leaks engine cache shared memory.
        while any(alive(process, created) for process, created in processes.values()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Owned actor processes did not stop before the deadline")
            time.sleep(0.01)
    finally:
        # A failed confirmation must not leave owned processes suspended.
        for process, created in processes.values():
            if alive(process, created):
                try:
                    process.resume()
                except psutil.NoSuchProcess:
                    pass


def terminate_actor_and_wait(actor: ray.actor.ActorHandle, identity: dict[str, Any] | None, timeout: float) -> None:
    """Kill without restart and return only after consumer processes stopped.

    Missing identity, unavailable nodes and incomplete termination fail closed;
    callers must retain any producer resources those consumers may still use.
    """
    if identity is None:
        ray.kill(actor, no_restart=True)
        raise RuntimeError("Actor killed but subprocess termination cannot be confirmed without identity")
    cleanup = _terminate_actor_process_tree.options(
        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=identity["node_id"], soft=False)
    ).remote(actor, identity, timeout)
    try:
        ray.get(cleanup, timeout=timeout)
    except ray.exceptions.GetTimeoutError:
        ray.cancel(cleanup)
        raise
