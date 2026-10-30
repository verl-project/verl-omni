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


import hashlib
import os
import re

_VERL_COLOCATE_PREFIX = "ipc:///tmp/rl-colocate-zmq-"


def _colocate_namespace(*, inherit: bool = True) -> str:
    namespace = f"u{os.getuid()}-"
    inherited = os.environ.get("VERL_ZMQ_NAMESPACE") if inherit else None
    if inherited:
        if not re.fullmatch(re.escape(namespace) + r"(?:n[0-9a-f]{16}-)?", inherited):
            raise ValueError("Invalid inherited colocated IPC namespace")
        return inherited
    try:
        import ray

        if getattr(ray, "is_initialized", lambda: False)():
            node_id = ray.get_runtime_context().get_node_id()
            if node_id:
                namespace += "n" + hashlib.blake2s(str(node_id).encode(), digest_size=8).hexdigest() + "-"
    except (ImportError, RuntimeError):
        # Non-Ray callers retain the user namespace. Never substitute a PID:
        # colocated sender and receiver are different processes.
        pass
    return namespace


def export_colocate_zmq_namespace() -> None:
    """Export this server Ray node's identity before spawning non-Ray engines."""
    os.environ["VERL_ZMQ_NAMESPACE"] = _colocate_namespace(inherit=False)


def _bounded_ipc_handle(handle: str) -> str:
    if len(os.fsencode(handle.removeprefix("ipc://"))) < 108:
        return handle
    digest = hashlib.blake2s(handle.encode(), digest_size=16).hexdigest()
    return f"{_VERL_COLOCATE_PREFIX}h{digest}.sock"


def colocate_zmq_handle(job_id: str, replica_rank: int | str, rank: int | str) -> str:
    """Rank-local IPC handle between a trainer rank and its colocated rollout worker.

    Scope by uid and Ray node identity: independent clusters under the same user
    can both start at job ``01000000`` on one host. Colocated workers share the
    node identity. Bound encoded paths to Linux's Unix-domain socket limit.
    """
    return _bounded_ipc_handle(
        f"{_VERL_COLOCATE_PREFIX}{_colocate_namespace()}{job_id}-replica-{replica_rank}-rank-{rank}.sock"
    )


def namespace_colocate_zmq_handle(handle: str) -> str:
    """Rewrite a handle built by verl's ``ServerAdapter`` into ``colocate_zmq_handle``'s form."""
    if not handle.startswith(_VERL_COLOCATE_PREFIX):
        return handle
    suffix = handle[len(_VERL_COLOCATE_PREFIX) :]
    if re.match(r"(?:u\d+-n[0-9a-f]{16}-|h[0-9a-f]{32}(?:-update-|\.sock$))", suffix):
        return handle
    suffix = suffix.removeprefix(f"u{os.getuid()}-")
    return _bounded_ipc_handle(f"{_VERL_COLOCATE_PREFIX}{_colocate_namespace()}{suffix}")


def make_update_zmq_id(global_steps: int | None, update_seq: int) -> str:
    """Return an id that is unique for each weight update in a worker lifetime."""
    step = "none" if global_steps is None else str(global_steps)
    return f"step-{step}-seq-{update_seq}"


def make_update_zmq_handle(base_handle: str, update_id: str) -> str:
    """Append a per-update id while preserving the job/rank-specific base handle."""
    if not base_handle.startswith("ipc://"):
        return base_handle

    path = base_handle.removeprefix("ipc://")
    if path.endswith(".sock"):
        path = path[: -len(".sock")]
    return _bounded_ipc_handle(f"ipc://{path}-update-{update_id}.sock")
