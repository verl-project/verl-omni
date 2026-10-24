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


import os

_VERL_COLOCATE_PREFIX = "ipc:///tmp/rl-colocate-zmq-"


def colocate_zmq_handle(job_id: str, replica_rank: int | str, rank: int | str) -> str:
    """Rank-local IPC handle between a trainer rank and its colocated rollout worker.

    Carries the uid: ``/tmp`` is shared by every user on the node and the first job of any
    fresh Ray cluster is ``01000000``, so without it another user's stale socket owns our
    path (``EADDRINUSE``) and the sticky bit stops us from unlinking it.
    """
    return f"{_VERL_COLOCATE_PREFIX}u{os.getuid()}-{job_id}-replica-{replica_rank}-rank-{rank}.sock"


def namespace_colocate_zmq_handle(handle: str) -> str:
    """Rewrite a handle built by verl's ``ServerAdapter`` into ``colocate_zmq_handle``'s form."""
    if not handle.startswith(_VERL_COLOCATE_PREFIX):
        return handle
    suffix = handle[len(_VERL_COLOCATE_PREFIX) :]
    if suffix.startswith(f"u{os.getuid()}-"):
        return handle
    return f"{_VERL_COLOCATE_PREFIX}u{os.getuid()}-{suffix}"


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
    return f"ipc://{path}-update-{update_id}.sock"
