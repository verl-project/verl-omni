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
from pathlib import Path


def _ipc_handle(path: str) -> str:
    if "\0" in path or len(os.fsencode(path)) >= 108:
        raise ValueError("ZMQ IPC socket path must contain no NUL and be shorter than 108 bytes.")
    return f"ipc://{path}"


def make_base_zmq_handle(job_id: str, replica_rank: int | str, local_rank: int) -> str:
    """Return a colocated weight route under VERL_ZMQ_DIR, defaulting to /tmp."""
    directory = os.environ.get("VERL_ZMQ_DIR") or "/tmp"
    directory_path = Path(directory)
    if os.environ.get("VERL_ZMQ_DIR") and (
        not os.path.isabs(directory)
        or not os.path.isdir(directory)
        or any(path.is_symlink() for path in (directory_path, *directory_path.parents))
    ):
        raise ValueError("VERL_ZMQ_DIR must be an existing absolute directory without symbolic-link components.")
    name = f"rl-colocate-zmq-{job_id}-replica-{replica_rank}-rank-{local_rank}.sock"
    return _ipc_handle(os.path.join(directory, name))


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
    return _ipc_handle(f"{path}-update-{update_id}.sock")
