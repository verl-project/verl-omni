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
