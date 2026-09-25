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

"""Cross-writer serialization for one trajectory artifact folder.

Several code paths write into the *same* ``rollout_images/<relpath>/`` folder:

* ``image_gen._save_images`` — the live ``generate_image`` tool body, executed in
  ``asyncio.to_thread``; concurrent calls inside one rollout (parallel plan
  subtasks / forced reflection) share a folder.
* ``image_gen_rollout_dump.materialize_rollout_images`` — post-processing that
  re-publishes ``meta.json`` for the folder the live tool just wrote.

Both do an ``image_NN`` index allocation plus a ``meta.json`` read-modify-write.
Without a shared lock they picked the same index (duplicate ``image_NN`` files)
and interleaved the JSON write (concatenated, unparsable ``meta.json``).

These helpers intentionally live in the ``trajectory`` package rather than in
``image_gen.py`` so importers share one lock domain without importing the tool
module (which re-executes ``@function_tool`` registration).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:  # POSIX-only cross-process guard for the ``meta.json`` read-modify-write.
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX keeps the intra-process lock
    fcntl = None

__all__ = ["meta_lock_path", "traj_dir_exclusive", "write_json_atomic"]

_traj_dir_locks: dict[str, threading.RLock] = {}
_traj_dir_locks_guard = threading.Lock()

#: Suffix for the lock files. Lock files live outside the artifact tree so the
#: dump readers never mistake them for artifacts.
_LOCK_ROOT_NAME = "verlomni_meta_locks"


def traj_dir_lock(traj_dir: Path) -> threading.RLock:
    """Return the process-local reentrant lock for one trajectory folder.

    Args:
        traj_dir: Folder holding ``image_NN_*.png`` and ``meta.json``.

    Returns:
        Shared ``threading.RLock`` for that folder.
    """
    key = str(traj_dir)
    with _traj_dir_locks_guard:
        lock = _traj_dir_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _traj_dir_locks[key] = lock
        return lock


def meta_lock_path(traj_dir: Path) -> Path:
    """Return the cross-process lock file guarding one trajectory folder.

    Args:
        traj_dir: Folder holding ``meta.json``.

    Returns:
        Path under the system temp dir, keyed by the resolved folder path.
    """
    digest = hashlib.sha256(str(Path(traj_dir).resolve()).encode("utf-8")).hexdigest()[:24]
    base = Path(tempfile.gettempdir()) / _LOCK_ROOT_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{digest}.lock"


@contextlib.contextmanager
def traj_dir_exclusive(traj_dir: Path) -> Iterator[None]:
    """Serialize ``image_NN`` allocation and ``meta.json`` writes for one folder.

    Combines a process-local reentrant lock with a POSIX ``flock`` so concurrent
    ``asyncio.to_thread`` calls (and, defensively, a second writer process) can
    neither pick the same index nor interleave the ``meta.json`` read-modify-write.

    Args:
        traj_dir: Folder holding ``image_NN_*.png`` and ``meta.json``.

    Yields:
        None. The folder is locked for the duration of the ``with`` block.
    """
    with traj_dir_lock(traj_dir):
        if fcntl is None:
            yield
            return
        with open(meta_lock_path(traj_dir), "w") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_json_atomic(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Publish ``payload`` as JSON via ``os.replace`` so readers never see a partial file.

    Args:
        path: Destination ``*.json`` path.
        payload: JSON-serializable object.
        indent: Pretty-print indent.

    Returns:
        None.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=indent, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)
