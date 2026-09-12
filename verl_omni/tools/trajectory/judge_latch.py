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

"""Per-rollout ``good_enough=YES`` latch (env hard-stop for further generate_image)."""

from __future__ import annotations

import contextvars
import threading

from .context import get_active_rollout_id

__all__ = [
    "clear_good_enough_yes_reached",
    "get_good_enough_yes_reached",
    "set_good_enough_yes_reached",
]

# After judge_image returns good_enough=YES, further generate_image is blocked
# for that rollout_id only (thread-pool workers are reused across samples).
_good_enough_yes_reached: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agentic_good_enough_yes_reached", default=False
)
_good_enough_yes_lock = threading.Lock()
_good_enough_yes_by_scope: dict[object, bool] = {}


def _rollout_scope_key() -> object | None:
    """Stable cross-context key for the active agent rollout, or ``None``.

    Prefers ``rollout_id`` (bound per rollout by the worker/loop) and falls back
    to the running asyncio task. There is deliberately **no** OS-thread fallback:
    ``threading.get_ident()`` is recycled once a pool thread exits, and a tool
    that sets the latch from an executor thread would key it on a *different*
    ident than the loop's ``clear`` — leaking YES into the next sample scheduled
    onto that ident. Without a rollout/task scope the shared dict is unused and
    the ``ContextVar`` alone governs the current context (sync tests / legacy
    standalone calls).
    """
    rid = get_active_rollout_id()
    if rid:
        return ("rollout", rid)
    try:
        import asyncio

        task = asyncio.current_task()
        if task is not None:
            return ("task", id(task))
    except RuntimeError:
        pass
    return None


def set_good_enough_yes_reached(reached: bool) -> contextvars.Token:
    """Mark that a live judge returned good_enough=YES on this rollout scope.

    Args:
        reached: True to set the latch, False to clear it.

    Returns:
        ContextVar token for the process-local flag.
    """
    flag = bool(reached)
    key = _rollout_scope_key()
    if key is not None:
        with _good_enough_yes_lock:
            if flag:
                _good_enough_yes_by_scope[key] = True
            else:
                _good_enough_yes_by_scope.pop(key, None)
    return _good_enough_yes_reached.set(flag)


def get_good_enough_yes_reached() -> bool:
    """Return whether good_enough=YES has been reached for this rollout.

    Returns:
        True if further ``generate_image`` should be blocked.
    """
    if _good_enough_yes_reached.get():
        return True
    key = _rollout_scope_key()
    if key is None:
        return False
    with _good_enough_yes_lock:
        return bool(_good_enough_yes_by_scope.get(key, False))


def clear_good_enough_yes_reached() -> None:
    """Reset the YES latch for the current rollout scope.

    Returns:
        None.
    """
    key = _rollout_scope_key()
    if key is not None:
        with _good_enough_yes_lock:
            _good_enough_yes_by_scope.pop(key, None)
    _good_enough_yes_reached.set(False)
