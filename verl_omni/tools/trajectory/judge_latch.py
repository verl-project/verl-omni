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
# (env hard-stop — not token force). Key by **rollout_id** (not asyncio task /
# thread): ``FunctionTool.call`` runs in ``asyncio.to_thread``, where
# ``current_task()`` is None, so a thread-scoped latch leaked YES across
# concurrent samples sharing a thread-pool worker and blocked the next sample's
# first generate (then judge_image(…, "last") had no PNG).
_good_enough_yes_reached: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "agentic_good_enough_yes_reached", default=False
)
_good_enough_yes_tls = threading.local()
_good_enough_yes_lock = threading.Lock()
_good_enough_yes_by_scope: dict[object, bool] = {}


def _rollout_scope_key() -> object:
    """Stable key for the active agent rollout (prefer rollout_id)."""
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
    return ("thread", threading.get_ident())


def set_good_enough_yes_reached(reached: bool) -> contextvars.Token:
    """Mark that a live judge returned good_enough=YES on this rollout scope."""
    flag = bool(reached)
    key = _rollout_scope_key()
    with _good_enough_yes_lock:
        if flag:
            _good_enough_yes_by_scope[key] = True
        else:
            _good_enough_yes_by_scope.pop(key, None)
    _good_enough_yes_tls.reached = flag
    return _good_enough_yes_reached.set(flag)


def get_good_enough_yes_reached() -> bool:
    if _good_enough_yes_reached.get():
        return True
    key = _rollout_scope_key()
    with _good_enough_yes_lock:
        if _good_enough_yes_by_scope.get(key, False):
            return True
    # Sync unit-test / no-running-task fallback only.
    if isinstance(key, tuple) and key[0] == "thread":
        return bool(getattr(_good_enough_yes_tls, "reached", False))
    return False


def clear_good_enough_yes_reached() -> None:
    """Reset YES latch for the current rollout scope (call at trajectory start/end)."""
    key = _rollout_scope_key()
    with _good_enough_yes_lock:
        _good_enough_yes_by_scope.pop(key, None)
    _good_enough_yes_tls.reached = False
    _good_enough_yes_reached.set(False)
