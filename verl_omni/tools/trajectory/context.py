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

"""Per-rollout ContextVars: trajectory relpath, rollout id, user prompt."""

from __future__ import annotations

import contextvars

from .paths import rollout_id_from_relpath

__all__ = [
    "get_active_rollout_id",
    "get_active_trajectory_relpath",
    "get_active_user_prompt",
    "reset_active_trajectory_relpath",
    "reset_active_user_prompt",
    "set_active_trajectory_relpath",
    "set_active_user_prompt",
]

# Relative path under the images/trajectories roots.
_active_trajectory_relpath: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_trajectory_relpath", default=None
)
# Stable short id derived from trajectory_relpath (copied into asyncio.to_thread).
_active_rollout_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_rollout_id", default=None
)
# Dataset / task user request for the active trajectory (written into meta.json).
_active_user_prompt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_user_prompt", default=None
)


def set_active_trajectory_relpath(
    relpath: str | None,
) -> tuple[contextvars.Token, contextvars.Token]:
    """Bind (or clear) the relative artifact path + matching rollout_id.

    Returns ``(path_token, rollout_token)`` for ``reset_active_trajectory_relpath``.
    Both must be restored together so judge/artifact lookups do not leak across
    rollouts after the path ContextVar alone is reset.
    """
    rid = rollout_id_from_relpath(relpath)
    path_token = _active_trajectory_relpath.set(relpath)
    rollout_token = _active_rollout_id.set(rid)
    return path_token, rollout_token


def reset_active_trajectory_relpath(
    tokens: tuple[contextvars.Token, contextvars.Token] | contextvars.Token,
) -> None:
    """Restore trajectory path + rollout_id bindings that preceded ``tokens``."""
    if isinstance(tokens, tuple):
        path_token, rollout_token = tokens
        _active_trajectory_relpath.reset(path_token)
        _active_rollout_id.reset(rollout_token)
        return
    # Legacy single-token callers: still clear stale rollout_id from this set().
    _active_trajectory_relpath.reset(tokens)
    _active_rollout_id.set(rollout_id_from_relpath(_active_trajectory_relpath.get()))


def get_active_trajectory_relpath() -> str | None:
    return _active_trajectory_relpath.get()


def get_active_rollout_id() -> str | None:
    rid = _active_rollout_id.get()
    if rid:
        return rid
    return rollout_id_from_relpath(get_active_trajectory_relpath())


def set_active_user_prompt(prompt: str | None) -> contextvars.Token:
    """Bind the dataset user request so ``meta.json`` can record it per call."""
    return _active_user_prompt.set(prompt)


def reset_active_user_prompt(token: contextvars.Token) -> None:
    """Restore the user-prompt binding that preceded ``token``."""
    _active_user_prompt.reset(token)


def get_active_user_prompt() -> str | None:
    return _active_user_prompt.get()
