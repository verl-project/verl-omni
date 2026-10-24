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

"""Per-rollout values (trajectory path, rollout id, user prompt) held in ContextVars."""

from __future__ import annotations

import contextvars

from .paths import rollout_id_from_relpath

__all__ = [
    "active_trajectory_relpath",
    "active_user_prompt",
    "clear_pending_generate_provenance",
    "consume_pending_generate_provenance",
    "get_active_rollout_id",
    "get_active_user_prompt",
    "get_pending_generate_provenance",
    "merge_pending_generate_provenance",
    "reset_active_trajectory_relpath",
    "reset_active_user_prompt",
    "set_active_trajectory_relpath",
    "set_active_user_prompt",
    "set_pending_generate_provenance",
]

# Relative path under the images/trajectories roots.
active_trajectory_relpath: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_trajectory_relpath", default=None
)
# Stable short id derived from trajectory_relpath (copied into asyncio.to_thread).
_active_rollout_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_rollout_id", default=None
)
# Dataset / task user request for the active trajectory (written into meta.json).
active_user_prompt: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_active_user_prompt", default=None
)


def set_active_trajectory_relpath(
    relpath: str | None,
) -> tuple[contextvars.Token, contextvars.Token]:
    """Bind the relative artifact path and matching rollout id.

    Args:
        relpath: Path under rollout_images / rollout_trajectories, or ``None``.

    Returns:
        ``(path_token, rollout_token)`` for ``reset_active_trajectory_relpath``.
    """
    rid = rollout_id_from_relpath(relpath)
    path_token = active_trajectory_relpath.set(relpath)
    rollout_token = _active_rollout_id.set(rid)
    return path_token, rollout_token


def reset_active_trajectory_relpath(
    tokens: tuple[contextvars.Token, contextvars.Token],
) -> None:
    """Restore trajectory path and rollout id bindings.

    Args:
        tokens: ``(path_token, rollout_token)`` from ``set_active_trajectory_relpath``.

    Returns:
        None.
    """
    path_token, rollout_token = tokens
    active_trajectory_relpath.reset(path_token)
    _active_rollout_id.reset(rollout_token)


def get_active_rollout_id() -> str | None:
    """Return the active rollout id.

    Returns:
        Rollout id string, or ``None`` if unbound.
    """
    rid = _active_rollout_id.get()
    if rid:
        return rid
    return rollout_id_from_relpath(active_trajectory_relpath.get())


def set_active_user_prompt(prompt: str | None) -> contextvars.Token:
    """Bind the dataset user request so ``meta.json`` can record it per call."""
    return active_user_prompt.set(prompt)


def reset_active_user_prompt(token: contextvars.Token) -> None:
    """Restore the user-prompt binding that preceded ``token``."""
    active_user_prompt.reset(token)


def get_active_user_prompt() -> str | None:
    """Return the active dataset user request, or ``None`` if unbound."""
    return active_user_prompt.get()


# Optional rewrite provenance written before generate_image executes.
_pending_generate_provenance: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "agentic_pending_generate_provenance", default=None
)


def set_pending_generate_provenance(data: dict | None) -> contextvars.Token:
    """Bind rewrite provenance consumed by the next ``generate_image`` call."""
    return _pending_generate_provenance.set(dict(data) if data else None)


def merge_pending_generate_provenance(updates: dict) -> contextvars.Token:
    """Merge non-None ``updates`` into the pending provenance dict."""
    current = dict(_pending_generate_provenance.get() or {})
    current.update({key: value for key, value in updates.items() if value is not None})
    return _pending_generate_provenance.set(current)


def get_pending_generate_provenance() -> dict | None:
    """Return the pending provenance dict, or ``None``."""
    pending = _pending_generate_provenance.get()
    return dict(pending) if pending else None


def consume_pending_generate_provenance() -> dict:
    """Return and clear the pending provenance (one-shot per generate call)."""
    pending = dict(_pending_generate_provenance.get() or {})
    _pending_generate_provenance.set(None)
    return pending


def clear_pending_generate_provenance() -> None:
    """Drop any pending provenance without consuming it."""
    _pending_generate_provenance.set(None)
