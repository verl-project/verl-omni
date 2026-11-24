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

"""Compatibility accessors for the episode binding; new callers use episode_scope."""

from __future__ import annotations

from contextvars import Token

from verl_omni.utils.agentic.artifacts import rollout_id_from_relpath

from .episode import EpisodeState, _current_episode, _EpisodeBinding, current_episode


def get_active_rollout_id() -> str | None:
    """Return the current episode identity, or None when unbound."""
    state = current_episode()
    return state.episode_id if state else None


def get_active_trajectory_relpath() -> str | None:
    """Return the current episode artifact path, or None when unbound."""
    state = current_episode()
    return state.artifact_relpath if state else None


def get_active_user_prompt() -> str | None:
    """Return the callback-local prompt, including any scoped override."""
    binding = _current_episode.get()
    return binding.user_prompt if binding else None


def set_active_trajectory_relpath(relpath: str | None) -> Token[_EpisodeBinding | None]:
    """Adapt legacy path scopes without retaining a process-wide rollout registry."""
    state = EpisodeState(
        episode_id=rollout_id_from_relpath(relpath), artifact_relpath=relpath, user_prompt=get_active_user_prompt()
    )
    return _current_episode.set(_EpisodeBinding(state, state.user_prompt))


def reset_active_trajectory_relpath(token: Token[_EpisodeBinding | None]) -> None:
    """Restore the previous binding and close the owner created for the path scope.

    Args:
        token: Token returned by ``set_active_trajectory_relpath`` in this context.

    Returns:
        None.
    """
    state = current_episode()
    _current_episode.reset(token)
    if state is not None and state is not current_episode():
        state.close()


def set_active_user_prompt(prompt: str | None) -> Token[_EpisodeBinding | None]:
    """Bind a callback-local prompt override without changing the shared owner."""
    binding = _current_episode.get()
    state = binding.state if binding else EpisodeState()
    return _current_episode.set(_EpisodeBinding(state, prompt))


def reset_active_user_prompt(token: Token[_EpisodeBinding | None]) -> None:
    """Restore the caller's binding and close a prompt-only scratch owner."""
    state = current_episode()
    _current_episode.reset(token)
    if state is not None and state is not current_episode():
        state.close()
