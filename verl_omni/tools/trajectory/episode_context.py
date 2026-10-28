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

from verl_omni.utils.agentic.artifacts import rollout_id_from_relpath

from .episode import EpisodeState, _current_episode, _EpisodeBinding, current_episode


def get_active_rollout_id() -> str | None:
    state = current_episode()
    return state.episode_id if state else None


def get_active_trajectory_relpath() -> str | None:
    state = current_episode()
    return state.artifact_relpath if state else None


def get_active_user_prompt() -> str | None:
    binding = _current_episode.get()
    return binding.user_prompt if binding else None


def set_active_trajectory_relpath(relpath: str | None):
    """Adapt legacy path scopes without retaining a process-wide rollout registry."""
    state = EpisodeState(
        episode_id=rollout_id_from_relpath(relpath), artifact_relpath=relpath, user_prompt=get_active_user_prompt()
    )
    return _current_episode.set(_EpisodeBinding(state, state.user_prompt))


def reset_active_trajectory_relpath(token) -> None:
    state = current_episode()
    _current_episode.reset(token)
    if state is not None and state is not current_episode():
        state.close()


class _FieldView:
    """Keep legacy get/set/reset callers on the same owner rather than a second ContextVar."""

    def __init__(self, name):
        self.name = name

    def get(self):
        return get_active_user_prompt() if self.name == "user_prompt" else get_active_trajectory_relpath()

    def set(self, value):
        if self.name == "artifact_relpath":
            return set_active_trajectory_relpath(value)
        state = current_episode(create=True)
        return _current_episode.set(_EpisodeBinding(state, value))

    def reset(self, token):
        if self.name == "artifact_relpath":
            reset_active_trajectory_relpath(token)
        else:
            _current_episode.reset(token)


active_trajectory_relpath = _FieldView("artifact_relpath")
active_user_prompt = _FieldView("user_prompt")
set_active_user_prompt = active_user_prompt.set
reset_active_user_prompt = active_user_prompt.reset
