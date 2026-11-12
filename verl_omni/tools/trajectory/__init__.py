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

"""Episode-owned tool state, with compatibility exports for worker configuration."""

from verl_omni.utils.agentic.artifacts import (
    bind_run_artifacts,
    build_artifact_id,
    build_trajectory_relpath,
    clear_run_artifacts,
    resolve_rollout_images_root,
    resolve_run_dir,
)
from verl_omni.utils.agentic.config import (
    bind_agentic_image_gen,
    clear_agentic_image_gen,
    merge_agentic_scorer_knobs,
)

from .episode import EpisodeState, GenerationRecord, JudgmentRecord, current_episode, episode_scope
from .episode_artifacts import (
    clear_tool_artifacts_for_active_rollout,
    count_live_generate_artifacts_for_active_rollout,
    get_latest_generate_prompt_for_active_rollout,
    register_tool_artifact,
    resolve_tool_image_path,
    set_latest_tool_image_path,
)
from .episode_context import (
    get_active_rollout_id,
    get_active_trajectory_relpath,
    get_active_user_prompt,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from .episode_judge import (
    clear_good_enough_yes_reached,
    get_good_enough_yes_reached,
    set_good_enough_yes_reached,
)

__all__ = [
    "EpisodeState",
    "GenerationRecord",
    "JudgmentRecord",
    "current_episode",
    "episode_scope",
    "bind_agentic_image_gen",
    "merge_agentic_scorer_knobs",
    "bind_run_artifacts",
    "build_artifact_id",
    "build_trajectory_relpath",
    "clear_agentic_image_gen",
    "clear_good_enough_yes_reached",
    "clear_tool_artifacts_for_active_rollout",
    "clear_run_artifacts",
    "count_live_generate_artifacts_for_active_rollout",
    "get_active_trajectory_relpath",
    "get_active_user_prompt",
    "set_active_user_prompt",
    "reset_active_user_prompt",
    "get_active_rollout_id",
    "get_good_enough_yes_reached",
    "get_latest_generate_prompt_for_active_rollout",
    "register_tool_artifact",
    "reset_active_trajectory_relpath",
    "resolve_rollout_images_root",
    "resolve_run_dir",
    "resolve_tool_image_path",
    "set_active_trajectory_relpath",
    "set_good_enough_yes_reached",
    "set_latest_tool_image_path",
]
