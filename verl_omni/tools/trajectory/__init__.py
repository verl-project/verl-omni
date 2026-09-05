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

"""Rollout artifact paths, ContextVars, PNG registry, and YES latch.

Import from this package (not ``image_gen.py``) so helpers can be shared without
re-executing ``@function_tool`` registration.

Artifact layout (under the e2e run dir)::

    rollout_trajectories/step_{S:06d}/sample_{index}.{rollout_n:02d}.json
    rollout_images/step_{S:06d}/sample_{index}.{rollout_n:02d}/
        image_00_<artifact_id>.png ...
        meta.json
"""

from .artifacts import (
    clear_latest_tool_image_for_active_rollout,
    count_live_generate_artifacts_for_active_rollout,
    get_latest_generate_prompt_for_active_rollout,
    register_tool_artifact,
    resolve_tool_image_path,
    set_latest_tool_image_path,
)
from .context import (
    get_active_rollout_id,
    get_active_trajectory_relpath,
    get_active_user_prompt,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)
from .judge_latch import (
    clear_good_enough_yes_reached,
    get_good_enough_yes_reached,
    set_good_enough_yes_reached,
)
from .paths import (
    bind_run_artifact_env,
    build_artifact_id,
    build_trajectory_relpath,
    resolve_rollout_images_root,
    resolve_run_dir,
)

__all__ = [
    "bind_run_artifact_env",
    "build_artifact_id",
    "build_trajectory_relpath",
    "clear_good_enough_yes_reached",
    "clear_latest_tool_image_for_active_rollout",
    "count_live_generate_artifacts_for_active_rollout",
    "get_active_rollout_id",
    "get_active_trajectory_relpath",
    "get_active_user_prompt",
    "get_good_enough_yes_reached",
    "get_latest_generate_prompt_for_active_rollout",
    "register_tool_artifact",
    "reset_active_trajectory_relpath",
    "reset_active_user_prompt",
    "resolve_rollout_images_root",
    "resolve_run_dir",
    "resolve_tool_image_path",
    "set_active_trajectory_relpath",
    "set_active_user_prompt",
    "set_good_enough_yes_reached",
    "set_latest_tool_image_path",
]
