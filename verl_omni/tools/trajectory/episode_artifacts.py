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

"""Legacy image-tool adapters over episode-owned generation records."""

from __future__ import annotations

import re
from pathlib import Path

from .episode import GenerationRecord, current_episode


def register_tool_artifact(
    *,
    prompt: str,
    paths: list[str],
    backend: str = "",
    tool_stubbed: bool = False,
    artifact_id: str | None = None,
    trajectory_relpath: str | None = None,
    rollout_id: str | None = None,
    call_id: str | None = None,
) -> GenerationRecord | None:
    """Adapt a completed image call; explicit scope overrides cannot cross episode owners."""
    state = current_episode(create=True)
    if rollout_id is not None and rollout_id != state.episode_id:
        return None
    if trajectory_relpath is not None and trajectory_relpath != state.artifact_relpath:
        return None
    identifiers = []
    for index, path in enumerate(paths):
        match = re.search(r"image_\d+_([0-9a-f]{12})\.png$", Path(path).name, re.IGNORECASE)
        identifiers.append(artifact_id if index == 0 and artifact_id else match[1] if match else None)
    return state.record_generation(
        prompt=prompt,
        paths=paths,
        artifact_ids=identifiers,
        call_id=call_id,
        backend=backend,
        tool_stubbed=tool_stubbed,
    )


def clear_tool_artifacts_for_active_rollout() -> None:
    state = current_episode()
    if state:
        state.clear_artifacts()


def count_live_generate_artifacts_for_active_rollout() -> int:
    state = current_episode()
    if state is None or state.episode_id is None:
        return 0
    return sum(
        not record.tool_stubbed
        and record.backend.lower() != "fewshot"
        and any(str(p).endswith(".png") and Path(p).is_file() for p in record.paths)
        for record in state.generations
    )


def get_latest_generate_prompt_for_active_rollout() -> str | None:
    state = current_episode()
    record = state.latest_generation if state else None
    return record.prompt if record else None


def set_latest_tool_image_path(path: str | None) -> None:
    current_episode(create=True).set_latest_image(path)


def get_latest_tool_image_path() -> str | None:
    state = current_episode()
    return state.resolve_image() if state else None


clear_latest_tool_image_for_active_rollout = clear_tool_artifacts_for_active_rollout


def resolve_tool_image_path(*, image_prompt: str | None = None, artifact_id: str | None = None) -> str | None:
    state = current_episode()
    return state.resolve_image(image_prompt=image_prompt, artifact_id=artifact_id) if state else None


def build_generate_call_meta(*, prompt: str, user_prompt: str) -> dict:
    """Preserve the existing metadata format while deriving call history from its owner."""
    previous = get_latest_generate_prompt_for_active_rollout() or ""
    rewrite = bool(previous)
    return {
        "call_role": "rewrite" if rewrite else "initial",
        "controlled_by_reflection": rewrite,
        "reflection": "",
        "prev_tool_prompt": previous,
        "source_image_for_reflection": (get_latest_tool_image_path() or "") if rewrite else "",
        "rewritten_prompt": prompt if rewrite else "",
        "image_generated_from_reflected_prompt": rewrite,
        "tool_prompt_equals_rewritten_prompt": rewrite,
        "content_source": "reflection" if rewrite else "initial",
        "llm_reflection": "",
        "llm_prompt": prompt,
        "model_decode": "",
        "user_prompt": user_prompt or "",
        "tool_prompt": prompt,
    }
