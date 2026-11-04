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
    call_id: str | None = None,
) -> GenerationRecord:
    """Record a completed image call on the current episode owner.

    Args:
        prompt: Generation prompt.
        paths: Saved sample paths in backend order.
        backend: Service name; ``fewshot`` is excluded from live-call counts.
        tool_stubbed: Whether this call used a stub.
        artifact_id: Explicit ID for the first sample; other IDs come from filenames.
        call_id: Episode-local call ID, generated when omitted.

    Returns:
        The generation record registered on the current owner.
    """
    state = current_episode(create=True)
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
    """Clear generation records and image lookup for the current owner.

    Returns:
        None. Does nothing when no episode is bound.
    """
    state = current_episode()
    if state:
        state.clear_artifacts()


def count_live_generate_artifacts_for_active_rollout() -> int:
    """Count live generation calls with at least one existing PNG.

    Returns:
        Call count, excluding stubs and fewshot records; zero for an unbound
        episode or one without an episode ID.
    """
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
    """Read the most recently registered generation prompt.

    Returns:
        Prompt text, or None when the owner has no generation records.
    """
    state = current_episode()
    record = state.latest_generation if state else None
    return record.prompt if record else None


def set_latest_tool_image_path(path: str | None) -> None:
    """Set the current owner's latest image path, creating a local owner if needed.

    Args:
        path: Image path, or None to clear the direct image reference.

    Returns:
        None.
    """
    current_episode(create=True).set_latest_image(path)


def get_latest_tool_image_path() -> str | None:
    """Resolve the current owner's latest available image.

    Returns:
        Existing PNG path, or None when no image is available.
    """
    state = current_episode()
    return state.resolve_image() if state else None


def resolve_tool_image_path(*, image_prompt: str | None = None, artifact_id: str | None = None) -> str | None:
    """Resolve an existing PNG within the current episode.

    Args:
        image_prompt: Prompt used to match generation records.
        artifact_id: Explicit sample ID; an unknown ID never falls back.

    Returns:
        Matching PNG path, a latest-image fallback when no ID was supplied,
        or None when no image matches or no owner is bound.
    """
    state = current_episode()
    return state.resolve_image(image_prompt=image_prompt, artifact_id=artifact_id) if state else None
