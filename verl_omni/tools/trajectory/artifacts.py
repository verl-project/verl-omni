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

"""Live generate_image PNG registry and judge path lookup (rollout-scoped)."""

from __future__ import annotations

import contextvars
import re
import threading
from pathlib import Path

from .context import active_trajectory_relpath, get_active_rollout_id
from .paths import rollout_id_from_relpath

__all__ = [
    "clear_latest_tool_image_for_active_rollout",
    "count_live_generate_artifacts_for_active_rollout",
    "get_latest_generate_prompt_for_active_rollout",
    "register_tool_artifact",
    "resolve_tool_image_path",
    "set_latest_tool_image_path",
]

# Live tool saves register here for judge lookup.
_artifact_registry_lock = threading.Lock()
_artifact_registry: list[dict] = []
# Direct artifact_id → png path (survives thread hops better than prompt match).
_artifact_by_id: dict[str, str] = {}
# Last PNG for each rollout_id (replaces bare thread-local latest-path).
_latest_image_by_rollout: dict[str, str] = {}

_latest_tool_image_path: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_latest_tool_image_path", default=None
)


def register_tool_artifact(
    *,
    prompt: str,
    paths: list[str],
    backend: str = "",
    tool_stubbed: bool = False,
    artifact_id: str | None = None,
    trajectory_relpath: str | None = None,
    rollout_id: str | None = None,
) -> None:
    """Record a live ``generate_image`` save for later judge lookup.

    Args:
        prompt: Diffusion prompt for this call.
        paths: Saved image paths.
        backend: Sidecar name (logging).
        tool_stubbed: If True, skip for max-pass counting.
        artifact_id: Optional short id recovered from the filename when omitted.
        trajectory_relpath: Override for the active ContextVar relpath.
        rollout_id: Override for the active rollout id.

    Returns:
        None.
    """
    relpath = trajectory_relpath or active_trajectory_relpath.get()
    rid = rollout_id or rollout_id_from_relpath(relpath) or get_active_rollout_id()
    png = _first_existing_png(paths)
    aid = (artifact_id or "").strip() or None
    if not aid and png:
        # Recover id from ``image_00_<hash>.png`` when callers omit it.
        m = re.search(r"image_\d+_([0-9a-f]{12})\.png$", Path(png).name, re.IGNORECASE)
        if m:
            aid = m.group(1)
    entry = {
        "prompt": (prompt or "").strip(),
        "paths": [str(p) for p in paths],
        "backend": backend,
        "tool_stubbed": bool(tool_stubbed),
        "thread_id": threading.get_ident(),
        "trajectory_relpath": relpath,
        "rollout_id": rid,
        "artifact_id": aid,
    }
    with _artifact_registry_lock:
        _artifact_registry.append(entry)
        if aid and png:
            _artifact_by_id[aid] = png
        if rid and png:
            _latest_image_by_rollout[rid] = png


def _entry_belongs_to_active_rollout(entry: dict, rid: str | None, relpath: str | None) -> bool:
    if rid and entry.get("rollout_id") == rid:
        return True
    if not rid and relpath and entry.get("trajectory_relpath") == relpath:
        return True
    return False


def clear_tool_artifacts_for_active_rollout() -> None:
    """Drop registry rows and indexes for the active rollout.

    Returns:
        None.
    """
    rid = get_active_rollout_id()
    relpath = active_trajectory_relpath.get()
    if not rid and not relpath:
        set_latest_tool_image_path(None)
        return
    with _artifact_registry_lock:
        kept: list[dict] = []
        drop_ids: list[str] = []
        for entry in _artifact_registry:
            if _entry_belongs_to_active_rollout(entry, rid, relpath):
                aid = entry.get("artifact_id")
                if aid:
                    drop_ids.append(str(aid))
            else:
                kept.append(entry)
        _artifact_registry[:] = kept
        for aid in drop_ids:
            _artifact_by_id.pop(aid, None)
        if rid:
            _latest_image_by_rollout.pop(rid, None)
    set_latest_tool_image_path(None)


def count_live_generate_artifacts_for_active_rollout() -> int:
    """Count successful live ``generate_image`` PNGs for the active rollout.

    Returns:
        Count for the bound ``rollout_id``, or ``0`` if none is bound.
    """
    rid = get_active_rollout_id()
    if not rid:
        return 0
    n = 0
    with _artifact_registry_lock:
        for entry in _artifact_registry:
            if entry.get("rollout_id") != rid:
                continue
            if entry.get("tool_stubbed"):
                continue
            if str(entry.get("backend") or "").lower() == "fewshot":
                continue
            if not _first_existing_png(entry.get("paths")):
                continue
            n += 1
    return n


def get_latest_generate_prompt_for_active_rollout() -> str | None:
    """Return the diffusion prompt from the latest live artifact on this rollout.

    Returns:
        Prompt string, or ``None`` if none.
    """
    rid = get_active_rollout_id()
    with _artifact_registry_lock:
        for entry in reversed(_artifact_registry):
            if rid and entry.get("rollout_id") != rid:
                continue
            prompt = str(entry.get("prompt") or "").strip()
            if prompt:
                return prompt
    return None


def set_latest_tool_image_path(path: str | None) -> contextvars.Token:
    """Remember the most recent ``generate_image`` PNG for ``judge_image``.

    Args:
        path: Absolute PNG path, or ``None`` to clear.

    Returns:
        ContextVar token for the path binding.
    """
    rid = get_active_rollout_id()
    if rid:
        with _artifact_registry_lock:
            if path:
                _latest_image_by_rollout[rid] = str(path)
            else:
                _latest_image_by_rollout.pop(rid, None)
    return _latest_tool_image_path.set(path)


def get_latest_tool_image_path() -> str | None:
    """Return the latest tool image path for the active rollout.

    Returns:
        PNG path string, or ``None``.
    """
    rid = get_active_rollout_id()
    if rid:
        with _artifact_registry_lock:
            scoped = _latest_image_by_rollout.get(rid)
        if scoped and Path(scoped).is_file():
            return scoped
    path = _latest_tool_image_path.get()
    if path and Path(path).is_file():
        return path
    return None


def clear_latest_tool_image_for_active_rollout() -> None:
    """Drop this rollout's latest-image pointer and related registry rows.

    Returns:
        None.
    """
    clear_tool_artifacts_for_active_rollout()


def _first_existing_png(paths: list[str] | None) -> str | None:
    for path in paths or []:
        text = str(path)
        if text.endswith(".png") and Path(text).is_file():
            return text
    return None


def _normalize_prompt(text: str | None) -> str:
    """Collapse whitespace/punctuation drift between generate and judge args."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _prompts_match(a: str | None, b: str | None) -> bool:
    na, nb = _normalize_prompt(a), _normalize_prompt(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Agent often lightly rewrites commas/wording when echoing image_prompt.
    return na[:96] == nb[:96] or na in nb or nb in na


def resolve_tool_image_path(
    *,
    image_prompt: str | None = None,
    artifact_id: str | None = None,
) -> str | None:
    """Resolve the PNG that ``judge_image`` should score.

    Args:
        prompt: Optional diffusion prompt used to match a registry entry.
        artifact_id: Optional short artifact id.

    Returns:
        Absolute PNG path, or ``None`` if unresolved.
    """
    aid = (artifact_id or "").strip()
    if aid:
        with _artifact_registry_lock:
            png = _artifact_by_id.get(aid)
        if png and Path(png).is_file():
            return png

    rid = get_active_rollout_id()
    direct = get_latest_tool_image_path()
    if direct and Path(direct).is_file():
        # If we have a rollout_id, only accept the direct hit when it belongs
        # to this rollout (path contains the active trajectory folder).
        relpath = active_trajectory_relpath.get()
        if not rid or not relpath or relpath in direct.replace("\\", "/"):
            return direct

    if not rid:
        # Smoke / unbound tools: keep same-thread latest only (no cross-prompt).
        return direct if direct and Path(direct).is_file() else None

    want = image_prompt or ""
    prompt_hit: str | None = None
    latest_hit: str | None = None
    with _artifact_registry_lock:
        for entry in reversed(_artifact_registry):
            if entry.get("rollout_id") != rid:
                continue
            png = _first_existing_png(entry.get("paths"))
            if not png:
                continue
            if latest_hit is None:
                latest_hit = png
            if want and _prompts_match(entry.get("prompt"), want):
                prompt_hit = png
                break
    if prompt_hit:
        return prompt_hit
    return latest_hit
