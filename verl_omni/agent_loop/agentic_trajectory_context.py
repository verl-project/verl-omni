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

"""Artifact path helpers and optional per-task diffusion-tool bindings.

Kept in a tiny module with no ``@function_tool`` registration so path helpers
can be shared by the stock-loop manager and diffusion tool safely.

Artifact layout (under the e2e run dir)::

    rollout_trajectories/step_{S:06d}/sample_{index}.{rollout_n:02d}.json
    rollout_images/step_{S:06d}/sample_{index}.{rollout_n:02d}/
        image_00_<artifact_id>.png ...
        meta.json

``artifact_id`` is ``sha256(relpath\\0index\\0prompt)[:12]`` — identity of the
generate call, not pixel content (overfit often reuses identical PNGs).

Judge lookup is **rollout-scoped** only. Cross-thread / cross-step fuzzy prompt
matching is intentionally removed: concurrent GRPO + identical overfit prompts
previously caused ``judge_image`` to score another rollout's (even previous
step's) PNG, corrupting live C/A rewards.
"""

from __future__ import annotations

import contextvars
import hashlib
import os
import re
import threading
from pathlib import Path


def default_e2e_root() -> Path:
    """Repo ``outputs/e2e`` (absolute). Prefer ``VERLOMNI_ROOT`` when set."""
    verlomni = os.getenv("VERLOMNI_ROOT", "").strip()
    if verlomni:
        return Path(verlomni).expanduser().resolve() / "outputs" / "e2e"
    # <repo>/verl_omni/agent_loop/this_file.py → parents[2] == repo root
    return Path(__file__).resolve().parents[2] / "outputs" / "e2e"


def resolve_e2e_root() -> Path:
    """Shared e2e artifact root for traj / images / hermes_actions."""
    explicit = os.getenv("AGENTIC_E2E_ROOT", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return default_e2e_root()


def resolve_run_dir() -> Path:
    """Per-run dir: ``<e2e_root>/<experiment_name>/``."""
    image_dir = os.getenv("AGENTIC_DIFFUSION_IMAGE_DIR", "").strip()
    if image_dir:
        return Path(image_dir).expanduser().resolve().parent
    run = os.getenv("AGENTIC_E2E_RUN_NAME", "").strip() or "agentic_run"
    return resolve_e2e_root() / run


def resolve_rollout_images_root() -> Path:
    """``<run_dir>/rollout_images`` (or explicit ``AGENTIC_DIFFUSION_IMAGE_DIR``)."""
    explicit = os.getenv("AGENTIC_DIFFUSION_IMAGE_DIR", "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return resolve_run_dir() / "rollout_images"


def bind_run_artifact_env(config) -> None:
    """Bind ``AGENTIC_E2E_{ROOT,RUN_NAME}`` so driver + Ray workers share one run dir.

    Must run on each AgentLoop worker before ``generate_image``: Ray's
    ``runtime_env`` is snapshotted at job start (often without RUN_NAME), while
    traj dumps use the manager's process env. Without this, images historically
    fell back to ``/tmp/agentic_qwen_image_t2i/...`` while traj landed under
    ``outputs/e2e/<experiment>/``.
    """
    if config is not None:
        experiment_name = config.trainer.get("experiment_name")
        if experiment_name:
            os.environ["AGENTIC_E2E_RUN_NAME"] = str(experiment_name)
    # Avoid a stale explicit image path from a previously sourced launcher.
    os.environ.pop("AGENTIC_DIFFUSION_IMAGE_DIR", None)
    os.environ["AGENTIC_E2E_ROOT"] = str(resolve_e2e_root())


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

# Live tool saves register here for judge lookup.
_artifact_registry_lock = threading.Lock()
_artifact_registry: list[dict] = []
# Direct artifact_id → png path (survives thread hops better than prompt match).
_artifact_by_id: dict[str, str] = {}
# Last PNG for each rollout_id (replaces bare thread-local latest-path).
_latest_image_by_rollout: dict[str, str] = {}


def rollout_id_from_relpath(relpath: str | None) -> str | None:
    """Short stable id for a trajectory folder (``sha256(relpath)[:16]``)."""
    text = (relpath or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_artifact_id(*, relpath: str, index: int, prompt: str) -> str:
    """Identity hash for one generate_image save (not a pixel content hash)."""
    blob = f"{relpath}\0{int(index)}\0{(prompt or '').strip()}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


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
    """Record a live generate_image save for judge lookup (rollout-scoped)."""
    relpath = trajectory_relpath or get_active_trajectory_relpath()
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
    """Drop registry rows and indexes for the active rollout (not other in-flight samples)."""
    rid = get_active_rollout_id()
    relpath = get_active_trajectory_relpath()
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

    Used by ``AGENTIC_BLOCK_GENERATE_AFTER_MAX_PASSES`` so the 4th+ generate is
    refused even when force-reflection is off for RL.
    """
    rid = get_active_rollout_id()
    n = 0
    with _artifact_registry_lock:
        for entry in _artifact_registry:
            if rid and entry.get("rollout_id") != rid:
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
    """Full diffusion prompt from the latest live artifact on this rollout."""
    rid = get_active_rollout_id()
    with _artifact_registry_lock:
        for entry in reversed(_artifact_registry):
            if rid and entry.get("rollout_id") != rid:
                continue
            prompt = str(entry.get("prompt") or "").strip()
            if prompt:
                return prompt
    return None


_latest_tool_image_path: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "agentic_latest_tool_image_path", default=None
)
# Kept only as a same-thread fallback for unit tests / smoke without rollout_id.
_latest_tool_image_tls = threading.local()

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


def set_latest_tool_image_path(path: str | None) -> contextvars.Token:
    """Remember the most recent generate_image PNG for judge_image (this rollout)."""
    rid = get_active_rollout_id()
    if rid:
        with _artifact_registry_lock:
            if path:
                _latest_image_by_rollout[rid] = str(path)
            else:
                _latest_image_by_rollout.pop(rid, None)
    _latest_tool_image_tls.path = path
    return _latest_tool_image_path.set(path)


def get_latest_tool_image_path() -> str | None:
    rid = get_active_rollout_id()
    if rid:
        with _artifact_registry_lock:
            scoped = _latest_image_by_rollout.get(rid)
        if scoped and Path(scoped).is_file():
            return scoped
    path = _latest_tool_image_path.get()
    if path and Path(path).is_file():
        return path
    tls = getattr(_latest_tool_image_tls, "path", None)
    if tls and Path(tls).is_file():
        return tls
    return None


def clear_latest_tool_image_for_active_rollout() -> None:
    """Drop this rollout's latest-image pointer, artifact registry rows, and id index."""
    clear_tool_artifacts_for_active_rollout()


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

    Strictly scoped to the **active rollout**. Order:
      1. Explicit ``artifact_id`` (from tool args / prior obs) if registered
      2. Latest PNG registered for this ``rollout_id``
      3. Same-rollout registry row with fuzzy-matching ``image_prompt``
      4. Same-rollout registry row (most recent)

    Never falls back to another rollout/thread/step via prompt match alone —
    that path corrupted live C/A rewards under concurrent overfit GRPO.
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
        relpath = get_active_trajectory_relpath()
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


def _sanitize_sample_index(sample_index: object | None) -> str:
    if sample_index is None:
        return "unknown"
    try:
        return str(int(sample_index))
    except (TypeError, ValueError):
        raw = str(sample_index)
        return re.sub(r"[^\w.\-]+", "_", raw)[:64] or "unknown"


def build_trajectory_relpath(*, step: int | None, sample_index: object | None, rollout_n: int) -> str:
    """Build ``step_XXXXXX/sample_{index}.{rollout_n:02d}``."""
    try:
        step_i = int(step) if step is not None else -1
    except (TypeError, ValueError):
        step_i = -1
    step_part = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
    sample_part = f"sample_{_sanitize_sample_index(sample_index)}.{int(rollout_n):02d}"
    return f"{step_part}/{sample_part}"
