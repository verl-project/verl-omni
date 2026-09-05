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

"""Run-dir roots, artifact ids, and ``step_*/sample_*.*`` relpaths."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

__all__ = [
    "bind_run_artifact_env",
    "build_artifact_id",
    "build_trajectory_relpath",
    "resolve_rollout_images_root",
    "resolve_run_dir",
    "rollout_id_from_relpath",
]


def default_e2e_root() -> Path:
    """Repo ``outputs/e2e`` (absolute). Prefer ``VERLOMNI_ROOT`` when set."""
    verlomni = os.getenv("VERLOMNI_ROOT", "").strip()
    if verlomni:
        return Path(verlomni).expanduser().resolve() / "outputs" / "e2e"
    # <repo>/verl_omni/tools/trajectory/this_file.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3] / "outputs" / "e2e"


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
