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

from omegaconf import DictConfig, OmegaConf

__all__ = [
    "bind_run_artifacts",
    "build_artifact_id",
    "build_trajectory_relpath",
    "clear_run_artifacts",
    "project_root",
    "resolve_rollout_images_root",
    "get_run_name",
    "resolve_run_dir",
    "rollout_id_from_relpath",
]

_run_config: DictConfig | None = None


def project_root() -> Path:
    """Return the verl-omni checkout root (honours ``VERLOMNI_ROOT`` when set).

    Returns:
        Absolute repo root.
    """
    verlomni = os.getenv("VERLOMNI_ROOT", "").strip()
    if verlomni:
        return Path(verlomni).expanduser().resolve()
    # <repo>/verl_omni/utils/agentic/this_file.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def default_e2e_root() -> Path:
    """Return the default e2e artifact root.

    Returns:
        Absolute ``outputs/e2e`` path (honours ``VERLOMNI_ROOT`` when set).
    """
    return project_root() / "outputs" / "e2e"


def clear_run_artifacts() -> None:
    """Clear the process-local artifact configuration binding."""
    global _run_config
    _run_config = None


def get_run_name() -> str:
    """Return the configured experiment name or the standalone-tool default."""
    name = OmegaConf.select(_run_config, "trainer.experiment_name") if _run_config is not None else None
    return str(name) if name else "agentic_run"


def resolve_e2e_root() -> Path:
    """Return the configured artifact root, defaulting to project outputs/e2e."""
    root = OmegaConf.select(_run_config, "agentic_image_gen.e2e_root") if _run_config is not None else None
    return Path(str(root)).expanduser().resolve() if root else default_e2e_root()


def resolve_run_dir() -> Path:
    """Return explicit run_dir verbatim, or e2e_root / experiment_name."""
    run_dir = OmegaConf.select(_run_config, "agentic_image_gen.run_dir") if _run_config is not None else None
    return Path(str(run_dir)).expanduser().resolve() if run_dir else resolve_e2e_root() / get_run_name()


def resolve_rollout_images_root() -> Path:
    """Return the rollout_images directory under the configured run directory."""
    return resolve_run_dir() / "rollout_images"


def bind_run_artifacts(config: DictConfig | None) -> None:
    """Snapshot the incoming Hydra config for function-tool artifact readers.

    Each worker binds its own config before tools run. Replace the whole binding
    so omitted fields cannot inherit a previous run's directory settings.
    """
    global _run_config
    _run_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False)) if config is not None else None


def rollout_id_from_relpath(relpath: str | None) -> str | None:
    """Derive a short stable id for a trajectory folder.

    Args:
        relpath: Trajectory relative path.

    Returns:
        ``sha256(relpath)[:16]``, or ``None`` if empty.
    """
    text = (relpath or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def build_artifact_id(*, relpath: str, index: int, prompt: str) -> str:
    """Build an identity hash for one ``generate_image`` save.

    Args:
        relpath: Trajectory relative path.
        index: Image index within the trajectory.
        prompt: Diffusion prompt.

    Returns:
        12-char hex id (not a pixel content hash).
    """
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
    """Build a trajectory relative path for one sample/rollout.

    Args:
        step: Global step (``None`` → ``step_unknown``).
        sample_index: Dataset sample index.
        rollout_n: Rollout index within the sample.

    Returns:
        Path like ``step_XXXXXX/sample_{index}.{rollout_n:02d}``.
    """
    try:
        step_i = int(step) if step is not None else -1
    except (TypeError, ValueError):
        step_i = -1
    step_part = f"step_{step_i:06d}" if step_i >= 0 else "step_unknown"
    sample_part = f"sample_{_sanitize_sample_index(sample_index)}.{int(rollout_n):02d}"
    return f"{step_part}/{sample_part}"
