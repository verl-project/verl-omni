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
from typing import Any

from verl_omni.tools.trajectory.hydra_env import agentic_get

__all__ = [
    "bind_run_artifacts",
    "build_artifact_id",
    "build_trajectory_relpath",
    "clear_run_artifacts",
    "project_relpath",
    "project_root",
    "resolve_rollout_images_root",
    "run_name",
    "resolve_run_dir",
    "rollout_id_from_relpath",
]

run_name: str = "agentic_run"
_e2e_root: Path | None = None
_diffusion_image_dir: Path | None = None
# Explicit ``agentic_image_gen.run_dir``: used verbatim as the run dir, so a caller who
# names the folder has chosen the layout and nothing is appended to it.
_run_dir_override: Path | None = None


def project_root() -> Path:
    """Return the verl-omni checkout root (honours ``VERLOMNI_ROOT`` when set).

    Returns:
        Absolute repo root.
    """
    verlomni = os.getenv("VERLOMNI_ROOT", "").strip()
    if verlomni:
        return Path(verlomni).expanduser().resolve()
    # <repo>/verl_omni/tools/trajectory/this_file.py → parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def project_relpath(path: str | Path) -> str:
    """Render ``path`` relative to the project root for artifact records.

    Artifacts written under the checkout (``outputs/<run>/...``) stay valid after the
    checkout moves or is reviewed on another host. A path outside the checkout has no
    project-relative form, so it is returned absolute rather than as a ``../`` chain.

    Args:
        path: File or directory path.

    Returns:
        POSIX path relative to ``project_root()``, or the absolute path when outside it.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(project_root()).as_posix()
    except ValueError:
        return str(resolved)


def default_e2e_root() -> Path:
    """Return the default e2e artifact root.

    Returns:
        Absolute ``outputs/e2e`` path (honours ``VERLOMNI_ROOT`` when set).
    """
    return project_root() / "outputs" / "e2e"


def clear_run_artifacts() -> None:
    """Reset process-local run-dir bindings (tests).

    Returns:
        None.
    """
    global run_name, _e2e_root, _diffusion_image_dir, _run_dir_override
    run_name = "agentic_run"
    _e2e_root = None
    _diffusion_image_dir = None
    _run_dir_override = None


def resolve_e2e_root() -> Path:
    """Return the shared e2e artifact root.

    Returns:
        Absolute path for traj / images / hermes dumps.
    """
    if _e2e_root is not None:
        return _e2e_root
    return default_e2e_root()


def resolve_run_dir() -> Path:
    """Return the directory that holds the three per-run artifact trees.

    This is the root of ``rollout_trajectories/``, ``rollout_images/`` and
    ``hermes_actions/``. Resolution order:

    1. An explicit ``agentic_image_gen.run_dir``, returned verbatim. The caller has
       named the folder, so appending ``experiment_name`` would nest a run inside the
       directory it just asked for.
    2. ``<e2e_root>/<experiment_name>/`` (the default). Namespacing by experiment name
       is what keeps two runs that share an ``experiment_name`` from overwriting each
       other's ``step_*`` dumps in place.

    Returns:
        Absolute run dir.
    """
    if _run_dir_override is not None:
        return _run_dir_override
    if _diffusion_image_dir is not None:
        return _diffusion_image_dir.parent
    return resolve_e2e_root() / (run_name or "agentic_run")


def resolve_rollout_images_root() -> Path:
    """Return the rollout images directory.

    Returns:
        ``<run_dir>/rollout_images`` (or an explicit diffusion override).
    """
    if _diffusion_image_dir is not None:
        return _diffusion_image_dir
    return resolve_run_dir() / "rollout_images"


def _node_get(node: Any, key: str) -> Any:
    """Read one ``agentic_image_gen`` key from a Hydra node.

    Handles both the dict-style (``OmegaConf``) and attribute-style objects that
    ``config.get("agentic_image_gen")`` can hand back.

    Args:
        node: The ``agentic_image_gen`` config node, or ``None``.
        key: Field name.

    Returns:
        The value, or ``None`` when the node or key is absent.
    """
    if node is None:
        return None
    try:
        return node.get(key)
    except Exception:  # noqa: BLE001 — a plain object has no ``.get``
        return getattr(node, key, None)


def bind_run_artifacts(config: Any) -> None:
    """Bind run-dir knobs from Hydra so driver and Ray workers share one layout.

    Args:
        config: Hydra config. ``trainer.experiment_name`` sets the run name;
            ``agentic_image_gen.run_dir`` sets the run dir verbatim;
            ``agentic_image_gen.e2e_root`` overrides the default ``outputs/e2e``.

    Returns:
        None.
    """
    global run_name, _e2e_root, _diffusion_image_dir, _run_dir_override
    # Drop stale explicit overrides from a previous bind/test.
    _diffusion_image_dir = None
    _run_dir_override = None
    if config is None:
        _e2e_root = default_e2e_root()
        return

    try:
        experiment_name = config.trainer.get("experiment_name")
    except Exception:  # noqa: BLE001
        experiment_name = None
    if experiment_name:
        run_name = str(experiment_name)

    e2e_root = None
    try:
        node = config.get("agentic_image_gen")
    except Exception:  # noqa: BLE001
        node = getattr(config, "agentic_image_gen", None)

    # ``run_dir`` is authoritative when set: it is the run dir itself, not a root to
    # append ``experiment_name`` to. The bagel recipes set it to ``$RUN_DIR`` so the
    # three artifact trees land beside ``.hydra/`` and ``main_omni.log`` instead of
    # under the extra ``e2e/<experiment_name>/`` nesting. Reading it here (rather than
    # reusing ``e2e_root``) leaves ``e2e_root``'s namespacing semantics intact for
    # every caller that relies on ``<e2e_root>/<experiment_name>/``.
    run_dir = _node_get(node, "run_dir")
    if run_dir is None:
        try:
            run_dir = agentic_get("run_dir", None)
        except RuntimeError:
            # Mirror the e2e_root fallback: bind_agentic_image_gen may not have run yet.
            run_dir = None
    if run_dir:
        _run_dir_override = Path(str(run_dir)).expanduser().resolve()

    if node is not None:
        e2e_root = _node_get(node, "e2e_root")
    if e2e_root is None:
        try:
            e2e_root = agentic_get("e2e_root")
        except RuntimeError:
            # bind_agentic_image_gen has not run in this process yet (a caller bound
            # run artifacts first). Fall back to the documented default root instead
            # of crashing worker/manager init (audit blocker: bind-order crash).
            e2e_root = None
    if e2e_root:
        _e2e_root = Path(str(e2e_root)).expanduser().resolve()
    else:
        _e2e_root = default_e2e_root()


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
