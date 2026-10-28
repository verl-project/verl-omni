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
"""Load, validate, and precheck declarative L4 convergence recipes.

An L4 recipe is a YAML document that declares, for one real training recipe:

* which existing example launcher to run and which extra Hydra overrides to append;
* which *real* model checkpoints and which *real* dataset shards are required;
* which hardware the recipe needs;
* which per-step metrics define convergence, in which direction, and with what
  tolerance;
* which release baseline artifact the result must be compared against.

The registry deliberately refuses to run a recipe until every declared
precondition is satisfied by something on disk.  It never substitutes a default
that would let a run pass without real weights or a real dataset, and it refuses
synthetic ``tiny-random`` models outright: those belong to L2/L3.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = 1
LAYER = "L4"

CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
DIRECTIONS = ("higher", "lower")

#: Mixed-precision settings a recipe may declare.  The value is part of the
#: comparability contract: two runs in different precisions are not comparable.
PRECISIONS = ("bf16", "fp32", "fp16")

#: Substrings that identify the tiny/synthetic models used by L2/L3.  L4 must
#: never accept these as "real weights": a convergence verdict computed from a
#: random model is meaningless.
SYNTHETIC_WEIGHT_MARKERS = ("tiny-random", "tiny_random", "tinyrandom", "dummy-model", "random-init")

GPU_QUERY_FIELDS = ("index", "name", "memory.total", "memory.free", "compute_cap")

_TOP_LEVEL_KEYS = (
    "schema_version",
    "layer",
    "case_id",
    "title",
    "algorithm",
    "precision",
    "recipe",
    "models",
    "dataset",
    "hardware",
    "budget",
    "metrics",
    "convergence",
    "baseline",
)


class RecipeError(Exception):
    """Raised when a recipe file cannot be parsed, validated, or resolved."""


@dataclass(frozen=True)
class ValidationIssue:
    """A single schema or semantic problem found in a recipe document."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


@dataclass
class Recipe:
    """A validated L4 recipe plus the provenance of the document it came from."""

    data: dict[str, Any]
    source: Path | None = None
    sha256: str = ""

    @property
    def case_id(self) -> str:
        return str(self.data["case_id"])

    @property
    def title(self) -> str:
        return str(self.data["title"])

    @property
    def algorithm(self) -> str:
        return str(self.data["algorithm"])

    @property
    def precision(self) -> str:
        return str(self.data["precision"])

    @property
    def launcher(self) -> str:
        return str(self.data["recipe"]["launcher"])

    @property
    def overrides(self) -> list[str]:
        return [str(item) for item in self.data["recipe"].get("overrides", [])]

    @property
    def release_gate(self) -> bool:
        return bool(self.data.get("release_gate", True))

    @property
    def metrics(self) -> list[dict[str, Any]]:
        return list(self.data["metrics"]["tracked"])

    @property
    def required_metric_names(self) -> list[str]:
        return [str(item["name"]) for item in self.metrics if not item.get("optional", False)]

    def metric_direction(self, name: str) -> str | None:
        for item in self.metrics:
            if str(item["name"]) == name:
                return str(item["direction"])
        return None


def compute_recipe_sha256(payload: dict[str, Any]) -> str:
    """Return a stable content hash for a recipe document."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def _issue(issues: list[ValidationIssue], path: str, message: str) -> None:
    issues.append(ValidationIssue(path=path, message=message))


def _require_mapping(issues: list[ValidationIssue], value: Any, path: str) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        _issue(issues, path, f"must be a mapping, got {type(value).__name__}")
        return None
    return value


def _require_str(issues: list[ValidationIssue], value: Any, path: str, *, allow_empty: bool = False) -> str | None:
    if not isinstance(value, str):
        _issue(issues, path, f"must be a string, got {type(value).__name__}")
        return None
    if not allow_empty and not value.strip():
        _issue(issues, path, "must not be empty")
        return None
    return value


def _require_int(issues: list[ValidationIssue], value: Any, path: str, *, minimum: int | None = None) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        _issue(issues, path, f"must be an integer, got {type(value).__name__}")
        return None
    if minimum is not None and value < minimum:
        _issue(issues, path, f"must be >= {minimum}, got {value}")
        return None
    return value


def _require_number(
    issues: list[ValidationIssue], value: Any, path: str, *, minimum: float | None = None
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _issue(issues, path, f"must be a number, got {type(value).__name__}")
        return None
    number = float(value)
    if minimum is not None and number < minimum:
        _issue(issues, path, f"must be >= {minimum}, got {number}")
        return None
    return number


def _validate_recipe_section(issues: list[ValidationIssue], section: Any) -> None:
    payload = _require_mapping(issues, section, "recipe")
    if payload is None:
        return
    _require_str(issues, payload.get("launcher"), "recipe.launcher")
    overrides = payload.get("overrides", [])
    if not isinstance(overrides, list):
        _issue(issues, "recipe.overrides", f"must be a list, got {type(overrides).__name__}")
        return
    for index, item in enumerate(overrides):
        _require_str(issues, item, f"recipe.overrides[{index}]")


def _validate_models(issues: list[ValidationIssue], section: Any) -> None:
    models = _require_mapping(issues, section, "models")
    if models is None:
        return
    if not models:
        _issue(issues, "models", "must declare at least one real model")
        return
    for name, entry in models.items():
        path = f"models.{name}"
        payload = _require_mapping(issues, entry, path)
        if payload is None:
            continue
        kind = _require_str(issues, payload.get("kind"), f"{path}.kind")
        if kind is not None and kind != "real":
            _issue(issues, f"{path}.kind", f"must be 'real' for L4, got '{kind}'")
        source = _require_str(issues, payload.get("source"), f"{path}.source")
        if source is not None:
            lowered = source.lower()
            for marker in SYNTHETIC_WEIGHT_MARKERS:
                if marker in lowered:
                    _issue(
                        issues,
                        f"{path}.source",
                        f"looks synthetic ('{marker}'); L4 requires real released checkpoints",
                    )
        _require_str(issues, payload.get("revision"), f"{path}.revision")
        if "local_path" in payload:
            _require_str(issues, payload["local_path"], f"{path}.local_path")
        if "env_path" in payload:
            _require_str(issues, payload["env_path"], f"{path}.env_path")


def _validate_dataset(issues: list[ValidationIssue], section: Any) -> None:
    dataset = _require_mapping(issues, section, "dataset")
    if dataset is None:
        return
    kind = _require_str(issues, dataset.get("kind"), "dataset.kind")
    if kind is not None and kind != "real":
        _issue(issues, "dataset.kind", f"must be 'real' for L4, got '{kind}'")
        return
    for key in ("train", "val"):
        _require_str(issues, dataset.get(key), f"dataset.{key}")
    if "env_root" in dataset:
        _require_str(issues, dataset["env_root"], "dataset.env_root")
    if "train_max_samples" in dataset:
        _require_int(issues, dataset["train_max_samples"], "dataset.train_max_samples", minimum=1)


def _validate_hardware(issues: list[ValidationIssue], section: Any) -> None:
    hardware = _require_mapping(issues, section, "hardware")
    if hardware is None:
        return
    _require_int(issues, hardware.get("min_gpus"), "hardware.min_gpus", minimum=1)
    _require_int(issues, hardware.get("min_gpu_memory_gb"), "hardware.min_gpu_memory_gb", minimum=1)
    architectures = hardware.get("gpu_architectures", [])
    if not isinstance(architectures, list) or not architectures:
        _issue(issues, "hardware.gpu_architectures", "must be a non-empty list of SM architecture names")
    else:
        for index, item in enumerate(architectures):
            _require_str(issues, item, f"hardware.gpu_architectures[{index}]")
    if "min_free_disk_gb" in hardware:
        _require_int(issues, hardware["min_free_disk_gb"], "hardware.min_free_disk_gb", minimum=1)


def _validate_budget(issues: list[ValidationIssue], section: Any) -> None:
    budget = _require_mapping(issues, section, "budget")
    if budget is None:
        return
    _require_number(issues, budget.get("timeout_minutes"), "budget.timeout_minutes", minimum=1.0)
    _require_int(issues, budget.get("total_training_steps"), "budget.total_training_steps", minimum=1)


def _validate_metrics(issues: list[ValidationIssue], section: Any) -> None:
    metrics = _require_mapping(issues, section, "metrics")
    if metrics is None:
        return
    tracked = metrics.get("tracked")
    if not isinstance(tracked, list) or not tracked:
        _issue(issues, "metrics.tracked", "must be a non-empty list of metric declarations")
        return
    seen: set[str] = set()
    for index, entry in enumerate(tracked):
        path = f"metrics.tracked[{index}]"
        payload = _require_mapping(issues, entry, path)
        if payload is None:
            continue
        name = _require_str(issues, payload.get("name"), f"{path}.name")
        if name is not None:
            if name in seen:
                _issue(issues, f"{path}.name", f"duplicate tracked metric '{name}'")
            seen.add(name)
        direction = _require_str(issues, payload.get("direction"), f"{path}.direction")
        if direction is not None and direction not in DIRECTIONS:
            _issue(issues, f"{path}.direction", f"must be one of {list(DIRECTIONS)}, got '{direction}'")
        if "optional" in payload and not isinstance(payload["optional"], bool):
            _issue(issues, f"{path}.optional", f"must be a boolean, got {type(payload['optional']).__name__}")
    if not any(not entry.get("optional", False) for entry in tracked if isinstance(entry, dict)):
        _issue(issues, "metrics.tracked", "at least one metric must be required (optional: false)")

    warmup_steps = _require_int(issues, metrics.get("warmup_steps"), "metrics.warmup_steps", minimum=0)
    final_window = _require_int(issues, metrics.get("final_window"), "metrics.final_window", minimum=1)
    min_points = _require_int(issues, metrics.get("min_points"), "metrics.min_points", minimum=1)
    if None not in (warmup_steps, final_window, min_points):
        if min_points < warmup_steps + final_window:
            _issue(
                issues,
                "metrics.min_points",
                f"must be >= warmup_steps + final_window ({warmup_steps} + {final_window} = "
                f"{warmup_steps + final_window}), got {min_points}; otherwise the final window cannot be scored",
            )


def _validate_convergence(issues: list[ValidationIssue], section: Any) -> None:
    convergence = _require_mapping(issues, section, "convergence")
    if convergence is None:
        return
    _require_number(issues, convergence.get("rtol"), "convergence.rtol", minimum=0.0)
    _require_number(issues, convergence.get("atol"), "convergence.atol", minimum=0.0)
    _require_number(issues, convergence.get("min_improvement"), "convergence.min_improvement")
    if "require_finite" in convergence and not isinstance(convergence["require_finite"], bool):
        _issue(
            issues,
            "convergence.require_finite",
            f"must be a boolean, got {type(convergence['require_finite']).__name__}",
        )


def _validate_baseline(issues: list[ValidationIssue], section: Any) -> None:
    baseline = _require_mapping(issues, section, "baseline")
    if baseline is None:
        return
    _require_str(issues, baseline.get("artifact_name"), "baseline.artifact_name")
    if "branch" in baseline:
        _require_str(issues, baseline["branch"], "baseline.branch")


def validate_recipe_dict(payload: Any) -> list[ValidationIssue]:
    """Return every schema/semantic issue found in a recipe document.

    An empty list means the recipe is structurally valid.  Validation is
    intentionally independent of the filesystem so that it can be unit tested
    without real checkpoints.
    """
    issues: list[ValidationIssue] = []
    document = _require_mapping(issues, payload, "<root>")
    if document is None:
        return issues

    for key in _TOP_LEVEL_KEYS:
        if key not in document:
            _issue(issues, key, "is required")

    version = _require_int(issues, document.get("schema_version"), "schema_version", minimum=1)
    if version is not None and version != SCHEMA_VERSION:
        _issue(issues, "schema_version", f"unsupported schema version {version}, expected {SCHEMA_VERSION}")

    layer = _require_str(issues, document.get("layer"), "layer")
    if layer is not None and layer != LAYER:
        _issue(issues, "layer", f"must be '{LAYER}', got '{layer}'")

    case_id = _require_str(issues, document.get("case_id"), "case_id")
    if case_id is not None and not CASE_ID_RE.match(case_id):
        _issue(issues, "case_id", f"must match {CASE_ID_RE.pattern!r}, got '{case_id}'")

    _require_str(issues, document.get("title"), "title")
    _require_str(issues, document.get("algorithm"), "algorithm")
    precision = _require_str(issues, document.get("precision"), "precision")
    if precision is not None and precision not in PRECISIONS:
        _issue(issues, "precision", f"must be one of {list(PRECISIONS)}, got '{precision}'")
    if "release_gate" in document and not isinstance(document["release_gate"], bool):
        _issue(issues, "release_gate", f"must be a boolean, got {type(document['release_gate']).__name__}")

    _validate_recipe_section(issues, document.get("recipe"))
    _validate_models(issues, document.get("models"))
    _validate_dataset(issues, document.get("dataset"))
    _validate_hardware(issues, document.get("hardware"))
    _validate_budget(issues, document.get("budget"))
    _validate_metrics(issues, document.get("metrics"))
    _validate_convergence(issues, document.get("convergence"))
    _validate_baseline(issues, document.get("baseline"))
    return issues


def load_recipe(path: Path) -> Recipe:
    """Parse and validate a single recipe file.

    Raises:
        RecipeError: the file is missing, is not YAML, or fails validation.
    """
    path = Path(path)
    if not path.is_file():
        raise RecipeError(f"recipe file does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:  # pragma: no cover - exercised via malformed input
        raise RecipeError(f"recipe file is not valid YAML: {path}: {error}") from error

    issues = validate_recipe_dict(payload)
    if issues:
        rendered = "\n".join(f"  - {issue}" for issue in issues)
        raise RecipeError(f"recipe {path} failed validation:\n{rendered}")

    assert isinstance(payload, dict)  # guaranteed by validate_recipe_dict
    return Recipe(data=payload, source=path, sha256=compute_recipe_sha256(payload))


def load_registry(recipes_dir: Path) -> dict[str, Recipe]:
    """Load every ``*.yaml`` recipe in a directory, keyed by ``case_id``.

    Raises:
        RecipeError: a file is invalid or two files declare the same ``case_id``.
    """
    recipes_dir = Path(recipes_dir)
    if not recipes_dir.is_dir():
        raise RecipeError(f"recipe directory does not exist: {recipes_dir}")

    registry: dict[str, Recipe] = {}
    origins: dict[str, Path] = {}
    problems: list[str] = []
    for path in sorted(recipes_dir.glob("*.yaml")):
        try:
            recipe = load_recipe(path)
        except RecipeError as error:
            problems.append(str(error))
            continue
        if recipe.case_id in registry:
            problems.append(f"duplicate case_id '{recipe.case_id}' in {path} and {origins[recipe.case_id]}")
            continue
        registry[recipe.case_id] = recipe
        origins[recipe.case_id] = path

    if problems:
        rendered = "\n".join(f"  - {problem}" for problem in problems)
        raise RecipeError(f"recipe registry {recipes_dir} failed validation:\n{rendered}")
    if not registry:
        raise RecipeError(f"recipe registry {recipes_dir} contains no recipes")
    if not any(recipe.release_gate for recipe in registry.values()):
        raise RecipeError(f"recipe registry {recipes_dir} has no release_gate recipe; nothing would gate a release")
    return registry


# --------------------------------------------------------------------------------------
# Preconditions
# --------------------------------------------------------------------------------------


@dataclass
class PreconditionCheck:
    """One concrete, human-readable precondition and whether it holds."""

    name: str
    ok: bool
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class PreconditionReport:
    """Aggregate precondition outcome for one recipe."""

    checks: list[PreconditionCheck] = field(default_factory=list)

    @property
    def unmet(self) -> list[PreconditionCheck]:
        return [check for check in self.checks if not check.ok]

    @property
    def status(self) -> str:
        return "ready" if not self.unmet else "skipped"

    @property
    def reason(self) -> str:
        if not self.unmet:
            return ""
        return "; ".join(f"{check.name}: {check.detail}" for check in self.unmet)

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "checks": [check.as_dict() for check in self.checks],
        }


def _resolve_model_path(entry: dict[str, Any], env: dict[str, str]) -> tuple[str | None, str]:
    """Resolve the on-disk path of a declared real checkpoint.

    Returns ``(path_or_None, detail)``.  ``env_path`` names an environment
    variable that overrides ``local_path``; when it is set but wrong, the
    override is reported rather than silently falling back, because a silent
    fallback could compare two different checkpoints.
    """
    env_path = entry.get("env_path")
    local_path = entry.get("local_path")
    if env_path and env.get(env_path):
        candidate = env[env_path]
        return (candidate if Path(candidate).is_dir() else None, f"{env_path}={candidate}")
    if local_path:
        return (str(local_path) if Path(local_path).is_dir() else None, f"local_path={local_path}")
    return None, "no local_path declared and no env override set"


def _parse_gpu_csv(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != len(GPU_QUERY_FIELDS):
            continue
        try:
            memory_total_mib = int(float(fields[2]))
            memory_free_mib = int(float(fields[3]))
        except ValueError:
            continue
        gpus.append(
            {
                "index": fields[0],
                "name": fields[1],
                "memory_total_mib": memory_total_mib,
                "memory_free_mib": memory_free_mib,
                "compute_cap": fields[4],
                "architecture": "sm" + fields[4].replace(".", "") if fields[4] else "",
            }
        )
    return gpus


def query_gpus(timeout_s: float = 30.0) -> list[dict[str, Any]]:
    """Query local GPUs via ``nvidia-smi``; return ``[]`` when unavailable."""
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode != 0:
        return []
    return _parse_gpu_csv(completed.stdout)


def check_preconditions(
    recipe: Recipe,
    *,
    repo_root: Path,
    env: dict[str, str] | None = None,
    gpus: list[dict[str, Any]] | None = None,
    disk_path: Path | None = None,
) -> PreconditionReport:
    """Check every declared prerequisite without starting any training work.

    ``env``, ``gpus`` and ``disk_path`` are injectable so the semantics can be
    unit tested on CPU without real hardware.
    """
    environment = dict(os.environ if env is None else env)
    repo_root = Path(repo_root)
    report = PreconditionReport()

    launcher = repo_root / recipe.launcher
    report.checks.append(
        PreconditionCheck(
            name="launcher",
            ok=launcher.is_file(),
            detail=f"{recipe.launcher}" if launcher.is_file() else f"missing launcher script {launcher}",
        )
    )

    for name, entry in recipe.data["models"].items():
        path, detail = _resolve_model_path(entry, environment)
        report.checks.append(
            PreconditionCheck(
                name=f"model:{name}",
                ok=path is not None,
                detail=f"{detail}" if path is None else f"{detail} (source={entry.get('source')})",
            )
        )

    dataset = recipe.data["dataset"]
    root = Path(environment.get(dataset["env_root"], "")) if dataset.get("env_root") else Path()
    for key in ("train", "val"):
        declared = Path(dataset[key])
        candidate = declared if declared.is_absolute() else root / declared
        report.checks.append(
            PreconditionCheck(
                name=f"dataset:{key}",
                ok=candidate.is_file(),
                detail=str(candidate) if candidate.is_file() else f"missing dataset shard {candidate}",
            )
        )

    hardware = recipe.data["hardware"]
    observed = query_gpus() if gpus is None else list(gpus)
    if not observed:
        report.checks.append(
            PreconditionCheck(
                name="gpu",
                ok=False,
                detail="no GPU visible (nvidia-smi unavailable or returned no devices)",
            )
        )
    else:
        required_mib = int(hardware["min_gpu_memory_gb"]) * 1024
        architectures = {str(item) for item in hardware["gpu_architectures"]}
        capable = [
            gpu
            for gpu in observed
            if gpu.get("memory_total_mib", 0) >= required_mib and gpu.get("architecture") in architectures
        ]
        idle = [gpu for gpu in capable if gpu.get("memory_free_mib", 0) >= required_mib]
        detail = (
            f"need {hardware['min_gpus']} x >= {hardware['min_gpu_memory_gb']}GiB on {sorted(architectures)}; "
            f"capable={len(capable)} idle={len(idle)} of {len(observed)} visible"
        )
        report.checks.append(PreconditionCheck(name="gpu", ok=len(idle) >= int(hardware["min_gpus"]), detail=detail))

    if hardware.get("min_free_disk_gb"):
        target = Path(disk_path) if disk_path is not None else repo_root
        try:
            free_gib = shutil.disk_usage(target).free / (1024**3)
        except OSError as error:
            report.checks.append(PreconditionCheck(name="disk", ok=False, detail=f"could not stat {target}: {error}"))
        else:
            needed = float(hardware["min_free_disk_gb"])
            report.checks.append(
                PreconditionCheck(
                    name="disk",
                    ok=free_gib >= needed,
                    detail=f"{free_gib:.1f} GiB free at {target}, need {needed:.1f} GiB",
                )
            )

    return report
