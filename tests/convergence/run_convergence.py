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
"""Run L4 convergence recipes and emit per-case results plus a release report.

Usage::

    # Static check only: validate recipes and declared preconditions.
    python3 -m tests.convergence.run_convergence --registry tests/convergence/recipes \
        --output-root outputs/l4 --mode preflight

    # Create reviewed baselines (release owners only).
    python3 -m tests.convergence.run_convergence --registry tests/convergence/recipes \
        --output-root outputs/l4 --mode baseline

    # Verify against the downloaded baselines and write the release report.
    python3 -m tests.convergence.run_convergence --registry tests/convergence/recipes \
        --output-root outputs/l4 --mode verify

Design rules enforced here:

* a recipe never starts training before every declared precondition is satisfied;
* a launcher is executed with an argument list (never a shell string) and its own
  process group, so a timeout kills only the processes this runner started;
* every terminal state is written to ``result.json`` even when the run fails or
  times out, and a missing baseline is ``invalid`` rather than "no regression";
* the process exit code reflects the mode's own success criterion, while
  ``release_readiness.json`` always carries the honest release verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compare import build_contract, compare_curves
from .curves import CurveError, CurveSummary, build_curves, completed_steps, load_step_records, summarize_curve
from .recipe_registry import (
    Recipe,
    RecipeError,
    check_preconditions,
    load_recipe,
    load_registry,
    query_gpus,
)
from .report import (
    build_release_report,
    case_result_from_payload,
    exit_code_for_status,
    load_case_results,
    write_release_report,
)

RESULT_SCHEMA_VERSION = 1
EVIDENCE_STATIC = "static"
EVIDENCE_RUN = "run"
EVIDENCE_COMPARED = "compared"

#: Environment variable names that are safe to persist in the run manifest.
#: Anything matching a secret-ish pattern is dropped even if it matches a prefix.
ENV_PREFIX_ALLOWLIST = ("L4_", "CUDA_", "HF_", "OMP_", "MKL_", "TORCH", "NCCL_", "RAY_", "VLLM_")
ENV_SECRET_MARKERS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")

EXIT_OK = 0
EXIT_GATE_NOT_SATISFIED = 1
EXIT_USAGE = 2


class RunnerError(Exception):
    """Raised for runner configuration problems (as opposed to run failures)."""


# --------------------------------------------------------------------------------------
# Placeholders and environment
# --------------------------------------------------------------------------------------


def resolved_paths(recipe: Recipe, env: dict[str, str], output_root: Path | None = None) -> dict[str, str]:
    """Resolve every placeholder a recipe may use.

    ``{model:<name>}``, ``{dataset:<key>}``, plus ``{output_root}`` and
    ``{case_dir}`` so a recipe can keep checkpoints and rollout dumps inside the
    L4 artifact tree instead of polluting the repository.
    """
    resolved: dict[str, str] = {}
    for name, entry in recipe.data["models"].items():
        env_path = entry.get("env_path")
        if env_path and env.get(env_path):
            resolved[f"model:{name}"] = env[env_path]
        elif entry.get("local_path"):
            resolved[f"model:{name}"] = str(entry["local_path"])
        else:
            resolved[f"model:{name}"] = str(entry.get("source", ""))

    dataset = recipe.data["dataset"]
    root = env.get(dataset["env_root"], "") if dataset.get("env_root") else ""
    for key in ("train", "val"):
        declared = str(dataset[key])
        if os.path.isabs(declared) or not root:
            resolved[f"dataset:{key}"] = declared
        else:
            resolved[f"dataset:{key}"] = str(Path(root) / declared)

    if output_root is not None:
        resolved["output_root"] = str(output_root)
        resolved["case_dir"] = str(current_dir(Path(output_root), recipe.case_id))
    return resolved


#: Placeholders are `{kind:name}` for models/datasets and bare `{output_root}` /
#: `{case_dir}` for run state.  They may appear anywhere inside a value, so a
#: recipe can write `{case_dir}/checkpoints` and have the suffix preserved.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][\w.:]*)\}")


def render_placeholder(value: str, resolved: dict[str, str]) -> str:
    """Substitute every known placeholder inside ``value``.

    A value may embed placeholders in a larger string (paths, Hydra lists).  An
    unrecognised ``{...}`` token is a hard error rather than being passed through,
    because a literal brace reaching the launcher would silently write run state
    to a directory literally named ``{case_dir}``.

    Raises:
        RunnerError: a placeholder is unknown.
    """

    def substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in resolved:
            raise RunnerError(f"unknown placeholder {{{key}}} in {value!r}; known keys are {sorted(resolved)}")
        return resolved[key]

    return _PLACEHOLDER_RE.sub(substitute, value)


def render_overrides(recipe: Recipe, resolved: dict[str, str]) -> list[str]:
    """Resolve placeholders inside every declared Hydra override."""
    rendered: list[str] = []
    for override in recipe.overrides:
        if "=" not in override:
            raise RunnerError(f"override {override!r} is not a Hydra key=value assignment")
        key, _, value = override.partition("=")
        rendered.append(f"{key}={render_placeholder(value, resolved)}")
    return rendered


def safelisted_env(env: dict[str, str]) -> dict[str, str]:
    """Return the subset of ``env`` that is safe to write into the run manifest."""
    safe: dict[str, str] = {}
    for name, value in sorted(env.items()):
        if not name.startswith(ENV_PREFIX_ALLOWLIST):
            continue
        if any(marker in name.upper() for marker in ENV_SECRET_MARKERS):
            continue
        safe[name] = value
    return safe


def observed_hardware(gpus: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the visible accelerators for the comparability contract."""
    if not gpus:
        return {}
    return {
        "gpu_architecture": sorted({str(gpu.get("architecture", "")) for gpu in gpus if gpu.get("architecture")}),
        "gpu_count": len(gpus),
        "gpu_names": sorted({str(gpu.get("name", "")) for gpu in gpus if gpu.get("name")}),
    }


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def run_launcher(
    *,
    repo_root: Path,
    recipe: Recipe,
    overrides: list[str],
    log_path: Path,
    env: dict[str, str],
    timeout_s: float,
) -> tuple[int | None, bool, float]:
    """Execute a recipe launcher in its own process group.

    Returns ``(exit_code, timed_out, duration_s)``.  On timeout only the process
    group started here is signalled; a shared machine's other jobs are untouched.
    ``exit_code`` is ``None`` when the process was killed by the timeout.
    """
    launcher = repo_root / recipe.launcher
    command = ["bash", str(launcher), *overrides]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"# command: {' '.join(shlex.quote(part) for part in command)}\n")
        log_handle.write(f"# cwd: {repo_root}\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_process_group(process)
            exit_code = None
        else:
            if exit_code < 0:
                # Killed by a signal: make that explicit instead of reporting a
                # bare negative code as if training had completed.
                timed_out = False
        duration = time.monotonic() - started
    return exit_code, timed_out, duration


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Terminate then kill only the process group created for this run."""
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=60)
            return
        except subprocess.TimeoutExpired:
            continue


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def baseline_path(output_root: Path, case_id: str) -> Path:
    return output_root / "baseline" / case_id / "baseline.json"


def current_dir(output_root: Path, case_id: str) -> Path:
    return output_root / "current" / case_id


def load_baseline(output_root: Path, case_id: str) -> dict[str, Any] | None:
    """Load a reviewed baseline, or ``None`` when it does not exist yet."""
    path = baseline_path(output_root, case_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RunnerError(f"baseline at {path} is not valid JSON: {error}") from error


def _base_result(recipe: Recipe, *, commit_sha: str | None, release_gate: bool) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "layer": "L4",
        "case_id": recipe.case_id,
        "title": recipe.title,
        "algorithm": recipe.algorithm,
        "precision": recipe.precision,
        "release_gate": release_gate,
        "status": "not_run",
        "evidence_level": EVIDENCE_STATIC,
        "converged": False,
        "commit_sha": commit_sha,
        "recipe_sha256": recipe.sha256,
        "recipe_source": str(recipe.source) if recipe.source else None,
        "started_at": datetime.now(UTC).isoformat(),
        "duration_s": None,
        "run": {
            "exit_code": None,
            "timed_out": False,
            "expected_steps": int(recipe.data["budget"]["total_training_steps"]),
            "completed_steps": None,
            "log_path": None,
            "command": None,
        },
        "preconditions": None,
        "curves": {},
        "comparison": None,
        "failure_reason": None,
    }


def _run_case(
    *,
    recipe: Recipe,
    repo_root: Path,
    output_root: Path,
    mode: str,
    commit_sha: str | None,
    env: dict[str, str],
    gpus: list[dict[str, Any]],
    timeout_override_minutes: float | None,
    release_gate: bool,
) -> dict[str, Any]:
    """Execute one case and return its result payload (never raises for run failures)."""
    result = _base_result(recipe, commit_sha=commit_sha, release_gate=release_gate)
    contract = build_contract(
        algorithm=recipe.algorithm,
        precision=recipe.precision,
        recipe_data=recipe.data,
        observed_hardware=observed_hardware(gpus),
        commit_sha=commit_sha,
    )
    result["contract"] = contract

    report = check_preconditions(recipe, repo_root=repo_root, env=env, gpus=gpus)
    result["preconditions"] = report.as_dict()

    if report.status != "ready":
        result["status"] = "skipped"
        result["evidence_level"] = EVIDENCE_STATIC
        result["failure_reason"] = f"preconditions unmet: {report.reason}"
        return result

    if mode == "preflight":
        result["status"] = "not_run"
        result["evidence_level"] = EVIDENCE_STATIC
        result["failure_reason"] = "preflight mode: preconditions are satisfied but no run was requested"
        return result

    resolved = resolved_paths(recipe, env, output_root)
    overrides = render_overrides(recipe, resolved)
    log_path = current_dir(output_root, recipe.case_id) / "train.log"
    result["run"]["command"] = ["bash", recipe.launcher, *overrides]
    result["run"]["log_path"] = str(log_path)
    result["resolved_paths"] = resolved
    result["env"] = safelisted_env(env)

    timeout_minutes = timeout_override_minutes or float(recipe.data["budget"]["timeout_minutes"])
    exit_code, timed_out, duration = run_launcher(
        repo_root=repo_root,
        recipe=recipe,
        overrides=overrides,
        log_path=log_path,
        env=env,
        timeout_s=timeout_minutes * 60.0,
    )
    result["duration_s"] = duration
    result["run"]["exit_code"] = exit_code
    result["run"]["timed_out"] = timed_out

    metrics_jsonl = _metrics_jsonl_path(recipe, output_root, env)
    try:
        records = load_step_records(metrics_jsonl=metrics_jsonl, log_file=log_path)
    except CurveError as error:
        # No usable curve.  The exit status still decides what happened: a crash
        # that produced nothing is a failure, not merely unparseable output.
        result["evidence_level"] = EVIDENCE_RUN
        if timed_out:
            result["status"] = "timeout"
            result["failure_reason"] = (
                f"run exceeded the {timeout_minutes:.0f} minute budget and produced no step records: {error}"
            )
        elif exit_code not in (0, None):
            result["status"] = "failed"
            result["failure_reason"] = f"launcher exited with code {exit_code} before producing step records: {error}"
        else:
            result["status"] = "invalid"
            result["failure_reason"] = str(error)
        return result

    result["evidence_level"] = EVIDENCE_RUN
    result["run"]["completed_steps"] = completed_steps(records)
    summaries = _summarize(recipe, records)
    result["curves"] = {name: summary.as_dict() for name, summary in summaries.items()}

    missing_required = [name for name in recipe.required_metric_names if summaries[name].num_points == 0]
    if missing_required:
        result["status"] = "invalid"
        result["failure_reason"] = f"required metric(s) never appeared in the run output: {missing_required}"
        return result

    if timed_out:
        result["status"] = "timeout"
        result["failure_reason"] = (
            f"run exceeded the {timeout_minutes:.0f} minute budget after {result['run']['completed_steps']} step(s)"
        )
        return result

    if exit_code not in (0, None):
        # A non-zero launcher exit is always reported as a failure.  Reaching the
        # step budget does not excuse it, otherwise a real training error could be
        # laundered into a green convergence run.
        result["status"] = "failed"
        result["failure_reason"] = f"launcher exited with code {exit_code}"
        return result

    if mode == "baseline":
        _write_baseline(recipe, output_root, contract, summaries, log_path)
        result["status"] = "baseline_created"
        result["failure_reason"] = (
            "baseline mode: this run created or refreshed the reviewed baseline and does not by itself "
            "verify convergence"
        )
        return result

    baseline = load_baseline(output_root, recipe.case_id)
    baseline_summaries = _baseline_summaries(baseline)
    comparison = compare_curves(
        current_summaries=summaries,
        baseline_summaries=baseline_summaries,
        metric_specs=recipe.metrics,
        convergence=recipe.data["convergence"],
        min_points=int(recipe.data["metrics"]["min_points"]),
        baseline_contract=(baseline or {}).get("contract"),
        current_contract=contract,
        baseline_present=baseline is not None,
    )
    result["comparison"] = comparison.as_dict()
    result["status"] = comparison.status
    result["evidence_level"] = EVIDENCE_COMPARED if baseline is not None else EVIDENCE_RUN
    result["converged"] = comparison.status == "passed"
    if result["status"] == "invalid":
        result["failure_reason"] = (
            f"no reviewed baseline found at {baseline_path(output_root, recipe.case_id)}; "
            "run mode=baseline first on the release runner"
        )
    elif result["status"] == "incomparable":
        rendered = ", ".join(f"{item['field']}" for item in comparison.mismatches)
        result["failure_reason"] = f"run contract differs from the baseline in: {rendered}"
    elif result["status"] == "failed":
        result["failure_reason"] = f"convergence regressed on: {comparison.failed_metrics}"
    return result


def _summarize(recipe: Recipe, records: list[dict[str, Any]]) -> dict[str, CurveSummary]:
    specs = recipe.metrics
    warmup = int(recipe.data["metrics"]["warmup_steps"])
    window = int(recipe.data["metrics"]["final_window"])
    curves = build_curves(records, [str(spec["name"]) for spec in specs])
    return {
        str(spec["name"]): summarize_curve(
            curves[str(spec["name"])],
            metric=str(spec["name"]),
            direction=str(spec["direction"]),
            warmup_steps=warmup,
            final_window=window,
        )
        for spec in specs
    }


def _baseline_summaries(baseline: dict[str, Any] | None) -> dict[str, CurveSummary]:
    """Rebuild ``CurveSummary`` objects from a stored baseline payload."""
    if not baseline:
        return {}
    summaries: dict[str, CurveSummary] = {}
    for name, payload in (baseline.get("summary") or {}).items():
        summaries[name] = CurveSummary(
            metric=name,
            direction=str(payload.get("direction", "higher")),
            num_points=int(payload.get("num_points", 0)),
            num_finite_points=int(payload.get("num_finite_points", 0)),
            warmup_steps=int(payload.get("warmup_steps", 0)),
            final_window=int(payload.get("final_window", 1)),
            first_window_mean=payload.get("first_window_mean"),
            final_window_mean=payload.get("final_window_mean"),
            improvement=payload.get("improvement"),
            min_value=payload.get("min_value"),
            max_value=payload.get("max_value"),
            non_finite_steps=list(payload.get("non_finite_steps", [])),
            steps=list(payload.get("steps", [])),
        )
    return summaries


def _write_baseline(
    recipe: Recipe,
    output_root: Path,
    contract: dict[str, Any],
    summaries: dict[str, Any],
    log_path: Path,
) -> Path:
    path = baseline_path(output_root, recipe.case_id)
    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "layer": "L4",
        "case_id": recipe.case_id,
        "title": recipe.title,
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe.sha256,
        "contract": contract,
        "summary": {name: summary.as_dict() for name, summary in summaries.items()},
        "source_log": str(log_path),
        "note": (
            "This baseline is only usable after human review of the originating run. "
            "A run never becomes its own baseline."
        ),
    }
    _write_json(path, payload)
    return path


def _metrics_jsonl_path(recipe: Recipe, output_root: Path, env: dict[str, str]) -> Path | None:
    """Resolve the optional structured metrics side channel declared by a recipe."""
    declared = recipe.data["recipe"].get("metrics_jsonl")
    if not declared:
        return None
    candidate = Path(render_placeholder(str(declared), resolved_paths(recipe, env, output_root)))
    return candidate if candidate.is_file() else None


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def run_all(
    *,
    recipes: dict[str, Recipe],
    repo_root: Path,
    output_root: Path,
    mode: str,
    case_ids: list[str] | None,
    env: dict[str, str],
    commit_sha: str | None,
    timeout_override_minutes: float | None,
    gpus: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the selected cases and return ``(results, release_report)``."""
    observed = query_gpus() if gpus is None else gpus
    selected = list(recipes.values()) if not case_ids else [recipes[case_id] for case_id in case_ids]
    results: list[dict[str, Any]] = []
    for recipe in selected:
        result = _run_case(
            recipe=recipe,
            repo_root=repo_root,
            output_root=output_root,
            mode=mode,
            commit_sha=commit_sha,
            env=env,
            gpus=observed,
            timeout_override_minutes=timeout_override_minutes,
            release_gate=recipe.release_gate,
        )
        _write_json(current_dir(output_root, recipe.case_id) / "result.json", result)
        _print_case(result)
        results.append(result)

    # Build the report from the cases that actually ran in this invocation.  Reading
    # `current/` back from disk would silently fold stale results from an earlier,
    # differently-filtered run into this run's release verdict.
    case_results = [
        case_result_from_payload(result, result_path=current_dir(output_root, recipe.case_id) / "result.json")
        for recipe, result in zip(selected, results, strict=True)
    ]
    report = build_release_report(case_results, commit_sha=commit_sha)
    write_release_report(report, output_root)
    return results, report


def _print_case(result: dict[str, Any]) -> None:
    print("=" * 88)
    print(f"[L4] {result['case_id']}: {result['status']} (evidence={result['evidence_level']})")
    if result.get("failure_reason"):
        print(f"[L4] reason: {result['failure_reason']}")
    run = result.get("run") or {}
    if run.get("completed_steps") is not None:
        print(f"[L4] steps: {run.get('completed_steps')}/{run.get('expected_steps')}")
    for name, curve in (result.get("curves") or {}).items():
        print(
            f"[L4] curve {name}: points={curve['num_points']} finite={curve['num_finite_points']} "
            f"final_window_mean={curve['final_window_mean']} improvement={curve['improvement']}"
        )
    comparison = result.get("comparison") or {}
    for name, verdict in (comparison.get("metrics") or {}).items():
        print(
            f"[L4] score {name}: passed={verdict['passed']} "
            f"baseline={verdict['baseline_final_window_mean']} current={verdict['current_final_window_mean']}"
        )
        for reason in verdict.get("reasons", []):
            print(f"[L4]   - {reason}")
    print("=" * 88)


def mode_exit_code(mode: str, results: list[dict[str, Any]], report: dict[str, Any]) -> int:
    """Return the exit code that reflects the mode's own success criterion."""
    if mode == "preflight":
        return EXIT_OK if all(item["status"] != "skipped" for item in results) else EXIT_GATE_NOT_SATISFIED
    if mode == "baseline":
        created = [item for item in results if item.get("release_gate", True) and item["status"] == "baseline_created"]
        gated = [item for item in results if item.get("release_gate", True)]
        return EXIT_OK if gated and len(created) == len(gated) else EXIT_GATE_NOT_SATISFIED
    return exit_code_for_status(report["overall_status"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L4 convergence recipes and write a release report.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--registry", type=Path, help="directory of *.yaml L4 recipes")
    source.add_argument("--recipe", type=Path, help="a single L4 recipe file")
    parser.add_argument("--case", action="append", dest="cases", default=None, help="case_id to run (repeatable)")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/l4_convergence"))
    parser.add_argument(
        "--repo-root", type=Path, default=None, help="repository root (default: two levels above this file)"
    )
    parser.add_argument("--mode", choices=("preflight", "baseline", "verify", "report"), default="verify")
    parser.add_argument("--timeout-minutes", type=float, default=None, help="override the recipe timeout")
    parser.add_argument("--commit-sha", default=None, help="commit under test (default: git rev-parse HEAD)")
    return parser.parse_args(argv)


def _load_recipes(args: argparse.Namespace) -> dict[str, Recipe]:
    if args.registry:
        return load_registry(args.registry)
    single = load_recipe(args.recipe)
    return {single.case_id: single}


def _detect_commit_sha(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root or Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root).expanduser().resolve()
    env = dict(os.environ)
    commit_sha = args.commit_sha or _detect_commit_sha(repo_root)

    try:
        recipes = _load_recipes(args)
    except RecipeError as error:
        print(f"[L4] ERROR: {error}", file=sys.stderr)
        return EXIT_USAGE

    if args.cases:
        unknown = [case for case in args.cases if case not in recipes]
        if unknown:
            print(f"[L4] ERROR: unknown case(s) {unknown}; registry has {sorted(recipes)}", file=sys.stderr)
            return EXIT_USAGE

    if args.mode == "report":
        try:
            case_results = load_case_results(output_root / "current")
        except Exception as error:  # noqa: BLE001 - surfaced as a CLI error
            print(f"[L4] ERROR: {error}", file=sys.stderr)
            return EXIT_USAGE
        report = build_release_report(case_results, commit_sha=commit_sha)
        write_release_report(report, output_root)
        print(f"[L4] release readiness: {report['overall_status']}")
        return exit_code_for_status(report["overall_status"])

    try:
        results, report = run_all(
            recipes=recipes,
            repo_root=repo_root,
            output_root=output_root,
            mode=args.mode,
            case_ids=args.cases,
            env=env,
            commit_sha=commit_sha,
            timeout_override_minutes=args.timeout_minutes,
        )
    except RunnerError as error:
        print(f"[L4] ERROR: {error}", file=sys.stderr)
        return EXIT_USAGE

    print(f"[L4] release readiness: {report['overall_status']} ({report['rationale'][0]})")
    return mode_exit_code(args.mode, results, report)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
