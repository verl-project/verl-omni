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
"""End-to-end L4 runner semantics driven by a synthetic launcher.

The launcher below is a test double: it only prints synthetic ``verl``-style step
records.  These tests therefore prove the runner's state machine — preflight,
baseline creation, comparison, timeout, non-zero exit, and fail-closed reporting —
and prove nothing about real training convergence.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml

from tests.convergence.recipe_registry import load_registry
from tests.convergence.report import exit_code_for_status
from tests.convergence.run_convergence import mode_exit_code, run_all

FAKE_LAUNCHER = """#!/usr/bin/env bash
exec python3 - "$@" <<'PY'
import os
import sys

base = float(os.environ.get("FAKE_BASE", "0.10"))
delta = float(os.environ.get("FAKE_DELTA", "0.01"))
steps = int(os.environ.get("FAKE_STEPS", "8"))
sleep_s = float(os.environ.get("FAKE_SLEEP", "0"))

for step in range(1, steps + 1):
    reward = base + (step - 1) * delta
    payload = {"training/global_step": step, "critic/rewards/mean": reward, "actor/loss": 0.5}
    print(f"2026-01-01 00:00:00 INFO step:{step} - {payload}", flush=True)

if sleep_s:
    import time

    time.sleep(sleep_s)

sys.exit(int(os.environ.get("FAKE_EXIT", "0")))
PY
"""


def recipe_document(timeout_minutes: float = 1.0, steps: int = 8) -> dict:
    return {
        "schema_version": 1,
        "layer": "L4",
        "case_id": "synthetic_runner_case",
        "title": "synthetic runner case",
        "algorithm": "flow_grpo",
        "precision": "bf16",
        "recipe": {
            "launcher": "launcher.sh",
            "overrides": [
                "data.train_files={dataset:train}",
                "data.val_files={dataset:val}",
                "actor_rollout_ref.model.path={model:policy}",
            ],
        },
        "models": {
            "policy": {
                "kind": "real",
                "source": "stabilityai/stable-diffusion-3.5-medium",
                "revision": "main",
                "env_path": "L4_TEST_MODEL",
            }
        },
        "dataset": {
            "kind": "real",
            "train": "train.parquet",
            "val": "test.parquet",
            "env_root": "L4_TEST_DATA",
        },
        "hardware": {"min_gpus": 1, "min_gpu_memory_gb": 1, "gpu_architectures": ["sm89"]},
        "budget": {"timeout_minutes": timeout_minutes, "total_training_steps": steps},
        "metrics": {
            "warmup_steps": 1,
            "final_window": 2,
            "min_points": 4,
            "tracked": [
                {"name": "critic/rewards/mean", "direction": "higher"},
                {"name": "actor/loss", "direction": "lower", "optional": True},
            ],
        },
        "convergence": {"atol": 0.01, "rtol": 0.05, "min_improvement": -1.0, "require_finite": True},
        "baseline": {"artifact_name": "l4-convergence-baseline", "branch": "main"},
    }


def fake_gpu() -> dict:
    return {
        "index": "0",
        "name": "NVIDIA L20",
        "memory_total_mib": 46068,
        "memory_free_mib": 46068,
        "compute_cap": "8.9",
        "architecture": "sm89",
    }


@pytest.fixture
def workspace(tmp_path: Path) -> dict:
    """Create a synthetic repository root with a launcher, assets, and registry."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "launcher.sh").write_text(FAKE_LAUNCHER, encoding="utf-8")
    model_dir = tmp_path / "models" / "sd35"
    model_dir.mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "train.parquet").write_bytes(b"synthetic")
    (data_dir / "test.parquet").write_bytes(b"synthetic")

    registry_dir = repo_root / "recipes"
    registry_dir.mkdir()
    (registry_dir / "synthetic.yaml").write_text(yaml.safe_dump(recipe_document()), encoding="utf-8")

    env = {
        **os.environ,
        "L4_TEST_MODEL": str(model_dir),
        "L4_TEST_DATA": str(data_dir),
        "WANDB_MODE": "disabled",
    }
    return {
        "repo_root": repo_root,
        "registry": registry_dir,
        "output_root": tmp_path / "out",
        "env": env,
        "model_dir": model_dir,
    }


def run(workspace: dict, mode: str, env_extra: dict | None = None, timeout_minutes: float | None = None):
    env = dict(workspace["env"])
    env.update(env_extra or {})
    recipes = load_registry(workspace["registry"])
    return run_all(
        recipes=recipes,
        repo_root=workspace["repo_root"],
        output_root=workspace["output_root"],
        mode=mode,
        case_ids=None,
        env=env,
        commit_sha="deadbeef",
        timeout_override_minutes=timeout_minutes,
        gpus=[fake_gpu()],
    )


def test_preflight_reports_ready_without_running(workspace: dict) -> None:
    results, report = run(workspace, "preflight")
    assert results[0]["status"] == "not_run"
    assert results[0]["evidence_level"] == "static"
    assert report["overall_status"] == "incomplete"
    assert mode_exit_code("preflight", results, report) == 0
    assert (workspace["output_root"] / "current" / "synthetic_runner_case" / "result.json").is_file()


def test_baseline_then_verify_passes(workspace: dict) -> None:
    results, report = run(workspace, "baseline")
    assert results[0]["status"] == "baseline_created"
    assert mode_exit_code("baseline", results, report) == 0
    assert report["overall_status"] == "incomplete"
    baseline_path = workspace["output_root"] / "baseline" / "synthetic_runner_case" / "baseline.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["contract"]["algorithm"] == "flow_grpo"
    assert baseline["summary"]["critic/rewards/mean"]["num_points"] == 8
    assert "never becomes its own baseline" in baseline["note"]

    results, report = run(workspace, "verify")
    assert results[0]["status"] == "passed"
    assert results[0]["converged"] is True
    assert results[0]["evidence_level"] == "compared"
    assert report["overall_status"] == "ready"
    assert mode_exit_code("verify", results, report) == 0
    assert results[0]["comparison"]["metrics"]["critic/rewards/mean"]["passed"] is True


def test_verify_without_baseline_is_invalid(workspace: dict) -> None:
    results, report = run(workspace, "verify")
    assert results[0]["status"] == "invalid"
    assert results[0]["converged"] is False
    assert "no reviewed baseline" in results[0]["failure_reason"]
    assert report["overall_status"] == "incomplete"
    assert mode_exit_code("verify", results, report) == 1


def test_regression_blocks_the_release(workspace: dict) -> None:
    run(workspace, "baseline")
    results, report = run(workspace, "verify", {"FAKE_BASE": "0.01"})
    assert results[0]["status"] == "failed"
    assert report["overall_status"] == "blocked"
    assert results[0]["comparison"]["failed_metrics"] == ["critic/rewards/mean"]
    assert mode_exit_code("verify", results, report) == 1


def test_non_zero_launcher_exit_is_never_a_pass(workspace: dict) -> None:
    run(workspace, "baseline")
    results, _ = run(workspace, "verify", {"FAKE_EXIT": "3"})
    assert results[0]["status"] == "failed"
    assert "exited with code 3" in results[0]["failure_reason"]
    # The curve was still captured as evidence, but it cannot certify a release.
    assert results[0]["curves"]["critic/rewards/mean"]["num_points"] == 8


def test_crash_without_step_records_is_a_failure_not_invalid(workspace: dict) -> None:
    """A launcher that dies before logging anything must be `failed`, not `invalid`."""
    workspace["repo_root"].joinpath("launcher.sh").write_text(
        '#!/usr/bin/env bash\necho "boom: config error" >&2\nexit 4\n', encoding="utf-8"
    )
    results, report = run(workspace, "verify")
    assert results[0]["status"] == "failed"
    assert "exited with code 4" in results[0]["failure_reason"]
    assert results[0]["run"]["exit_code"] == 4
    assert report["overall_status"] == "blocked"


def test_timeout_is_reported_and_only_kills_its_own_process(workspace: dict) -> None:
    # The recipe keeps a realistic 1-minute budget; the runner's explicit
    # override is what makes this test fast.
    started = time.monotonic()
    results, report = run(workspace, "verify", {"FAKE_SLEEP": "120"}, timeout_minutes=0.02)
    elapsed = time.monotonic() - started
    assert results[0]["status"] == "timeout"
    assert results[0]["run"]["timed_out"] is True
    assert "exceeded the 0 minute budget" in results[0]["failure_reason"]
    assert report["overall_status"] == "incomplete"
    # 0.02 minutes is ~1.2s; anything close to the 120s sleep means the kill failed.
    assert elapsed < 60, f"timeout handling took {elapsed:.1f}s"


def test_missing_assets_skip_with_a_precise_reason(workspace: dict, tmp_path: Path) -> None:
    moved = tmp_path / "models-moved"
    workspace["model_dir"].rename(moved)
    results, report = run(workspace, "verify")
    assert results[0]["status"] == "skipped"
    assert results[0]["evidence_level"] == "static"
    assert "model:policy" in results[0]["failure_reason"]
    assert results[0]["run"]["exit_code"] is None, "no training may start when preconditions fail"
    assert report["overall_status"] == "incomplete"


def test_contract_mismatch_against_an_old_baseline_is_incomparable(workspace: dict) -> None:
    run(workspace, "baseline")
    # A different algorithm changes the contract, so the stored baseline must be
    # rejected instead of being quietly reused.
    document = recipe_document()
    document["algorithm"] = "diffusion_nft"
    workspace["registry"].joinpath("synthetic.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
    results, report = run(workspace, "verify")
    assert results[0]["status"] == "incomparable"
    assert "algorithm" in results[0]["failure_reason"]
    assert report["overall_status"] == "incomplete"


def test_result_payload_records_provenance_and_command(workspace: dict) -> None:
    results, _ = run(workspace, "baseline")
    result = results[0]
    assert result["commit_sha"] == "deadbeef"
    assert result["recipe_sha256"]
    assert result["run"]["command"][0] == "bash"
    assert result["run"]["command"][1].endswith("launcher.sh")
    assert result["env"]["L4_TEST_MODEL"]
    assert json.dumps(result)  # every result must be a persistable artifact


def test_selected_case_run_does_not_report_stale_cases(workspace: dict) -> None:
    """A filtered run must not fold an earlier run's results into its verdict."""
    run(workspace, "baseline")
    # A second recipe that never runs in the filtered invocation.
    other = recipe_document()
    other["case_id"] = "synthetic_other_case"
    workspace["registry"].joinpath("other.yaml").write_text(yaml.safe_dump(other), encoding="utf-8")

    env = dict(workspace["env"])
    recipes = load_registry(workspace["registry"])
    results, report = run_all(
        recipes=recipes,
        repo_root=workspace["repo_root"],
        output_root=workspace["output_root"],
        mode="baseline",
        case_ids=["synthetic_runner_case"],
        env=env,
        commit_sha="deadbeef",
        timeout_override_minutes=None,
        gpus=[fake_gpu()],
    )
    assert [item["case_id"] for item in results] == ["synthetic_runner_case"]
    assert report["gated_cases"] == ["synthetic_runner_case"]
    assert "synthetic_other_case" not in json.dumps(report)


def test_exit_code_helper_matches_release_verdict() -> None:
    assert exit_code_for_status("ready") == 0
    assert exit_code_for_status("blocked") == 1
    assert exit_code_for_status("incomplete") == 1
