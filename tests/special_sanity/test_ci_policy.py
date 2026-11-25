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

"""Exercise the CI policy against pull request diffs in temporary Git repositories."""

import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from shlex import split

import pytest
import yaml

CHECKER = Path(__file__).with_name("check_ci_policy.py").resolve()
WORKFLOWS = CHECKER.parents[2] / ".github" / "workflows"
LINT_WORKFLOW = "ci-check-policy.yml"
SANITY_CHECK = "tests/special_sanity/check_existing.py"
WORKFLOW = ".github/workflows/sanity.yml"
RUFF_CONFIG = '[project]\nname = "example"\n[tool.ruff.lint]\nselect = ["F"]\n'
SOURCE_PATH = "verl_omni/example.py"


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def commit(repo: Path, changes: dict[str, str | None]) -> str:
    for name, content in changes.items():
        path = repo / name
        if content is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    git(repo, "add", "--all")
    git(repo, "-c", "user.name=CI Test", "-c", "user.email=ci@example.invalid", "commit", "-qm", "fixture")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    commit(tmp_path, {SANITY_CHECK: "original\n", WORKFLOW: "original\n", "pyproject.toml": RUFF_CONFIG})
    return tmp_path


@pytest.mark.parametrize(
    ("changes", "blocked_path"),
    [
        ({SOURCE_PATH: "feature\n"}, None),
        ({SANITY_CHECK: "updated\n"}, None),
        ({SOURCE_PATH: "feature\n", "tests/special_sanity/check_new.py": "new\n"}, None),
        ({SOURCE_PATH: "feature\n", SANITY_CHECK: "weakened\n"}, SANITY_CHECK),
        ({SOURCE_PATH: "feature\n", SANITY_CHECK: None}, SANITY_CHECK),
        (
            {SOURCE_PATH: "feature\n", SANITY_CHECK: None, "tests/special_sanity/check_moved.py": "original\n"},
            SANITY_CHECK,
        ),
        ({SOURCE_PATH: "feature\n", WORKFLOW: "changed\n"}, WORKFLOW),
        ({SOURCE_PATH: "feature\n", ".pre-commit-config.yaml": "repos: []\n"}, ".pre-commit-config.yaml"),
        ({SOURCE_PATH: "feature\n", "ruff.toml": "lint.ignore = ['F401']\n"}, "ruff.toml"),
        ({SOURCE_PATH: "feature\n", ".ruff.toml": "lint.ignore = ['F401']\n"}, ".ruff.toml"),
        ({SOURCE_PATH: "feature\n", "mypy.ini": "[mypy]\nignore_errors = True\n"}, "mypy.ini"),
        ({"tests/workers/test_example_on_cpu.py": "def test_ok(): pass\n", WORKFLOW: "updated\n"}, None),
        ({SANITY_CHECK: "updated\n", "tests/special_sanity/test_existing.py": "def test_ok(): pass\n"}, None),
        ({SOURCE_PATH: "feature\n", "pyproject.toml": RUFF_CONFIG.replace('"F"', '"E"')}, "pyproject.toml"),
        ({SOURCE_PATH: "feature\n", "pyproject.toml": RUFF_CONFIG + "[tool.mypy]\nignore_errors = true\n"}, "mypy"),
        ({SOURCE_PATH: "feature\n", "pyproject.toml": RUFF_CONFIG.replace('"example"', '"other"')}, None),
    ],
)
def test_check_separates_implementation_from_check_maintenance(
    repo: Path, changes: dict[str, str | None], blocked_path: str | None
) -> None:
    base = git(repo, "rev-parse", "HEAD")
    head = commit(repo, changes)
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=repo, capture_output=True, text=True)
    assert result.returncode == int(blocked_path is not None), result.stderr
    if blocked_path:
        assert blocked_path in result.stdout


def test_check_ignores_changes_only_on_target_branch(repo: Path) -> None:
    ancestor = git(repo, "rev-parse", "HEAD")
    base = commit(repo, {SANITY_CHECK: "changed on target branch\n"})
    git(repo, "switch", "--detach", ancestor)
    head = commit(repo, {SOURCE_PATH: "feature\n"})
    result = subprocess.run([sys.executable, str(CHECKER), base, head], cwd=repo, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def test_lint_runs_on_every_pr() -> None:
    lint = _workflow(LINT_WORKFLOW)
    assert lint["on"]["pull_request"] == ""
    assert "if" not in lint["jobs"]["lint"]


def test_lint_workflow_collects_every_checker_regression_module() -> None:
    steps = _workflow(LINT_WORKFLOW)["jobs"]["lint"]["steps"]
    regression = next(step["run"] for step in steps if step.get("name") == "Exercise checker regressions")
    patterns = [argument for argument in split(regression) if argument.startswith("tests/")]
    suites = set(CHECKER.parent.glob("test_check_*.py")) | {CHECKER.with_name("test_ci_policy.py")}
    for suite in suites:
        relative = suite.relative_to(CHECKER.parents[2]).as_posix()
        assert any(fnmatch(relative, pattern) for pattern in patterns), relative


def test_renamed_checker_is_wired_into_local_and_trusted_ci_entries() -> None:
    root = CHECKER.parents[2]
    config = yaml.load((root / ".pre-commit-config.yaml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    hook = next(hook for repo in config["repos"] for hook in repo["hooks"] if hook["id"] == "check-code")
    entry = split(hook["entry"])
    assert entry == ["python", "tests/special_sanity/check_code.py"]
    assert (root / entry[-1]).is_file()
    trusted_steps = _workflow("protect-ci-checks.yml")["jobs"]["protect-ci-checks"]["steps"]
    trusted_step = next(step for step in trusted_steps if "run" in step)
    trusted_run = trusted_step["run"]
    assert entry[-1] in split(trusted_run)
    assert "refs/pull/${PR_NUMBER}/head" in trusted_run
    assert "PR_NUMBER" in trusted_step["env"]


@pytest.mark.parametrize(
    ("workflow", "accelerator_job"), [("gpu_smoke.yml", "setup_runner"), ("npu_smoke.yml", "npu_smoke")]
)
def test_accelerator_jobs_wait_for_lint(workflow: str, accelerator_job: str) -> None:
    jobs = _workflow(workflow)["jobs"]
    assert jobs["pr_lint"]["uses"] == f"./.github/workflows/{LINT_WORKFLOW}"
    assert "pr_lint" in jobs[accelerator_job]["needs"]
    assert "needs.pr_lint.result == 'success'" in jobs[accelerator_job]["if"]
