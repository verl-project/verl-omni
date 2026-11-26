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

"""Keep changes to existing CI checks out of implementation PRs."""

import os
import re
import subprocess
import tomllib
from argparse import ArgumentParser
from pathlib import PurePosixPath

STRICT_PROTECTED_DIRECTORIES = (".github/workflows/", ".github/actions/")
SANITY_DIRECTORY = "tests/special_sanity/"
NEW_SANITY_FILE_RE = re.compile(r"tests/special_sanity/(?:check|test)_[a-z0-9_]+\.py")
PROTECTED_FILES = {
    ".pre-commit-config.yaml",
    ".github/CODEOWNERS",
    "scripts/generate_trainer_config.sh",
    "conftest.py",
    "tests/conftest.py",
}
PROTECTED_ANY_DEPTH_FILENAMES = {
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    ".mypy.ini",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
}
IMPLEMENTATION_DIRECTORIES = ("verl_omni/", "examples/")


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def _lint_config(revision: str) -> tuple[dict | None, dict | None, dict | None] | object:
    try:
        tools = tomllib.loads(_git("show", f"{revision}:pyproject.toml").decode("utf-8")).get("tool", {})
    except (tomllib.TOMLDecodeError, UnicodeDecodeError):
        return object()
    return tools.get("ruff"), tools.get("mypy"), tools.get("pytest")


def _is_protected_check(status: str, path: str) -> str | None:
    if path in PROTECTED_FILES or path.startswith(STRICT_PROTECTED_DIRECTORIES):
        return path
    basename = PurePosixPath(path).name
    if basename in PROTECTED_ANY_DEPTH_FILENAMES or (basename == "pyproject.toml" and path != "pyproject.toml"):
        return path
    if path.startswith(SANITY_DIRECTORY):
        if status != "A" or not NEW_SANITY_FILE_RE.fullmatch(path):
            return path
    return None


def mixed_changes(base: str, head: str) -> tuple[list[str], list[str]]:
    """Return the checks and implementation paths when the PR changes both."""
    ancestor = _git("merge-base", base, head).decode("ascii").strip()
    diff = _git("diff", "--name-status", "--no-renames", "-z", ancestor, head, "--")
    fields = os.fsdecode(diff).split("\0")[:-1]
    implementation = sorted(path for path in fields[1::2] if path.startswith(IMPLEMENTATION_DIRECTORIES))
    if not implementation:
        return [], []

    checks = []
    for status, path in zip(fields[::2], fields[1::2], strict=True):
        protected = _is_protected_check(status, path)
        if protected is not None:
            checks.append(protected)
        elif path == "pyproject.toml" and (status != "M" or _lint_config(ancestor) != _lint_config(head)):
            checks.append("pyproject.toml (Ruff, mypy, or pytest config)")
    return (sorted(checks), implementation) if checks else ([], [])


def main() -> int:
    """Check the PR diff without importing or running its code."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args()
    checks, implementation = mixed_changes(args.base, args.head)
    if checks:
        print("Fix the flagged implementation or propose check maintenance in a separate PR for review.")
        print("Changed checks:")
        for path in checks:
            print(f"  {path!r}")
        print("Implementation paths:")
        for path in implementation:
            print(f"  {path!r}")
        return 1
    print("No mixed check and implementation changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
