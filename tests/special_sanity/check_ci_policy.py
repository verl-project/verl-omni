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
import subprocess
import tomllib
from argparse import ArgumentParser

PROTECTED_DIRECTORIES = ("tests/special_sanity/", ".github/workflows/", ".github/actions/")
PROTECTED_FILES = {
    ".pre-commit-config.yaml",
    ".github/CODEOWNERS",
    "scripts/generate_trainer_config.sh",
    "ruff.toml",
    ".ruff.toml",
    "mypy.ini",
    ".mypy.ini",
}
IMPLEMENTATION_DIRECTORIES = ("verl_omni/", "examples/")


def _is_implementation(path: str) -> bool:
    return path.startswith(IMPLEMENTATION_DIRECTORIES) or (
        path.startswith("tests/") and not path.startswith("tests/special_sanity/")
    )


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def _lint_config(revision: str) -> tuple[dict | None, dict | None]:
    tools = tomllib.loads(_git("show", f"{revision}:pyproject.toml").decode("utf-8")).get("tool", {})
    return tools.get("ruff"), tools.get("mypy")


def mixed_changes(base: str, head: str) -> tuple[list[str], list[str]]:
    """Return the checks and implementation paths when the PR changes both."""
    ancestor = _git("merge-base", base, head).decode("ascii").strip()
    diff = _git("diff", "--name-status", "--no-renames", "-z", ancestor, head, "--")
    fields = os.fsdecode(diff).split("\0")[:-1]
    implementation = sorted(path for path in fields[1::2] if _is_implementation(path))
    if not implementation:
        return [], []

    checks = []
    for status, path in zip(fields[::2], fields[1::2], strict=True):
        if path in PROTECTED_FILES or (status != "A" and path.startswith(PROTECTED_DIRECTORIES)):
            checks.append(path)
        elif path == "pyproject.toml" and (status != "M" or _lint_config(ancestor) != _lint_config(head)):
            checks.append("pyproject.toml (Ruff or mypy config)")
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
