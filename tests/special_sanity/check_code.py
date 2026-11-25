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

"""Check staged or PR source with snapshot comparisons, Python rules and Ruff."""

import ast
import json
import os
import subprocess
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import PurePosixPath
from typing import NamedTuple

from check_comments import findings as comment_findings
from check_examples import findings as example_findings
from check_python import SHARED_PIPELINE_MODULES
from check_python import findings as python_findings

RUFF_RULES = (
    "F401,F811,F841,F822,RET505,RET506,SIM114,C901,S110,S112,E722,B006,B012,B018,B023,"
    "PLE0101,PLE0604,PLE0605,PLW0127,PLW1510,PIE794,SIM115,RUF006,S307,S602,T100,ERA001"
)
TEST_RUFF_RULES = "PT015,T100"


class Finding(NamedTuple):
    """A diagnostic in one source snapshot."""

    line: int
    code: str
    message: str
    scope: tuple[str, ...] = ()


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def _changed_files(ancestor: str, head: str) -> Iterator[tuple[str | None, str]]:
    diff = _git(
        "diff", "--name-status", "--find-renames", "-z", ancestor, head, "--", "verl_omni/", "tests/", "examples/"
    )
    fields = iter(os.fsdecode(diff).split("\0")[:-1])
    for status, path in zip(fields, fields, strict=True):
        previous = path
        if status.startswith("R"):
            path = next(fields)
        if status != "D" and (path.endswith(".py") or path.startswith("examples/")):
            yield (None if status == "A" else previous), path


def _scopes(tree: ast.AST) -> dict[int, tuple[str, ...]]:
    scopes: dict[int, tuple[str, ...]] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            scope = (*scope, node.name)
            for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                scopes[line] = scope
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return scopes


def _ruff_findings(path: str, source: bytes) -> Iterator[Finding]:
    rules = RUFF_RULES if path.startswith("verl_omni/") else TEST_RUFF_RULES
    if path.startswith("verl_omni/") and not path.startswith("verl_omni/utils/reward_score/"):
        rules += ",PLW0603"
    lint = subprocess.run(
        [
            "ruff",
            "check",
            "--no-cache",
            "--isolated",
            "--ignore-noqa",
            "--target-version=py311",
            "--config",
            "lint.mccabe.max-complexity=15",
            "--select",
            rules,
            "--output-format=json",
            "--stdin-filename",
            path,
            "-",
        ],
        input=source,
        stdout=subprocess.PIPE,
        check=False,
    )
    if lint.returncode not in (0, 1):
        lint.check_returncode()
    for item in json.loads(lint.stdout):
        yield Finding(item["location"]["row"], item["code"] or "E999", item["message"])


def _findings(path: str, source: bytes) -> list[Finding]:
    hygiene = [Finding(*finding) for finding in example_findings(path, source)]
    if not path.endswith(".py"):
        return hygiene
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        return [Finding(error.lineno or 1, "E999", error.msg)]

    findings = hygiene + [Finding(*finding) for finding in python_findings(path, tree)]
    findings.extend(Finding(*finding) for finding in comment_findings(path, source))
    if path.startswith(("verl_omni/", "tests/")):
        findings.extend(_ruff_findings(path, source))
    if not findings:
        return []
    scopes = _scopes(tree)
    return [finding._replace(scope=scopes.get(finding.line, ())) for finding in findings]


def _pipeline_findings(snapshot: str) -> Iterator[tuple[str, Finding]]:
    paths = _git("ls-tree", "-r", "--name-only", "-z", snapshot, "--", "verl_omni/pipelines/")
    names = set(os.fsdecode(paths).split("\0")[:-1])
    packages = {
        PurePosixPath(path).parts[2]
        for path in names
        if len(PurePosixPath(path).parts) == 4 and path.endswith("/__init__.py")
    } - SHARED_PIPELINE_MODULES
    if not packages:
        return
    entrypoint = "verl_omni/pipelines/__init__.py"
    try:
        tree = ast.parse(_git("show", f"{snapshot}:{entrypoint}")) if entrypoint in names else ast.Module(body=[])
    except SyntaxError:
        return
    imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level <= 1:
            parent = resolve_name("." * node.level + (node.module or ""), "verl_omni.pipelines")
            imports.add(parent)
            imports.update(f"{parent}.{alias.name}" for alias in node.names)
    for package in sorted(packages):
        module = f"verl_omni.pipelines.{package}"
        if not any(imported == module or imported.startswith(module + ".") for imported in imports):
            yield (
                f"verl_omni/pipelines/{package}/__init__.py",
                Finding(1, "VO010", f"Import {module} in {entrypoint} so its adapters can register."),
            )


def _new_findings(ancestor: str, head: str) -> Iterator[tuple[str, Finding]]:
    for previous, path in _changed_files(ancestor, head):
        source = _git("show", f"{head}:{path}")
        current = _findings(path, source)
        if not current:
            continue
        baseline = Counter()
        if previous:
            old_source = _git("show", f"{ancestor}:{previous}")
            baseline.update(finding[1:] for finding in _findings(previous, old_source))
        for finding in current:
            key = finding[1:]
            if baseline[key]:
                baseline[key] -= 1
            else:
                yield path, finding
    legacy_pipelines = set(_pipeline_findings(ancestor))
    for diagnostic in _pipeline_findings(head):
        if diagnostic not in legacy_pipelines:
            yield diagnostic


def main() -> int:
    """Check staged or PR source without importing it or loading its lint configuration."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?")
    parser.add_argument("head", nargs="?")
    args = parser.parse_args()
    base = args.base or os.environ.get("PRE_COMMIT_FROM_REF")
    head = args.head or os.environ.get("PRE_COMMIT_TO_REF")
    if bool(base) != bool(head):
        parser.error("provide both base and head, or neither to check staged changes")
    if base and head:
        ancestor = _git("merge-base", base, head).decode("ascii").strip()
    else:
        ancestor = "HEAD"
        head = _git("write-tree").decode("ascii").strip()
    findings = list(_new_findings(ancestor, head))
    for path, finding in findings:
        print(f"{path!r}:{finding.line}: {finding.code} {finding.message!r}")
    if not findings:
        print("No new code findings.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
