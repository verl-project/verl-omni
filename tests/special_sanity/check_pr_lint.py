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
import re
import subprocess
from argparse import ArgumentParser
from collections import Counter
from collections.abc import Iterator
from importlib.util import resolve_name
from pathlib import PurePosixPath
from typing import NamedTuple

from check_ast_rules import SHARED_PIPELINE_MODULES, _static_exports
from check_ast_rules import findings as python_findings
from check_comments import findings as comment_findings
from check_example_recipes import ACCEL_VARIANT_SCRIPT, delegates_to_base_script
from check_example_recipes import findings as example_findings

RUFF_RULES = (
    "F401,F811,F841,F822,RET505,RET506,SIM114,C901,S110,S112,E722,B006,B012,B018,B023,"
    "PLE0101,PLE0604,PLE0605,PLW0127,PLW1510,PIE794,SIM115,RUF006,S307,S602,T100,ERA001"
)
TEST_RUFF_RULES = "PT015,T100"


_EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
_LINE_NUMBER_RE = re.compile(r"\bline \d+\b")


class Finding(NamedTuple):
    """A diagnostic in one source snapshot."""

    line: int
    code: str
    message: str
    scope: tuple[str, ...] = ()


def _repo_root() -> str:
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()


def _git(*args: str, root: str | None = None) -> bytes:
    target_root = root or _repo_root()
    return subprocess.check_output(["git", "-C", target_root, *args])


def _changed_files(ancestor: str, head: str, root: str | None = None) -> Iterator[tuple[str | None, str]]:
    diff = _git(
        "diff",
        "--raw",
        "--find-renames",
        "-z",
        ancestor,
        head,
        "--",
        ":(top)verl_omni/",
        ":(top)tests/",
        ":(top)examples/",
        root=root,
    )
    fields = iter(os.fsdecode(diff).split("\0")[:-1])
    for header in fields:
        old_mode, new_mode, _old_sha, _new_sha, status = header.removeprefix(":").split()
        previous = next(fields)
        path = next(fields) if status.startswith(("R", "C")) else previous
        if status == "D" or new_mode in {"000000", "160000"}:
            continue
        if path.endswith(".py") or path.startswith("examples/"):
            prev_path = None if status.startswith(("A", "C")) or old_mode in {"000000", "160000"} else previous
            yield prev_path, path


def _scopes(tree: ast.AST) -> dict[int, tuple[str, ...]]:
    scopes: dict[int, tuple[str, ...]] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            scope = (*scope, node.name)
            start_line = min((decorator.lineno for decorator in node.decorator_list), default=node.lineno)
            for line in range(start_line, (node.end_lineno or node.lineno) + 1):
                scopes[line] = scope
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return scopes


def _ruff_findings(path: str, source: bytes) -> Iterator[Finding]:
    try:
        source.decode("utf-8")
    except UnicodeDecodeError:
        yield Finding(1, "E999", "Source file is not valid UTF-8.")
        return

    rules = RUFF_RULES if path.startswith("verl_omni/") else TEST_RUFF_RULES
    if path.startswith("verl_omni/") and not path.startswith("verl_omni/utils/reward_score/"):
        rules += ",PLW0603"
    try:
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
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ruff executable not found on PATH; install ruff==0.12.2 or run via uv --with ruff==0.12.2."
        ) from exc
    if lint.returncode not in (0, 1):
        lint.check_returncode()
    for item in json.loads(lint.stdout):
        yield Finding(item["location"]["row"], item["code"] or "E999", item["message"])


def _findings(path: str, source: bytes) -> list[Finding]:
    hygiene = [Finding(*finding) for finding in example_findings(path, source)]
    if not path.endswith(".py"):
        if path.startswith("examples/") and path.endswith(".sh"):
            hygiene.extend(Finding(*finding) for finding in comment_findings(path, source))
        return hygiene
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        return [Finding(error.lineno or 1, "E999", error.msg)]

    findings = hygiene + [Finding(*finding) for finding in python_findings(path, tree)]
    findings.extend(Finding(*finding) for finding in comment_findings(path, source, tree=tree))
    if path.startswith(("verl_omni/", "tests/")):
        findings.extend(_ruff_findings(path, source))
    if not findings:
        return []
    scopes = _scopes(tree)
    return [finding._replace(scope=scopes.get(finding.line, ())) for finding in findings]


def _ls_tree_blobs(snapshot: str, directory: str, root: str | None = None) -> set[str]:
    raw = _git("ls-tree", "-r", "-z", snapshot, "--", f":(top){directory}", root=root)
    paths: set[str] = set()
    for entry in os.fsdecode(raw).split("\0")[:-1]:
        meta, path = entry.split("\t", 1)
        if meta.split()[0].startswith("100"):
            paths.add(path)
    return paths


def _collect_module_imports(
    statements: list[ast.stmt], imports: set[str], package: str = "verl_omni.pipelines"
) -> None:
    max_level = len(package.split("."))
    for node in statements:
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level <= max_level:
            parent = resolve_name("." * node.level + (node.module or ""), package)
            imports.add(parent)
            imports.update(f"{parent}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Try | ast.TryStar):
            _collect_module_imports(node.body, imports, package=package)
            _collect_module_imports(node.orelse, imports, package=package)


def _defines_registered_adapter(source: bytes) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.ClassDef)
        and any(
            isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr == "register"
            for dec in node.decorator_list
        )
        for node in tree.body
    )


def _pipeline_findings(snapshot: str, root: str | None = None) -> Iterator[tuple[str, Finding]]:
    names = _ls_tree_blobs(snapshot, "verl_omni/pipelines/", root=root)
    entrypoint = "verl_omni/pipelines/__init__.py"
    if entrypoint not in names and not any(path.endswith("/__init__.py") for path in names):
        return
    packages = {
        PurePosixPath(path).parts[2] for path in names if len(PurePosixPath(path).parts) >= 4 and path.endswith(".py")
    } - SHARED_PIPELINE_MODULES
    if not packages:
        return
    try:
        tree = (
            ast.parse(_git("show", f"{snapshot}:{entrypoint}", root=root))
            if entrypoint in names
            else ast.Module(body=[])
        )
    except SyntaxError:
        return
    imports: set[str] = set()
    _collect_module_imports(tree.body, imports)
    for package in sorted(packages):
        module = f"verl_omni.pipelines.{package}"
        if not any(imported == module or imported.startswith(module + ".") for imported in imports):
            yield (
                f"verl_omni/pipelines/{package}/__init__.py",
                Finding(
                    1,
                    "VO010",
                    f"Import and re-export {module} (e.g. via __all__) in {entrypoint} so its adapters can register.",
                ),
            )
        pkg_prefix = f"verl_omni/pipelines/{package}/"
        pkg_init = f"{pkg_prefix}__init__.py"
        adapter_mods = [
            path
            for path in sorted(names)
            if path.startswith(pkg_prefix)
            and path != pkg_init
            and path.endswith(".py")
            and _defines_registered_adapter(_git("show", f"{snapshot}:{path}", root=root))
        ]
        if not adapter_mods:
            continue
        try:
            pkg_tree = (
                ast.parse(_git("show", f"{snapshot}:{pkg_init}", root=root))
                if pkg_init in names
                else ast.Module(body=[])
            )
        except SyntaxError:
            continue
        pkg_imports: set[str] = set()
        _collect_module_imports(pkg_tree.body, pkg_imports, package=module)
        for mod_path in adapter_mods:
            mod_fqn = ".".join(PurePosixPath(mod_path).with_suffix("").parts)
            if not any(imp == mod_fqn or imp.startswith(mod_fqn + ".") for imp in pkg_imports):
                yield (
                    mod_path,
                    Finding(
                        1,
                        "VO010",
                        f"Import {PurePosixPath(mod_path).stem} in {pkg_init} "
                        "so its registered adapter loads at pipeline import time.",
                    ),
                )


def _example_variant_findings(snapshot: str, root: str | None = None) -> Iterator[tuple[str, Finding]]:
    names = _ls_tree_blobs(snapshot, "examples/", root=root)
    for path in sorted(names):
        path_obj = PurePosixPath(path)
        match = ACCEL_VARIANT_SCRIPT.match(path_obj.name)
        if not match:
            continue
        base_name = match.group(1) + (match.group(2) or "") + ".sh"
        base_path = str(path_obj.parent / base_name)
        if base_path not in names:
            continue
        source = _git("show", f"{snapshot}:{path}", root=root)
        if not delegates_to_base_script(source, base_name):
            yield (
                path,
                Finding(
                    1,
                    "EX009",
                    f"Reuse sibling GPU recipe '{base_name}' via 'exec bash \"$SCRIPT_DIR/{base_name}\" ... \"$@\"' "
                    "instead of duplicating it.",
                ),
            )


def _dead_function_kind(node: ast.stmt, exported: set[str] | None) -> str | None:
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or node.decorator_list:
        return None
    name = node.name
    if (name.startswith("__") and name.endswith("__")) or name == "main":
        return None
    if name.startswith("_"):
        return "private"
    if exported is not None and name not in exported:
        return "unexported"
    return None


def _is_symbol_referenced_in_module(tree: ast.Module, owner: ast.AST, name: str) -> bool:
    word_re = re.compile(rf"\b{re.escape(name)}\b")
    for stmt in tree.body:
        if stmt is owner:
            continue
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id == name:
                return True
            if isinstance(sub, ast.Attribute) and sub.attr == name:
                return True
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) and word_re.search(sub.value):
                return True
    return False


def _dead_function_candidates(source: bytes) -> list[tuple[int, str, str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    if not isinstance(tree, ast.Module):
        return []
    exported = _static_exports(tree)
    candidates: list[tuple[int, str, str]] = []
    for node in tree.body:
        kind = _dead_function_kind(node, exported)
        if kind is not None and not _is_symbol_referenced_in_module(tree, node, node.name):
            candidates.append((node.lineno, node.name, kind))
    return candidates


def _has_external_reference(snapshot: str, own_path: str, name: str, root: str | None = None) -> bool:
    target_root = root or _repo_root()
    result = subprocess.run(
        ["git", "-C", target_root, "grep", "-I", "-F", "-w", "-l", "-z", "-e", name, snapshot],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode not in (0, 1):
        result.check_returncode()
    if result.returncode == 1 or not result.stdout:
        return False
    prefix = f"{snapshot}:"
    for entry in os.fsdecode(result.stdout).split("\0")[:-1]:
        rel = entry.removeprefix(prefix) if entry.startswith(prefix) else entry.split(":", 1)[-1]
        if rel != own_path:
            return True
    return False


def _dead_function_findings(
    ancestor: str,
    head: str,
    changed: list[tuple[str | None, str]],
    root: str | None = None,
) -> Iterator[tuple[str, Finding]]:
    for previous, path in changed:
        if not (path.startswith("verl_omni/") and path.endswith(".py") and not path.endswith("/__init__.py")):
            continue
        head_candidates = _dead_function_candidates(_git("show", f"{head}:{path}", root=root))
        if not head_candidates:
            continue
        old_keys: set[tuple[str, str]] | None = None
        for lineno, name, kind in head_candidates:
            if _has_external_reference(head, path, name, root=root):
                continue
            if previous is not None:
                if old_keys is None:
                    old_source = _git("show", f"{ancestor}:{previous}", root=root)
                    old_keys = {
                        (old_name, old_kind) for _ln, old_name, old_kind in _dead_function_candidates(old_source)
                    }
                if (name, kind) in old_keys and not _has_external_reference(ancestor, previous, name, root=root):
                    continue
            message = (
                f"Remove unused private function '{name}' or wire it into a caller."
                if kind == "private"
                else f"Remove unused unexported function '{name}', call it, or export it in __all__."
            )
            yield path, Finding(lineno, "VO019", message)


def _baseline_key(finding: Finding) -> tuple[str, str, tuple[str, ...]]:
    return (finding.code, _LINE_NUMBER_RE.sub("line <N>", finding.message), finding.scope)


def _new_findings(ancestor: str, head: str, root: str | None = None) -> Iterator[tuple[str, Finding]]:
    changed = list(_changed_files(ancestor, head, root=root))
    for previous, path in changed:
        source = _git("show", f"{head}:{path}", root=root)
        current = _findings(path, source)
        if not current:
            continue
        baseline = Counter()
        if previous:
            old_source = _git("show", f"{ancestor}:{previous}", root=root)
            baseline.update(_baseline_key(finding) for finding in _findings(previous, old_source))
        for finding in current:
            key = _baseline_key(finding)
            if baseline[key]:
                baseline[key] -= 1
            else:
                yield path, finding
    legacy_pipelines = set(_pipeline_findings(ancestor, root=root))
    for diagnostic in _pipeline_findings(head, root=root):
        if diagnostic not in legacy_pipelines:
            yield diagnostic
    changed_head_paths = {path for _previous, path in changed}
    rename_map = {path: previous for previous, path in changed if previous and previous != path}
    legacy_variant_paths = {path for path, _finding in _example_variant_findings(ancestor, root=root)}
    for path, finding in _example_variant_findings(head, root=root):
        if (
            path in legacy_variant_paths
            or rename_map.get(path) in legacy_variant_paths
            or path not in changed_head_paths
        ):
            continue
        yield path, finding
    yield from _dead_function_findings(ancestor, head, changed, root=root)


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
    root = _repo_root()
    if base and head:
        ancestor = _git("merge-base", base, head, root=root).decode("ascii").strip()
    else:
        head_exists = (
            subprocess.run(
                ["git", "-C", root, "rev-parse", "--verify", "--quiet", "HEAD"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        ancestor = "HEAD" if head_exists else _EMPTY_TREE_SHA
        head = _git("write-tree", root=root).decode("ascii").strip()
    findings = list(_new_findings(ancestor, head, root=root))
    for path, finding in findings:
        print(f"{path!r}:{finding.line}: {finding.code} {finding.message!r}")
    if not findings:
        print("No new code findings.")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
