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
"""Ensure every examples/**/README.md is exposed via a docs/examples git symlink.

PR #316 made docs/examples/*.md git symlinks (mode 120000) pointing at the
corresponding examples/**/README.md so ReadTheDocs can render them. This
checker keeps that invariant from drifting.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EXEMPT_DOCS_PAGES = frozenset(
    {
        "docs/examples/config.md",
        "docs/examples/flowgrpo_trainer_sd35_drm.md",
    }
)

GIT_SYMLINK_MODE = "120000"


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def staged_files(prefix: str = "") -> dict[str, tuple[str, str]]:
    """Return path -> (mode, object_sha) for staged files."""
    out = _run_git("ls-files", "--stage")
    files: dict[str, tuple[str, str]] = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        mode, sha, _stage = meta.split()
        if prefix and not path.startswith(prefix):
            continue
        files[path] = (mode, sha)
    return files


def blob_text(sha: str) -> str:
    return _run_git("cat-file", "-p", sha).rstrip("\n")


def _normalize_relpath(docs_path: str, target: str) -> str:
    parts: list[str] = []
    for part in Path(docs_path).parent.parts + Path(target).parts:
        if part == "..":
            if parts:
                parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def expected_docs_path_and_target(readme_path: str) -> tuple[str, str]:
    """Map examples/.../README.md to docs/examples/... path and symlink target."""
    parts = Path(readme_path).parts
    if len(parts) < 3 or parts[0] != "examples" or parts[-1] != "README.md":
        raise ValueError(f"unexpected README path layout: {readme_path}")

    mid = parts[1:-1]
    if len(mid) == 1:
        trainer = mid[0]
        docs_path = f"docs/examples/{trainer}.md"
        target = f"../../examples/{trainer}/README.md"
        return docs_path, target
    if len(mid) == 2:
        trainer, model = mid
        docs_path = f"docs/examples/{model}/{trainer}_{model}.md"
        target = f"../../../examples/{trainer}/{model}/README.md"
        return docs_path, target
    raise ValueError(f"unsupported nested README depth (only trainer or trainer/model allowed): {readme_path}")


def parse_examples_toctree(index_text: str) -> set[str]:
    """Return toctree entries under the recipe Examples caption (maxdepth 2)."""
    pattern = re.compile(
        r"```\{toctree\}\n:maxdepth: 2\n:caption: Examples\n\n(.*?)```",
        re.DOTALL,
    )
    matches = pattern.findall(index_text)
    if not matches:
        raise ValueError("could not find Examples toctree (maxdepth: 2) in docs/index.md")
    block = matches[-1]
    return {line.strip() for line in block.splitlines() if line.strip()}


def main() -> int:
    errors: list[str] = []
    all_staged = staged_files()
    example_readmes = sorted(
        path for path in all_staged if path.startswith("examples/") and path.endswith("/README.md")
    )
    docs_example_files = {
        path: meta for path, meta in all_staged.items() if path.startswith("docs/examples/") and path.endswith(".md")
    }

    expected_docs_for_readme: dict[str, str] = {}
    for readme in example_readmes:
        try:
            docs_path, expected_target = expected_docs_path_and_target(readme)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        expected_docs_for_readme[readme] = docs_path

        if docs_path not in docs_example_files:
            errors.append(f"missing docs symlink for {readme}: expected {docs_path}")
            continue

        mode, sha = docs_example_files[docs_path]
        if mode != GIT_SYMLINK_MODE:
            errors.append(f"{docs_path} must be a git symlink (mode {GIT_SYMLINK_MODE}), found mode {mode}")
            continue

        actual_target = blob_text(sha)
        if actual_target != expected_target:
            errors.append(f"{docs_path} symlink target mismatch: expected {expected_target!r}, got {actual_target!r}")
            continue

        rel = _normalize_relpath(docs_path, actual_target)
        if rel not in all_staged:
            errors.append(f"{docs_path} points to missing target {actual_target} ({rel})")

    index_path = REPO_ROOT / "docs" / "index.md"
    index_text = index_path.read_text(encoding="utf-8")
    try:
        toctree_entries = parse_examples_toctree(index_text)
    except ValueError as exc:
        errors.append(str(exc))
        toctree_entries = set()

    for _readme, docs_path in expected_docs_for_readme.items():
        toctree_key = docs_path.removeprefix("docs/")
        if toctree_key not in toctree_entries:
            errors.append(
                f"{docs_path} is not listed in the Examples toctree in docs/index.md (expected entry {toctree_key!r})"
            )

    example_readme_set = set(example_readmes)
    for docs_path, (mode, sha) in docs_example_files.items():
        if docs_path in EXEMPT_DOCS_PAGES:
            continue
        if mode != GIT_SYMLINK_MODE:
            continue
        target = blob_text(sha)
        rel = _normalize_relpath(docs_path, target)
        if not rel.startswith("examples/") or not rel.endswith("/README.md"):
            errors.append(
                f"{docs_path} is a git symlink but does not point at an "
                f"examples/**/README.md (target {target!r} -> {rel})"
            )
        elif rel not in example_readme_set:
            errors.append(f"{docs_path} is a dangling git symlink to missing {rel}")

    if errors:
        print("Example README docs-symlink check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print(f"OK: {len(example_readmes)} examples/**/README.md pages are git-symlinked under docs/examples/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
