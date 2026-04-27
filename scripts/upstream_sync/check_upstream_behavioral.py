#!/usr/bin/env python3
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
"""
Track 2: detect behavioral drift in upstream symbols that verl-omni mirrors.

Uses the GitHub compare API (no upstream clone needed) to find commits that
touched watched files since the last recorded SHA, then filters the diff to
only hunks touching the watched symbols.

Reads:
  .github/upstream_sync/upstream_watch.yaml

Environment variables:
  UPSTREAM_VERL_SHA       — last processed verl commit SHA (from GH Actions variable)
  UPSTREAM_VLLM_OMNI_SHA  — last processed vllm-omni commit SHA
  GH_TOKEN / GITHUB_TOKEN — GitHub API auth

Writes:
  behavioral_changes.json — list of {entry, patch, current_sha} for affected symbols

Exit codes:
  0 — no relevant upstream changes
  1 — behavioral drift detected (workflow proceeds to generate prompt + Cursor)
  2 — API error (missing token, rate limit, etc.)
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
WATCH_FILE = REPO_ROOT / ".github" / "upstream_sync" / "upstream_watch.yaml"
CHANGES_FILE = REPO_ROOT / "behavioral_changes.json"

# Maps upstream_repo slug → GitHub API base + SHA env var
REPO_CONFIG = {
    "verl-project/verl": {
        "sha_env": "UPSTREAM_VERL_SHA",
        "default_branch": "main",
    },
    "vllm-project/vllm-omni": {
        "sha_env": "UPSTREAM_VLLM_OMNI_SHA",
        "default_branch": "main",
    },
}

GH_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------


def _gh_request(path: str, token: str) -> dict:
    url = f"{GH_API}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"GitHub API error {e.code} for {url}: {body}") from e


def get_head_sha(repo: str, branch: str, token: str) -> str:
    data = _gh_request(f"repos/{repo}/commits/{branch}", token)
    return data["sha"]


def get_comparison(repo: str, base_sha: str, head_sha: str, token: str) -> dict:
    """Return GitHub comparison object between base and head."""
    return _gh_request(f"repos/{repo}/compare/{base_sha}...{head_sha}", token)


def get_file_content(repo: str, file_path: str, ref: str, token: str) -> str | None:
    """Fetch raw file content at a specific ref (for context in the Cursor prompt)."""
    try:
        data = _gh_request(f"repos/{repo}/contents/{file_path}?ref={ref}", token)
        import base64

        return base64.b64decode(data["content"]).decode()
    except (RuntimeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Diff filtering
# ---------------------------------------------------------------------------


def _split_hunks(patch: str) -> list[str]:
    """Split a unified diff patch into individual @@ hunks."""
    hunks = []
    current: list[str] = []
    for line in patch.splitlines(keepends=True):
        if line.startswith("@@") and current:
            hunks.append("".join(current))
            current = []
        current.append(line)
    if current:
        hunks.append("".join(current))
    return hunks


def filter_patch_to_symbols(patch: str, symbols: list[str]) -> str:
    """
    Keep only hunks where at least one watched symbol name appears.

    Heuristic: symbol appears in the hunk header line (e.g. `@@ ... @@ class Foo`)
    OR in any changed (+/-) line in the hunk body.
    Errs on the side of inclusion — the AI can discard irrelevant context.
    """
    if not patch:
        return ""

    symbol_pattern = re.compile(r"\b(" + "|".join(re.escape(s) for s in symbols) + r")\b")
    kept: list[str] = []

    for hunk in _split_hunks(patch):
        lines = hunk.splitlines(keepends=True)
        # Check hunk header (first line) and any modified lines
        header = lines[0] if lines else ""
        changed_lines = "".join(line for line in lines[1:] if line.startswith(("+", "-")))
        if symbol_pattern.search(header) or symbol_pattern.search(changed_lines):
            kept.append(hunk)

    return "".join(kept)


def try_alt_paths(changed_files: dict[str, str], file_path: str) -> str | None:
    """Try the given path and its __init__.py variant."""
    patch = changed_files.get(file_path)
    if patch is not None:
        return patch
    # e.g. verl/single_controller/base.py → verl/single_controller/base/__init__.py
    alt = file_path.replace(".py", "/__init__.py")
    return changed_files.get(alt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GH_TOKEN or GITHUB_TOKEN must be set", file=sys.stderr)
        return 2

    with open(WATCH_FILE) as f:
        watch_config = yaml.safe_load(f)
    entries = watch_config.get("watch", [])

    # Group entries by upstream repo
    by_repo: dict[str, list[dict]] = {}
    for entry in entries:
        by_repo.setdefault(entry["upstream_repo"], []).append(entry)

    all_results: list[dict] = []
    current_shas: dict[str, str] = {}

    for repo, repo_entries in by_repo.items():
        config = REPO_CONFIG.get(repo)
        if not config:
            print(f"  WARNING: no config for repo {repo}, skipping")
            continue

        print(f"\nChecking {repo}...")

        # Get current HEAD SHA
        try:
            current_sha = get_head_sha(repo, config["default_branch"], token)
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 2
        current_shas[repo] = current_sha
        print(f"  Current HEAD: {current_sha[:12]}")

        # Get last processed SHA
        last_sha = os.environ.get(config["sha_env"], "").strip()
        if not last_sha:
            print(f"  No prior SHA ({config['sha_env']} not set) — using 7-day lookback")
            # Fall back: get SHA from 7 days ago as base
            from datetime import datetime, timedelta, timezone

            since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
            commits = _gh_request(
                f"repos/{repo}/commits?sha={config['default_branch']}&since={since}&per_page=1",
                token,
            )
            if not commits:
                print("  No commits in last 7 days — nothing to check")
                continue
            last_sha = commits[-1]["sha"]
            print(f"  Using 7-day lookback base: {last_sha[:12]}")

        if last_sha == current_sha:
            print("  No new commits since last run — skipping")
            continue

        # Get comparison diff via GitHub API
        print(f"  Comparing {last_sha[:12]}...{current_sha[:12]}")
        try:
            comparison = get_comparison(repo, last_sha, current_sha, token)
        except RuntimeError as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            return 2

        num_commits = len(comparison.get("commits", []))
        changed_files: dict[str, str] = {f["filename"]: f.get("patch", "") for f in comparison.get("files", [])}
        print(f"  {num_commits} commit(s), {len(changed_files)} file(s) changed")

        # Check each watched entry
        for entry in repo_entries:
            file_path = entry["file"]
            symbols = entry["symbols"]

            patch = try_alt_paths(changed_files, file_path)
            if patch is None:
                continue  # file not touched in this window

            relevant_patch = filter_patch_to_symbols(patch, symbols)
            if not relevant_patch:
                print(f"  {file_path}: changed but no watched symbols affected — skipping")
                continue

            print(f"  {file_path}: changes to {symbols} detected")

            # Fetch current upstream file content for Cursor context
            upstream_content = get_file_content(repo, file_path, current_sha, token)

            all_results.append(
                {
                    "entry": entry,
                    "upstream_patch": relevant_patch,
                    "upstream_content": upstream_content,
                    "current_sha": current_sha,
                    "last_sha": last_sha,
                }
            )

    # Write current SHAs for the workflow to update variables with
    current_shas_file = REPO_ROOT / "upstream_current_shas.json"
    current_shas_file.write_text(json.dumps(current_shas, indent=2) + "\n")

    if not all_results:
        print("\nNo behavioral drift detected.")
        CHANGES_FILE.write_text(json.dumps({"has_changes": False, "changes": []}, indent=2) + "\n")
        return 0

    print(f"\nBehavioral drift in {len(all_results)} upstream file(s):")
    for r in all_results:
        print(f"  {r['entry']['file']} → {r['entry']['verl_omni_file']}")

    CHANGES_FILE.write_text(json.dumps({"has_changes": True, "changes": all_results}, indent=2) + "\n")

    # Set GHA output
    if gha_output := os.environ.get("GITHUB_OUTPUT"):
        with open(gha_output, "a") as f:
            f.write("has_changes=true\n")
            f.write(f"change_count={len(all_results)}\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
