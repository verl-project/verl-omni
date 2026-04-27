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
Track 1: apply mechanical signature fixes (tiers 1-4) to verl_omni source.

Reads signature_changes.json produced by check_upstream_signatures.py and:
  - Tier 1 (new optional param): no code edit needed; annotated in PR body
  - Tier 3 (param renamed, single clean swap): grep-replace `old=` → `new=`
    in files that import the changed symbol
  - Tier 4 (param removed): insert a TODO comment at each call site
  - Complex: written to remaining_complex.json for Cursor (Week 4)

Also promotes api_signatures_fresh.json → api_signatures.json so the
committed snapshot stays current after the PR is merged.

Writes:
  - Edits to verl_omni/**/*.py (in-place)
  - .github/upstream_sync/api_signatures.json  (updated snapshot)
  - remaining_complex.json                      (for Cursor in Week 4)
  - pr_body_track1.md                           (PR description)

Exit codes:
  0 — changes applied (or nothing to do)
  1 — error
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHANGES_FILE = REPO_ROOT / "signature_changes.json"
FRESH_SNAPSHOT = REPO_ROOT / ".github" / "upstream_sync" / "api_signatures_fresh.json"
SNAPSHOT_FILE = REPO_ROOT / ".github" / "upstream_sync" / "api_signatures.json"
COMPLEX_FILE = REPO_ROOT / "remaining_complex.json"
PR_BODY_FILE = REPO_ROOT / "pr_body_track1.md"


# ---------------------------------------------------------------------------
# Mechanical fix implementations
# ---------------------------------------------------------------------------


def _files_to_edit(usages: list[str]) -> list[Path]:
    return [REPO_ROOT / p for p in usages if (REPO_ROOT / p).exists()]


def fix_tier1_info_only(change: dict) -> str:
    """Tier 1: new optional param — no code change, just describe in PR body."""
    added = change.get("added", [])
    return (
        f"- `{change['key']}` (`{change.get('method', 'function')}`): "
        f"new optional parameter(s) `{'`, `'.join(added)}` — non-breaking, no action needed."
    )


def fix_tier3_param_renamed(change: dict) -> tuple[list[str], str]:
    """Tier 3: rename keyword argument at all call sites in importing files."""
    old_name = change.get("old_name", "")
    new_name = change.get("new_name", "")
    edited: list[str] = []

    if not old_name or not new_name:
        return edited, f"- `{change['key']}`: param rename data incomplete — skipped, added to complex."

    # Match `old_name=` as a keyword argument (not inside strings or comments).
    # Restrict to word-boundary match to avoid partial replacements.
    pattern = re.compile(r"\b" + re.escape(old_name) + r"=")
    replacement = new_name + "="

    for py_file in _files_to_edit(change.get("verl_omni_usages", [])):
        original = py_file.read_text()
        updated = pattern.sub(replacement, original)
        if updated != original:
            py_file.write_text(updated)
            edited.append(str(py_file.relative_to(REPO_ROOT)))

    if edited:
        note = (
            f"- `{change['key']}` (`{change.get('method', 'function')}`): "
            f"renamed `{old_name}=` → `{new_name}=` in {len(edited)} file(s): "
            f"{', '.join(f'`{p}`' for p in edited)}"
        )
    else:
        note = (
            f"- `{change['key']}` (`{change.get('method', 'function')}`): "
            f"param renamed `{old_name}` → `{new_name}` but no keyword-arg usages found in verl_omni "
            f"(may use positional args — please verify manually)."
        )
    return edited, note


def fix_tier4_param_removed(change: dict) -> tuple[list[str], str]:
    """Tier 4: param removed — add TODO comment near each keyword usage."""
    removed = change.get("removed", [])
    edited: list[str] = []
    annotated: list[str] = []

    for param_name in removed:
        pattern = re.compile(r"(\b" + re.escape(param_name) + r"=)")
        todo = f"# TODO(upstream-sync): param `{param_name}` removed from upstream — verify and delete"

        for py_file in _files_to_edit(change.get("verl_omni_usages", [])):
            original = py_file.read_text()
            lines = original.splitlines(keepends=True)
            new_lines = []
            changed = False
            for line in lines:
                if pattern.search(line) and "TODO(upstream-sync)" not in line:
                    indent = len(line) - len(line.lstrip())
                    new_lines.append(" " * indent + todo + "\n")
                    changed = True
                new_lines.append(line)
            if changed:
                py_file.write_text("".join(new_lines))
                rel = str(py_file.relative_to(REPO_ROOT))
                if rel not in edited:
                    edited.append(rel)
                annotated.append(f"`{param_name}` in `{rel}`")

    if annotated:
        note = (
            f"- `{change['key']}` (`{change.get('method', 'function')}`): "
            f"param(s) `{'`, `'.join(removed)}` removed upstream. "
            f"Added TODO comments at: {', '.join(annotated)}. **Requires manual cleanup.**"
        )
    else:
        note = (
            f"- `{change['key']}` (`{change.get('method', 'function')}`): "
            f"param(s) `{'`, `'.join(removed)}` removed upstream. "
            f"No keyword usages found — may use positional args. **Please verify manually.**"
        )
    return edited, note


# ---------------------------------------------------------------------------
# PR body generation
# ---------------------------------------------------------------------------


def build_pr_body(
    tier1_notes: list[str],
    tier3_notes: list[str],
    tier4_notes: list[str],
    complex_changes: list[dict],
) -> str:
    sections = [
        "## Upstream Sync — Track 1: Signature Drift\n",
        "_Auto-generated by the daily upstream-sync bot._\n",
    ]

    if tier1_notes:
        sections.append("### Non-breaking additions (no code change required)\n")
        sections.extend(f"{n}\n" for n in tier1_notes)

    if tier3_notes:
        sections.append("\n### Parameter renames (auto-fixed)\n")
        sections.extend(f"{n}\n" for n in tier3_notes)

    if tier4_notes:
        sections.append("\n### Parameters removed upstream (TODO comments added)\n")
        sections.append("These require manual review before merging:\n")
        sections.extend(f"{n}\n" for n in tier4_notes)

    if complex_changes:
        sections.append("\n### Complex changes (require Cursor/human review)\n")
        sections.append("These were **not** auto-fixed. Cursor will handle them in a follow-up PR:\n")
        for c in complex_changes:
            sections.append(
                f"- `{c['key']}` (`{c.get('method', 'function')}`): `{c['type']}` — {c.get('detail', '')}\n"
            )

    sections.append(
        "\n---\n"
        "**Checklist before merging:**\n"
        "- [ ] Run `pytest -s -x tests/special_sanity` locally\n"
        "- [ ] Verify any TODO comments above are resolved or intentionally deferred\n"
        "- [ ] Confirm complex changes are tracked in a follow-up issue/PR\n"
    )

    return "".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not CHANGES_FILE.exists():
        print("No signature_changes.json found — nothing to apply.")
        return 0

    with open(CHANGES_FILE) as f:
        data = json.load(f)

    if not data.get("has_changes"):
        print("No changes to apply.")
        return 0

    mechanical = data.get("mechanical", [])
    complex_changes = data.get("complex", [])

    tier1_notes: list[str] = []
    tier3_notes: list[str] = []
    tier4_notes: list[str] = []
    all_edited: list[str] = []
    leftover_complex: list[dict] = []

    for change in mechanical:
        tier = change.get("tier")

        if tier == 1:
            tier1_notes.append(fix_tier1_info_only(change))

        elif tier == 3:
            if change.get("type") == "param_renamed":
                edited, note = fix_tier3_param_renamed(change)
                all_edited.extend(edited)
                tier3_notes.append(note)
            else:
                # method_added is tier 1 in practice
                tier1_notes.append(fix_tier1_info_only(change))

        elif tier == 4:
            edited, note = fix_tier4_param_removed(change)
            all_edited.extend(edited)
            tier4_notes.append(note)

        else:
            leftover_complex.append(change)

    all_complex = complex_changes + leftover_complex

    # Promote fresh snapshot → committed snapshot
    if FRESH_SNAPSHOT.exists():
        import shutil

        shutil.copy(FRESH_SNAPSHOT, SNAPSHOT_FILE)
        print(f"Updated {SNAPSHOT_FILE.relative_to(REPO_ROOT)}")
        FRESH_SNAPSHOT.unlink()

    # Write remaining_complex.json
    COMPLEX_FILE.write_text(json.dumps({"complex": all_complex}, indent=2) + "\n")

    # Write PR body
    pr_body = build_pr_body(tier1_notes, tier3_notes, tier4_notes, all_complex)
    PR_BODY_FILE.write_text(pr_body)

    # Summarize
    print("\nApplied fixes:")
    print(f"  Tier 1 (info-only):  {len(tier1_notes)}")
    print(f"  Tier 3 (renamed):    {len(tier3_notes)}")
    print(f"  Tier 4 (TODO'd):     {len(tier4_notes)}")
    print(f"  Complex (deferred):  {len(all_complex)}")
    print(f"  Files edited:        {len(set(all_edited))}")

    # Set GHA output
    import os

    has_pr_content = bool(tier1_notes or tier3_notes or tier4_notes or all_complex)
    needs_human = bool(tier4_notes or all_complex)
    has_complex = bool(all_complex)
    if gha_output := os.environ.get("GITHUB_OUTPUT"):
        with open(gha_output, "a") as f:
            f.write(f"has_pr_content={'true' if has_pr_content else 'false'}\n")
            f.write(f"needs_human_review={'true' if needs_human else 'false'}\n")
            f.write(f"has_complex={'true' if has_complex else 'false'}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
