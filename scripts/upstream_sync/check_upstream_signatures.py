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
Signature-drift: detect upstream API signature drift.

Generates fresh signatures from the currently installed upstream packages,
diffs against the committed .github/upstream_sync/api_signatures.json snapshot,
and writes:
  - signature_changes.json  — classified list of every changed symbol
  - api_signatures_fresh.json — new snapshot (committed by the workflow after fixes)

Change classifications:
  non_breaking_addition — new optional param; callers need no update
  param_rename          — one param renamed; same kind, required, and annotation — no AI needed
  param_removal         — param removed; AI agent verifies and fixes
  others                — structural/complex change; AI agent verifies and fixes

Exit codes:
  0 — no drift detected
  1 — drift detected (workflow proceeds to apply_signature_fixes.py)
  2 — upstream packages not resolvable (ImportErrors for all symbols)
"""

import json
import os
import sys
from pathlib import Path

# Allow running from repo root or this script's directory
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "upstream_sync"))

from generate_api_signatures import collect_upstream_imports, get_symbol_info  # noqa: E402

SNAPSHOT_FILE = Path(
    os.environ.get(
        "UPSTREAM_SYNC_SNAPSHOT_FILE",
        str(REPO_ROOT / ".github" / "upstream_sync" / "api_signatures.json"),
    )
)
FRESH_FILE = REPO_ROOT / ".github" / "upstream_sync" / "api_signatures_fresh.json"
CHANGES_FILE = REPO_ROOT / "signature_changes.json"


# ---------------------------------------------------------------------------
# Change classification
# ---------------------------------------------------------------------------


def _classify_param_change(method_name: str, old_params: dict, new_params: dict) -> dict | None:
    """Classify a change to a single method's parameter list."""
    # Strip annotation before comparison: annotation-only changes don't affect call-site
    # behavior, and old snapshots may predate the annotation field in the schema.
    old_params = {n: {k: v for k, v in p.items() if k != "annotation"} for n, p in old_params.items()}
    new_params = {n: {k: v for k, v in p.items() if k != "annotation"} for n, p in new_params.items()}

    if old_params == new_params:
        return None

    old_names = set(old_params)
    new_names = set(new_params)
    added = new_names - old_names
    removed = old_names - new_names

    if not added and not removed:
        # Same names, different attributes (kind changed, required changed)
        changed = {n for n in old_names if old_params[n] != new_params[n]}
        return {
            "method": method_name,
            "type": "param_attrs_changed",
            "classification": "others",
            "changed": sorted(changed),
        }

    if added and not removed:
        all_optional = all(not new_params[p]["required"] for p in added)
        if all_optional:
            return {
                "method": method_name,
                "type": "param_added_optional",
                "classification": "non_breaking_addition",
                "added": sorted(added),
            }
        return {
            "method": method_name,
            "type": "param_added_required",
            "classification": "others",
            "added": sorted(added),
        }

    if removed and not added:
        return {
            "method": method_name,
            "type": "param_removed",
            "classification": "param_removal",
            "removed": sorted(removed),
        }

    # Both added and removed
    if len(added) == 1 and len(removed) == 1:
        old_p = old_params[next(iter(removed))]
        new_p = new_params[next(iter(added))]
        # Classify as a mechanical rename only when kind, required, AND annotation
        # all match — annotation equality rules out cases where the param was
        # restructured and a coincidentally same-shaped param appeared.
        if (
            old_p.get("kind") == new_p.get("kind")
            and old_p.get("required") == new_p.get("required")
            and old_p.get("annotation") == new_p.get("annotation")
        ):
            return {
                "method": method_name,
                "type": "param_renamed",
                "classification": "param_rename",
                "old_name": next(iter(removed)),
                "new_name": next(iter(added)),
            }

    return {
        "method": method_name,
        "type": "params_restructured",
        "classification": "others",
        "added": sorted(added),
        "removed": sorted(removed),
    }


def _compare_info(key: str, old: dict, new: dict) -> list[dict]:
    """Return a list of change records for one symbol (may span multiple methods)."""
    if "error" in new:
        return [
            {
                "key": key,
                "method": None,
                "type": "symbol_gone",
                "classification": "others",
                "detail": new["error"],
            }
        ]

    changes = []

    # Compare method signatures (classes)
    old_methods = old.get("methods", {})
    new_methods = new.get("methods", {})
    for method in sorted(set(old_methods) | set(new_methods)):
        if method not in old_methods:
            changes.append(
                {
                    "key": key,
                    "method": method,
                    "type": "method_added",
                    "classification": "non_breaking_addition",
                }
            )
            continue
        if method not in new_methods:
            changes.append(
                {
                    "key": key,
                    "method": method,
                    "type": "method_removed",
                    "classification": "others",
                }
            )
            continue
        old_p = old_methods[method].get("params", {})
        new_p = new_methods[method].get("params", {})
        change = _classify_param_change(method, old_p, new_p)
        if change:
            changes.append({"key": key, **change})

    # Compare plain function signatures
    old_sig = old.get("signature", {}).get("params", {})
    new_sig = new.get("signature", {}).get("params", {})
    if old_sig or new_sig:
        change = _classify_param_change("(function)", old_sig, new_sig)
        if change:
            changes.append({"key": key, **change})

    # Compare abstract methods list
    old_abs = set(old.get("abstract_methods", []))
    new_abs = set(new.get("abstract_methods", []))
    if old_abs != new_abs:
        added_abs = sorted(new_abs - old_abs)
        removed_abs = sorted(old_abs - new_abs)
        changes.append(
            {
                "key": key,
                "method": None,
                "type": "abstract_methods_changed",
                "classification": "others",
                "added": added_abs,
                "removed": removed_abs,
            }
        )

    return changes


def find_importing_files(module_path: str, symbol_name: str) -> list[str]:
    """Return relative paths of verl_omni files that import (module_path, symbol_name)."""
    import ast

    verl_omni_dir = REPO_ROOT / "verl_omni"
    results = []
    for py_file in sorted(verl_omni_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == module_path:
                for alias in node.names:
                    if alias.name == symbol_name:
                        results.append(str(py_file.relative_to(REPO_ROOT)))
                        break
    return results


def diff_snapshots(old_snap: dict, new_snap: dict) -> list[dict]:
    """Compare old and new signature snapshots, return all classified changes."""
    old_sigs = old_snap.get("signatures", {})
    new_sigs = new_snap.get("signatures", {})
    all_changes = []

    for key in sorted(set(old_sigs) | set(new_sigs)):
        if key not in old_sigs:
            continue  # new symbol added to verl_omni imports — not a drift, just a new import
        if key not in new_sigs:
            # Symbol we import is completely gone from upstream
            module_path, _, symbol = key.rpartition(".")
            all_changes.append(
                {
                    "key": key,
                    "method": None,
                    "type": "symbol_gone",
                    "classification": "others",
                    "detail": "symbol disappeared from snapshot",
                    "verl_omni_usages": find_importing_files(module_path, symbol),
                }
            )
            continue

        old_info = old_sigs[key]
        new_info = new_sigs[key]
        if old_info == new_info:
            continue

        changes = _compare_info(key, old_info, new_info)
        for change in changes:
            module_path, _, symbol = key.rpartition(".")
            change["verl_omni_usages"] = find_importing_files(module_path, symbol)
            # Include old/new snapshot data for AI-bound changes so the AI agent
            # prompt can show a concrete before/after without re-loading the snapshot.
            if change.get("classification") in ("param_removal", "others"):
                change["old_signature"] = old_info
                change["new_signature"] = new_info
            all_changes.append(change)

    return all_changes


def main() -> int:
    # Load committed snapshot
    if not SNAPSHOT_FILE.exists():
        print(f"ERROR: {SNAPSHOT_FILE} does not exist. Run generate_api_signatures.py first.")
        return 2

    with open(SNAPSHOT_FILE) as f:
        old_snap = json.load(f)

    if not old_snap.get("signatures"):
        print("Committed snapshot is empty — treating as no prior baseline. Generating fresh snapshot.")
        old_snap = {"signatures": {}}

    # Generate fresh signatures
    print("Generating fresh upstream signatures...")
    imports = collect_upstream_imports()
    signatures: dict = {}
    import_errors = 0

    for module_path, symbols in sorted(imports.items()):
        for symbol in sorted(symbols):
            key = f"{module_path}.{symbol}"
            info = get_symbol_info(module_path, symbol)
            if "error" in info:
                import_errors += 1
            else:
                signatures[key] = info

    if import_errors == len(imports):
        print("ERROR: All upstream imports failed. Are verl/vllm-omni installed?")
        return 2

    if import_errors:
        print(f"  Warning: {import_errors} symbols unresolved (partial upstream install)")

    new_snap = {**old_snap, "signatures": signatures}

    # Write fresh snapshot for the workflow to commit later
    with open(FRESH_FILE, "w") as f:
        json.dump(new_snap, f, indent=2)
        f.write("\n")

    # Diff
    print("Diffing against committed snapshot...")
    changes = diff_snapshots(old_snap, new_snap)

    if not changes:
        print("No drift detected.")
        CHANGES_FILE.write_text(json.dumps({"has_changes": False, "changes": []}, indent=2) + "\n")
        return 0

    ai_bound = [c for c in changes if c["classification"] in ("param_removal", "others")]
    non_ai = [c for c in changes if c["classification"] not in ("param_removal", "others")]

    print(f"\nDrift detected: {len(changes)} change(s)")
    print(f"  Non-AI (info/rename):  {len(non_ai)}")
    print(f"  AI-bound:              {len(ai_bound)}")
    for c in changes:
        print(f"  [{c['classification']:25s}] {c['key']} — {c['type']}")

    output = {
        "has_changes": True,
        "summary": f"{len(changes)} change(s): {len(non_ai)} non-ai, {len(ai_bound)} ai-bound",
        "changes": changes,
        "non_ai": non_ai,
        "ai_bound": ai_bound,
    }
    CHANGES_FILE.write_text(json.dumps(output, indent=2) + "\n")

    # Set GitHub Actions output for conditional workflow steps
    if gha_output := os.environ.get("GITHUB_OUTPUT"):
        with open(gha_output, "a") as f:
            f.write("has_changes=true\n")
            f.write(f"has_mechanical={'true' if non_ai else 'false'}\n")
            f.write(f"has_complex={'true' if ai_bound else 'false'}\n")

    return 1


if __name__ == "__main__":
    sys.exit(main())
