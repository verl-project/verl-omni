#!/usr/bin/env python3
"""
Generate the Cursor prompt for Track 1 complex signature fixes.

"Complex" means the change cannot be handled by a word-boundary regex alone:
  - params_restructured : multiple params changed simultaneously (possible API redesign)
  - param_added_required: a new REQUIRED param was added (all callers must supply it)
  - symbol_gone         : symbol disappeared from its module (moved or deleted upstream)
  - abstract_methods_changed: abstract interface on a base class changed
  - method_removed      : a method that verl-omni calls no longer exists upstream
  - param_attrs_changed : same param names, different kinds/required status

Mechanical fixes (tier 1/3/4) have already been applied by apply_signature_fixes.py.
This script generates a prompt for the remaining cases.

Reads : remaining_complex.json
Writes: track1_complex_cursor_prompt.md

Usage:
    python scripts/upstream_sync/generate_track1_complex_prompt.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COMPLEX_FILE = REPO_ROOT / "remaining_complex.json"
PROMPT_FILE = REPO_ROOT / "track1_complex_cursor_prompt.md"
MAX_FILE_CHARS = 5000  # truncation limit per file shown to Cursor


def _format_params(info: dict) -> str:
    """Render a signature snapshot dict as a human-readable param list."""
    if not info:
        return "(unavailable)"
    methods = info.get("methods", {})
    sig = info.get("signature", {})
    lines = []
    if sig:
        params = sig.get("params", {})
        param_str = ", ".join(
            f"{n}" + ("?" if not v.get("required") else "") + f" [{v.get('kind', '')}]"
            for n, v in params.items()
        )
        lines.append(f"  (function)({param_str})")
    for method, minfo in sorted(methods.items()):
        params = minfo.get("params", {})
        param_str = ", ".join(
            f"{n}" + ("?" if not v.get("required") else "") + f" [{v.get('kind', '')}]"
            for n, v in params.items()
        )
        lines.append(f"  .{method}({param_str})")
    return "\n".join(lines) if lines else "(no tracked methods)"


def _read_verl_omni_files(usages: list[str]) -> dict[str, str]:
    """Return {relative_path: content} for each usage file."""
    result = {}
    for rel in usages:
        path = REPO_ROOT / rel
        if not path.exists():
            result[rel] = f"(file not found: {rel})"
            continue
        content = path.read_text()
        if len(content) > MAX_FILE_CHARS:
            content = content[:MAX_FILE_CHARS] + "\n... (truncated)"
        result[rel] = content
    return result


PREAMBLE = """\
# Upstream Sync — Track 1 Complex: Signature Drift Fixes

You are updating **verl-omni** to match upstream API changes that require
judgment beyond simple parameter renaming. The mechanical fixes have already
been applied (renamed params, optional additions, removed-param TODOs). These
remaining cases need deeper analysis.

## Change type glossary
| Type | Meaning | What to do |
|------|---------|------------|
| `params_restructured` | Multiple params changed at once — possible API redesign | Reconstruct call sites; may need to build a new config object |
| `param_added_required` | New REQUIRED param — all callers must supply a value | Find instantiation sites, supply an appropriate value |
| `symbol_gone` | Symbol disappeared from its module | Check if it moved; update the import; add TODO if not found |
| `abstract_methods_changed` | Abstract interface of a base class changed | Implement any new abstract methods in the verl-omni subclass |
| `method_removed` | A method verl-omni calls no longer exists upstream | Remove the call or find the replacement |
| `param_attrs_changed` | Same param names, different `required`/`kind` status | Adjust how the param is passed at call sites |

## Constraints (strictly enforced)
- Only modify the **verl-omni files listed** in each change section.
- Do NOT refactor, rename variables, or touch unrelated code.
- Do NOT add new imports unless strictly required by the fix.
- Preserve existing class hierarchy and diffusion-model specialisation.
- If you cannot determine the correct fix, add a `# TODO(upstream-sync): <reason>`
  comment at the affected line and do NOT guess.

---
"""


def format_change(idx: int, change: dict) -> str:
    key = change["key"]
    change_type = change.get("type", "unknown")
    method = change.get("method")
    usages = change.get("verl_omni_usages", [])
    old_info = change.get("old_signature", {})
    new_info = change.get("new_signature", {})

    # Specific detail fields
    added = change.get("added", [])
    removed = change.get("removed", [])
    changed_params = change.get("changed", [])
    detail = change.get("detail", "")

    files = _read_verl_omni_files(usages)

    parts = [
        f"---",
        f"## Change {idx}: `{key}`",
        f"",
        f"**Type:** `{change_type}`  ",
        f"**Affected method:** `{method or '(class-level)'}`  ",
        f"**verl-omni files to edit:** {', '.join(f'`{u}`' for u in usages) or '(none found — check manually)'}  ",
        f"",
    ]

    # Type-specific guidance
    if change_type == "params_restructured":
        parts += [
            f"**Old signature:**",
            f"```",
            _format_params(old_info),
            f"```",
            f"**New signature:**",
            f"```",
            _format_params(new_info),
            f"```",
            f"**Parameters added:** {added}  ",
            f"**Parameters removed:** {removed}  ",
            f"",
            f"Find all call sites that pass `{'`, `'.join(removed)}` and update them to use",
            f"the new API. If the new params require constructing a config/dataclass object,",
            f"inspect the upstream module for its definition and build it from existing data.",
        ]

    elif change_type == "param_added_required":
        parts += [
            f"**Old signature:**",
            f"```",
            _format_params(old_info),
            f"```",
            f"**New signature:**",
            f"```",
            _format_params(new_info),
            f"```",
            f"**New required parameter(s):** {added}  ",
            f"",
            f"Find every place the affected class/function is **instantiated or called**",
            f"in verl-omni and supply an appropriate value for `{'`, `'.join(added)}`.  ",
            f"Use the surrounding context to infer the correct value (config fields,",
            f"constructor arguments, existing attributes).",
        ]

    elif change_type == "symbol_gone":
        parts += [
            f"**Detail:** {detail}  ",
            f"",
            f"The symbol `{key.rpartition('.')[2]}` can no longer be imported from",
            f"`{key.rpartition('.')[0]}`. Possible causes:",
            f"1. It was moved to a different module (update the import path).",
            f"2. It was renamed (update both import and usage).",
            f"3. It was deleted (add `# TODO(upstream-sync): symbol removed — needs replacement`).",
            f"",
            f"Check the import statement in the listed verl-omni files and update accordingly.",
            f"If you cannot determine the new location, add a TODO comment and do NOT guess.",
        ]

    elif change_type == "abstract_methods_changed":
        parts += [
            f"**Abstract methods added upstream:** {added}  ",
            f"**Abstract methods removed upstream:** {removed}  ",
            f"",
            f"For each **added** abstract method: check whether the verl-omni subclass",
            f"already has an equivalent implementation (possibly under a different name).",
            f"If not, implement the method with the diffusion-model equivalent logic.",
            f"For each **removed** abstract method: check if the verl-omni subclass still",
            f"calls `super().<method>` and remove that call if so.",
        ]

    elif change_type == "method_removed":
        parts += [
            f"**Method removed:** `{method}`  ",
            f"",
            f"Find every call to `.<method>(...)` in the listed verl-omni files.",
            f"Either remove the call (if it was optional), find the replacement method",
            f"in the new upstream API, or add a TODO comment if unclear.",
        ]

    elif change_type == "param_attrs_changed":
        parts += [
            f"**Parameters with changed attributes:** {changed_params}  ",
            f"**Old signature:**",
            f"```",
            _format_params(old_info),
            f"```",
            f"**New signature:**",
            f"```",
            _format_params(new_info),
            f"```",
            f"",
            f"The parameter names are the same but their `required` or `kind` changed",
            f"(e.g. positional-or-keyword → keyword-only, or optional → required).",
            f"Find call sites and adjust how the parameter is passed.",
        ]

    else:
        parts += [
            f"**Detail:** {detail or '(no additional detail)'}  ",
            f"",
            f"Review the old and new signatures below and apply the minimal fix:",
            f"```",
            f"OLD:\n{_format_params(old_info)}",
            f"NEW:\n{_format_params(new_info)}",
            f"```",
        ]

    # Always append the file contents so Cursor has context
    if files:
        parts.append(f"\n### verl-omni file(s) to edit")
        for rel_path, content in files.items():
            parts += [
                f"\n**`{rel_path}`**",
                f"```python",
                content.rstrip(),
                f"```",
            ]

    parts.append("")
    return "\n".join(parts)


def main() -> int:
    if not COMPLEX_FILE.exists():
        print("No remaining_complex.json found — nothing to do.")
        return 0

    with open(COMPLEX_FILE) as f:
        data = json.load(f)

    complex_changes = data.get("complex", [])
    if not complex_changes:
        print("No complex changes to process.")
        return 0

    sections = [PREAMBLE]
    for i, change in enumerate(complex_changes, 1):
        sections.append(format_change(i, change))

    sections.append(
        "---\n"
        "## Required summary\n\n"
        "After all edits, append this block:\n\n"
        "```\n"
        "### Changes made\n"
        "- Change 1: <file>:<line> — <what you changed or 'TODO added — reason'>\n"
        "- Change 2: ...\n"
        "```\n"
    )

    prompt = "\n".join(sections)
    PROMPT_FILE.write_text(prompt)
    print(f"Wrote complex prompt ({len(prompt)} chars, {len(complex_changes)} case(s)) to {PROMPT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
