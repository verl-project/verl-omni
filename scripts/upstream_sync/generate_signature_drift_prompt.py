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
Generate AI agent prompt batches for signature-drift fixes.

Reads remaining_for_ai.json (produced by apply_signature_fixes.py) and
writes signature_drift_ai_prompt_batch{N}.md files. Each batch file contains
at most BATCH_SIZE change sections plus preamble and summary footer.

The preamble tells the AI agent exactly where upstream packages are installed so it
can use Read() to look up moved symbols or verify parameter names directly from
the source.

Usage:
    python scripts/upstream_sync/generate_signature_drift_prompt.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
AI_QUEUE_FILE = REPO_ROOT / "remaining_for_ai.json"
BATCH_SIZE = 3


def get_upstream_path() -> str:
    """Return the site-packages directory where upstream deps are installed."""
    import site

    pkgs = site.getsitepackages()
    return pkgs[0] if pkgs else "/usr/local/lib/python3.10/site-packages"


def render_params(snapshot_info: dict) -> str:
    """Render old/new signature snapshot as readable text.

    For classes: show each tracked method with its params.
    For functions: show the function params.
    Format: '  .method_name(param [KIND, required])' or '  (function)(param [KIND])'
    If no info: return '(unavailable)'
    """
    if not snapshot_info:
        return "(unavailable)"
    methods = snapshot_info.get("methods", {})
    sig = snapshot_info.get("signature", {})
    lines = []
    if sig:
        params = sig.get("params", {})
        param_str = ", ".join(
            f"{n}" + ("?" if not v.get("required") else "") + f" [{v.get('kind', '')}]" for n, v in params.items()
        )
        lines.append(f"  (function)({param_str})")
    for method, minfo in sorted(methods.items()):
        params = minfo.get("params", {})
        param_str = ", ".join(
            f"{n}" + ("?" if not v.get("required") else "") + f" [{v.get('kind', '')}]" for n, v in params.items()
        )
        lines.append(f"  .{method}({param_str})")
    return "\n".join(lines) if lines else "(no tracked methods)"


def hint_sentence(change: dict) -> str:
    """One-sentence description of our classification guess."""
    cls = change.get("classification", "")
    if cls == "param_rename":
        return f"our snapshot suggests a rename: `{change.get('old_name', '?')}` → `{change.get('new_name', '?')}`"
    elif cls == "param_removal":
        removed = change.get("removed", [])
        return f"our snapshot suggests removal of: `{'`, `'.join(removed)}`"
    elif cls == "non_breaking_addition":
        added = change.get("added", [])
        return f"new optional parameter(s) added: `{'`, `'.join(added)}` — likely non-breaking"
    else:
        t = change.get("type", "structural change")
        return f"complex change (`{t}`) — verify from upstream source before editing"


def build_preamble(upstream_path: str) -> str:
    """Static header injected once per batch file."""
    return f"""# Upstream Sync — Signature Drift: Verify and Fix

You are updating **verl-omni** to stay compatible with upstream API changes.
The upstream packages are installed at:

  verl:      {upstream_path}/verl/
  vllm_omni: {upstream_path}/vllm_omni/

You **may** use Read() on any file under those paths to look up types,
locate moved symbols, or verify parameter names and annotations. Use it.

## Constraints
- Only modify the verl-omni files listed in each change section.
- Do NOT refactor, rename variables, or touch unrelated code.
- Do NOT add new imports unless strictly required by the fix.
- Preserve existing class hierarchy and diffusion-model specialisation.
- If you cannot determine the correct action with confidence, add a TODO comment:
    # TODO(upstream-sync): <what you found and why you are unsure>
  and do NOT guess.
"""


def make_summary_footer(batch_n: int) -> str:
    return f"""---
## Summary of edits made

After completing all edits above, write the following block to `ai_edits_summary_batch{batch_n}.md`:

```
### Edits made (batch {batch_n})
- Fix 1: <file>:<line> — <what you changed> OR "no change needed — <reason>"
- Fix 2: ...
```
"""


def format_change(idx: int, change: dict, upstream_path: str) -> str:
    """Produce one change section: dynamic header, procedure, dynamic data."""
    key = change.get("key", "")
    method = change.get("method", "(class-level)")
    usages = change.get("verl_omni_usages", [])
    old_info = change.get("old_signature", {})
    new_info = change.get("new_signature", {})

    module_path = key.rpartition(".")[0]
    module_file = module_path.replace(".", "/") + ".py"

    files_content = {}
    for rel in usages:
        path = REPO_ROOT / rel
        if path.exists():
            content = path.read_text()
            if len(content) > 5000:
                content = content[:5000] + "\n... (truncated)"
            files_content[rel] = content
        else:
            files_content[rel] = f"(file not found: {rel})"

    file_list = ", ".join(f"`{u}`" for u in usages) if usages else "(none found — check manually)"

    # --- Minimal dynamic header ---
    parts = [
        "---",
        f"## Change {idx}: `{key}` (`{method}`)",
        f"**Hint:** {hint_sentence(change)}",
        f"**Files to edit:** {file_list}",
        "",
    ]

    # --- Procedure block ---
    parts += [
        "### Procedure",
        "",
        "**Step 1 — Read the upstream source**",
        "Using the upstream module path in the Data section below, read the source file.",
        "Find the symbol named in the Data section and note its full parameter list and type annotations.",
        "Check whether anything was moved, renamed, or removed.",
        "",
        "**Step 2 — Compare signatures**",
        "Review the old and new signatures in the Data section below.",
        "Identify exactly what changed: params added, removed, renamed,",
        "kind changes (e.g. positional→keyword-only), or required-status changes.",
        "",
        "**Step 3 — Categorize**",
        "  (A) New optional param only → call sites unchanged",
        "  (B) Param renamed → update `old=` → `new=` at keyword-arg call sites",
        "  (C) Param removed → remove argument from call sites",
        "  (D) New required param → supply a value at all call sites",
        "  (E) Multiple params restructured → reconstruct call sites",
        "  (F) Symbol moved / renamed / deleted upstream → update import",
        "",
        "**Step 4 — Edit**",
        '  (A) No edit needed → write in summary: "no change — new optional param"',
        "  (B) Rename `old_name=` → `new_name=` at every keyword-arg call site.",
        "      Positional usage: add `# positional: maps to new_name (was old_name)`",
        "  (C) Remove the argument. If stored as an attribute, trace and remove dependents.",
        "  (D) Supply a value inferred from context (config fields, self.* attributes).",
        "      Cannot infer: `# TODO(upstream-sync): new required param NAME — infer value`",
        "  (E) Reconstruct each call using the new parameter structure.",
        "  (F) Update the import. Symbol gone entirely:",
        "      `# TODO(upstream-sync): SYMBOL removed from upstream — needs replacement`",
        "",
    ]

    # --- Dynamic data section (at the end, after all static text) ---
    parts += [
        "### Data",
        "",
        f"**Upstream module:** `{upstream_path}/{module_file}`",
        "",
        "**Old signature (committed snapshot):**",
        "```",
        render_params(old_info),
        "```",
        "",
        "**New signature (current upstream):**",
        "```",
        render_params(new_info),
        "```",
    ]

    if files_content:
        parts.append("")
        parts.append("**verl-omni file(s) to edit:**")
        for rel_path, content in files_content.items():
            parts += [
                "",
                f"`{rel_path}`:",
                "```python",
                content.rstrip(),
                "```",
            ]

    parts.append("")
    return "\n".join(parts)


def main() -> int:
    if not AI_QUEUE_FILE.exists():
        print("No remaining_for_ai.json — nothing to do.")
        return 0
    data = json.loads(AI_QUEUE_FILE.read_text())
    changes = data.get("changes", [])
    if not changes:
        print("No changes queued for AI agent.")
        return 0

    upstream_path = get_upstream_path()
    batches = [changes[i : i + BATCH_SIZE] for i in range(0, len(changes), BATCH_SIZE)]

    written = []
    for n, batch in enumerate(batches, 1):
        sections = [build_preamble(upstream_path)]
        for i, ch in enumerate(batch, 1):
            sections.append(format_change(i, ch, upstream_path))
        sections.append(make_summary_footer(n))
        out_path = REPO_ROOT / f"signature_drift_ai_prompt_batch{n}.md"
        out_path.write_text("\n".join(sections))
        written.append(out_path)
        print(f"  Wrote {out_path.name} ({len(batch)} change(s))")

    print(f"Wrote {len(written)} batch file(s) for {len(changes)} change(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
