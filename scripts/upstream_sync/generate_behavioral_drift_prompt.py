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
Generate AI agent prompt batches for behavioral-drift fixes.

Reads behavioral_changes.json (produced by check_upstream_behavioral.py) and
writes behavioral_drift_ai_prompt_batch{N}.md files. Each batch contains
at most BATCH_SIZE change sections.

Key improvements over the previous behavioral prompt generator:
  - Reads ALL comma-separated verl_omni_file paths (not just the first)
  - Injects per-criteria step-by-step procedures for the AI agent to follow
  - Batches into multiple files (max 3 per file)
  - Includes the upstream site-packages install path in the preamble

Usage:
    python scripts/upstream_sync/generate_behavioral_drift_prompt.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHANGES_FILE = REPO_ROOT / "behavioral_changes.json"
BATCH_SIZE = 3
MAX_FILE_CHARS = 6000


def get_upstream_path() -> str:
    """Return the site-packages directory where upstream deps are installed."""
    import site

    pkgs = site.getsitepackages()
    return pkgs[0] if pkgs else "/usr/local/lib/python3.10/site-packages"


def read_all_verl_omni_files(verl_omni_file_field: str) -> dict[str, str]:
    """Read ALL comma-separated paths. Returns {rel_path: content}."""
    paths = [p.strip() for p in verl_omni_file_field.split(",") if p.strip()]
    result = {}
    for rel in paths:
        full = REPO_ROOT / rel
        if full.exists():
            content = full.read_text()
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n... (truncated)"
            result[rel] = content
        else:
            result[rel] = f"(file not found: {rel})"
    return result


DIRECT_INHERITANCE_PROCEDURE = """### Procedure for `direct_inheritance`

Step 1 — Classify the upstream change from the diff below:
  (a) Method signature changed (params added/removed/renamed)
  (b) New non-abstract method added
  (c) New ABSTRACT method added (all subclasses must implement it)
  (d) Method removed or renamed
  (e) Behavior-only change inside an existing method body

Step 2 — Check which upstream methods the verl-omni subclass overrides.
  Read the full verl-omni counterpart file(s) shown below.

Step 3 — Apply the correct action:
  (a) Signature changed, method IS overridden → update the override signature to match.
      Signature changed, NOT overridden → inherited automatically; no edit needed.
  (b) New non-abstract method → inherited automatically; no edit needed unless
      it conflicts with diffusion specialisation.
  (c) New ABSTRACT method → MUST implement in the verl-omni subclass.
      Check if an equivalent method exists under a different name; if so, add an alias.
      If not, implement the method using diffusion-domain equivalent logic.
  (d) Method removed/renamed, IS overridden → remove or rename the override.
      Method NOT overridden → no edit needed.
  (e) Body change, NOT overridden → inherited automatically; no edit needed.
      Body change, IS overridden → evaluate whether the behavioral change must be mirrored.

Step 4 — Edit only if Step 3 requires it. Make one minimal change per affected site.
  If no edit needed: write in your summary: "no change needed — <reason>"
"""

STRUCTURAL_PARALLEL_PROCEDURE = """### Procedure for `structural_parallel`

These classes do NOT inherit from upstream — they are parallel diffusion-domain
implementations of the same concept. Do NOT copy LLM code verbatim.

Step 1 — Classify the upstream change from the diff below:
  (a) New metric or diagnostic logging added
  (b) New lifecycle hook or callback point
  (c) Config/parameter structure change
  (d) Performance optimization (caching, async, etc.)
  (e) Bug fix or correctness change in core logic

Step 2 — Determine whether a diffusion equivalent is needed:
  (a) New metric → find where verl-omni logs its equivalent metric.
      Add the same metric using the same simple_timer / logging pattern if applicable.
      If the metric is LLM-specific (e.g. token counts, KV cache hits), no edit needed.
  (b) New lifecycle hook → find the equivalent method in the verl-omni class.
      Add the hook at the equivalent lifecycle point using diffusion primitives.
  (c) Config change → check if Diffusion*Config has an equivalent field.
      Add it if missing, following the style of existing fields.
  (d) Performance optimization:
      LLM-specific (KV cache, token budgets) → no edit needed.
      General (batch sizing, timer, async pattern) → apply the diffusion equivalent.
  (e) Bug fix → determine if the same bug can occur in the diffusion path.
      If yes: apply the semantic equivalent fix. If no: no edit needed.

Step 3 — Edit or annotate. Make the minimal semantically equivalent change.
  If no change needed: write in your summary: "no change needed — <reason>"
"""

def criteria_procedure(criteria: str) -> str:
    if criteria == "direct_inheritance":
        return DIRECT_INHERITANCE_PROCEDURE
    elif criteria == "structural_parallel":
        return STRUCTURAL_PARALLEL_PROCEDURE
    else:
        return (
            "### Procedure\n"
            "Review the upstream diff and apply the minimal necessary change to the verl-omni counterpart.\n"
        )


def build_preamble(upstream_path: str) -> str:
    return f"""# Upstream Sync — Behavioral Drift: Verify and Adapt

You are helping maintain **verl-omni**, a multimodal RL training framework that
extends upstream **verl** (LLM training) and **vllm-omni** (diffusion rollout).

verl-omni classes do NOT always subclass upstream classes directly. They also implement
*parallel* diffusion-domain equivalents (e.g. DiffusionAgentLoopWorker mirrors
AgentLoopWorker) and inject into upstream registries at import time via _patch.py.

The upstream packages are installed at:
  verl:      {upstream_path}/verl/
  vllm_omni: {upstream_path}/vllm_omni/

You **may** use Read() on any file under those paths to look up implementations.

## Constraints
- Only modify the verl-omni counterpart file(s) listed in each section.
- Adapt semantically, not literally. Do NOT copy LLM code verbatim into diffusion classes.
- Preserve existing imports and class structure.
- If you determine no change is needed, do NOT edit the file.
  Write in your summary: "no change needed — <reason>"
"""


def make_summary_footer(batch_n: int) -> str:
    return f"""---
## Summary required

After completing all sections, write the following block to `behavioral_edits_summary_batch{batch_n}.md`:

```
### Changes made (batch {batch_n})
- Change 1: <file> — <what you did> OR "no change needed — <reason>"
- Change 2: ...
```
"""


def format_change(idx: int, change: dict, upstream_path: str) -> str:
    entry = change["entry"]
    verl_omni_files = read_all_verl_omni_files(entry.get("verl_omni_file", ""))
    upstream_content = change.get("upstream_content") or "(upstream content unavailable)"
    if len(upstream_content) > MAX_FILE_CHARS:
        upstream_content = upstream_content[:MAX_FILE_CHARS] + "\n... (truncated)"

    file_labels = ", ".join(f"`{p}`" for p in verl_omni_files)

    # --- Minimal dynamic header ---
    parts = [
        "---",
        f"## Change {idx}: `{entry['file']}` ({entry['criteria']})",
        f"**Upstream:** `{entry['upstream_repo']}` | "
        f"**Symbols:** {', '.join(f'`{s}`' for s in entry['symbols'])}",
        f"**verl-omni counterpart(s):** {file_labels}",
        f"**verl-omni symbols:** {', '.join(f'`{s}`' for s in entry.get('verl_omni_symbols', []))}",
    ]
    if entry.get("note"):
        parts.append(f"**Note:** {entry['note']}")
    parts.append("")

    # --- Static procedure block (criteria-specific, no dynamic values) ---
    # Kept before the diff/file content so the static text is cache-friendly.
    parts.append(criteria_procedure(entry["criteria"]))

    # --- Dynamic data section (diff and file contents at the end) ---
    parts += [
        "### Data",
        "",
        f"#### Upstream diff — `{entry['file']}`",
        "```diff",
        change.get("upstream_patch", "").rstrip(),
        "```",
        "",
        f"#### Current upstream file — `{entry['file']}`",
        "```python",
        upstream_content.rstrip(),
        "```",
    ]
    for rel_path, content in verl_omni_files.items():
        parts += [
            "",
            f"#### verl-omni counterpart — `{rel_path}`",
            "```python",
            content.rstrip(),
            "```",
        ]
    parts.append("")
    return "\n".join(parts)


def main() -> int:
    if not CHANGES_FILE.exists():
        print("No behavioral_changes.json found.")
        return 0
    data = json.loads(CHANGES_FILE.read_text())
    if not data.get("has_changes"):
        print("No behavioral changes to process.")
        return 0

    changes = data["changes"]
    upstream_path = get_upstream_path()
    batches = [changes[i : i + BATCH_SIZE] for i in range(0, len(changes), BATCH_SIZE)]

    written = []
    for n, batch in enumerate(batches, 1):
        sections = [build_preamble(upstream_path)]
        for i, ch in enumerate(batch, 1):
            sections.append(format_change(i, ch, upstream_path))
        sections.append(make_summary_footer(n))
        out_path = REPO_ROOT / f"behavioral_drift_ai_prompt_batch{n}.md"
        out_path.write_text("\n".join(sections))
        written.append(out_path)
        print(f"  Wrote {out_path.name} ({len(batch)} change(s))")

    print(f"Wrote {len(written)} batch file(s) for {len(changes)} change(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
