#!/usr/bin/env python3
"""
Generate the Cursor prompt for Track 2 behavioral drift fixes.

Reads behavioral_changes.json and writes track2_cursor_prompt.md — a single
prompt that covers all detected behavioral changes in one Cursor session.

Usage:
    python scripts/upstream_sync/generate_track2_prompt.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHANGES_FILE = REPO_ROOT / "behavioral_changes.json"
PROMPT_FILE = REPO_ROOT / "track2_cursor_prompt.md"


def read_verl_omni_file(rel_path: str) -> str:
    """Read a verl_omni source file. rel_path may be comma-separated (multi-file entry)."""
    # upstream_watch.yaml verl_omni_file can be "file1.py, file2.py" for multi-file entries
    first_path = rel_path.split(",")[0].strip()
    full = REPO_ROOT / first_path
    if full.exists():
        return full.read_text()
    return f"(file not found: {first_path})"


PREAMBLE = """\
# Upstream Sync — Track 2: Behavioral Drift Analysis

You are helping maintain **verl-omni**, a multimodal RL training framework that
extends the upstream **verl** framework (for LLMs) and **vllm-omni** (diffusion
rollout engine).

verl-omni classes do NOT always subclass upstream classes directly — they often
implement *parallel* diffusion-model equivalents of the LLM-side upstream classes.
For example, `DiffusionAgentLoopWorker` mirrors `AgentLoopWorker`, and
`RayFlowGRPOTrainer` mirrors `RayPPOTrainer`.

## Your task
For **each change below**, you must:
1. Read the upstream diff carefully.
2. Determine whether the upstream change requires an update to the corresponding
   verl-omni file (listed as "verl-omni counterpart").
3. If yes — apply the minimal necessary change to that verl-omni file.
4. If no — add a one-line comment `# upstream-sync: no change needed — <reason>`
   at the top of your analysis section and do NOT edit the file.

## Constraints (strictly enforced)
- Only modify files listed as "verl-omni counterpart" below.
- Do NOT refactor, rename, or clean up unrelated code.
- Preserve existing imports and class structure.
- The verl-omni implementation is specialized for **diffusion models**; adapt
  semantically, not literally. A timer added to an LLM method should be added
  to the equivalent diffusion method using the same `simple_timer` pattern.
- If a change involves an abstract method added to a base class, check whether
  the verl-omni subclass already implements it before adding it.

---
"""


def format_change(idx: int, change: dict) -> str:
    entry = change["entry"]
    upstream_repo = entry["upstream_repo"]
    upstream_file = entry["file"]
    symbols = entry["symbols"]
    criteria = entry.get("criteria", "")
    note = entry.get("note", "")
    verl_omni_file_label = entry.get("verl_omni_file", "")
    verl_omni_symbols = entry.get("verl_omni_symbols", [])

    upstream_patch = change["upstream_patch"]
    upstream_content = change.get("upstream_content") or "(upstream content unavailable)"
    verl_omni_content = read_verl_omni_file(verl_omni_file_label)

    # Truncate very long upstream content to keep prompt size manageable
    max_chars = 6000
    if len(upstream_content) > max_chars:
        upstream_content = upstream_content[:max_chars] + "\n... (truncated for brevity)"
    if len(verl_omni_content) > max_chars:
        verl_omni_content = verl_omni_content[:max_chars] + "\n... (truncated for brevity)"

    lines = [
        f"---",
        f"## Change {idx}: `{upstream_file}`",
        f"",
        f"**Upstream repo:** `{upstream_repo}`  ",
        f"**Watched symbols:** {', '.join(f'`{s}`' for s in symbols)}  ",
        f"**Criteria:** {criteria}  ",
        f"**verl-omni counterpart:** `{verl_omni_file_label}`  ",
        f"**verl-omni symbols:** {', '.join(f'`{s}`' for s in verl_omni_symbols)}  ",
    ]
    if note:
        lines.append(f"**Note:** {note}  ")

    lines += [
        f"",
        f"### Upstream diff (filtered to watched symbols)",
        f"```diff",
        upstream_patch.rstrip(),
        f"```",
        f"",
        f"### Current upstream file (`{upstream_file}`)",
        f"```python",
        upstream_content.rstrip(),
        f"```",
        f"",
        f"### verl-omni counterpart (`{verl_omni_file_label}`)",
        f"```python",
        verl_omni_content.rstrip(),
        f"```",
        f"",
    ]
    return "\n".join(lines)


def main() -> int:
    if not CHANGES_FILE.exists():
        print("No behavioral_changes.json found.")
        return 0

    with open(CHANGES_FILE) as f:
        data = json.load(f)

    if not data.get("has_changes"):
        print("No behavioral changes to process.")
        return 0

    changes = data["changes"]
    sections = [PREAMBLE]
    for i, change in enumerate(changes, 1):
        sections.append(format_change(i, change))

    sections.append(
        "---\n"
        "## Summary required\n\n"
        "After all edits, append a brief summary in this format:\n\n"
        "```\n"
        "### Changes made\n"
        "- Change 1: <file> — <what you did or 'no change needed — reason'>\n"
        "- Change 2: ...\n"
        "```\n"
    )

    prompt = "\n".join(sections)
    PROMPT_FILE.write_text(prompt)
    print(f"Wrote prompt ({len(prompt)} chars, {len(changes)} change(s)) to {PROMPT_FILE.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
