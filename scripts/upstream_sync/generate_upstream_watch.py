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
Bootstrap script: generate .github/upstream_sync/upstream_watch.yaml.

Detects symbols in verl_omni that depend on upstream (verl / vllm_omni / vllm) via:
  1. Direct subclassing   -- auto-detected via AST class-inheritance analysis
  2. Registry injection   -- hard-coded from _patch.py (cannot be auto-detected)
  3. Structural parallel  -- hard-coded duck-typed mirrors (cannot be auto-detected)

Run once, review the output, then commit. Re-run when adding new upstream-dependent
classes to verl_omni (or edit MANUAL_ENTRIES directly).

Usage:
    python scripts/upstream_sync/generate_upstream_watch.py
"""

import ast
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERL_OMNI_DIR = REPO_ROOT / "verl_omni"
OUTPUT_FILE = REPO_ROOT / ".github" / "upstream_sync" / "upstream_watch.yaml"

UPSTREAM_PREFIXES = ("verl.", "vllm_omni.", "vllm.")

UPSTREAM_REPO_MAP = {
    "verl.": "verl-project/verl",
    "vllm_omni.": "vllm-project/vllm-omni",
    "vllm.": "vllm-project/vllm",
}

# ---------------------------------------------------------------------------
# Entries that static AST analysis cannot detect:
#   - registry injection points (runtime dict/registry manipulation in _patch.py)
#   - structural parallels (duck-typed mirrors with no formal inheritance)
#
# Format: one entry per upstream file. If multiple criteria apply to the same
# file, list all symbols together — the bot fetches the file once.
# ---------------------------------------------------------------------------
MANUAL_ENTRIES = [
    # --- Registry injection points (from _patch.py) -------------------------
    # If the registry .register() API changes or the registry object is renamed,
    # the patch silently fails at runtime.
    {
        "upstream_repo": "verl-project/verl",
        "file": "verl/workers/rollout/replica.py",
        "symbols": ["RolloutReplicaRegistry"],
        "criteria": "registry_injection",
        "verl_omni_file": "verl_omni/_patch.py",
        "verl_omni_symbols": ["_patch_vllm_omni_replica"],
        "note": (
            "_patch_vllm_omni_replica calls RolloutReplicaRegistry.register('vllm_omni', ...); "
            "API or rename breaks the patch"
        ),
    },
    {
        "upstream_repo": "verl-project/verl",
        "file": "verl/workers/engine/base.py",
        "symbols": ["EngineRegistry"],
        "criteria": "registry_injection",
        "verl_omni_file": "verl_omni/_patch.py",
        "verl_omni_symbols": ["_patch_fsdp_diffusers_engine"],
        "note": "_patch_fsdp_diffusers_engine pops 'diffusion_model' key and re-registers DiffusersFSDPEngine",
    },
    {
        "upstream_repo": "verl-project/verl",
        "file": "verl/experimental/reward_loop/reward_manager/registry.py",
        "symbols": ["REWARD_MANAGER"],
        "criteria": "registry_injection",
        "verl_omni_file": "verl_omni/_patch.py",
        "verl_omni_symbols": ["_patch_visual_reward_manager"],
        "note": "_patch_visual_reward_manager replaces REWARD_MANAGER['visual'] with VisualRewardManager",
    },
    # --- Structural parallels (duck-typed mirrors, no formal subclassing) ----
    # DiffusionAgentLoopWorker mirrors the LLM-side AgentLoopWorker contract.
    # Upstream behavioral additions (e.g. timing wrappers in _compute_score,
    # new fields in AgentLoopMetrics) must be evaluated for diffusion parity.
    {
        "upstream_repo": "verl-project/verl",
        "file": "verl/experimental/agent_loop/agent_loop.py",
        "symbols": [
            "AgentLoopWorker",
            "AgentLoopMetrics",
            "_get_rollout_and_model_config",
            "DictConfigWrap",
            "_agent_loop_registry",
        ],
        "criteria": "structural_parallel",
        "verl_omni_file": "verl_omni/agent_loop/diffusion_agent_loop.py",
        "verl_omni_symbols": ["DiffusionAgentLoopWorker"],
        "note": (
            "DiffusionAgentLoopWorker is a parallel diffusion impl; "
            "upstream loop lifecycle or metric changes must be mirrored"
        ),
    },
    # RayFlowGRPOTrainer mirrors RayPPOTrainer's train-loop structure (checkpoint,
    # validation, reward extraction, timing). Upstream additions to the training
    # loop (new hooks, metric patterns) need evaluation for diffusion parity.
    {
        "upstream_repo": "verl-project/verl",
        "file": "verl/trainer/ppo/ray_trainer.py",
        "symbols": ["RayPPOTrainer"],
        "criteria": "structural_parallel",
        "verl_omni_file": "verl_omni/trainer/diffusion/ray_diffusion_trainer.py",
        "verl_omni_symbols": ["RayFlowGRPOTrainer"],
        "note": (
            "RayFlowGRPOTrainer mirrors RayPPOTrainer train-loop; "
            "upstream lifecycle or metric changes need diffusion equivalents"
        ),
    },
]


def get_upstream_repo(module: str) -> str:
    for prefix, repo in UPSTREAM_REPO_MAP.items():
        if module.startswith(prefix):
            return repo
    return "unknown"


def module_to_file(module: str) -> str:
    """Best-guess file path from a dotted module name.

    Returns the .py form; the bot should also try the __init__.py form
    (i.e. module/path -> module/path/__init__.py) when fetching from GitHub,
    since some modules are packages.
    """
    return module.replace(".", "/") + ".py"


def collect_upstream_imports(tree: ast.AST) -> dict[str, tuple[str, str]]:
    """Return {local_alias: (module, original_name)} for all upstream imports in a file."""
    result: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not any(node.module.startswith(p) for p in UPSTREAM_PREFIXES):
            continue
        for alias in node.names:
            local = alias.asname or alias.name
            result[local] = (node.module, alias.name)
    return result


def find_subclass_entries() -> list[dict]:
    """AST-scan verl_omni/ for classes that directly subclass upstream classes."""
    # key: (upstream_repo, file) -> accumulated entry dict
    by_file: dict[tuple[str, str], dict] = {}

    for py_file in sorted(VERL_OMNI_DIR.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue

        imports = collect_upstream_imports(tree)
        rel_path = str(py_file.relative_to(REPO_ROOT))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr

                if not base_name or base_name not in imports:
                    continue

                module, original_name = imports[base_name]
                upstream_file = module_to_file(module)
                repo = get_upstream_repo(module)
                key = (repo, upstream_file)

                if key not in by_file:
                    by_file[key] = {
                        "upstream_repo": repo,
                        "file": upstream_file,
                        "symbols": [],
                        "criteria": "direct_inheritance",
                        "verl_omni_file": rel_path,
                        "verl_omni_symbols": [],
                        "note": "",
                    }

                entry = by_file[key]
                if original_name not in entry["symbols"]:
                    entry["symbols"].append(original_name)
                if node.name not in entry["verl_omni_symbols"]:
                    entry["verl_omni_symbols"].append(node.name)
                # track unique verl_omni files that subclass from this upstream file
                existing_files = [f.strip() for f in entry["verl_omni_file"].split(",")]
                if rel_path not in existing_files:
                    entry["verl_omni_file"] = entry["verl_omni_file"] + ", " + rel_path

    return list(by_file.values())


def merge_all(manual: list[dict], auto: list[dict]) -> list[dict]:
    """Merge manual and auto entries by (upstream_repo, file), manual takes priority."""
    by_file: dict[tuple[str, str], dict] = {}

    # Manual entries first — they define the canonical entry for each file
    for entry in manual:
        key = (entry["upstream_repo"], entry["file"])
        if key not in by_file:
            by_file[key] = {**entry, "symbols": list(entry["symbols"])}
        else:
            for s in entry["symbols"]:
                if s not in by_file[key]["symbols"]:
                    by_file[key]["symbols"].append(s)

    # Auto-detected entries — add symbols not already present
    for entry in auto:
        key = (entry["upstream_repo"], entry["file"])
        if key not in by_file:
            by_file[key] = {**entry, "symbols": list(entry["symbols"])}
        else:
            existing = by_file[key]
            for s in entry["symbols"]:
                if s not in existing["symbols"]:
                    existing["symbols"].append(s)
            for s in entry.get("verl_omni_symbols", []):
                existing.setdefault("verl_omni_symbols", [])
                if s not in existing["verl_omni_symbols"]:
                    existing["verl_omni_symbols"].append(s)
            # upgrade criteria label if auto adds inheritance info to a manual entry
            if existing["criteria"] == "registry_injection" and entry["criteria"] == "direct_inheritance":
                existing["criteria"] = "registry_injection+direct_inheritance"

    return list(by_file.values())


def main() -> None:
    print(f"Scanning {VERL_OMNI_DIR} for upstream subclass relationships...")
    auto_entries = find_subclass_entries()
    print(f"  Auto-detected {len(auto_entries)} upstream file dependencies via subclassing")

    all_entries = merge_all(MANUAL_ENTRIES, auto_entries)
    print(f"  Total after merging with manual entries: {len(all_entries)} upstream files to watch")

    output = {
        "version": 1,
        "description": (
            "Upstream symbols that verl-omni subclasses, injects into, or structurally mirrors. "
            "Watched by the daily upstream-sync bot (Track 2: behavioral drift). "
            "Auto-generated — edit MANUAL_ENTRIES in generate_upstream_watch.py to add/remove entries."
        ),
        "watch": all_entries,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    print(f"\nWrote to {OUTPUT_FILE.relative_to(REPO_ROOT)}")
    print("\nEntries:")
    for e in all_entries:
        print(f"  [{e['criteria']:40s}] {e['file']}")
        print(f"    symbols:          {e['symbols']}")
        print(f"    verl_omni_file:   {e.get('verl_omni_file', '')}")


if __name__ == "__main__":
    main()
