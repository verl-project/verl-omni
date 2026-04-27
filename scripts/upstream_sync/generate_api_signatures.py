#!/usr/bin/env python3
"""
Bootstrap + daily refresh: generate .github/upstream_sync/api_signatures.json.

For every symbol imported from verl.*, vllm_omni.*, or vllm.* anywhere in
verl_omni/, captures:
  - For callable classes: __init__ signature + signatures of key public methods
  - For plain functions: their full signature
  - For non-callable objects (registries, dicts): their type only

Run with upstream deps installed:
    pip install "verl @ git+https://github.com/verl-project/verl.git@main"
    pip install "vllm-omni==0.18"
    python scripts/upstream_sync/generate_api_signatures.py

The daily upstream-sync bot runs this and diffs the output against the
previously committed api_signatures.json to detect Track 1 signature drift.

Exit codes:
  0 — all symbols resolved
  1 — some symbols could not be imported (upstream not fully installed)
"""

import ast
import importlib
import inspect
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VERL_OMNI_DIR = REPO_ROOT / "verl_omni"
OUTPUT_FILE = REPO_ROOT / ".github" / "upstream_sync" / "api_signatures.json"

UPSTREAM_PREFIXES = ("verl.", "vllm_omni.", "vllm.")

# Methods worth tracking on upstream classes (beyond __init__).
# These are the integration points most likely to have behavioral contracts.
TRACKED_METHODS = {
    "__init__",
    "run",
    "generate_sequences",
    "compute_score",
    "run_single",
    "forward",
    "generate",
    "register",        # registry APIs
    "load",
    "save",
}


def collect_upstream_imports() -> dict[str, set[str]]:
    """Return {module_path: {symbol_name, ...}} for all upstream imports in verl_omni/."""
    result: dict[str, set[str]] = {}

    for py_file in sorted(VERL_OMNI_DIR.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue
        try:
            tree = ast.parse(py_file.read_text())
        except (SyntaxError, OSError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not any(node.module.startswith(p) for p in UPSTREAM_PREFIXES):
                continue
            result.setdefault(node.module, set())
            for alias in node.names:
                result[node.module].add(alias.name)

    return result


def serialize_signature(sig: inspect.Signature) -> dict:
    """Extract parameter names, kinds, and whether they have defaults."""
    params = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        params[name] = {
            "kind": param.kind.name,
            "required": param.default is inspect.Parameter.empty
            and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ),
        }
    return {"params": params}


def get_symbol_info(module_path: str, symbol_name: str) -> dict:
    """Import symbol and return its signature info, or an error dict."""
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        return {"error": f"ImportError: {e}"}
    except Exception as e:
        return {"error": f"UnexpectedError on import: {e}"}

    obj = getattr(mod, symbol_name, None)
    if obj is None:
        return {"error": "AttributeError: symbol not found in module"}

    info: dict = {}

    if inspect.isclass(obj):
        info["kind"] = "class"
        methods: dict = {}
        for method_name in sorted(TRACKED_METHODS):
            method = getattr(obj, method_name, None)
            if method is None or not callable(method):
                continue
            # skip if inherited from builtins (object.__init__ etc.)
            if method_name == "__init__" and obj.__init__ is object.__init__:
                continue
            try:
                methods[method_name] = serialize_signature(inspect.signature(method))
            except (ValueError, TypeError):
                pass
        if methods:
            info["methods"] = methods
        abstracts = sorted(getattr(obj, "__abstractmethods__", set()))
        if abstracts:
            info["abstract_methods"] = abstracts

    elif callable(obj):
        info["kind"] = "function"
        try:
            info["signature"] = serialize_signature(inspect.signature(obj))
        except (ValueError, TypeError):
            info["kind"] = "function_no_sig"

    else:
        info["kind"] = type(obj).__name__

    return info


def diff_summary(old: dict, new: dict) -> list[str]:
    """Return human-readable lines describing what changed between two snapshots."""
    changes = []
    old_sigs = old.get("signatures", {})
    new_sigs = new.get("signatures", {})

    for key in sorted(set(old_sigs) | set(new_sigs)):
        if key not in old_sigs:
            changes.append(f"  NEW    {key}")
        elif key not in new_sigs:
            changes.append(f"  GONE   {key}")
        else:
            o, n = old_sigs[key], new_sigs[key]
            if o != n:
                changes.append(f"  DRIFT  {key}")
                # surface parameter-level changes for functions/methods
                for section in ("methods", "signature"):
                    if section in o or section in n:
                        o_sec = o.get(section, {})
                        n_sec = n.get(section, {})
                        if o_sec != n_sec:
                            changes.append(f"           {section}: {o_sec} → {n_sec}")

    return changes


def main() -> int:
    print(f"Collecting upstream imports from {VERL_OMNI_DIR.relative_to(REPO_ROOT)}/...")
    imports = collect_upstream_imports()
    total = sum(len(v) for v in imports.values())
    print(f"  {len(imports)} modules, {total} symbols")

    signatures: dict = {}
    errors: list[tuple[str, str]] = []

    for module_path, symbols in sorted(imports.items()):
        for symbol in sorted(symbols):
            key = f"{module_path}.{symbol}"
            info = get_symbol_info(module_path, symbol)
            if "error" in info:
                errors.append((key, info["error"]))
            else:
                signatures[key] = info

    output = {
        "_meta": {
            "description": (
                "Signatures of upstream symbols (verl/vllm_omni/vllm) imported by verl_omni. "
                "Regenerated by the daily upstream-sync bot. "
                "Track 1 diffs against this snapshot to detect signature drift."
            ),
            "generator": "scripts/upstream_sync/generate_api_signatures.py",
            "upstream_deps": {
                "verl": "git+https://github.com/verl-project/verl.git@main",
                "vllm_omni": "vllm-omni==0.18",
            },
        },
        "signatures": signatures,
    }

    # If an existing file exists, print a diff summary
    if OUTPUT_FILE.exists():
        try:
            with open(OUTPUT_FILE) as f:
                old = json.load(f)
            changes = diff_summary(old, output)
            if changes:
                print(f"\nChanges vs existing snapshot ({len(changes)} items):")
                for line in changes:
                    print(line)
            else:
                print("\nNo changes vs existing snapshot.")
        except (json.JSONDecodeError, KeyError):
            print("\n(Could not parse existing snapshot for diff)")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")

    print(f"\nWrote {len(signatures)} signatures to {OUTPUT_FILE.relative_to(REPO_ROOT)}")

    if errors:
        print(f"\n{len(errors)} symbols unresolved (upstream not fully installed?):")
        for key, err in errors[:15]:
            print(f"  {key}: {err}")
        if len(errors) > 15:
            print(f"  ... and {len(errors) - 15} more")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
