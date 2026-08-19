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
"""Collect dependency and runtime metadata for nightly regression artifacts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GIT_PIN_FILES = {
    "vllm_omni_git": _REPO_ROOT / ".github" / "vllm_omni_pin.txt",
    "verl_git": _REPO_ROOT / ".github" / "verl_pin.txt",
}
_ENV_PIN_KEYS = {
    "kernels_pip": "NIGHTLY_PIN_KERNELS",
    "fa3_fwd_pip": "NIGHTLY_PIN_FA3_FWD",
    "flash_attn_pip": "NIGHTLY_PIN_FLASH_ATTN",
    "transformers_pip": "NIGHTLY_PIN_TRANSFORMERS",
}


def _read_pin(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_version(module_name: str, attr: str = "__version__") -> str | None:
    if importlib.util.find_spec(module_name) is None:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return "import_failed"
    value = getattr(module, attr, None)
    return str(value) if value is not None else None


def _read_git_pins() -> dict[str, str | None]:
    return {key: _read_pin(path) for key, path in _GIT_PIN_FILES.items()}


def _read_env_pins() -> dict[str, str | None]:
    return {key: os.environ.get(env_key) for key, env_key in _ENV_PIN_KEYS.items()}


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def collect_env_metadata(*, attn_backend: str | None = None, rollout_attn_backend: str | None = None) -> dict[str, Any]:
    """Return pinned and installed versions plus FA3 availability flags."""
    try:
        from verl_omni.utils import diffusion_attention as da

        fa3_flags = {
            "fa3_available": da.fa3_available(),
            "actor_fa3_available": da.actor_fa3_available(),
            "rollout_fa3_available": da.rollout_fa3_available(),
        }
    except Exception as exc:
        fa3_flags = {"fa3_check_error": str(exc)}

    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "git_head": _git_head(),
        "pins": {**_read_git_pins(), **_read_env_pins()},
        "installed": {
            "torch": _module_version("torch"),
            "transformers": _distribution_version("transformers"),
            "vllm": _distribution_version("vllm"),
            "vllm_omni": _distribution_version("vllm-omni"),
            "verl": _distribution_version("verl"),
            "verl_omni": _distribution_version("verl-omni"),
            "kernels": _distribution_version("kernels"),
            "fa3_fwd": _distribution_version("fa3-fwd"),
            "flash_attn": _distribution_version("flash-attn"),
            "ray": _distribution_version("ray"),
            "diffusers": _distribution_version("diffusers"),
            "peft": _distribution_version("peft"),
            "accelerate": _distribution_version("accelerate"),
        },
        "attention": {
            "attn_backend": attn_backend or os.environ.get("NIGHTLY_ATTN_BACKEND"),
            "rollout_attn_backend": rollout_attn_backend or os.environ.get("NIGHTLY_ROLLOUT_ATTN_BACKEND"),
            **fa3_flags,
        },
        "nightly": {
            "deterministic_seed": os.environ.get("NIGHTLY_DETERMINISTIC_SEED"),
            "require_fa3": os.environ.get("NIGHTLY_REQUIRE_FA3"),
        },
    }
    return metadata


def write_env_metadata(output: Path, **kwargs: Any) -> dict[str, Any]:
    payload = collect_env_metadata(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write nightly environment metadata JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attn-backend", default=None)
    parser.add_argument("--rollout-attn-backend", default=None)
    args = parser.parse_args()
    payload = write_env_metadata(
        args.output.expanduser().resolve(),
        attn_backend=args.attn_backend,
        rollout_attn_backend=args.rollout_attn_backend,
    )
    print(f"[NIGHTLY] env metadata: {args.output}")
    print(
        "[NIGHTLY] attention:",
        payload.get("attention", {}),
        "transformers=",
        payload.get("installed", {}).get("transformers"),
    )


if __name__ == "__main__":
    main()
