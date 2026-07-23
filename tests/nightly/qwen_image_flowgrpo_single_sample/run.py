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
"""Nightly entrypoint for the Qwen-Image FlowGRPO single-sample regression."""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent

import install_debug_hooks

_DEBUG_ENV_PREFIXES = ("DEBUG_DUMP_",)
_DEBUG_ENV_NAMES = {
    "DEBUG_METRICS_JSONL",
}


def _copy_debug_env(env_vars: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(env_vars or {})
    for key, value in os.environ.items():
        if key in _DEBUG_ENV_NAMES or key.startswith(_DEBUG_ENV_PREFIXES):
            merged[key] = value
    python_path = merged.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    path_parts = [str(_SCRIPT_DIR), str(_REPO_ROOT)]
    path_parts.extend(part for part in python_path.split(os.pathsep) if part and part not in path_parts)
    merged["PYTHONPATH"] = os.pathsep.join(path_parts)
    return merged


def _patch_ray_init() -> None:
    """Install debug hooks in every Ray worker process without touching product code."""
    import ray

    original_ray_init = ray.init
    if getattr(original_ray_init, "__nightly_debug_wrapped__", False):
        return

    def ray_init_with_debug_hooks(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        runtime_env["env_vars"] = _copy_debug_env(runtime_env.get("env_vars"))
        runtime_env["worker_process_setup_hook"] = install_debug_hooks.install_debug_hooks
        kwargs["runtime_env"] = runtime_env
        return original_ray_init(*args, **kwargs)

    ray_init_with_debug_hooks.__nightly_debug_wrapped__ = True
    ray.init = ray_init_with_debug_hooks


def main() -> None:
    install_debug_hooks.install_debug_hooks()
    _patch_ray_init()
    runpy.run_module("verl_omni.trainer.main_diffusion", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
