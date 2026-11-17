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
"""Nightly entrypoint for L3 regressions on the v1 trainer.

Qwen-Image cases use ``verl_omni.trainer.main_diffusion_v1``.
MiniCPM-o and Qwen3-Omni cases use ``verl_omni.trainer.main_omni``.
"""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path

import install_debug_hooks

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent


_DEFAULT_TRAINER_MODULE = "verl_omni.trainer.main_diffusion_v1"
_ALLOWED_TRAINER_MODULES = frozenset(
    {
        "verl_omni.trainer.main_diffusion_v1",
        "verl_omni.trainer.main_omni",
    }
)

_FORWARDED_ENV_PREFIXES = ("DEBUG_DUMP_", "GENRM_OCR_")
_FORWARDED_ENV_NAMES = {
    "CUBLAS_WORKSPACE_CONFIG",
    "DEBUG_METRICS_JSONL",
    "NIGHTLY_DETERMINISTIC_SEED",
    "NIGHTLY_TRAINER_MODULE",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
}


def _copy_debug_env(env_vars: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(env_vars or {})
    for key, value in os.environ.items():
        if key in _FORWARDED_ENV_NAMES or key.startswith(_FORWARDED_ENV_PREFIXES):
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


def _trainer_module() -> str:
    """Return the v1 trainer module selected by ``NIGHTLY_TRAINER_MODULE``."""
    module = os.environ.get("NIGHTLY_TRAINER_MODULE", _DEFAULT_TRAINER_MODULE)
    if module not in _ALLOWED_TRAINER_MODULES:
        allowed = ", ".join(sorted(_ALLOWED_TRAINER_MODULES))
        raise ValueError(f"NIGHTLY_TRAINER_MODULE={module!r} is not one of: {allowed}")
    return module


def main() -> None:
    install_debug_hooks.install_debug_hooks()
    _patch_ray_init()
    module = _trainer_module()
    print(f"[NIGHTLY] trainer module: {module}", flush=True)
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
