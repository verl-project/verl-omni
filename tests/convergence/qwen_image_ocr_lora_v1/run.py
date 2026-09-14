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
"""L4 entrypoint: v1 trainer plus a metrics JSONL hook on the Ray driver worker."""

from __future__ import annotations

import os
import runpy
import sys
from collections.abc import Mapping
from pathlib import Path

import install_metrics_hooks

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_FORWARDED_ENV_NAMES = {
    "DEBUG_METRICS_JSONL",
    "L4_METRICS_JSONL",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
}


def _copy_l4_env(env_vars: Mapping[str, str] | None) -> dict[str, str]:
    merged = dict(env_vars or {})
    for key, value in os.environ.items():
        if key in _FORWARDED_ENV_NAMES or key.startswith("GENRM_OCR_"):
            merged[key] = value
    python_path = merged.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    path_parts = [str(_SCRIPT_DIR), str(_REPO_ROOT)]
    path_parts.extend(part for part in python_path.split(os.pathsep) if part and part not in path_parts)
    merged["PYTHONPATH"] = os.pathsep.join(path_parts)
    return merged


def _patch_ray_init() -> None:
    """Install the JSONL hook inside Ray workers (v1 Tracking lives on TaskRunner)."""
    import ray

    original_ray_init = ray.init
    if getattr(original_ray_init, "__l4_metrics_wrapped__", False):
        return

    def ray_init_with_metrics_hooks(*args, **kwargs):
        runtime_env = dict(kwargs.get("runtime_env") or {})
        runtime_env["env_vars"] = _copy_l4_env(runtime_env.get("env_vars"))
        runtime_env["worker_process_setup_hook"] = install_metrics_hooks.install_metrics_hooks
        kwargs["runtime_env"] = runtime_env
        return original_ray_init(*args, **kwargs)

    ray_init_with_metrics_hooks.__l4_metrics_wrapped__ = True
    ray.init = ray_init_with_metrics_hooks


def main() -> None:
    install_metrics_hooks.install_metrics_hooks()
    _patch_ray_init()
    runpy.run_module("verl_omni.trainer.main_diffusion_v1", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
