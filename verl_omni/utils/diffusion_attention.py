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
"""FA3 availability checks and fallback for matched actor/rollout attention."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

ACTOR_FA3_BACKEND = "_flash_3_varlen_hub"
ACTOR_NATIVE_BACKEND = "native"
ROLLOUT_FA3_BACKEND = "FLASH_ATTN"
ROLLOUT_NATIVE_BACKEND = "TORCH_SDPA"
DIFFUSION_ATTENTION_ENV = "DIFFUSION_ATTENTION_BACKEND"


def actor_fa3_available() -> bool:
    return importlib.util.find_spec("kernels") is not None


def _cuda_supports_rollout_fa3() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        compute_capability = major + minor / 10.0
        return 8.0 <= compute_capability < 10.0
    except Exception:
        return False


def rollout_fa3_available() -> bool:
    if not _cuda_supports_rollout_fa3():
        return False
    for module_name in ("fa3_fwd_interface", "flash_attn"):
        if importlib.util.find_spec(module_name) is not None:
            return True
    return False


def fa3_available() -> bool:
    return actor_fa3_available() and rollout_fa3_available()


def fallback_fa3_if_unavailable(config: Any) -> None:
    """Downgrade explicit FA3 settings to native/SDPA when deps are missing."""
    attn_backend = config.actor_rollout_ref.model.get("attn_backend", ACTOR_FA3_BACKEND)
    if attn_backend != ACTOR_FA3_BACKEND:
        return

    if fa3_available():
        if config.actor_rollout_ref.rollout.get("name") == "vllm_omni":
            os.environ.setdefault(DIFFUSION_ATTENTION_ENV, ROLLOUT_FA3_BACKEND)
    else:
        logger.warning(
            "FA3 requested but unavailable for matched actor+rollout (kernels=%s, rollout_fa3=%s); "
            "falling back to actor=%s rollout=%s.",
            actor_fa3_available(),
            rollout_fa3_available(),
            ACTOR_NATIVE_BACKEND,
            ROLLOUT_NATIVE_BACKEND,
        )
        config.actor_rollout_ref.model.attn_backend = ACTOR_NATIVE_BACKEND
        os.environ[DIFFUSION_ATTENTION_ENV] = ROLLOUT_NATIVE_BACKEND

    # Inject into Ray runtime_env.env_vars to ensure propagation to Ray workers in distributed clusters
    if DIFFUSION_ATTENTION_ENV in os.environ:
        if "ray_kwargs" in config and "ray_init" in config.ray_kwargs:
            ray_init = config.ray_kwargs.ray_init
            if "runtime_env" not in ray_init:
                ray_init["runtime_env"] = {}
            runtime_env = ray_init["runtime_env"]
            if "env_vars" not in runtime_env:
                runtime_env["env_vars"] = {}
            runtime_env["env_vars"][DIFFUSION_ATTENTION_ENV] = os.environ[DIFFUSION_ATTENTION_ENV]
