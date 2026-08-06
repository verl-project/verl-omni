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
"""Attention backend helpers for GPU smoke / CI tests only."""

from __future__ import annotations

from verl_omni.utils.diffusion_attention import (
    ACTOR_FA3_BACKEND,
    ACTOR_NATIVE_BACKEND,
    ROLLOUT_SDPA_BACKEND,
    fa3_available,
)

ROLLOUT_LOCAL_FA_BACKEND = "FLASH_ATTN"


def resolve_smoke_attention_backends() -> tuple[str, str]:
    """Return ``(attn_backend, rollout_attn_backend)`` for GPU smoke tests.

    Product defaults pair kernels Hub FA3 (``FLASH_ATTN_3_HUB``), but smoke uses
    local ``FLASH_ATTN`` when FA packages are available so vllm-omni does not
    download Hub kernels at engine init. Falls back to native/SDPA when FA3 deps
    are missing (same idea as ``tests/workers/test_diffusers_fsdp_engine.py``).
    """
    if not fa3_available():
        return ACTOR_NATIVE_BACKEND, ROLLOUT_SDPA_BACKEND
    return ACTOR_FA3_BACKEND, ROLLOUT_LOCAL_FA_BACKEND
