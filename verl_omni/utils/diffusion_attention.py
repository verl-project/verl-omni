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
from typing import Any

logger = logging.getLogger(__name__)

ACTOR_FA3_BACKEND = "_flash_3_varlen_hub"
ACTOR_NATIVE_BACKEND = "native"
ROLLOUT_SDPA_BACKEND = "TORCH_SDPA"

# Keep in sync with vllm-omni diffusion attention backends for FA train/rollout pairs.
FA3_ROLLOUT_BACKENDS = ("FLASH_ATTN", "FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB")
KERNELS_HUB_ROLLOUT_BACKENDS = ("FLASH_ATTN_HUB", "FLASH_ATTN_3_HUB")


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
    """True when local FA packages can back ``FLASH_ATTN`` rollout."""
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

    rollout_backend = config.actor_rollout_ref.rollout.get("rollout_attn_backend")
    if rollout_backend in KERNELS_HUB_ROLLOUT_BACKENDS:
        if actor_fa3_available():
            return
    elif fa3_available():
        return

    logger.warning(
        "FA3 requested but unavailable for matched actor+rollout (kernels=%s, rollout_fa3=%s); "
        "falling back to actor=%s, rollout=%s.",
        actor_fa3_available(),
        rollout_fa3_available(),
        ACTOR_NATIVE_BACKEND,
        ROLLOUT_SDPA_BACKEND,
    )
    config.actor_rollout_ref.model.attn_backend = ACTOR_NATIVE_BACKEND
    if rollout_backend in FA3_ROLLOUT_BACKENDS:
        config.actor_rollout_ref.rollout.rollout_attn_backend = ROLLOUT_SDPA_BACKEND


def validate_attention_consistency(config: Any) -> None:
    """Validate that rollout and training attention backends match.

    Called after ``fallback_fa3_if_unavailable`` so any FA3→native downgrade
    has already updated both config fields.

    Rules:
        - If the training engine is VeOmni, skip validation.
        - If ``attn_backend`` is ``_flash_3_varlen_hub``, rollout must be one of
          ``FA3_ROLLOUT_BACKENDS`` (default ``FLASH_ATTN_3_HUB`` for kernels FA3
          train/rollout consistency).
        - If ``attn_backend`` is ``native`` or ``_native_npu``, rollout must be
          ``TORCH_SDPA``.

    Raises:
        ValueError: If the rollout attention backend does not match the training
            attention backend.
    """
    actor_cfg = config.actor_rollout_ref.actor
    strategy = actor_cfg.get("strategy") if hasattr(actor_cfg, "get") else None
    if strategy == "veomni":
        logger.warning("strategy=veomni: attention consistency is not validated; ensure backends match manually.")
        return  # VeOmni engine manages its own attention independently

    model_cfg = config.actor_rollout_ref.model
    attn_backend = model_cfg.get("attn_backend", ACTOR_FA3_BACKEND)
    rollout_backend = config.actor_rollout_ref.rollout.get("rollout_attn_backend")

    if attn_backend == ACTOR_FA3_BACKEND:
        if rollout_backend in FA3_ROLLOUT_BACKENDS:
            return
        expected = ", ".join(FA3_ROLLOUT_BACKENDS)
    elif attn_backend in (ACTOR_NATIVE_BACKEND, "_native_npu"):
        expected = ROLLOUT_SDPA_BACKEND
    else:
        logger.warning("Unknown attn_backend=%r; skipping attention consistency check.", attn_backend)
        return

    if rollout_backend != expected:
        raise ValueError(
            f"Attention backend mismatch: attn_backend={attn_backend!r} requires "
            f"rollout_attn_backend={expected!r}, but got {rollout_backend!r}. "
            "Both must use the same attention implementation. "
            "Set rollout_attn_backend via --diffusion-attention-backend flag "
            "or in the rollout config."
        )
