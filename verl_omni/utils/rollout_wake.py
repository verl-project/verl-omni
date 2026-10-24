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
"""One classifier for the rollout engine's transient sleep-mode VRAM failure.

A rollout wake re-maps the memory the engine backed up during its level-1 sleep, and it does so
through vLLM's sleep-mode allocator (``cuMemCreate``/``cuMemMap``; see ``cumem_allocator.cpp``).
That call needs a large *physically contiguous* region, so on a card shared with another tenant it
can fail even when the free total is ample, and the engine reports it as::

    Wake-up failed on Rank 0: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:163

This module is a leaf on purpose -- it imports nothing from ``verl_omni`` -- because two layers
need the same judgement and neither can import the other:

* :mod:`verl_omni.workers.engine_workers` retries the ``update_weights`` resume path, flushing the
  *actor's* cached blocks between attempts (``_resume_rollout_vram_safe``).
* :mod:`verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server` must retry *itself*, because
  ``wake_up`` is also reached directly by the trainer (``BagelCoRLTrainer._wake_rollout_replicas``)
  with no caller-side retry at all, so that path is unreachable from the retry helper.

Retrying is safe because a *failed* wake samples nothing -- the weights are still in the CPU backup
and the engine stays asleep -- so re-issuing the same request is idempotent.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

#: Retries *after* the first attempt. Four covers the measured ladder below.
WAKE_OOM_RETRIES: int = int(os.getenv("VERL_OMNI_WAKE_OOM_RETRIES", "4"))
WAKE_OOM_BACKOFF_S: tuple[float, ...] = (3.0, 8.0, 20.0, 40.0)


def wake_oom_backoff(attempt: int) -> float:
    """Seconds to wait before retry ``attempt`` (1-based). The last delay repeats."""
    return WAKE_OOM_BACKOFF_S[min(attempt - 1, len(WAKE_OOM_BACKOFF_S) - 1)]


def is_cuda_oom(exc: BaseException) -> bool:
    """True when ``exc`` or anything it wraps is a CUDA out-of-memory failure.

    The engine's rejection arrives wrapped -- ``ray.exceptions.RayTaskError`` around the stage's
    ``RuntimeError`` -- and ``RayTaskError.__str__`` renders the *task* repr, so the marker has to
    be searched through the whole cause chain, and through both spellings of the wrapper (Python's
    ``__cause__``/``__context__`` and Ray's own ``cause`` attribute).
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__}: {current}"
        if "out of memory" in text or "cumem_allocator" in text:
            return True
        wrapped = getattr(current, "cause", None)
        if not isinstance(wrapped, BaseException):
            wrapped = current.__cause__ if isinstance(current.__cause__, BaseException) else current.__context__
        current = wrapped if isinstance(wrapped, BaseException) else None
    return False


async def wake_with_oom_retry(
    wake_once: Callable[[], Awaitable[Any]],
    *,
    flush: Callable[[], Any] | None = None,
    label: str = "wake_up",
    retries: int = WAKE_OOM_RETRIES,
    backoff: Callable[[int], float] = wake_oom_backoff,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Any:
    """Await ``wake_once()``, retrying while it keeps failing with a CUDA OOM.

    ``flush`` runs before every *retry* (never before the first attempt), because the point of a
    retry is to hand the driver back the blocks the process is merely caching -- that is what makes
    a later ``cuMemCreate`` more likely to find a contiguous window than the one that failed. Any
    non-OOM error is re-raised immediately, and a persistent OOM still fails loud once the schedule
    is exhausted: the retries buy back a shared card's transient window, they do not paper over a
    genuine shortage.
    """
    for attempt in range(retries + 1):
        if attempt:
            if flush is not None:
                flush()
            delay = backoff(attempt)
            logger.warning(
                "%s failed with CUDA OOM (attempt %d/%d); flushed the engine cache and retrying in %.0fs",
                label,
                attempt,
                retries,
                delay,
            )
            await sleep(delay)
        try:
            return await wake_once()
        except Exception as exc:  # noqa: BLE001 -- only the transient VRAM failure is retried
            if not is_cuda_oom(exc) or attempt >= retries:
                raise
    raise AssertionError("unreachable: the loop returns or raises on its last attempt")  # pragma: no cover
