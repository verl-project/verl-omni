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

"""UND AR decode policies: per-turn token budget, engine-wake retry, and the replica health canary."""

from __future__ import annotations

import asyncio
import math
import traceback

#: Fraction of the *remaining* context a single UND turn may consume.
UND_TURN_FRACTION: float = 0.5

#: Substring of the rejection ``AsyncOmni.generate`` raises while any sleep tag is still set.
ENGINE_ASLEEP_MARKER: str = "partially or fully asleep"

#: Backoff for re-issuing a decode rejected by a still-waking engine (~62 s total; a never-waking
#: engine still fails loudly).
ENGINE_ASLEEP_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 30.0)

#: AR-replica health canary: trivial prompt, judged on length and token repetition.
UND_HEALTH_CANARY: str = "Name three fruits."
UND_HEALTH_MIN_WORDS: int = 6
UND_HEALTH_REPEAT_SHARE: float = 0.4

#: A UND turn must carry at least this many tokens before a repeat can be called a loop.
UND_LOOP_MIN_TOKENS: int = 6


def _iter_exception_chain(exc: BaseException):
    """Yield ``exc`` and everything it wraps (Python ``__cause__``/``__context__`` and Ray's ``cause``)."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        wrapped = getattr(current, "cause", None)
        if not isinstance(wrapped, BaseException):
            wrapped = current.__cause__ if isinstance(current.__cause__, BaseException) else current.__context__
        current = wrapped if isinstance(wrapped, BaseException) else None


def exception_search_text(exc: BaseException) -> str:
    """Flatten ``exc`` and everything it wraps into one searchable string."""
    parts: list[str] = []
    for current in _iter_exception_chain(exc):
        parts.append(f"{type(current).__name__}: {current}")
        parts.append(repr(current))
    try:
        parts.append("".join(traceback.format_exception(exc)))
    except Exception:  # noqa: BLE001 -- classification must never raise
        pass
    return "\n".join(parts)


def engine_asleep_retry_delay(attempt: int, message: str) -> float | None:
    """Seconds to wait before re-issuing a decode rejected by a still-waking engine, or ``None``."""
    if ENGINE_ASLEEP_MARKER not in message or attempt >= len(ENGINE_ASLEEP_BACKOFF_S):
        return None
    return ENGINE_ASLEEP_BACKOFF_S[attempt]


async def generate_retrying_engine_wake(attempt_fn, *, on_retry=None, sleep=None):
    """Await ``attempt_fn()``, re-issuing while the engine reports a still-waking sleep tag."""
    sleeper = sleep or asyncio.sleep
    attempt = 0
    while True:
        try:
            return await attempt_fn()
        except Exception as exc:  # noqa: BLE001 -- only the wake-window rejection is swallowed
            delay = engine_asleep_retry_delay(attempt, exception_search_text(exc))
            if delay is None:
                raise
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            await sleeper(delay)
            attempt += 1


def und_turn_max_tokens(*, context_used: int, max_context_tokens: int, fraction: float = UND_TURN_FRACTION) -> int:
    """Per-turn ``max_tokens`` for the UND lane: ``fraction`` of the remaining budget (``1.0`` disables the cap)."""
    remaining = int(max_context_tokens) - int(context_used)
    if remaining <= 0:
        return 0
    if fraction >= 1.0:
        return remaining
    if fraction <= 0.0:
        raise ValueError(f"und turn fraction must be in (0, 1], got {fraction!r}")
    return min(remaining, max(1, math.ceil(remaining * float(fraction))))


def und_health_verdict(
    text: str, *, min_words: int = UND_HEALTH_MIN_WORDS, repeat_share: float = UND_HEALTH_REPEAT_SHARE
) -> tuple[bool, float]:
    """Judge an AR-replica canary answer: returns ``(degenerate, top_token_share)``."""
    words = text.split()
    if not words:
        return True, 1.0
    counts: dict[str, int] = {}
    for word in words:
        key = word.lower().strip(",.")
        counts[key] = counts.get(key, 0) + 1
    top_share = max(counts.values()) / len(words)
    return len(words) < min_words or top_share > repeat_share, top_share
