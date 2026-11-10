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

"""Episode-owned generation and judgment state, independent of training and disk layout.

The loop owns ``episode_scope``; nested tools reuse its owner across awaits and
``asyncio.to_thread`` callbacks. Each concurrent episode needs a fresh state.
Scope exit releases references, leaves saved files alone, and rejects late writes.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Iterator, Sequence


@dataclass(frozen=True)
class GenerationRecord:
    """Immutable metadata for one generation call; contains no training tensors.

    Attributes:
        call_id: Unique call identifier within the owning episode.
        prompt: Stripped generation prompt.
        paths: Saved sample paths in backend order; recording does not create files.
        artifact_ids: Optional identifiers aligned one-to-one with ``paths``.
        backend: Generation service name, used for provenance and live-call counting.
        tool_stubbed: Whether the call used a stub rather than live generation.
    """

    call_id: str
    prompt: str
    paths: tuple[str, ...]
    artifact_ids: tuple[str | None, ...]
    backend: str = ""
    tool_stubbed: bool = False


@dataclass(frozen=True)
class JudgmentRecord:
    """Immutable feedback for a generation registered in the same episode.

    Attributes:
        call_id: Identifier of the generation being judged; unknown IDs are rejected.
        good_enough: Whether this judgment requests the sticky YES latch.
    """

    call_id: str
    good_enough: bool


@dataclass
class EpisodeState:
    """Own generation history, judgments, image lookup, and the episode stop latch.

    Attributes:
        episode_id: Optional rollout identity; independent episodes need distinct owners.
        artifact_relpath: Optional path relative to the run's artifact trees.
        user_prompt: Original user request shared with tool callbacks.

    ``record_judgment`` rejects unknown call IDs. Explicit image IDs resolve only
    within this owner and return ``None`` when unknown. Closing clears records and
    the latch without deleting saved files; subsequent record/latch/image writes
    raise ``RuntimeError``. The owning ``episode_scope`` closes on every exit.
    """

    episode_id: str | None = None
    artifact_relpath: str | None = None
    user_prompt: str | None = None
    _generations: dict[str, GenerationRecord] = field(default_factory=dict, init=False, repr=False)
    _judgments: list[JudgmentRecord] = field(default_factory=list, init=False, repr=False)
    _latest_image: str | None = field(default=None, init=False, repr=False)
    _good_enough: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False, compare=False)

    @property
    def generations(self) -> tuple[GenerationRecord, ...]:
        """Return records in generation order, without exposing mutable indexes."""
        with self._lock:
            return tuple(self._generations.values())

    @property
    def judgments(self) -> tuple[JudgmentRecord, ...]:
        """Return the judgments received by this episode."""
        with self._lock:
            return tuple(self._judgments)

    @property
    def latest_generation(self) -> GenerationRecord | None:
        """Return the latest generation call, if one exists."""
        with self._lock:
            return next(reversed(self._generations.values()), None)

    @property
    def good_enough(self) -> bool:
        """Whether a successful judge has requested the episode's YES latch."""
        with self._lock:
            return self._good_enough

    @property
    def closed(self) -> bool:
        """Whether the owning scope has released this episode."""
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("episode state is closed")

    def record_generation(
        self,
        *,
        prompt: str,
        paths: Sequence[str],
        call_id: str | None = None,
        artifact_ids: Sequence[str | None] = (),
        backend: str = "",
        tool_stubbed: bool = False,
    ) -> GenerationRecord:
        """Record one call, keeping all sample paths under the same owner."""
        with self._lock:
            self._require_open()
            paths = tuple(str(path) for path in paths)
            identifiers = tuple(artifact_ids) or (None,) * len(paths)
            if len(identifiers) != len(paths):
                raise ValueError("artifact IDs must align with generation paths")
            key = call_id or f"{self.episode_id or 'unbound'}:call{len(self._generations)}"
            if key in self._generations:
                raise ValueError(f"duplicate generation call {key!r}")
            record = GenerationRecord(key, prompt.strip(), paths, identifiers, backend, bool(tool_stubbed))
            self._generations[key] = record
            self._latest_image = next((p for p in paths if _existing_png(p)), self._latest_image)
            return record

    def record_judgment(self, call_id: str, *, good_enough: bool) -> JudgmentRecord:
        """Associate feedback with a real generation, retaining the sticky YES latch."""
        with self._lock:
            self._require_open()
            if call_id not in self._generations:
                raise ValueError(f"unknown generation call {call_id!r}")
            record = JudgmentRecord(call_id, bool(good_enough))
            self._judgments.append(record)
            self._good_enough |= record.good_enough
            return record

    def set_good_enough(self, reached: bool) -> None:
        """Adapt a legacy latch update without creating another state store."""
        with self._lock:
            self._require_open()
            self._good_enough = bool(reached)

    def set_latest_image(self, path: str | None) -> None:
        """Adapt a tool's direct image result before a full call record is available."""
        with self._lock:
            self._require_open()
            self._latest_image = str(path) if path else None

    def resolve_image(self, *, artifact_id: str | None = None, image_prompt: str | None = None) -> str | None:
        """Resolve an image strictly within this episode; explicit IDs never fall back."""
        with self._lock:
            records = tuple(reversed(self._generations.values()))
            if artifact_id:
                return next(
                    (
                        path
                        for record in records
                        for identifier, path in zip(record.artifact_ids, record.paths, strict=True)
                        if identifier == artifact_id and _existing_png(path)
                    ),
                    None,
                )
            if self._latest_image and _existing_png(self._latest_image):
                return self._latest_image
            latest = None
            for record in records:
                path = next((p for p in record.paths if _existing_png(p)), None)
                if path is None:
                    continue
                latest = latest or path
                if image_prompt and _prompts_match(record.prompt, image_prompt):
                    return path
            return latest

    def clear_artifacts(self) -> None:
        """Release artifact references for this episode only."""
        with self._lock:
            self._generations.clear()
            self._judgments.clear()
            self._latest_image = None

    def close(self) -> None:
        """Release runtime references and reject late results after scope exit."""
        with self._lock:
            self.clear_artifacts()
            self._good_enough = False
            self._closed = True


def _existing_png(path: str) -> bool:
    return str(path).endswith(".png") and Path(path).is_file()


def _prompts_match(first: str, second: str) -> bool:
    a, b = (re.sub(r"\s+", " ", text.strip().lower()) for text in (first, second))
    return bool(a and b and (a == b or a[:96] == b[:96] or a in b or b in a))


@dataclass(frozen=True)
class _EpisodeBinding:
    """Share the owner across callbacks while keeping legacy prompt overrides local."""

    state: EpisodeState
    user_prompt: str | None


_current_episode: ContextVar[_EpisodeBinding | None] = ContextVar("agentic_episode", default=None)


def current_episode(*, create: bool = False) -> EpisodeState | None:
    """Return this callback's owner; unbound standalone tools can request a local owner."""
    binding = _current_episode.get()
    if binding is None and create:
        binding = _EpisodeBinding(EpisodeState(), None)
        _current_episode.set(binding)
    return binding.state if binding else None


@contextmanager
def episode_scope(state: EpisodeState) -> Iterator[EpisodeState]:
    """Bind an owner across awaits/thread copies and always restore its parent."""
    state._require_open()
    parent = current_episode()
    # Reusing an owner must preserve a context-local legacy prompt override.
    binding = _current_episode.get() if parent is state else _EpisodeBinding(state, state.user_prompt)
    token = _current_episode.set(binding)
    try:
        yield state
    finally:
        _current_episode.reset(token)
        if parent is not state:
            state.close()
