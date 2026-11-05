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

"""Opt-in parameter identity for one atomic update; no tensor math is replaced.

This covers registered engine writers and explicit worker lifecycle access.
It cannot detect arbitrary writes through ``tensor.data``. Engines overriding
an unannotated writer must not enable the optimization.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import uuid
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps

_ACCESS_SCOPE: ContextVar[_AccessScope | None] = ContextVar("parameter_access_scope", default=None)
_DELEGATED_SCOPE: ContextVar[_AccessScope | None] = ContextVar("parameter_delegated_scope", default=None)


class _AccessScope:
    def __init__(self, tracker):
        self.tracker = tracker
        self.active = True
        self.tasks = set()
        self.managed_depth = 0
        self.thread_lock = tracker._delegated_lock


def parameter_to_thread(function, *args):
    """Start a worker thread whose lifetime stays inside parameter access."""
    scope = _ACCESS_SCOPE.get()
    if scope is not None and (not scope.active or not scope.managed_depth):
        raise RuntimeError("Parameter worker requires a live async access guard")

    def invoke():
        if scope is None:
            return function(*args)
        with scope.thread_lock:
            if not scope.active or scope.tracker._poisoned:
                raise RuntimeError("Parameter worker access has expired or is poisoned")
            return function(*args)

    async def run():
        token = _DELEGATED_SCOPE.set(scope)
        try:
            return await asyncio.to_thread(invoke)
        finally:
            _DELEGATED_SCOPE.reset(token)

    task = asyncio.create_task(run())
    if scope is None:
        return task
    scope.tasks.add(task)
    # Callers can cancel their wait without cancelling the task that owns the
    # thread; cancelling an asyncio.to_thread task does not stop its thread.
    handle = asyncio.shield(task)

    def observe(done):
        if not done.cancelled():
            done.exception()

    handle.add_done_callback(observe)
    return handle


async def _drain_parameter_tasks(scope):
    cancelled = None
    failure = None
    observed: set[asyncio.Task] = set()
    while tasks := scope.tasks - observed:
        for task in tasks:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError as exc:
                    cancelled = exc
                except BaseException:
                    break
            if task.cancelled():
                failure = failure or RuntimeError("A parameter worker task was cancelled")
            else:
                error = task.exception()
                if failure is None:
                    failure = error
            observed.add(task)
    return cancelled, failure


@dataclass(frozen=True)
class ParameterVersion:
    incarnation: str
    revision: int


class ParameterRevision:
    def __init__(self):
        self._incarnation = uuid.uuid4().hex
        self._revision = 0
        self._lock = threading.RLock()
        self._delegated_lock = threading.RLock()
        self._owner = None
        self._depth = 0
        self._poisoned = False
        self._writer = None
        self._access_depth = 0
        self._external_access_depth = 0
        self._update = None
        self._updates = 0
        self._consumed = set()
        self._reservation = None

    def _reject_reserved(self):
        if self._reservation is not None:
            raise RuntimeError("An update is reserved")

    def reserve_update(self, expected: ParameterVersion, cohort_id: str):
        """Fence parameter work until the coordinator commits or aborts."""
        with self._exclusive():
            self._reject_reserved()
            self._validate_update(expected, cohort_id)
            self._reservation = (expected, cohort_id)

    def abort_reserved_update(self, expected: ParameterVersion, cohort_id: str):
        """Release only a matching reservation that has not entered update."""
        with self._exclusive():
            if self._reservation != (expected, cohort_id):
                raise ValueError("Reservation does not match")
            self._reservation = None

    @contextmanager
    def commit_reserved_update(self, expected: ParameterVersion, cohort_id: str):
        """Transfer a matching reservation to the tracked update atomically."""
        with self._exclusive():
            if self._reservation != (expected, cohort_id):
                raise ValueError("Reservation does not match")
            self._reservation = None
            with self.update(expected, cohort_id):
                yield

    @contextmanager
    def _exclusive(self):
        # A synchronous RLock alone also admits another asyncio task on the
        # same thread while a weight export is awaiting an RPC.
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        owner = (threading.get_ident(), task)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent parameter access")
        entered = False
        try:
            if self._owner is not None and self._owner != owner:
                raise RuntimeError("Concurrent async parameter access")
            if self._poisoned:
                raise RuntimeError("Partial actor operation; use a fresh engine")
            self._owner = owner
            self._depth += 1
            entered = True
            yield
        finally:
            if entered:
                self._depth -= 1
                if not self._depth:
                    self._owner = None
            self._lock.release()

    def version(self):
        with self._exclusive():
            if self._update is not None or self._writer is not None or self._access_depth:
                raise RuntimeError("Cannot prepare a version during parameter work")
            return ParameterVersion(self._incarnation, self._revision)

    @contextmanager
    def mutation(self, kind):
        if kind not in {"initialize", "load", "optimizer", "restore"}:
            raise ValueError("Unknown parameter writer")
        with self._exclusive():
            self._reject_reserved()
            if self._writer is not None or self._external_access_depth:
                raise RuntimeError("Nested parameter writer or active lifecycle operation")
            if self._update is not None and (kind != "optimizer" or self._updates):
                raise RuntimeError("Expected exactly one optimizer call during the cohort")
            self._writer = kind
            self._revision += 1  # Invalidate tickets before a possibly partial write.
            try:
                yield
                if self._update is not None:
                    self._updates += 1
            except BaseException:
                self._poisoned = True
                raise
            finally:
                self._writer = None

    @contextmanager
    def access(self, *, internal=False):
        with self._exclusive():
            self._reject_reserved()
            if not internal and (self._update is not None or self._writer is not None):
                raise RuntimeError("External lifecycle operation during an atomic update")
            self._access_depth += 1
            self._external_access_depth += not internal
            scope = _AccessScope(self)
            token = _ACCESS_SCOPE.set(scope)
            try:
                yield
            finally:
                _ACCESS_SCOPE.reset(token)
                scope.active = False
                self._access_depth -= 1
                self._external_access_depth -= not internal

    @contextmanager
    def device_transition(self):
        """Fail closed if moving model or optimizer state is interrupted."""
        delegated = _DELEGATED_SCOPE.get()
        if delegated is not None and delegated.tracker is self:
            self._reject_reserved()
            if not delegated.active or self._poisoned:
                raise RuntimeError("Parameter worker access has expired")
            try:
                yield
            except BaseException:
                self._poisoned = True
                raise
            return
        with self._exclusive():
            self._reject_reserved()
            if not self._delegated_lock.acquire(blocking=False):
                raise RuntimeError("Concurrent parameter thread")
            try:
                try:
                    yield
                except BaseException:
                    self._poisoned = True
                    raise
            finally:
                self._delegated_lock.release()

    def assert_update_pending(self):
        """Require an active scoped update before its optimizer write."""
        with self._exclusive():
            if self._update is None or self._updates != 0 or self._writer is not None:
                raise RuntimeError("No pending atomic update before the optimizer write")

    def assert_update_complete(self):
        with self._exclusive():
            if self._update is None or self._updates != 1 or self._writer is not None:
                raise RuntimeError("The complete cohort must perform exactly one optimizer call")

    @contextmanager
    def preflight_update(self, expected: ParameterVersion, cohort_id: str):
        """Hold parameter identity stable while all ranks validate an update."""
        with self._exclusive():
            self._reject_reserved()
            self._validate_update(expected, cohort_id)
            yield

    def _validate_update(self, expected, cohort_id):
        if self._update is not None or self._writer is not None or self._access_depth:
            raise RuntimeError("An actor operation is already active")
        if expected != ParameterVersion(self._incarnation, self._revision):
            raise ValueError("Stale parameter revision or different worker incarnation")
        if not isinstance(cohort_id, str) or not cohort_id or cohort_id in self._consumed:
            raise ValueError("Missing or consumed cohort identity")

    @contextmanager
    def update(self, expected: ParameterVersion, cohort_id: str):
        with self._exclusive():
            self._reject_reserved()
            self._validate_update(expected, cohort_id)
            self._consumed.add(cohort_id)
            self._update, self._updates = cohort_id, 0
            try:
                yield
                self.assert_update_complete()
            except BaseException:
                self._poisoned = True
                raise
            finally:
                self._update = None


def parameter_writer(kind, *, preflight_backend=False):
    """Annotate the actual backend writer; disabled unless explicitly enabled."""

    def decorate(method):
        @wraps(method)
        def wrapped(engine, *args, **kwargs):
            tracker = getattr(engine, "_parameter_revision", None)
            if tracker is not None and preflight_backend:
                # A Python argument error happens before the backend can write.
                backend = getattr(super(ParameterTrackingMixin, engine), method.__name__)
                try:
                    signature = inspect.signature(backend)
                except (TypeError, ValueError):
                    signature = None
                if signature is not None:
                    signature.bind(*args, **kwargs)
            context = tracker.mutation(kind) if tracker is not None else nullcontext()
            with context:
                return method(engine, *args, **kwargs)

        wrapped.parameter_writer_kind = kind  # type: ignore[attr-defined]
        return wrapped

    return decorate


def parameter_access(engine_path="engine", *, internal=False):
    """Keep an entire synchronous/async lifecycle operation outside updates.

    The engine path is relative to the method's owner. ``internal`` is reserved
    for native training entries and device moves within a qualified update.
    """

    def decorate(method):
        @contextmanager
        def context(owner):
            engine = owner
            for attr in engine_path.split(".") if engine_path else ():
                engine = getattr(engine, attr)
            tracker = getattr(engine, "_parameter_revision", None)
            if tracker is not None:
                with tracker.access(internal=internal):
                    yield
            else:
                token = _ACCESS_SCOPE.set(None)
                try:
                    yield
                finally:
                    _ACCESS_SCOPE.reset(token)

        if inspect.iscoroutinefunction(method):

            @wraps(method)
            async def wrapped(owner, *args, **kwargs):
                with context(owner):
                    scope = _ACCESS_SCOPE.get()
                    if scope is not None:
                        scope.managed_depth += 1
                    try:
                        try:
                            result = await method(owner, *args, **kwargs)
                        except BaseException:
                            if scope is not None:
                                await _drain_parameter_tasks(scope)
                            raise
                        if scope is not None:
                            cancelled, failure = await _drain_parameter_tasks(scope)
                            if cancelled is not None:
                                raise cancelled
                            if failure is not None:
                                raise failure
                        return result
                    finally:
                        if scope is not None:
                            scope.managed_depth -= 1
        else:

            @wraps(method)
            def wrapped(owner, *args, **kwargs):
                with context(owner):
                    return method(owner, *args, **kwargs)

        return wrapped

    return decorate


class ParameterTrackingMixin:
    """Track Omni-owned writer calls while delegating backend math unchanged.

    Worker lifecycle coverage is also required before reuse can be enabled.
    Constructing an engine does not opt in to tracking or probability reuse.
    """

    def enable_parameter_tracking(self):
        for method, kind in (
            ("initialize", "initialize"),
            ("load_checkpoint", "load"),
            ("optimizer_step", "optimizer"),
        ):
            writer = getattr(self, method)
            if getattr(writer, "parameter_writer_kind", None) != kind:
                raise ValueError(f"Untracked parameter writer: {type(self).__name__}.{method}")
            # A native dependency that starts tracking these writers must be
            # integrated deliberately, rather than nesting two write guards.
            backend = getattr(super(), method)
            if getattr(backend, "parameter_writer_kind", None) is not None:
                raise ValueError(f"Backend already tracks {method}; duplicate instrumentation is unsupported")
        tracker = getattr(self, "_parameter_revision", None)
        if tracker is None:
            tracker = self._parameter_revision = ParameterRevision()
        return tracker

    def parameter_mutation(self, kind):
        tracker = getattr(self, "_parameter_revision", None)
        return tracker.mutation(kind) if tracker is not None else nullcontext()

    def to(self, *args, **kwargs):
        tracker = getattr(self, "_parameter_revision", None)
        # The concrete backend follows this mixin in the engine MRO.
        move = super().to  # type: ignore[misc]
        if tracker is None:
            return move(*args, **kwargs)
        # A Python signature error happens before the backend is entered.
        # Reject it without treating the model as partially moved.
        try:
            signature = inspect.signature(move)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            signature.bind(*args, **kwargs)
        with tracker.device_transition():
            return move(*args, **kwargs)

    @parameter_writer("initialize", preflight_backend=True)
    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)  # type: ignore[misc]

    @parameter_writer("load", preflight_backend=True)
    def load_checkpoint(self, *args, **kwargs):
        return super().load_checkpoint(*args, **kwargs)  # type: ignore[misc]

    @parameter_writer("optimizer", preflight_backend=True)
    def optimizer_step(self, *args, **kwargs):
        return super().optimizer_step(*args, **kwargs)  # type: ignore[misc]
