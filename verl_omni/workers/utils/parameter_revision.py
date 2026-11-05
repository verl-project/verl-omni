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

import asyncio
import inspect
import threading
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from functools import wraps


@dataclass(frozen=True)
class ParameterVersion:
    incarnation: str
    revision: int


class ParameterRevision:
    def __init__(self):
        self._incarnation = uuid.uuid4().hex
        self._revision = 0
        self._lock = threading.RLock()
        self._owner = None
        self._depth = 0
        self._poisoned = False
        self._writer = None
        self._access_depth = 0
        self._external_access_depth = 0
        self._update = None
        self._updates = 0
        self._consumed = set()

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
            if not internal and (self._update is not None or self._writer is not None):
                raise RuntimeError("External lifecycle operation during an atomic update")
            self._access_depth += 1
            self._external_access_depth += not internal
            try:
                yield
            except BaseException:
                self._poisoned = True
                raise
            finally:
                self._access_depth -= 1
                self._external_access_depth -= not internal

    def assert_update_pending(self):
        """The loss may consume the anchor only before the sole optimizer write."""
        with self._exclusive():
            if self._update is None or self._updates != 0 or self._writer is not None:
                raise RuntimeError("No pending atomic update for the proximal anchor")

    def assert_update_complete(self):
        with self._exclusive():
            if self._update is None or self._updates != 1 or self._writer is not None:
                raise RuntimeError("The complete cohort must perform exactly one optimizer call")

    @contextmanager
    def update(self, expected: ParameterVersion, cohort_id: str):
        with self._exclusive():
            if self._update is not None or self._writer is not None or self._access_depth:
                raise RuntimeError("An actor operation is already active")
            if expected != ParameterVersion(self._incarnation, self._revision):
                raise ValueError("Stale parameter revision or different worker incarnation")
            if not isinstance(cohort_id, str) or not cohort_id or cohort_id in self._consumed:
                raise ValueError("Missing or consumed cohort identity")
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


def parameter_writer(kind):
    """Annotate the actual backend writer; disabled unless explicitly enabled."""

    def decorate(method):
        @wraps(method)
        def wrapped(engine, *args, **kwargs):
            tracker = getattr(engine, "_parameter_revision", None)
            context = tracker.mutation(kind) if tracker is not None else nullcontext()
            with context:
                return method(engine, *args, **kwargs)

        wrapped.parameter_writer_kind = kind
        return wrapped

    return decorate


def parameter_access(engine_path="engine", *, internal=False):
    """Keep an entire synchronous/async lifecycle operation outside updates.

    The engine path is relative to the method's owner. ``internal`` is reserved
    for native training entries and device moves within a qualified update.
    """

    def decorate(method):
        def context(owner):
            engine = owner
            for attr in engine_path.split(".") if engine_path else ():
                engine = getattr(engine, attr)
            tracker = getattr(engine, "_parameter_revision", None)
            return tracker.access(internal=internal) if tracker is not None else nullcontext()

        if inspect.iscoroutinefunction(method):

            @wraps(method)
            async def wrapped(owner, *args, **kwargs):
                with context(owner):
                    return await method(owner, *args, **kwargs)
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

    @parameter_writer("initialize")
    def initialize(self, *args, **kwargs):
        return super().initialize(*args, **kwargs)

    @parameter_writer("load")
    def load_checkpoint(self, *args, **kwargs):
        return super().load_checkpoint(*args, **kwargs)

    @parameter_writer("optimizer")
    def optimizer_step(self, *args, **kwargs):
        return super().optimizer_step(*args, **kwargs)
