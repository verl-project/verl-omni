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
"""Worker-side executors for named reward models and the scorer microbatcher."""

from __future__ import annotations

import asyncio
import contextvars
import gc
import inspect
import logging
from collections import deque
from functools import cached_property
from time import perf_counter
from typing import Any, Callable

import torch
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fs import copy_to_local

from verl_omni.utils.asyncio_utils import await_owned
from verl_omni.utils.reward_score.scorers.base import RewardScorer
from verl_omni.utils.reward_score.scorers.registry import import_from_spec
from verl_omni.workers.config.reward import RewardModelSpec, is_engine_backend

__all__ = [
    "EngineRewardExecutor",
    "NativeRewardExecutor",
    "ScorerMicrobatcher",
    "build_engine_reward_executors",
    "build_native_reward_executors",
]

logger = logging.getLogger(__name__)


class _ScorerRuntimeStats:
    """Lifetime counters updated only by the owning microbatcher's event loop."""

    def __init__(self):
        self.enqueued_requests = 0
        self.accepted_requests = 0
        self.withdrawn_requests = 0
        self.discarded_requests = 0
        self.capacity_wait_requests = 0
        self.batches = 0
        self.batch_requests = 0
        self.failed_batches = 0
        self.failed_requests = 0
        self.peak_queue_size = 0
        self.peak_batch_size = 0
        self.capacity_wait_seconds = 0.0
        self.queue_wait_seconds = 0.0
        self.forward_seconds = 0.0

    def snapshot(self) -> dict[str, int | float]:
        return vars(self).copy()


def _future_error(error: Exception) -> Exception:
    # asyncio Futures reject StopIteration rather than delivering it to the
    # awaiting request. Preserve the scorer's error as the explicit cause.
    if isinstance(error, StopIteration):
        normalized = RuntimeError("Reward scorer raised StopIteration")
        normalized.__cause__ = error
        return normalized
    return error


async def _run_sync_owned(method, *args, **kwargs):
    """Join a synchronous reward call even during global event-loop shutdown."""
    context = contextvars.copy_context()

    def forward():
        try:
            return context.run(method, *args, **kwargs)
        except StopIteration as error:
            # asyncio cannot chain StopIteration into its executor Future.
            raise _future_error(error) from error

    return await await_owned(asyncio.get_running_loop().run_in_executor(None, forward))


class EngineRouterClient:
    """Expose one engine router to a configured reward function."""

    def __init__(self, router_address: str, model_path: str):
        self.router_address = router_address
        self.model_path = model_path

    def reward_kwargs(self) -> dict[str, str]:
        return {
            "reward_router_address": self.router_address,
            "model_name": self.model_path,
        }


class EngineRewardExecutor:
    """Worker-side request contract for one engine-backed reward model."""

    def __init__(self, spec: RewardModelSpec):
        self.spec = spec
        self._client = EngineRouterClient(
            router_address=spec.router_address,
            model_path=spec.model_path,
        )

    def reward_kwargs(self) -> dict[str, str]:
        return self._client.reward_kwargs()

    @cached_property
    def tokenizer(self):
        """Tokenizer of the served model, loaded on first use."""
        return hf_tokenizer(copy_to_local(self.spec.model_path), trust_remote_code=True)


class NativeRewardExecutor:
    """Worker-local owner of a native reward model or scorer, exposing ``score()`` or ``infer()``.

    Two hosting paths share the wake/sleep fencing, in-flight accounting, and
    fail-closed disposal:

    - ``RewardScorer`` (the target path): ``load(device)`` construction,
      per-request :meth:`score` through the framework microbatcher.
    - Legacy native model (frozen): an ``infer()`` adapter constructed with
      injected ``model_path``/``device`` kwargs, driven by :meth:`infer`.
      Dropped at post-window cleanup (#723) once the remaining adapters are
      ported to class-based scorers.

    Concurrency model — each guard pins one race:

    - ``_lifecycle_lock`` serializes wake/sleep/close so two lifecycle RPCs
      never interleave.
    - ``_lock`` guards the ``_awake``/``_closed``/``_inflight`` fields; the
      ``_inflight`` count plus the ``_idle`` event is a readers-many gate —
      concurrent scoring proceeds, while sleep/close wait on ``_idle`` for
      every in-flight call to settle. On the scorer path they stop microbatcher
      admission *before* that wait, so queued and capacity-blocked work fails
      retryably instead of being scored out under it.
    - Caller cancellation stops at the microbatcher's acceptance boundary: a
      request not yet dequeued is withdrawn and never scored, while an accepted
      one — including its synchronous ``score_batch`` thread, owned through
      consumer cancellation — settles before the cancellation re-raises, so
      sleep/close never offload weights under a live forward. Lifecycle ops
      and the legacy ``infer()`` path keep their ``await_owned`` shield.
    - The executor binds to the loop of its first use and never crosses loops:
      the sync entrypoint pins one loop per process, and the worker path runs
      on the worker's stable loop.
    - The scorer path adds the microbatcher's admission condition. Forwards
      need no cross-executor lock: placement validation grants each native
      model exclusive device indices, and one consumer task per scorer
      already serializes its own forwards.
    """

    def __init__(self, spec: RewardModelSpec):
        if spec.backend != "native":
            raise ValueError(f"NativeRewardExecutor requires a native spec, got {spec.backend!r}")
        self.spec = spec
        self._model: Any | None = None
        self._inflight = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._awake = False
        self._closed = False
        self._close_failed = False
        self._model_cls: Any = None
        self._is_scorer = False
        self._microbatcher: ScorerMicrobatcher | None = None
        self._stats = _ScorerRuntimeStats()
        self._loop: asyncio.AbstractEventLoop | None = None

    def get_runtime_stats(self) -> dict[str, int | float]:
        """Return this executor's cumulative scorer counters without waking it."""
        return self._stats.snapshot() if self._is_scorer else {}

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("Native reward executor cannot be reused across event loops")

    def reward_kwargs(self) -> dict[str, Any]:
        return {"reward_model": self}

    async def wake_up(self) -> None:
        self._bind_loop()
        await await_owned(self._wake_up())

    async def sleep(self) -> None:
        self._bind_loop()
        await await_owned(self._sleep())

    async def close(self) -> None:
        self._bind_loop()
        await await_owned(self._close())

    async def score(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit one request to the hosted scorer's microbatcher and await its score dict."""
        self._bind_loop()
        async with self._lock:
            if not self._awake or self._closed:
                raise RuntimeError(f"Native reward model {self.spec.name!r} is not awake")
            if self._microbatcher is None:
                raise TypeError(f"Native reward model {self.spec.name!r} defines infer(); call infer() instead")
            self._inflight += 1
            self._idle.clear()
            microbatcher = self._microbatcher
        # Ownership lives in submit(): withdraw before acceptance, settle-then-cancel after.
        return await self._score_owned(microbatcher, request)

    async def infer(self, *args, **kwargs):
        """Run the frozen ``infer()`` adapter path — dropped at post-window cleanup (#723)."""
        self._bind_loop()
        async with self._lock:
            if not self._awake or self._closed:
                raise RuntimeError(f"Native reward model {self.spec.name!r} is not awake")
            if self._microbatcher is not None:
                raise TypeError(
                    f"Native reward scorer {self.spec.name!r} defines score_batch; call score(request) instead"
                )
            self._inflight += 1
            self._idle.clear()
            model = self._model
        return await await_owned(self._infer_owned(model, args, kwargs))

    async def _wake_up(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError(f"Native reward model {self.spec.name!r} is closed")
            if self._awake:
                return
            try:
                if self._model_cls is None:
                    self._model_cls = import_from_spec(self.spec.executor_config["model"])
                    self._is_scorer = isinstance(self._model_cls, type) and issubclass(self._model_cls, RewardScorer)
                if self._is_scorer:
                    await self._wake_up_scorer()
                else:
                    # Legacy adapter leg — dropped at post-window cleanup (#723).
                    await self._wake_up_legacy_model()
            except BaseException:
                await self._fail_closed()
                raise

    async def _wake_up_scorer(self) -> None:
        """Load (or re-wake) the hosted scorer and open its microbatcher."""
        if self._model is None:
            # Scorer constructors take configuration only; device arrives via load().
            kwargs = dict(self.spec.executor_config.get("kwargs", {}))
            if self.spec.model_path is not None:
                kwargs.setdefault("model_path", self.spec.model_path)
            device = self._placement_device()
            scorer = self._model_cls(**kwargs)
            # Ownership precedes load(): a load failing after placing weights
            # still reaches close() through _fail_closed.
            self._model = scorer
            await self._call(scorer.load, device)
        elif self._supports_cpu_offload(self._model):
            await self._call(self._model.wake_up)
        if self._microbatcher is None:
            self._microbatcher = ScorerMicrobatcher(
                self._model.score_batch,
                max_batch_size=self._model.max_batch_size,
                name=self.spec.name,
                _stats=self._stats,
            )
        await self._microbatcher.start()
        async with self._lock:
            self._awake = True

    async def _wake_up_legacy_model(self) -> None:
        """Wake the frozen ``infer()`` adapter — dropped at post-window cleanup (#723)."""
        if self._model is None:
            kwargs = dict(self.spec.executor_config.get("kwargs", {}))
            if self.spec.model_path is not None:
                kwargs.setdefault("model_path", self.spec.model_path)
            if "device" not in kwargs:
                kwargs["device"] = self._placement_device()
            self._model = self._model_cls(**kwargs)
        elif self._supports_cpu_offload(self._model):
            await self._call(self._model.wake_up)
        async with self._lock:
            self._awake = True

    async def _score_owned(self, microbatcher: ScorerMicrobatcher, request: dict[str, Any]) -> dict[str, Any]:
        try:
            return await microbatcher.submit(request)
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def _infer_owned(self, model, args, kwargs):
        """Run one frozen ``infer()`` adapter call — dropped at post-window cleanup (#723)."""
        try:
            infer_fn = getattr(model, "infer", None)
            if infer_fn is None:
                raise TypeError(f"Native reward model {type(self._model).__name__!r} must define infer()")
            if inspect.iscoroutinefunction(infer_fn):
                result = await infer_fn(*args, **kwargs)
            else:
                result = await _run_sync_owned(infer_fn, *args, **kwargs)
            return await result if inspect.isawaitable(result) else result
        finally:
            async with self._lock:
                self._inflight -= 1
                if self._inflight == 0:
                    self._idle.set()

    async def _sleep(self) -> None:
        async with self._lifecycle_lock:
            if self._closed or not self._awake:
                return
            async with self._lock:
                self._awake = False
            try:
                if self._microbatcher is not None:
                    # Admission stops first: queued and capacity-blocked submits fail
                    # retryably instead of being scored out under the idle wait.
                    await self._microbatcher.pause()
                await self._idle.wait()
                if self._microbatcher is not None:
                    await self._sleep_scorer()
                else:
                    # Legacy adapter leg — dropped at post-window cleanup (#723).
                    await self._sleep_legacy_model()
                if self.spec.device_type != "cpu":
                    _empty_accelerator_cache()
            except BaseException:
                await self._fail_closed()
                raise

    async def _sleep_scorer(self) -> None:
        """Park the hosted scorer weights; the caller already joined the consumer."""
        if self._supports_cpu_offload(self._model):
            await self._call(self._model.sleep)
        else:
            await self._drop_model(self._model)

    async def _sleep_legacy_model(self) -> None:
        """Park the frozen ``infer()`` adapter — dropped at post-window cleanup (#723)."""
        if self._supports_cpu_offload(self._model):
            await self._call(self._model.sleep)
        else:
            await self._drop_model(self._model)

    async def _drop_model(self, model) -> None:
        """Dispose weights that cannot be parked on the host; the microbatcher goes with them."""
        self._model = None
        self._microbatcher = None
        self._close_failed = True
        await self._dispose(model)
        self._close_failed = False

    async def _close(self) -> None:
        async with self._lifecycle_lock:
            if self._closed:
                if self._close_failed:
                    raise RuntimeError(f"Native reward model {self.spec.name!r} cleanup previously failed")
                return
            async with self._lock:
                self._awake = False
                self._closed = True
            # Detach and mark failed up front: any failure below must leave the
            # accurate failed-cleanup state, not a resident model reporting closed.
            model, self._model = self._model, None
            self._close_failed = True
            if self._microbatcher is not None:
                # Same ordering as sleep: unaccepted work fails before the idle wait.
                await self._microbatcher.close()
            await self._idle.wait()
            await self._teardown(model)
            self._close_failed = False

    async def _fail_closed(self) -> None:
        async with self._lock:
            self._awake = False
            self._closed = True
        model, self._model = self._model, None
        if model is None and self._close_failed:
            return
        self._close_failed = True
        try:
            await self._teardown(model)
        except BaseException:
            logger.exception("Failed to clean up native reward model %s", self.spec.name)
        else:
            self._close_failed = False

    async def _teardown(self, model) -> None:
        """Close the microbatcher (scorer path), dispose the model, clear the accelerator cache."""
        microbatcher, self._microbatcher = self._microbatcher, None
        if microbatcher is not None:
            await microbatcher.close()
        await self._dispose(model)
        if self.spec.device_type != "cpu":
            _empty_accelerator_cache()

    @classmethod
    async def _dispose(cls, model) -> None:
        if model is not None:
            close = getattr(model, "close", None)
            if callable(close):
                await cls._call(close)
        gc.collect()

    @staticmethod
    def _supports_cpu_offload(model) -> bool:
        return getattr(model, "supports_cpu_offload", None) is True and all(
            callable(getattr(model, method, None)) for method in ("sleep", "wake_up")
        )

    @staticmethod
    async def _call(method, *args):
        result = await method(*args) if inspect.iscoroutinefunction(method) else await _run_sync_owned(method, *args)
        return await result if inspect.isawaitable(result) else result

    def _placement_device(self) -> torch.device:
        if self.spec.device_type == "cpu":
            return torch.device("cpu")
        return torch.device(get_device_name(), get_device_id())


class ScorerMicrobatcher:
    """Group per-request kwarg dicts into microbatches for one ``score_batch``.

    Per-sample reward functions submit one kwarg dict each; the consumer groups
    requests up to ``max_batch_size`` and calls ``score_batch``, which returns
    one score dict per request, in input order. The scorer never owns a queue,
    consumer task, or thread — this is the framework side of that seam.

    Contract:

    - The trigger is opportunistic, never timeout-based: first arrival, one
      event-loop yield (``sleep(0)``), then drain — a lone request scores
      immediately at batch 1.
    - Admission is bounded: the queue holds at most ``max_inflight``
      not-yet-accepted requests; submitters beyond the cap await a free slot.
    - A request is **accepted** when the consumer dequeues it into a microbatch;
      accepted work settles whatever happens to its submitter.
    - Caller cancellation before acceptance withdraws the request — it is never
      scored; after acceptance, settlement completes first, its outcome is
      consumed, and the cancellation re-raises.
    - ``pause`` (executor sleep): stops admission, fails queued-but-unaccepted
      requests and capacity-blocked submitters with a retryable ``RuntimeError``
      (scoring may resume after wake), drains accepted work, joins the
      consumer; ``start`` re-opens admission.
    - ``close``: fails every not-yet-accepted request with a ``RuntimeError``,
      then awaits the in-flight forward.
    - Consumer-task cancellation (e.g. loop shutdown): every unsettled request
      — in the in-flight microbatch, in hand for packing, or still queued —
      fails with a ``RuntimeError`` and the microbatcher lands in ``closed``;
      the cancellation propagates, terminating the consumer. A synchronous
      forward stays owned until its thread settles, and joining a cancelled
      consumer never cancels the joiner. A consumer that dies without running
      its handler — cancellation delivered before the coroutine's first step —
      is settled by its done-callback. ``start`` re-spawns a consumer that was
      cancelled before it first ran; ``submit`` into that dead state fails
      like any other cancellation.
    - Per-request preprocessing failures (``Exception`` slots returned by
      ``score_batch``) are isolated to that request; a raised ``score_batch``
      failure fails each request in that microbatch.

    Args:
        score_batch: the scorer's callable (sync or async); receives the request
            list, must return one entry per request in input order (a score dict
            or a per-request ``Exception`` slot).
        max_batch_size: requests packed per ``score_batch`` call; ``None`` or
            ``1`` means unbatched.
        max_inflight: cap on queued-but-not-accepted requests (default 64).
        name: owning model name for error messages.
    """

    def __init__(
        self,
        score_batch: Callable[[list[dict[str, Any]]], list[Any]],
        *,
        max_batch_size: int | None = None,
        max_inflight: int = 64,
        name: str = "",
        _stats: _ScorerRuntimeStats | None = None,
    ):
        if max_batch_size is not None and (
            isinstance(max_batch_size, bool) or not isinstance(max_batch_size, int) or max_batch_size < 1
        ):
            raise ValueError(f"max_batch_size must be a positive integer or None, got {max_batch_size!r}")
        if isinstance(max_inflight, bool) or not isinstance(max_inflight, int) or max_inflight < 1:
            raise ValueError(f"max_inflight must be a positive integer, got {max_inflight!r}")
        self._score_batch = score_batch
        self._pack = max_batch_size or 1
        self._max_inflight = max_inflight
        self._name = name
        self._stats = _stats if _stats is not None else _ScorerRuntimeStats()
        self._enqueued_at: dict[asyncio.Future, float] = {}
        self._is_async = inspect.iscoroutinefunction(score_batch)
        self._cond = asyncio.Condition()
        self._queue: deque[tuple[dict[str, Any], asyncio.Future]] = deque()
        self._state = "idle"  # idle -> running -> (paused -> running)* | closed
        self._consumer: asyncio.Task | None = None
        self._lifecycle_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def get_runtime_stats(self) -> dict[str, int | float]:
        """Return a detached cumulative snapshot, including after close."""
        return self._stats.snapshot()

    def _bind_loop(self) -> None:
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("Reward scorer microbatcher cannot be reused across event loops")

    async def start(self) -> None:
        """Open admission and start the consumer (idempotent while running)."""
        self._bind_loop()
        async with self._lifecycle_lock:
            async with self._cond:
                if self._state == "closed":
                    raise RuntimeError(f"Reward scorer microbatcher {self._name!r} is closed")
                if self._state == "running" and self._consumer is not None and not self._consumer.done():
                    return
                # Also revives a consumer cancelled before its first step; the
                # settlement its done callback queued yields to the replacement.
                self._state = "running"
                self._consumer = asyncio.create_task(self._consumer_loop())
                self._consumer.add_done_callback(self._consumer_died)

    def _consumer_died(self, task: asyncio.Task) -> None:
        """Settle admission when a consumer dies without running its handler.

        A cancellation delivered before the coroutine's first step never
        reaches the handler, so this callback is the only settlement site.
        """
        if task.cancelled() and task is self._consumer and self._state == "running":
            asyncio.get_running_loop().create_task(self._fail_pending([], dead=task))

    async def submit(self, request: dict[str, Any]) -> dict[str, Any]:
        """Submit one request and await its score dict.

        Raises:
            RuntimeError: admission stopped by :meth:`pause` (retry after wake)
                or the microbatcher closed before this request was accepted —
                including via a consumer that was cancelled.
            asyncio.CancelledError: the caller left. A request not yet accepted
                is withdrawn (never scored); an accepted one settles first —
                its outcome is consumed — and the cancellation re-raises after.
        """
        self._bind_loop()
        entry: tuple[dict[str, Any], asyncio.Future] | None = None
        future: asyncio.Future | None = None
        try:
            async with self._cond:
                if self._state == "running" and self._consumer.done():
                    # A consumer cancelled before its first step settles nothing itself.
                    for _, done in self._settle_closed_locked([]):
                        if not done.done():
                            done.set_exception(RuntimeError(f"Reward scorer {self._name!r} consumer was cancelled"))
                    raise RuntimeError(f"Reward scorer {self._name!r} consumer was cancelled")
                capacity_wait_started = None
                try:
                    while self._state == "running" and len(self._queue) >= self._max_inflight:
                        if capacity_wait_started is None:
                            capacity_wait_started = perf_counter()
                            self._stats.capacity_wait_requests += 1
                        await self._cond.wait()
                finally:
                    if capacity_wait_started is not None:
                        self._stats.capacity_wait_seconds += perf_counter() - capacity_wait_started
                if self._state == "paused":
                    raise RuntimeError(f"Reward scorer {self._name!r} is asleep; retry after wake_up")
                if self._state == "closed":
                    raise RuntimeError(f"Reward scorer {self._name!r} is closed")
                if self._state != "running":
                    raise RuntimeError(f"Reward scorer microbatcher {self._name!r} is not started")
                future = asyncio.get_running_loop().create_future()
                # An abandoned (caller-cancelled) future never awaits its exception;
                # marking it retrieved at creation keeps every settlement site plain.
                future.add_done_callback(lambda f: f.cancelled() or f.exception())
                entry = (request, future)
                self._queue.append(entry)
                self._enqueued_at[future] = perf_counter()
                self._stats.enqueued_requests += 1
                self._stats.peak_queue_size = max(self._stats.peak_queue_size, len(self._queue))
                self._cond.notify_all()
            # Shielded: a caller cancel must not cancel the future itself — it
            # is the settlement signal an accepted request waits on.
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            # Unaccepted work is withdrawn; accepted work settles before the
            # cancellation re-raises — the boundary is the consumer's dequeue.
            if entry is not None:
                if not await self._withdraw_owned(entry):
                    try:
                        await await_owned(future)
                    except BaseException:
                        pass
            raise

    async def _withdraw_owned(self, entry: tuple[dict[str, Any], asyncio.Future]) -> bool:
        """Withdrawal that repeated caller cancellation cannot abort."""
        while True:
            try:
                return await self._withdraw(entry)
            except asyncio.CancelledError:
                continue

    async def _withdraw(self, entry: tuple[dict[str, Any], asyncio.Future]) -> bool:
        """Remove a queued-but-unaccepted entry; False once the consumer dequeues it."""
        async with self._cond:
            for index, queued_entry in enumerate(self._queue):
                if queued_entry is entry:
                    del self._queue[index]
                    self._enqueued_at.pop(entry[1])
                    self._stats.withdrawn_requests += 1
                    self._cond.notify_all()
                    return True
            return False

    async def pause(self) -> None:
        """Stop admission and join accepted work before permitting another start."""
        self._bind_loop()
        await self._stop(closed=False)

    async def close(self) -> None:
        """Stop admission permanently and join accepted work for every caller."""
        self._bind_loop()
        await self._stop(closed=True)

    async def _stop(self, *, closed: bool) -> None:
        # Keep the lifecycle owner until settlement. Concurrent start/close/pause
        # callers cannot mistake a detached, still-running consumer for idle.
        async with self._lifecycle_lock:
            async with self._cond:
                if not closed and self._state != "running":
                    return
                self._state = "closed" if closed else "paused"
                queued = list(self._queue)
                self._queue.clear()
                self._stats.discarded_requests += len(queued)
                for _, future in queued:
                    self._enqueued_at.pop(future)
                consumer = self._consumer
                self._cond.notify_all()
            message = (
                f"Reward scorer {self._name!r} closed before this request was accepted"
                if closed
                else f"Reward scorer {self._name!r} went to sleep before this request was accepted; retry after wake_up"
            )
            for _, future in queued:
                if not future.done():
                    future.set_exception(RuntimeError(message))
            if consumer is not None:
                try:
                    await await_owned(consumer)
                except asyncio.CancelledError:
                    # Joining an already cancelled consumer does not cancel the
                    # joiner. Cancellation of the lifecycle caller still propagates
                    # after its owned consumer has settled.
                    if asyncio.current_task().cancelling():
                        raise
                finally:
                    if self._consumer is consumer:
                        self._consumer = None

    async def _consumer_loop(self) -> None:
        batch: list[tuple[dict[str, Any], asyncio.Future]] = []
        try:
            while True:
                async with self._cond:
                    while self._state == "running" and not self._queue:
                        await self._cond.wait()
                    if not self._queue:
                        return
                    batch = [self._accept_one()]
                await asyncio.sleep(0)
                async with self._cond:
                    while self._state == "running" and self._queue and len(batch) < self._pack:
                        batch.append(self._accept_one())
                    self._cond.notify_all()
                await self._run_batch(batch)
                batch = []
        except asyncio.CancelledError:
            # Unsettled futures would hang shielded submitters and the executor's idle gate.
            await await_owned(self._fail_pending(batch))
            raise

    async def _fail_pending(
        self, batch: list[tuple[dict[str, Any], asyncio.Future]], *, dead: asyncio.Task | None = None
    ) -> None:
        """Settle every unsettled request after consumer cancellation and close admission.

        ``dead`` names the consumer whose done callback scheduled this settlement;
        a ``start`` revival that replaced it owns the queue, so the settlement
        steps aside instead of closing the new consumer.
        """
        async with self._cond:
            if dead is not None and dead is not self._consumer:
                return
            pending = self._settle_closed_locked(batch)
        for _, future in pending:
            if not future.done():
                future.set_exception(RuntimeError(f"Reward scorer {self._name!r} consumer was cancelled"))

    def _settle_closed_locked(self, batch: list[tuple[dict[str, Any], asyncio.Future]]) -> list[Any]:
        """Close admission and collect every unsettled request; the caller holds ``_cond``."""
        self._state = "closed"
        pending = list(batch) + list(self._queue)
        self._stats.failed_requests += sum(not future.done() for _, future in batch)
        self._stats.discarded_requests += len(self._queue)
        for _, future in self._queue:
            self._enqueued_at.pop(future)
        self._queue.clear()
        self._consumer = None
        self._cond.notify_all()
        return pending

    def _accept_one(self) -> tuple[dict[str, Any], asyncio.Future]:
        entry = self._queue.popleft()
        self._stats.accepted_requests += 1
        self._stats.queue_wait_seconds += perf_counter() - self._enqueued_at.pop(entry[1])
        return entry

    async def _run_batch(self, batch: list[tuple[dict[str, Any], asyncio.Future]]) -> None:
        self._stats.batches += 1
        self._stats.batch_requests += len(batch)
        self._stats.peak_batch_size = max(self._stats.peak_batch_size, len(batch))
        try:
            forward_started = perf_counter()
            try:
                results = await self._forward([request for request, _ in batch])
            finally:
                self._stats.forward_seconds += perf_counter() - forward_started
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError
            if not isinstance(results, list | tuple) or len(results) != len(batch):
                raise ValueError(
                    f"score_batch for {self._name!r} must return one entry per request in input order, "
                    f"got {len(results) if isinstance(results, list | tuple) else type(results).__name__} "
                    f"for {len(batch)} requests"
                )
        except BaseException as error:
            self._stats.failed_batches += 1
            # A scorer may translate or suppress cancellation while cleaning up.
            # The consumer still owns its requested cancellation after it settles.
            if asyncio.current_task().cancelling():
                raise asyncio.CancelledError from error
            if isinstance(error, asyncio.CancelledError):
                failure = RuntimeError(f"Reward scorer {self._name!r} cancelled its batch")
                failure.__cause__ = error
                error = failure
            logger.error("Reward scorer %r microbatch failed", self._name, exc_info=error)
            for _, future in batch:
                if not future.done():
                    self._stats.failed_requests += 1
                    future.set_exception(error)
            return
        failed_slot = any(isinstance(result, Exception) for result in results)
        for (_, future), result in zip(batch, results, strict=True):
            if isinstance(result, Exception) and not future.done():
                self._stats.failed_requests += 1
                future.set_exception(_future_error(result))
            elif not future.done():
                future.set_result(result)
        if failed_slot:
            self._stats.failed_batches += 1

    async def _forward(self, requests: list[dict[str, Any]]) -> list[Any]:
        # Only the single consumer task calls this, so forwards never overlap.
        if self._is_async:
            result = await self._score_batch(requests)
        else:
            # Keep the thread owned through consumer cancellation and loop shutdown,
            # so weights are never offloaded under a live forward.
            result = await _run_sync_owned(self._score_batch, requests)
        return await result if inspect.isawaitable(result) else result


def _empty_accelerator_cache() -> None:
    """Clear the accelerator cache only when the backend has one and is available."""
    accelerator = getattr(torch, get_device_name(), None)
    empty_cache = getattr(accelerator, "empty_cache", None)
    if callable(empty_cache) and getattr(accelerator, "is_available", lambda: False)():
        empty_cache()


def build_engine_reward_executors(specs: dict[str, RewardModelSpec]) -> dict[str, EngineRewardExecutor]:
    """Build worker-side router clients for engine-backed model specs."""
    return {name: EngineRewardExecutor(spec) for name, spec in specs.items() if is_engine_backend(spec.backend)}


def build_native_reward_executors(specs: dict[str, RewardModelSpec]) -> dict[str, NativeRewardExecutor]:
    """Build worker-local executors for native model specs."""
    return {name: NativeRewardExecutor(spec) for name, spec in specs.items() if spec.backend == "native"}
