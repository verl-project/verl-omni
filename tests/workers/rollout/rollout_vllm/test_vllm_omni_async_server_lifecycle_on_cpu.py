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
"""Async rollout server lifecycle contract.

Driven by a mocked ``AsyncOmni`` matching the pinned engine API: ``abort``
takes EXTERNAL ids in one batched acked call; ``pause_generation`` never
delivers tokens; diffusion sleep/wake ACKs are not engine-checked.
"""

import asyncio
import queue
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")

from verl.workers.rollout.replica import RolloutMode  # noqa: E402

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer  # noqa: E402
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import OmniStrategyBase  # noqa: E402

_SUCCESS_ACK = SimpleNamespace(status="SUCCESS")

pytestmark = pytest.mark.asyncio


class _FakeRequestState:
    def __init__(self, request_id: str, external_request_id: str):
        self.request_id = request_id
        self.external_request_id = external_request_id
        self.queue: asyncio.Queue = asyncio.Queue()


def _terminal(request_id: str, token_ids: list[int]):
    """Engine-side abort terminal: real cumulative partial tokens."""
    output = SimpleNamespace(finish_reason="abort", token_ids=list(token_ids))
    return SimpleNamespace(request_id=request_id, engine_outputs=SimpleNamespace(outputs=[output]), finished=True)


class _FakeAsyncOmni:
    def __init__(self, *, states=None, fail_abort=False, sleep_acks=None, wake_acks=None):
        self.request_states: dict[str, _FakeRequestState] = dict(states or {})
        self.fail_abort = fail_abort
        self.sleep_acks = sleep_acks if sleep_acks is not None else [_SUCCESS_ACK]
        self.wake_acks = wake_acks if wake_acks is not None else [_SUCCESS_ACK]
        self.calls: list[str] = []
        self.abort_calls: list = []
        self.pause_calls: list[dict] = []
        self.sleep_calls: list[dict] = []
        self.wake_calls: list[dict] = []
        self.resumed = 0
        self.sleeping_tags = set()
        # Real-shaped handle: the frontend multimodal cache lives on the
        # renderer, so the fake must expose clear_mm_cache_async there.
        self.mm_clears = 0

        async def _clear_mm_cache_async():
            self.mm_clears += 1

        self.renderer = SimpleNamespace(clear_mm_cache_async=_clear_mm_cache_async)

    async def abort(self, request_ids):
        self.calls.append("abort")
        self.abort_calls.append(request_ids if isinstance(request_ids, list) else [request_ids])
        if self.fail_abort:
            raise RuntimeError("abort rpc failed")
        ids = request_ids if isinstance(request_ids, list) else [request_ids]
        for state in self.request_states.values():
            if state.external_request_id in ids:
                state.queue.put_nowait(_terminal(state.request_id, token_ids=[7, 8, 9]))

    async def pause_generation(self, **kwargs):
        self.calls.append("pause")
        self.pause_calls.append(kwargs)

    async def sleep(self, stage_ids=None, level=2, mode="abort"):
        self.sleeping_tags = {"weights", "kv_cache"}
        self.calls.append("sleep")
        self.sleep_calls.append({"stage_ids": stage_ids, "level": level, "mode": mode})
        return self.sleep_acks

    async def wake_up(self, stage_ids=None, tags=None):
        self.sleeping_tags.difference_update(tags or {"weights", "kv_cache"})
        self.calls.append("wake_up")
        self.wake_calls.append({"stage_ids": stage_ids, "tags": tags})
        return self.wake_acks

    async def resume_generation(self, stage_ids=None):
        self.resumed += 1

    async def is_sleeping(self):
        return bool(self.sleeping_tags)

    async def collective_rpc(self, method, timeout=None, kwargs=None):
        self.calls.append(method)
        return [True]


def _make_server(engine, rollout_mode=RolloutMode.HYBRID, node_rank=0, free_cache_engine=True):
    # HYBRID default: release/resume_kv_cache are skipped in COLOCATED mode
    # (parent semantics), so their delegation is exercised via HYBRID.
    server = object.__new__(vLLMOmniHttpServer)
    server.engine = engine
    server._step_handoff_lock = asyncio.Lock()
    server.node_rank = node_rank
    server.replica_rank = 0
    server.global_steps = 0
    server.rollout_mode = rollout_mode
    server.config = SimpleNamespace(free_cache_engine=free_cache_engine, step_execution=False)
    server._lora_request_cache = None  # a valid cached value
    server._submission_paused = server._rejecting = False
    server._admitting = 0
    server._resume_event = asyncio.Event()
    server._resume_event.set()
    return server


async def test_diffusion_engine_shutdown_rejects_parked_and_late_requests():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)
    server._submission_paused = True
    server._resume_event.clear()

    def shutdown():
        assert server._submission_paused and server._rejecting
        engine.calls.append("shutdown")

    engine.shutdown = shutdown
    parked = asyncio.create_task(server._park_until_admitted("parked"))
    try:
        await asyncio.sleep(0)
        assert not parked.done()
        await server.close_diffusion_engine()
        assert (await asyncio.wait_for(parked, timeout=1)).stop_reason == "aborted"
        assert (await server._park_until_admitted("late")).stop_reason == "aborted"
        assert engine.calls == ["shutdown"]
    finally:
        if not parked.done():
            parked.cancel()
            await asyncio.gather(parked, return_exceptions=True)


# ---------------------------------------------------------------------------
# abort-then-pause ordering; single batched abort; no empty-list abort
# ---------------------------------------------------------------------------


async def test_abort_runs_before_pause_with_one_batched_call():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-1-def": _FakeRequestState("ext-1-def", "ext-1"),  # same external id
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
        "ext-none": _FakeRequestState("ext-none", None),  # None external id must be skipped
    }
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    result = await server.abort_all_requests()

    # Deduped EXTERNAL ids (request_states keys are internal ids).
    assert engine.abort_calls == [["ext-1", "ext-2"]]
    assert engine.calls == ["abort", "pause"], "abort must run while generate is live, pause after"
    assert engine.pause_calls == [{"mode": "abort", "wait_for_inflight_requests": False, "clear_cache": True}]
    assert result == {"aborted_count": 2, "request_ids": ["ext-1", "ext-2"]}


async def test_abort_with_no_in_flight_requests_still_pauses():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)

    result = await server.abort_all_requests()

    # The engine returns immediately on an empty id list.
    assert engine.abort_calls == [[]]
    assert engine.calls == ["abort", "pause"]
    assert result == {"aborted_count": 0, "request_ids": []}


@pytest.mark.parametrize("failure", [None, "abort", "cancel"])
async def test_stragglers_are_aborted_after_pause_and_failures_hold_admission(failure):
    late = _FakeRequestState("late-internal", "late-external")

    class Engine(_FakeAsyncOmni):
        async def pause_generation(self, **kwargs):
            await super().pause_generation(**kwargs)
            self.request_states[late.request_id] = late

        async def abort(self, request_ids):
            if request_ids:
                if failure == "abort":
                    raise RuntimeError("straggler failed")
                if failure == "cancel":
                    raise asyncio.CancelledError
            return await super().abort(request_ids)

    engine = Engine()
    server = _make_server(engine)
    if failure:
        with pytest.raises(RuntimeError if failure == "abort" else asyncio.CancelledError):
            await server.abort_all_requests()
    else:
        result = await server.abort_all_requests()
        assert result["request_ids"] == ["late-external"]
    assert not server._resume_event.is_set()
    assert late.queue.get_nowait().finished


async def test_straggler_retry_with_reused_external_id_is_aborted():
    from vllm_omni.entrypoints import AsyncOmni

    original = _FakeRequestState("original-internal", "sample")
    late = _FakeRequestState("retry-internal", "sample")
    engine = _FakeAsyncOmni(states={original.request_id: original})
    aborted = []

    async def settle_pending(internal_ids):
        # Native abort has already selected its internal IDs at this boundary.
        engine.request_states[late.request_id] = late

    async def abort_internal(internal_ids, *, timeout=None):
        aborted.append(internal_ids)
        for internal_id in internal_ids:
            engine.request_states[internal_id].queue.put_nowait(_terminal(internal_id, [7]))

    engine._settle_pending_submits = settle_pending
    engine._abort = abort_internal
    engine.abort = lambda request_ids: AsyncOmni.abort(engine, request_ids)
    server = _make_server(engine)
    await server.abort_all_requests()
    assert aborted == [[original.request_id], [original.request_id, late.request_id]]
    assert late.queue.get_nowait().finished
    assert server._submission_paused and not server._resume_event.is_set()


async def test_resident_rollout_pauses_until_explicit_resume_without_sleeping():
    """Quiescence includes late admissions before enumerating resident rollout aborts."""

    class LateAdmissionEngine(_FakeAsyncOmni):
        async def pause_generation(self, **kwargs):
            await super().pause_generation(**kwargs)
            if kwargs["mode"] == "keep":
                self.request_states["late"] = _FakeRequestState("late", "late-external")

    engine = LateAdmissionEngine()
    server = _make_server(engine, free_cache_engine=False)

    await server.abort_all_requests()
    await server.sleep()
    assert engine.calls == ["pause", "abort", "pause"]
    assert engine.pause_calls[0]["mode"] == "keep"
    assert engine.abort_calls == [["late-external"]]
    assert engine.sleep_calls == []
    assert engine.resumed == 0

    await server.resume_generation()
    assert engine.resumed == 1


@pytest.mark.parametrize("step_execution", [False, True])
async def test_resident_standalone_pause_uses_supported_engine_mode(step_execution):
    from vllm_omni.diffusion.diffusion_engine import DiffusionEngine, DiffusionExecutionMode
    from vllm_omni.entrypoints import AsyncOmni

    execution_mode = DiffusionExecutionMode.STEP_BATCH if step_execution else DiffusionExecutionMode.REQUEST_BATCH

    async def engine_rpc(method, *, stage_ids, kwargs):
        DiffusionEngine._check_pause_request(SimpleNamespace(execution_mode=execution_mode), kwargs)

    frontend = SimpleNamespace(
        _pause_cond=asyncio.Condition(),
        _paused_stage_ids=set(),
        _name="lifecycle-test",
        _split_stage_ids_by_type=lambda stage_ids: ([], [0]),
        _cancel_pending_submits=lambda: None,
        _engine_core_rpc=engine_rpc,
    )
    engine = _FakeAsyncOmni()

    async def pause(**kwargs):
        await AsyncOmni.pause_generation(frontend, **kwargs)
        engine.pause_calls.append(kwargs)

    engine.pause_generation = pause
    server = _make_server(engine, rollout_mode=RolloutMode.STANDALONE, free_cache_engine=False)
    server.config.step_execution = step_execution
    await server.abort_all_requests()
    assert [call["mode"] for call in engine.pause_calls] == (["abort"] if step_execution else ["keep", "abort"])
    assert frontend._paused and server._submission_paused


@pytest.mark.parametrize(
    "phase, outcome",
    [("pause", "success"), ("pause", "timeout"), ("pause", "cancel"), ("pause", "full"), ("resume", "timeout")],
)
async def test_scheduler_deadline_settles_real_rpc_waiter(monkeypatch, phase, outcome):
    from vllm_omni.engine.messages import CollectiveRPCResultMessage
    from vllm_omni.engine.omni_engine_base import OmniEngineBase
    from vllm_omni.engine.rpc_result_router import CorrelatedRpcClient

    monkeypatch.setenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "0.05")
    requests, results = queue.Queue(maxsize=1), queue.Queue()
    if outcome == "full":
        requests.put(object())
    transport = CorrelatedRpcClient(requests, results)
    engine = object.__new__(OmniEngineBase)
    engine._correlated_rpc_client = transport
    frontend = object.__new__(server_module._RolloutAsyncOmni)
    frontend.engine = engine
    frontend._pause_cond = asyncio.Condition()
    frontend._paused_stage_ids = {0} if phase == "resume" else set()
    frontend._split_stage_ids_by_type = lambda stage_ids: ([], [0])
    frontend._cancel_pending_submits = lambda: None
    state = _FakeRequestState("internal", "external")
    frontend.request_states = {state.request_id: state}

    async def abort(request_ids):
        pass

    frontend.abort = abort
    server = _make_server(frontend, free_cache_engine=False)
    server._submission_paused = True
    server._resume_event.clear()
    task = asyncio.create_task(server.abort_all_requests() if phase == "pause" else server.resume_generation())
    try:
        if outcome != "full":
            message = await asyncio.to_thread(requests.get, True, 2)
            assert message.timeout == 0.05
            if outcome == "success":
                results.put(
                    CollectiveRPCResultMessage(
                        rpc_id=message.rpc_id, method=message.method, stage_ids=message.stage_ids, results=[None]
                    )
                )
            elif outcome == "cancel":
                task.cancel()
        if outcome == "success":
            await asyncio.wait_for(asyncio.shield(task), 2)
            assert frontend._paused
        else:
            error = {"full": queue.Full, "cancel": asyncio.CancelledError, "timeout": TimeoutError}[outcome]
            with pytest.raises(error):
                await asyncio.wait_for(asyncio.shield(task), 2)
            if phase == "pause":
                assert state.queue.get_nowait().finished
        assert not transport._router._pending
        assert server._submission_paused and not server._resume_event.is_set()
    finally:
        if outcome == "full":
            requests.get_nowait()
        transport.close()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("free_cache", [False, True])
async def test_abort_only_aborts_without_pausing(free_cache):
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    server.config.free_cache_engine = free_cache
    result = await server.abort_all_requests(abort_only=True)

    # abort_only cancels current engine requests and keeps admission open: no pause.
    assert engine.calls == ["abort"]
    assert engine.pause_calls == []
    assert result == {"aborted_count": 1, "request_ids": ["ext-1"]}


async def test_abort_only_failure_unblocks_generators_without_pausing():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
    }
    engine = _FakeAsyncOmni(states=states, fail_abort=True)
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await server.abort_all_requests(abort_only=True)

    assert engine.pause_calls == []
    for state in states.values():
        terminal = state.queue.get_nowait()
        assert terminal.finished is True
        assert terminal.engine_outputs.outputs[0].finish_reason == "abort"


async def test_reject_request_aborts_late_calls_without_entering_the_engine():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    await server.abort_all_requests(reject_request=True)
    assert engine.calls == ["abort", "pause"]
    output = await server.generate(prompt_ids=[1], sampling_params={}, request_id="late")
    assert output.stop_reason == "aborted"
    assert not server._resume_event.is_set()


@pytest.mark.parametrize("cancel", [False, True])
async def test_parked_admission_is_bounded_and_cancellable(monkeypatch, cancel):
    monkeypatch.setenv("VERL_OMNI_ADMISSION_TIMEOUT_S", "0.02")
    server = _make_server(_FakeAsyncOmni())
    await server.abort_all_requests()
    request = asyncio.create_task(server.generate(prompt_ids=[1], sampling_params={}, request_id="late"))
    if cancel:
        await asyncio.sleep(0)
        request.cancel()
    with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
        await request
    assert not server._resume_event.is_set()


@pytest.mark.parametrize("failure", [False, True])
async def test_generate_accepts_session_id_without_forwarding(failure):
    engine = _FakeAsyncOmni()
    server = _make_server(engine)
    forwarded: dict = {}

    async def _record_generate(**kwargs):
        forwarded.update(kwargs)
        if failure:
            raise RuntimeError("pre-submit failed")
        kwargs["submitted"].set_result(None)
        return SimpleNamespace()

    server._generate_strategy = SimpleNamespace(generate=_record_generate)

    with pytest.raises(RuntimeError, match="pre-submit failed") if failure else nullcontext():
        await server.generate(prompt_ids=[1], sampling_params={"top_p": 1.0}, request_id="req-1", session_id="sess-1")
    await asyncio.sleep(0)

    # Cross-request server affinity has no AsyncOmni counterpart; routing
    # stickiness stays owned by the load balancer.
    assert forwarded["request_id"] == "req-1"
    assert "session_id" not in forwarded
    assert server._admitting == 0


async def test_replica_sleep_does_not_add_an_explicit_admission_hold():
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

    engine = _FakeAsyncOmni()
    server = _make_server(engine)
    replica = SimpleNamespace(
        servers=[
            SimpleNamespace(
                wait_for_requests_to_drain=SimpleNamespace(remote=server.wait_for_requests_to_drain),
                sleep=SimpleNamespace(remote=server.sleep),
            )
        ]
    )
    await vLLMReplica.sleep(replica)
    assert not server._submission_paused
    await server.wake_up()
    assert engine.resumed == 1


async def test_admission_rechecks_a_gate_closed_again_before_wakeup(monkeypatch):
    server = _make_server(_FakeAsyncOmni())
    server._submission_paused = True
    server._resume_event.clear()
    parked, reparked = asyncio.Event(), asyncio.Event()
    wait = server._resume_event.wait
    waits, calls = [], []

    async def observed_wait():
        waits.append(None)
        (parked if len(waits) == 1 else reparked).set()
        return await wait()

    async def generate(*, submitted=None, **kwargs):
        calls.append(kwargs["request_id"])
        if submitted is not None:
            submitted.set_result(None)
        return SimpleNamespace()

    monkeypatch.setattr(server._resume_event, "wait", observed_wait)
    server._generate_strategy = SimpleNamespace(generate=generate)
    request = asyncio.create_task(server.generate(prompt_ids=[1], sampling_params={}, request_id="waiting"))
    try:
        await asyncio.wait_for(parked.wait(), 1)
        await server.open_submission_gate()
        server._submission_paused = True
        server._resume_event.clear()
        await asyncio.wait_for(reparked.wait(), 1)
        assert calls == [] and server._admitting == 0
        await server.resume_generation()
        await asyncio.wait_for(request, 1)
        assert calls == ["waiting"] and server._admitting == 0
    finally:
        request.cancel()
        await asyncio.gather(request, return_exceptions=True)


# ---------------------------------------------------------------------------
# abort outputs arrive via the request queues; mapper untouched
# ---------------------------------------------------------------------------


async def test_abort_outputs_come_from_engine_queues_with_non_empty_tokens():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    await server.abort_all_requests()

    terminal = states["ext-1-abc"].queue.get_nowait()
    output = terminal.engine_outputs.outputs[0]
    assert output.token_ids == [7, 8, 9], "cumulative partial tokens must survive the abort"
    assert output.finish_reason == "abort"
    assert OmniStrategyBase._map_stop_reason(output.finish_reason) == "aborted"
    # The engine owns success-path terminals; the server synthesizes only on failure.
    assert states["ext-1-abc"].queue.qsize() == 0


# ---------------------------------------------------------------------------
# all four lifecycle methods delegate, validate ACKs, level=1, keyword tags
# ---------------------------------------------------------------------------


async def test_sleep_and_wake_delegate_with_level_one_and_keyword_tags():
    for mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
        engine = _FakeAsyncOmni()
        server = _make_server(engine, rollout_mode=mode)
        server._lora_request_cache = None

        await server.sleep()
        await server.wake_up()

        # Uniform delegation in BOTH modes (the old code split hybrid vs
        # colocated between raw RPC and engine.sleep).
        assert [c for c in engine.calls if c in ("sleep", "wake_up")] == ["sleep", "wake_up"]
        assert engine.sleep_calls == [{"stage_ids": None, "level": 1, "mode": "abort"}]
        # Keyword tags only: a positional call would have bound to stage_ids.
        # Full wake must include kv_cache; weights-only leaves generate() rejected.
        assert engine.wake_calls == [{"stage_ids": None, "tags": ["kv_cache", "weights"]}]
        assert server._lora_request_cache is server_module._LORA_REQUEST_CACHE_MISS


async def test_weight_version_commit_selects_exact_installed_adapter():
    engine = _FakeAsyncOmni()
    engine.list_loras = AsyncMock(return_value=[server_module.VLLM_LORA_INT_ID, 124])
    server = _make_server(engine)
    server.model_config = SimpleNamespace(lora_rank=32, lora={})
    server._lora_resolve_lock = asyncio.Lock()
    server._invalidate_lora_request_cache()
    assert (await server._resolve_lora_request()).lora_int_id == server_module.VLLM_LORA_INT_ID

    server._submission_paused = True
    await server.set_global_steps(7, lora_adapter_id=124)
    assert server.global_steps == 7
    assert (await server._resolve_lora_request()).lora_int_id == 124
    assert engine.list_loras.await_count == 2

    await server.set_global_steps(8, lora_adapter_id=125)
    with pytest.raises(RuntimeError, match="committed LoRA adapter 125"):
        await server._resolve_lora_request()


@pytest.mark.parametrize("drain_ok", [True, False])
@pytest.mark.parametrize("frontend_has_old_request", [True, False])
async def test_step_handoff_retires_previous_adapter_before_reusing_slot(drain_ok, frontend_has_old_request):
    engine = _FakeAsyncOmni(states={"old": _FakeRequestState("old", "prior-session")})
    engine.num_stages = 1
    server = _make_server(engine)
    server._lora_resolve_lock = asyncio.Lock()
    original = server_module.VLLM_LORA_INT_ID
    bundle = [{"old": b"snapshot"}]
    operations = []
    pending_backend = {"old", "current"}

    async def rpc(method, timeout, *, args=(), kwargs=None):
        operations.append((method, args))
        if method == "export_step_requests":
            return [[bundle]]
        if method == "weight_sharing_drain":
            assert kwargs == {"request_ids": ["old"]}
            assert engine.abort_calls == ([["prior-session"]] if frontend_has_old_request else [])
            if drain_ok:
                pending_backend.remove("old")
            return [drain_ok]
        if method == "remove_lora":
            assert pending_backend == {"current"}
        return [True]

    server._step_handoff_rpc = rpc
    server.abort_all_requests = AsyncMock()
    assert await server.suspend_diffusion_steps(3, 1024) == original + 1
    assert server._submission_paused
    await server.set_global_steps(1, lora_adapter_id=original + 1)
    await server.restore_diffusion_steps(3, 1024)
    assert operations == [
        ("pin_lora", (original,)),
        ("export_step_requests", ()),
        ("release_step_requests", (["old"],)),
        ("restore_step_requests", (bundle,)),
    ]
    if not frontend_has_old_request:
        engine.request_states.pop("old")
    engine.request_states["current"] = _FakeRequestState("current", "current-session")
    bundle = [{"current": b"current-snapshot"}]
    operations.clear()
    if drain_ok:
        assert await server.suspend_diffusion_steps(3, 1024) == original
        assert operations == [
            ("weight_sharing_drain", ()),
            ("remove_lora", (original,)),
            ("pin_lora", (original + 1,)),
            ("export_step_requests", ()),
            ("release_step_requests", (["current"],)),
        ]
        assert server._diffusion_step_handoff.request_ids == {"current"}
        assert server._diffusion_step_handoff.snapshots == bundle
    else:
        with pytest.raises(RuntimeError, match="drain"):
            await server.suspend_diffusion_steps(3, 1024)
        assert operations == [("weight_sharing_drain", ())]
        assert server._diffusion_step_handoff.adapter_id == original
    server.abort_all_requests.assert_not_awaited()
    assert server._submission_paused


@pytest.mark.parametrize(
    ("failed_method", "drain_acknowledged"),
    [("export_step_requests", True), ("release_step_requests", False)],
)
async def test_step_snapshot_fallback_waits_for_backend_drain(failed_method, drain_acknowledged):
    engine = _FakeAsyncOmni(states={"old": _FakeRequestState("old", "old-session")})
    engine.num_stages = 1
    server = _make_server(engine)
    server.config.step_execution = True
    original = server_module.VLLM_LORA_INT_ID
    backend_tasks = {"old"}
    calls = []
    abort = engine.abort

    async def abort_with_frontend_cleanup(request_ids):
        await abort(request_ids)
        engine.request_states.clear()

    async def rpc(method, timeout, *, args=(), kwargs=None):
        calls.append(method)
        if method == failed_method:
            raise TimeoutError("snapshot RPC did not acknowledge")
        if method == "export_step_requests":
            return [[[{"old": b"snapshot"}]]]
        if method == "weight_sharing_drain":
            assert kwargs == {"request_ids": ["old"]}
            assert engine.request_states == {} and engine.abort_calls == [["old-session"]]
            assert backend_tasks == {"old"}
            if drain_acknowledged:
                backend_tasks.clear()
            return [drain_acknowledged]
        return [True]

    engine.abort = abort_with_frontend_cleanup
    server._step_handoff_rpc = rpc
    if drain_acknowledged:
        assert await server.suspend_diffusion_steps(3, 1024) == original + 1
        assert backend_tasks == set()
        assert server._diffusion_step_handoff.snapshots is None
    else:
        with pytest.raises(RuntimeError, match="drain"):
            await server.suspend_diffusion_steps(3, 1024)
        assert backend_tasks == {"old"}
        assert server._diffusion_step_handoff is None
    assert calls[-1] == "weight_sharing_drain"
    assert server._submission_paused and not server._resume_event.is_set()


async def test_step_checkpoint_only_exports_mapped_inflight_sessions(tmp_path):
    engine = _FakeAsyncOmni()
    engine.request_states = {
        "old": SimpleNamespace(external_request_id="train_1_0"),
        "unrelated": SimpleNamespace(external_request_id="other_0_0"),
    }
    server = _make_server(engine)
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"base-v1")
    server._diffusion_checkpoint_model_fingerprint = server_module._DiffusionCheckpointModel(tmp_path).verify()
    server._diffusion_step_handoff = server_module._DiffusionStepHandoff(
        124,
        "rollout-step-3",
        frozenset(engine.request_states),
        [{"old": b"selected", "unrelated": b"other", "gone": b"canceled"}],
    )
    server._step_handoff_rpc = AsyncMock(return_value=[[b"installed-adapter"]])

    assert await server.export_diffusion_checkpoint({"none"}, 9, 1024) is None
    server._step_handoff_rpc.assert_not_awaited()
    handoff = await server.export_diffusion_checkpoint({"train_1_0"}, 9, 1024)
    model_fingerprint = server_module.fingerprint(server_module.inventory(tmp_path, [weights]))
    assert handoff == {
        "policy_ref": "rollout-step-3",
        "model_fingerprint": model_fingerprint,
        "requests": {"train_1_0": "old"},
        "snapshots": [{"old": b"selected"}],
        "adapters": [b"installed-adapter"],
    }
    server._step_handoff_rpc.assert_awaited_once_with("export_runtime_lora", 9, args=(124,), kwargs={"max_bytes": 1024})

    weights.write_bytes(b"base-v2")

    fresh_engine = _FakeAsyncOmni()
    fresh_engine.restore_diffusion_step_streaming = AsyncMock()
    fresh_server = _make_server(fresh_engine)
    fresh_server._submission_paused = True
    fresh_server._committed_lora_adapter_id = 125
    fresh_server._diffusion_checkpoint_model_fingerprint = server_module._DiffusionCheckpointModel(tmp_path).verify()
    fresh_server._step_handoff_rpc = AsyncMock()
    assert await fresh_server.recover_diffusion_checkpoint(handoff, 124, {}, 9, 1024) == {}
    fresh_server._step_handoff_rpc.assert_not_awaited()
    fresh_engine.restore_diffusion_step_streaming.assert_not_awaited()


async def test_release_and_resume_kv_cache_route_both_halves_through_delegation():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.release_kv_cache()
    await server.resume_kv_cache()

    assert [c for c in engine.calls if c in ("sleep", "wake_up")] == ["sleep", "wake_up", "wake_up"]
    assert engine.sleep_calls[0]["level"] == 1
    assert engine.wake_calls == [
        {"stage_ids": None, "tags": ["weights"]},  # release: keep weights awake
        {"stage_ids": None, "tags": ["kv_cache"]},  # resume: restore the cache
    ]


async def test_lifecycle_guards_skip_standalone_and_non_driver_ranks():
    engine = _FakeAsyncOmni()
    server = _make_server(engine, rollout_mode=RolloutMode.STANDALONE)
    await server.sleep()
    await server.wake_up()
    assert engine.calls == []

    engine = _FakeAsyncOmni()
    server = _make_server(engine, node_rank=1)
    await server.sleep()
    await server.wake_up()
    await server.release_kv_cache()
    await server.resume_kv_cache()
    assert await server.abort_all_requests() == {"aborted_count": 0, "request_ids": []}
    assert engine.calls == []


# ---------------------------------------------------------------------------
# fail-closed ack validation
# ---------------------------------------------------------------------------


async def test_ack_validation_fails_closed():
    bad_acks = [
        # diffusion worker error dict (collective_rpc error shape)
        [{"supported": False, "error": "diffusion stage failed"}, _SUCCESS_ACK],
        # dict without an error key is still a failure
        [{"supported": False}],
        # diffusion OmniACK failure
        [SimpleNamespace(status="ERROR", error_msg="camem pool corrupted")],
        # mixed-stage: AR raise happens engine-side; the dict/ACK mix must still raise here
        [_SUCCESS_ACK, {"error": "late diffusion failure"}],
    ]
    for acks in bad_acks:
        engine = _FakeAsyncOmni(sleep_acks=acks)
        server = _make_server(engine)
        with pytest.raises(RuntimeError, match="failed on a stage"):
            await server.sleep()

    # Empty ACK list is success ("already warm").
    engine = _FakeAsyncOmni(wake_acks=[])
    server = _make_server(engine)
    await server.wake_up()
    assert engine.wake_calls == [{"stage_ids": None, "tags": ["kv_cache", "weights"]}]


async def test_engine_side_rpc_failure_propagates_from_all_four_methods():
    class _RaisingEngine(_FakeAsyncOmni):
        async def sleep(self, **kwargs):
            raise RuntimeError("AR EngineCore control methods re-raise")

        async def wake_up(self, **kwargs):
            raise RuntimeError("AR EngineCore control methods re-raise")

    engine = _RaisingEngine()
    server = _make_server(engine)
    for method in (server.sleep, server.wake_up, server.release_kv_cache, server.resume_kv_cache):
        with pytest.raises(RuntimeError, match="re-raise"):
            await method()


# ---------------------------------------------------------------------------
# fail-closed abort: abort failure propagates; timeout raises
# ---------------------------------------------------------------------------


async def test_abort_failure_propagates_without_pausing():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
    }
    engine = _FakeAsyncOmni(states=states, fail_abort=True)
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await server.abort_all_requests()

    assert engine.pause_calls == []
    for state in states.values():
        terminal = state.queue.get_nowait()
        assert terminal.finished is True
        assert terminal.engine_outputs.outputs[0].finish_reason == "abort"


async def test_pause_failure_after_successful_abort_does_not_double_enqueue():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)

    async def failing_pause(**kwargs):
        raise RuntimeError("pause rpc failed")

    engine.pause_generation = failing_pause
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="pause rpc failed"):
        await server.abort_all_requests()

    # Exactly the engine's real terminal — no synthetic appended after it.
    assert states["ext-1-abc"].queue.qsize() == 1
    assert states["ext-1-abc"].queue.get_nowait().engine_outputs.outputs[0].token_ids == [7, 8, 9]


async def test_abort_ack_timeout_raises(monkeypatch):
    monkeypatch.setenv("VERL_OMNI_ABORT_ACK_TIMEOUT_S", "0.05")

    class _HangingEngine(_FakeAsyncOmni):
        async def abort(self, request_ids):
            await asyncio.sleep(999)

    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    server = _make_server(_HangingEngine(states=states))

    with pytest.raises(asyncio.TimeoutError):
        await server.abort_all_requests()

    for state in states.values():
        terminal = state.queue.get_nowait()
        assert terminal.finished is True
        assert terminal.engine_outputs.outputs[0].finish_reason == "abort"


# ---------------------------------------------------------------------------
# single-request abort shares the contract
# ---------------------------------------------------------------------------


async def test_abort_request_aborts_single_id_without_pausing():
    """Per-request abort must not pause the whole engine (pause_scheduler
    finishes ALL in-flight requests); pausing belongs to abort_all_requests."""
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    result = await server.abort_request("ext-1")

    assert engine.abort_calls == [["ext-1"]]  # external id, not internal
    assert engine.calls == ["abort"], "single-request abort must not pause the engine"
    assert states["ext-1-abc"].queue.qsize() == 1  # engine terminal only
    assert result == {"aborted": True, "request_id": "ext-1"}


async def test_abort_request_failure_enqueues_terminal():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states, fail_abort=True)
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await server.abort_request("ext-1")

    terminal = states["ext-1-abc"].queue.get_nowait()
    assert terminal.finished is True
    assert terminal.engine_outputs.outputs[0].finish_reason == "abort"


async def test_abort_request_unknown_id_is_a_noop():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)

    result = await server.abort_request("missing")

    assert engine.abort_calls == [["missing"]]
    assert engine.calls == ["abort"]
    assert result == {"aborted": True, "request_id": "missing"}


# ---------------------------------------------------------------------------
# the manager-level gather observes the server's raise
# ---------------------------------------------------------------------------


async def test_checkpoint_manager_gather_propagates_server_abort_raise():
    class _Replica:
        def __init__(self, server):
            self._server = server

        async def abort_all_requests(self):
            # Same seam CheckpointEngineManager.abort_replicas drives:
            # await asyncio.gather(*[r.abort_all_requests() for r in replicas])
            return await self._server.abort_all_requests()

    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    server = _make_server(_FakeAsyncOmni(states=states, fail_abort=True))

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await asyncio.gather(_Replica(server).abort_all_requests())


# ---------------------------------------------------------------------------
# diffusion-only sleep/wake contract — why diffusion v1 sync needs no
# resume bridge (drives the REAL AsyncOmni state machine, not a mock)
# ---------------------------------------------------------------------------


def _build_async_omni(stage_type: str):
    """Drive the real AsyncOmni state machine; only RPC plumbing is stubbed.

    ``stage_type`` selects a pure-diffusion ("diffusion") or pure-AR ("llm")
    single-stage engine.
    """
    from vllm_omni.entrypoints.async_omni import AsyncOmni

    ack = SimpleNamespace(status="SUCCESS")
    engine_rpc: list[str] = []

    async def _engine_core_rpc(method, stage_ids=None, args=(), kwargs=None):
        engine_rpc.append(method)
        return []

    async def _acks(*args, **kwargs):
        return [ack]

    omni = object.__new__(AsyncOmni)
    omni.engine = SimpleNamespace(
        stage_clients=[SimpleNamespace(stage_type=stage_type)],
        stage_configs=[SimpleNamespace(stage_type=stage_type)],
    )
    omni.input_processor = None
    omni._paused_stage_ids = set()
    omni._pause_cond = asyncio.Condition()
    omni._admitting = 0
    omni._paused = False
    omni._hold_admission_until_resume = False
    omni._level2_sleeping = False
    omni._sleeping_tags = {"weights"}
    omni._stage_sleeping_tags = {}
    # RPC plumbing stubs — not the contract under test.
    omni.reset_mm_cache = _acks
    omni._final_output_handler = lambda: None
    omni._sleep_diffusion = _acks
    omni._wake_diffusion = _acks
    omni._record_stage_sleep = lambda stage_ids, tags: None
    omni._sleeping_tags_for_stages = lambda stage_ids: {"weights"}
    omni._clear_stage_sleep = lambda stage_ids, tags: None
    omni._engine_core_rpc = _engine_core_rpc
    omni._engine_rpc_log = engine_rpc
    return omni


async def test_diffusion_only_sleep_wake_needs_no_resume_generation():
    """Pure-diffusion sleep must not set the admission hold.

    Diffusion v1 sync has no resume bridge; if the engine ever holds
    admission on diffusion sleep, that trainer hangs silently — this
    fails first.
    """
    omni = _build_async_omni("diffusion")

    await omni.sleep(level=1)

    assert omni._paused is True, "race guard set during sleep"
    assert omni._hold_admission_until_resume is False, "no AR stages — the hold must NOT be set"
    assert omni._engine_rpc_log == [], "diffusion sleep must not touch EngineCore RPCs"

    await omni.wake_up()

    assert omni._paused is False, "diffusion-only wake must restore admission by itself"


async def test_ar_sleep_wake_requires_resume_generation_contrast():
    """The AR-side contrast that makes the omni_sync bridge load-bearing."""
    omni = _build_async_omni("llm")

    await omni.sleep(level=1)
    assert omni._hold_admission_until_resume is True

    await omni.wake_up()
    assert omni._paused is True, "wake must keep the hold for AR stages"
    assert "sleep" in omni._engine_rpc_log and "wake_up" in omni._engine_rpc_log

    await omni.resume_generation()
    assert omni._paused is False
    assert omni._hold_admission_until_resume is False


# ---------------------------------------------------------------------------
# admission resume rides a *full* wake (wake_up / resume_kv_cache). A
# weights-only wake inside release_kv_cache must keep the hold: kv_cache
# is still asleep and generate() would be rejected.
# ---------------------------------------------------------------------------


async def test_wake_up_resumes_engine_admission():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.wake_up()

    assert engine.wake_calls == [{"stage_ids": None, "tags": ["kv_cache", "weights"]}]
    assert engine.resumed == 1


async def test_actor_weight_sync_keeps_admission_closed_until_full_wake():
    engine = _FakeAsyncOmni()
    server = _make_server(engine)
    await server.sleep()
    await server.wake_up(tags=["weights"])
    assert engine.resumed == 0
    assert engine.sleeping_tags == {"kv_cache"}
    await server.wake_up(tags=["kv_cache"])
    assert engine.resumed == 1


async def test_release_kv_cache_holds_admission_until_resume():
    """Weights-only wake must not admit generate: kv_cache is still asleep.

    CheckpointEngineManager.update_weights aborts, then release_kv_cache,
    then NCCL, then resume_kv_cache. If release resumes, FullyAsyncLLMServerClient
    reissues into AsyncOmni.generate() while kv_cache is still asleep and
    generation fails ("Currently sleeping tags: ['kv_cache']").
    """
    engine = _FakeAsyncOmni()
    server = _make_server(engine)

    await server.release_kv_cache()
    assert engine.wake_calls == [{"stage_ids": None, "tags": ["weights"]}]
    assert engine.resumed == 0

    await server.resume_kv_cache()
    assert engine.wake_calls[-1] == {"stage_ids": None, "tags": ["kv_cache"]}
    assert engine.resumed == 1


async def test_failed_wake_skips_admission_resume():
    engine = _FakeAsyncOmni(wake_acks=[SimpleNamespace(status="FAILED", error_msg="boom")])
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="wake_up failed"):
        await server.wake_up()

    assert engine.resumed == 0


async def test_step_suspend_reuses_the_snapshot_for_retried_handoff():
    engine = _FakeAsyncOmni(states={"request": _FakeRequestState("request", "session")})
    engine.num_stages = 1
    server = _make_server(engine)
    exporting, allow_export, retry_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
    operations = []
    bundle = [{"request": b"snapshot"}]

    async def rpc(method, timeout, *, args=(), kwargs=None):
        operations.append(method)
        if method == "export_step_requests":
            exporting.set()
            await allow_export.wait()
            return [[bundle]]
        return [True]

    async def retry():
        retry_entered.set()
        return await server.suspend_diffusion_steps(3, 1024)

    server._step_handoff_rpc = rpc
    async with asyncio.timeout(5), asyncio.TaskGroup() as jobs:
        initial = jobs.create_task(server.suspend_diffusion_steps(3, 1024))
        await exporting.wait()
        overlapping_retry = jobs.create_task(retry())
        await retry_entered.wait()
        allow_export.set()
    next_slot = server_module.VLLM_LORA_INT_ID + 1
    assert initial.result() == overlapping_retry.result() == next_slot
    assert await server.suspend_diffusion_steps(3, 1024) == next_slot
    assert operations == ["pin_lora", "export_step_requests", "release_step_requests"]
    assert server._diffusion_step_handoff.snapshots is bundle
    assert not engine.abort_calls
    assert server._submission_paused and not server._resume_event.is_set()
