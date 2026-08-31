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
"""Async rollout server lifecycle contract (issue #492 §5).

Covers the abort-then-pause ordering, the ACK-validated delegated sleep/wake
across all four lifecycle methods, the bounded drain, and the fail-closed
abort path — all against a mocked ``AsyncOmni`` matching the frozen-pin API
(``abort`` takes EXTERNAL ids in one batched acked call; ``pause_generation``
never delivers tokens; ``sleep``/``wake_up`` return ``OmniACK`` lists whose
diffusion entries are not engine-checked).
"""

import asyncio
from types import SimpleNamespace

import pytest

server_module = pytest.importorskip("verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server")

from verl.workers.rollout.replica import RolloutMode  # noqa: E402

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import vLLMOmniHttpServer  # noqa: E402
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_strategy_base import OmniStrategyBase  # noqa: E402

_SUCCESS_ACK = SimpleNamespace(status="SUCCESS")


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
    """AsyncOmni surface at the frozen pin, limited to the lifecycle methods."""

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

    async def abort(self, request_ids):
        self.calls.append("abort")
        self.abort_calls.append(request_ids if isinstance(request_ids, list) else [request_ids])
        if self.fail_abort:
            raise RuntimeError("abort rpc failed")
        ids = request_ids if isinstance(request_ids, list) else [request_ids]
        for state in self.request_states.values():
            if state.external_request_id in ids:
                # Real cumulative partial tokens, as the acked abort enqueues.
                state.queue.put_nowait(_terminal(state.request_id, token_ids=[7, 8, 9]))

    async def pause_generation(self, **kwargs):
        self.calls.append("pause")
        self.pause_calls.append(kwargs)

    async def sleep(self, stage_ids=None, level=2, mode="abort"):
        self.calls.append("sleep")
        self.sleep_calls.append({"stage_ids": stage_ids, "level": level, "mode": mode})
        return self.sleep_acks

    async def wake_up(self, stage_ids=None, tags=None):
        self.calls.append("wake_up")
        self.wake_calls.append({"stage_ids": stage_ids, "tags": tags})
        return self.wake_acks

    async def resume_generation(self, stage_ids=None):
        self.resumed += 1

    async def reset_encoder_cache(self):
        self.calls.append("reset_encoder_cache")

    async def reset_prefix_cache(self, reset_connector=False):
        self.calls.append("reset_prefix_cache")


def _make_server(engine, rollout_mode=RolloutMode.HYBRID, node_rank=0, free_cache_engine=True):
    # HYBRID default: release/resume_kv_cache are skipped in COLOCATED mode
    # (parent semantics), so their delegation is exercised via HYBRID.
    server = object.__new__(vLLMOmniHttpServer)
    server.engine = engine
    server.node_rank = node_rank
    server.rollout_mode = rollout_mode
    server.config = SimpleNamespace(free_cache_engine=free_cache_engine)
    server._lora_request_cache = None  # a valid cached value
    return server


# ---------------------------------------------------------------------------
# (a) abort-then-pause ordering; single batched abort; no empty-list abort
# ---------------------------------------------------------------------------


async def test_abort_runs_before_pause_with_one_batched_call():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-1-def": _FakeRequestState("ext-1-def", "ext-1"),  # same external id
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
    }
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    result = await server.abort_all_requests()

    # One batched call with the deduped EXTERNAL ids (request_states keys are
    # internal; engine.abort maps external -> internal).
    assert engine.abort_calls == [["ext-1", "ext-2"]]
    assert engine.calls == ["abort", "pause"], "abort must run while generate is live, pause after"
    assert engine.pause_calls == [{"mode": "abort", "wait_for_inflight_requests": False, "clear_cache": True}]
    assert result == {"aborted_count": 2, "request_ids": ["ext-1", "ext-2"]}


async def test_abort_with_no_in_flight_requests_still_pauses_without_calling_abort():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)

    result = await server.abort_all_requests()

    # An empty-list abort is a silent no-op engine-side — never send one.
    assert engine.abort_calls == []
    assert engine.calls == ["pause"]
    assert result == {"aborted_count": 0, "request_ids": []}


# ---------------------------------------------------------------------------
# (b) abort outputs arrive via the request queues; mapper untouched
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
    # The frozen #403 seam maps finish_reason="abort" to the client vocabulary.
    assert OmniStrategyBase._map_stop_reason(output.finish_reason) == "aborted"
    # The success-path synthetic enqueue is gone entirely (the engine owns
    # terminals — appending empty ones after real ones wipes the partials).
    assert states["ext-1-abc"].queue.qsize() == 0
    assert not hasattr(vLLMOmniHttpServer, "_enqueue_abort_output")


# ---------------------------------------------------------------------------
# (c) all four lifecycle methods delegate, validate ACKs, level=1, keyword tags
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
        assert engine.wake_calls == [{"stage_ids": None, "tags": ["weights"]}]
        assert server._lora_request_cache is server_module._LORA_REQUEST_CACHE_MISS


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
    assert engine.calls == []


async def test_ack_validation_fails_closed():
    bad_acks = [
        # diffusion worker error dict (collective_rpc error shape)
        [{"supported": False, "error": "diffusion stage failed"}, _SUCCESS_ACK],
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
    assert engine.wake_calls == [{"stage_ids": None, "tags": ["weights"]}]


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
# (e) fail-closed abort: enqueue terminals first, then raise; timeout raises
# ---------------------------------------------------------------------------


async def test_abort_failure_enqueues_terminals_then_raises():
    states = {
        "ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1"),
        "ext-2-xyz": _FakeRequestState("ext-2-xyz", "ext-2"),
    }
    engine = _FakeAsyncOmni(states=states, fail_abort=True)
    server = _make_server(engine)

    with pytest.raises(RuntimeError, match="abort rpc failed"):
        await server.abort_all_requests()

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

    # Exactly the engine's real terminal — no empty synthetic appended after it.
    assert states["ext-1-abc"].queue.qsize() == 1
    assert states["ext-1-abc"].queue.get_nowait().engine_outputs.outputs[0].token_ids == [7, 8, 9]


async def test_abort_ack_timeout_raises(monkeypatch):
    monkeypatch.setattr(server_module, "ABORT_ACK_TIMEOUT_S", 0.05)

    class _HangingEngine(_FakeAsyncOmni):
        async def abort(self, request_ids):
            await asyncio.sleep(999)

    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    server = _make_server(_HangingEngine(states=states))

    with pytest.raises(asyncio.TimeoutError):
        await server.abort_all_requests()


# ---------------------------------------------------------------------------
# (f) bounded drain that raises
# ---------------------------------------------------------------------------


async def test_drain_returns_immediately_when_no_requests_in_flight():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)
    await server.wait_for_requests_to_drain()
    assert engine.calls == []


async def test_drain_raises_on_lingering_requests(monkeypatch):
    monkeypatch.setattr(server_module, "ABORT_ACK_TIMEOUT_S", 0.05)
    monkeypatch.setattr(server_module, "_DRAIN_POLL_INTERVAL_S", 0.01)
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    server = _make_server(_FakeAsyncOmni(states=states))

    with pytest.raises(TimeoutError, match="in-flight"):
        await server.wait_for_requests_to_drain()


# ---------------------------------------------------------------------------
# (g) single-request abort shares the contract
# ---------------------------------------------------------------------------


async def test_abort_request_aborts_then_pauses_single_id():
    states = {"ext-1-abc": _FakeRequestState("ext-1-abc", "ext-1")}
    engine = _FakeAsyncOmni(states=states)
    server = _make_server(engine)

    result = await server.abort_request("ext-1")

    assert engine.abort_calls == [["ext-1"]]  # external id, not internal
    assert engine.calls == ["abort", "pause"]
    assert states["ext-1-abc"].queue.qsize() == 1  # engine terminal, no synthetic append
    assert result == {"aborted": True, "request_id": "ext-1"}


async def test_abort_request_unknown_id_skips_abort_but_pauses():
    engine = _FakeAsyncOmni(states={})
    server = _make_server(engine)

    result = await server.abort_request("missing")

    assert engine.abort_calls == []
    assert engine.calls == ["pause"]
    assert result == {"aborted": True, "request_id": "missing"}


# ---------------------------------------------------------------------------
# (h) the manager-level gather observes the server's raise
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
# (i) diffusion-only sleep/wake contract — why diffusion v1 sync needs no
#     resume bridge (drives the REAL AsyncOmni state machine, not a mock)
# ---------------------------------------------------------------------------


def _build_async_omni(stage_type: str):
    """Minimal AsyncOmni driving the real sleep/wake/resume state machine.

    Only the RPC plumbing (EngineCore/worker RPCs, tag bookkeeping, output
    handler) is stubbed; the admission-hold logic under test is the real
    class's. ``stage_type`` selects a pure-diffusion ("diffusion") or
    pure-AR ("llm") single-stage engine.
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
    omni.engine = SimpleNamespace(stage_clients=[SimpleNamespace(stage_type=stage_type)])
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
    """Pure-diffusion sleep must not set the admission hold (engine contract).

    Diffusion v1 sync has no omni-style resume bridge; it relies on
    ``AsyncOmni.sleep()`` setting ``_hold_admission_until_resume`` only for AR
    stages and ``wake_up()`` restoring admission otherwise. If upstream ever
    sets the hold on diffusion sleep (or stops clearing ``_paused`` on a
    diffusion wake), that trainer hangs silently — this tripwire fails first.
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
