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
"""CPU tests for the duplex episode driver against a fake DuplexOmni.

The fake mirrors the engine contract the driver consumes: open/close, paced
appends, events, per-response ``collect_action_records`` draining, and the
post-close terminal cache. No vLLM-Omni duplex symbols are imported, so the
suite runs at the pin.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from typing import Any

import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    record_set_wire,
)
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_duplex_strategy as strategy_module
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_duplex_strategy import (
    DuplexEpisodeError,
    DuplexStrategy,
)


@dataclass
class _FakeLedger:
    """Mirrors the engine ledger's collection semantics.

    ``collect(None)`` prefers sealed sets and only drains the open scope as
    an ``in_progress`` snapshot when nothing is sealed — the exact behavior
    the driver's choreography depends on.
    """

    sealed: list[dict] = field(default_factory=list)
    open_units: int = 2
    #: Whether a speak onset seals a listening scope (an empty listening
    #: scope closes silently engine-side — nothing to seal).
    listen_units: int = 2
    session_id: str = "duplex-0"

    def seal(self, response_id, status="completed"):
        self.sealed.append(record_set_wire(session_id=self.session_id, response_id=response_id, status=status))

    def collect(self, response_id=None):
        response_ids = [wire.get("response_id") for wire in self.sealed]
        if self.sealed and (response_id is None or response_id in response_ids):
            wire = self.sealed.pop() if response_id is None else self.sealed.pop(response_ids.index(response_id))
            return _RecordSet(wire)
        if not self.sealed and self.open_units:
            wire = record_set_wire(session_id=self.session_id, response_id=response_id, status="in_progress")
            return _RecordSet(wire)
        raise _SessionError("action_capture_not_retained")


class FakeHandle:
    """A duplex session handle fake with a scriptable event timeline."""

    def __init__(self, engine: FakeDuplexOmni, session_id: str, *, script: list[str]):
        self.session_id = session_id
        self._engine = engine
        self._script = list(script)
        self._queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.appends: list[tuple[bytes, bool]] = []
        self.collected = 0
        self.commits = 0
        self.ledger = _FakeLedger(session_id=session_id)
        for event_type in self._script:
            self._queue.put_nowait(_Event(event_type))

    async def submit(self, command):  # noqa: ARG002 - the fake records via helpers
        return None

    async def append_audio(self, audio: bytes, *, format: str, sample_rate_hz: int, is_speech=None, **kwargs):
        assert format == "pcm_f32le" and sample_rate_hz == 16000
        self.appends.append((audio, bool(is_speech)))

    async def commit(self, **kwargs):
        self.commits = getattr(self, "commits", 0) + 1

    async def collect_action_records(self, response_id=None, **kwargs):
        self.collected += 1
        return self.ledger.collect(response_id)

    async def close(self, *, reason: str = "client_close", **kwargs):
        if self.closed:
            return
        self.closed = True
        self.close_reason = reason
        self._queue.put_nowait(_Event("session.closed"))

    async def events(self):
        while True:
            event = await self._queue.get()
            if event.type == "response.speak" and self.ledger.listen_units > 0:
                # The engine's begin_response seals the listening scope before
                # any payload of the new response can arrive.
                self.ledger.seal(response_id=None)
            if event.type in ("response.done", "response.completed"):
                self.ledger.seal(response_id=f"resp-{self.collected + 1}")
            if event.type == "audio.cancelled":
                self.ledger.seal(response_id=None, status="truncated_interruption")
            yield event
            if event.type == "session.closed":
                return


@dataclass
class _Event:
    type: str


@dataclass
class _RecordSet:
    wire: dict

    @property
    def status(self):
        return self.wire.get("status")

    def to_wire(self):
        return self.wire


class FakeDuplexOmni:
    """Engine fake: opens scripted sessions, drains one terminal record set."""

    def __init__(self, *, script: list[str], terminal_sets: int = 1, open_error: Exception | None = None):
        self.script = script
        self.terminal_sets = terminal_sets
        self.open_error = open_error
        self.handles: list[FakeHandle] = []
        self.open_configs: list[dict] = []

    async def open_session(self, config, **kwargs):
        if self.open_error is not None:
            raise self.open_error
        self.open_configs.append(dict(config))
        handle = FakeHandle(self, f"duplex-{len(self.handles)}", script=self.script)
        self.handles.append(handle)
        return handle

    def collect_action_records(self, session_id, response_id=None, **kwargs):
        if self.terminal_sets <= 0:
            raise _SessionError("action_capture_not_retained")
        self.terminal_sets -= 1
        return _RecordSet(record_set_wire(session_id=session_id, response_id="terminal"))  # noqa: RET504


class _SessionError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class _Server:
    def __init__(self, engine):
        self.engine = engine


def _strategy(fake_engine, *, cadence=0.01) -> DuplexStrategy:
    strategy = DuplexStrategy(_Server(fake_engine))
    # Tests run at millisecond cadence; the driver only needs the ordering.
    original = strategy_module._DUPLEX_CHUNK_SECONDS
    strategy_module._DUPLEX_CHUNK_SECONDS = cadence
    strategy._restore_cadence = original
    return strategy


def _restore(strategy: DuplexStrategy) -> None:
    strategy_module._DUPLEX_CHUNK_SECONDS = strategy._restore_cadence


def _chunks(count: int, *, speech_from: int | None = None) -> list[dict]:
    return [
        {"pcm": [0.01] * 160, "is_speech": speech_from is not None and index >= speech_from} for index in range(count)
    ]


def _run_spec(**overrides) -> dict:
    spec: dict[str, Any] = {
        "episode_id": "ep-1",
        "axis": "turn",
        "seed": 11,
        "chunks": _chunks(3, speech_from=0),
        "event": {"axis": "turn", "user_turn_end_s": 1.0},
        "tail_horizon_s": 0.0,
    }
    spec.update(overrides)
    return spec


@pytest.mark.asyncio
async def test_turn_episode_feeds_chunks_and_drains_records():
    engine = FakeDuplexOmni(script=["response.done", "session.closed"])
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    assert len(handle.appends) == 3
    # The session opened with capture on and the derived seed.
    assert engine.open_configs[0]["action_capture"] is True
    assert engine.open_configs[0]["seed"] == 11
    # One in-flight collection (response.done) + one terminal drain.
    assert len(result["record_sets"]) == 2
    assert all(wire["session_id"] == handle.session_id for wire in result["record_sets"])
    # The fed chunks ride the result for trajectory assembly.
    assert len(result["client_chunks"]) == 3
    assert handle.closed


@pytest.mark.asyncio
async def test_speech_hints_reach_the_appends():
    engine = FakeDuplexOmni(script=["session.closed"])
    strategy = _strategy(engine)
    try:
        await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    speech_flags = [is_speech for _, is_speech in engine.handles[0].appends]
    assert speech_flags == [True, True, True]
    pcm = struct.unpack("<160f", engine.handles[0].appends[0][0])
    assert pcm[0] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_interruption_injection_waits_for_the_speak_onset():
    script = ["response.speak", "response.done", "session.closed"]
    engine = FakeDuplexOmni(script=script)
    strategy = _strategy(engine)
    spec = _run_spec(
        axis="interruption",
        event={"axis": "interruption", "interrupt_speech_end_s": 2.0},
        inject={"delay_s": 0.0, "tail_silence_chunks": 1},
        speak_wait_horizon_s=2.0,
    )
    try:
        result = await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    appends = engine.handles[0].appends
    # Pre-onset silence + speech + tail silence, all fed after the onset event.
    assert len(appends) == 3 + 1
    assert [is_speech for _, is_speech in appends] == [True, True, True, False]
    assert result["client_chunks"]


@pytest.mark.asyncio
async def test_engine_error_closes_the_session_and_fails_the_episode():
    engine = FakeDuplexOmni(script=["error", "session.closed"])
    strategy = _strategy(engine)
    try:
        with pytest.raises(DuplexEpisodeError, match="engine error"):
            await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    assert engine.handles[0].closed


@pytest.mark.asyncio
async def test_no_records_is_a_rollout_failure():
    engine = FakeDuplexOmni(script=["session.closed"], terminal_sets=0)
    # Remove the in-flight collection too.
    engine.handles = []
    strategy = _strategy(engine)
    original_collect = FakeHandle.collect_action_records

    async def refusing(self, response_id=None, **kwargs):
        raise _SessionError("action_capture_not_retained")

    FakeHandle.collect_action_records = refusing
    try:
        with pytest.raises(DuplexEpisodeError, match="no action-record scopes"):
            await strategy.run_duplex_episode(_run_spec())
    finally:
        FakeHandle.collect_action_records = original_collect
        _restore(strategy)


@pytest.mark.asyncio
async def test_generate_hooks_refuse():
    strategy = DuplexStrategy(_Server(None))
    with pytest.raises(RuntimeError, match="session-only"):
        strategy.preprocess_input(None, {}, None)
    with pytest.raises(RuntimeError, match="session-only"):
        await strategy.run_generation(None, None, "r", None, 0)
    with pytest.raises(RuntimeError, match="session-only"):
        strategy.process_output(None, None, {})


def test_requires_a_duplex_engine_client():
    strategy = DuplexStrategy(_Server(None))
    with pytest.raises(DuplexEpisodeError, match="did not build a duplex engine client"):
        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(strategy._require_duplex_client())


def test_strategy_selector_routes_duplex():
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_async_server import (
        resolve_output_mode_strategy,
    )
    from verl_omni.workers.rollout.vllm_rollout.vllm_omni_diffusion_strategy import DiffusionStrategy

    assert resolve_output_mode_strategy("duplex") is DuplexStrategy
    assert resolve_output_mode_strategy("ar") is ARStrategy
    assert resolve_output_mode_strategy("diffusion") is DiffusionStrategy
    assert resolve_output_mode_strategy("unknown") is DiffusionStrategy


@pytest.mark.asyncio
async def test_collection_choreography_matches_the_ledger_seals():
    # speak seals the listening scope (engine bind runs before any payload of
    # the new response); done seals the response; cancelled seals the
    # barge-in set. Each seal is collected before the next can overflow.
    script = ["response.speak", "audio.cancelled", "session.closed"]
    engine = FakeDuplexOmni(script=script, terminal_sets=0)
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    statuses = [wire["status"] for wire in result["record_sets"]]
    # speak -> sealed listening set (response_id None); cancelled ->
    # truncated_interruption set; the close sealed nothing more.
    assert statuses == ["completed", "truncated_interruption"]
    assert result["record_sets"][0]["response_id"] is None


@pytest.mark.asyncio
async def test_in_progress_drain_fails_the_episode_loudly():
    # No sealed set and a non-empty open scope: collect(None) drains an
    # in_progress snapshot — the driver must refuse to train on it.
    engine = FakeDuplexOmni(script=["response.speak", "session.closed"])
    strategy = _strategy(engine)
    original_open = engine.open_session

    async def _open_listenless(config, **kwargs):
        handle = await original_open(config, **kwargs)
        handle.ledger.listen_units = 0  # the session never listened: speak seals nothing
        handle.ledger.open_units = 2
        return handle

    engine.open_session = _open_listenless
    try:
        with pytest.raises(DuplexEpisodeError, match="in_progress"):
            await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
