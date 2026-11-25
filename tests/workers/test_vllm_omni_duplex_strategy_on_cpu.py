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
appends, events, the commit-affecting unit windows, and the post-close
terminal cache (retention spills every sealed scope there; the driver makes
no in-flight collects). No vLLM-Omni duplex symbols are imported, so the
suite runs at the pin.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from tests.utils.dataset.duplex._duplex_fixtures import (
    LISTEN_TOKEN_ID,
    SPEAK_CONTENT_TOKEN_ID,
    interruption_event,
    record_set_wire,
)
from verl_omni.workers.rollout.vllm_rollout import vllm_omni_duplex_strategy as strategy_module
from verl_omni.workers.rollout.vllm_rollout.vllm_omni_duplex_strategy import (
    DuplexEpisodeError,
    DuplexStrategy,
)


class FakeHandle:
    """A duplex session handle fake with a scriptable event timeline."""

    def __init__(self, engine: FakeDuplexOmni, session_id: str, *, script: list[str]):
        self.session_id = session_id
        self._engine = engine
        self._script = list(script)
        self._queue: asyncio.Queue = asyncio.Queue()
        self.closed = False
        self.appends: list[tuple[bytes, bool]] = []
        self.append_times: list[float] = []
        self.event_times: dict[str, float] = {}
        self.commits = 0
        #: Appends completed before each commit (commit ordering).
        self.commit_log: list[int] = []
        #: In-flight collection attempts — the driver must make none.
        self.collect_calls = 0
        for event_type in self._script:
            self._queue.put_nowait(_Event(event_type))

    async def submit(self, command):  # noqa: ARG002 - the fake records via helpers
        return None

    async def append_audio(self, audio: bytes, *, format: str, sample_rate_hz: int, is_speech=None, **kwargs):
        assert format == "pcm_f32le" and sample_rate_hz == 16000
        self.appends.append((audio, bool(is_speech)))
        self.append_times.append(asyncio.get_running_loop().time())

    async def commit(self, **kwargs):
        self.commits += 1
        self.commit_log.append(len(self.appends))

    async def collect_action_records(self, response_id=None, **kwargs):
        self.collect_calls += 1
        raise _SessionError("action_capture_not_retained")

    async def close(self, *, reason: str = "client_close", **kwargs):
        if self.closed:
            return
        self.closed = True
        self.close_reason = reason
        self._queue.put_nowait(_Event("session.closed"))

    async def events(self):
        while True:
            event = await self._queue.get()
            self.event_times[event.type] = asyncio.get_running_loop().time()
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
    """Engine fake: opens scripted sessions; the terminal cache drains post-close.

    ``terminal_wires`` holds the sealed scopes most-recent-first (the engine's
    drain order); ``open_errors`` replays one error per ``open_session`` call
    (``None`` entries open normally).
    """

    def __init__(
        self,
        *,
        script: list[str],
        terminal_wires: list[dict] | None = None,
        open_errors: list[Exception | None] | None = None,
    ):
        self.script = script
        if terminal_wires is None:
            # One sealed scope by default, like the engine's drain after close.
            terminal_wires = [record_set_wire(session_id="duplex-0", response_id="resp-1")]
        self.terminal_wires = list(terminal_wires)
        self.open_errors = list(open_errors) if open_errors is not None else []
        self.handles: list[FakeHandle] = []
        self.open_configs: list[dict] = []
        #: ``(session_id, session_closed_at_call)`` per terminal-cache collect.
        self.terminal_collect_log: list[tuple[str, bool]] = []

    async def open_session(self, config, **kwargs):
        if self.open_errors:
            error = self.open_errors.pop(0)
            if error is not None:
                raise error
        self.open_configs.append(dict(config))
        handle = FakeHandle(self, f"duplex-{len(self.handles)}", script=self.script)
        self.handles.append(handle)
        return handle

    def collect_action_records(self, session_id, response_id=None, **kwargs):
        handle = self.handles[-1] if self.handles else None
        self.terminal_collect_log.append((session_id, bool(handle is not None and handle.closed)))
        if handle is None or not handle.closed:
            raise AssertionError("terminal-cache collect before the session closed")
        if not self.terminal_wires:
            raise _SessionError("action_capture_not_retained")
        wire = dict(self.terminal_wires.pop(0))
        wire["session_id"] = session_id
        return _RecordSet(wire)


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


def _chunks(count: int, *, speech_from: int | None = None, speech_to: int | None = None) -> list[dict]:
    """``speech_from``/``speech_to`` bracket the speech-carrying chunks (half-open), as a mined clip does."""
    return [
        {"pcm": [0.01] * 160, "is_speech": speech_from is not None and speech_from <= index < (speech_to or count)}
        for index in range(count)
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
    engine = FakeDuplexOmni(
        script=["response.done", "session.closed"],
        terminal_wires=[record_set_wire(session_id="duplex-0", response_id="resp-1")],
    )
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    assert len(handle.appends) == 3
    # The session opened with capture on and the derived seed, and without
    # the ``instructions`` key (an explicit value silences the engine's
    # native duplex pipeline).
    assert engine.open_configs[0]["action_capture"] is True
    assert engine.open_configs[0]["seed"] == 11
    assert "instructions" not in engine.open_configs[0]
    # The scopes arrive through the post-close terminal drain only.
    assert handle.collect_calls == 0
    assert [closed for _, closed in engine.terminal_collect_log] == [True, True]
    assert len(result["record_sets"]) == 1
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
        event=interruption_event(),
        chunks=_chunks(3, speech_from=1),  # a non-speech lead chunk ahead of the speech
        inject={"delay_s": 0.0, "tail_silence_chunks": 1},
        speak_wait_horizon_s=2.0,
    )
    try:
        result = await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    appends = engine.handles[0].appends
    # Speech + tail silence, all fed after the onset event.
    assert len(appends) == 3 + 1
    assert [is_speech for _, is_speech in appends] == [False, True, True, False]
    assert result["client_chunks"]
    # 1-based fed-chunk indices: the clip's first speech-carrying chunk and
    # its last chunk (the tail silence is not part of the clip).
    assert (result["speech_chunk_start_index"], result["speech_chunk_end_index"]) == (2, 3)


@pytest.mark.asyncio
async def test_interruption_without_a_speech_chunk_fails_the_episode():
    engine = FakeDuplexOmni(script=["response.speak", "session.closed"])
    strategy = _strategy(engine)
    spec = _run_spec(
        axis="interruption",
        event=interruption_event(),
        chunks=[{"pcm": [0.0] * 160, "is_speech": False} for _ in range(2)],
        inject={"delay_s": 0.0},
        speak_wait_horizon_s=2.0,
    )
    try:
        with pytest.raises(DuplexEpisodeError, match="no is_speech chunk"):
            await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    assert engine.handles[0].appends == []  # nothing fed before the failure


@pytest.mark.asyncio
async def test_interruption_delay_reanchors_to_the_speak_onset():
    # delay_s > 0: the driver feeds pre-onset silence on the session's input
    # clock and injects the interrupting speech only after onset + delay.
    # The onset rides absolute event-loop time — the same base the feed
    # schedules run on; an episode-relative onset would skip the silence
    # phase and inject immediately.
    engine = FakeDuplexOmni(script=["response.speak", "session.closed"])
    strategy = _strategy(engine, cadence=0.01)
    spec = _run_spec(
        axis="interruption",
        event=interruption_event(),
        chunks=_chunks(3, speech_from=0, speech_to=2),  # a trailing silence chunk inside the clip
        inject={"delay_s": 0.15, "tail_silence_chunks": 1},
        speak_wait_horizon_s=2.0,
    )
    try:
        result = await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    onset_time = handle.event_times["response.speak"]
    first_speech_time = next(
        time for time, (_, is_speech) in zip(handle.append_times, handle.appends, strict=False) if is_speech
    )
    # Only the silences before the first speech append are the pre-onset feed.
    silence_times = [
        time
        for time, (_, is_speech) in zip(handle.append_times, handle.appends, strict=False)
        if not is_speech and time < first_speech_time
    ]
    # ~delay/cadence pre-onset silences (zero would be the clock-base mixup;
    # the onset wait itself polls at 50 ms, hence the wide band).
    assert 5 <= len(silence_times) <= 16
    # The interrupting speech starts no earlier than onset + delay.
    assert first_speech_time - onset_time >= 0.1
    # The end index is the clip's last speech-carrying chunk: the clip's own
    # trailing silence chunk and the injected tail both follow it.
    assert result["speech_chunk_end_index"] == len(result["client_chunks"]) - 2
    assert result["speech_chunk_start_index"] == result["speech_chunk_end_index"] - 1


@pytest.mark.asyncio
async def test_non_interruption_episodes_report_no_speech_indices():
    engine = FakeDuplexOmni(script=["session.closed"])
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    assert result["speech_chunk_start_index"] is None and result["speech_chunk_end_index"] is None


@pytest.mark.asyncio
async def test_commit_rides_the_last_speech_chunk():
    # E4: paced appends are each their own input unit, so a commit after the
    # whole feed lands in the engine's silent-commit drop. It must fire while
    # the last speech-carrying chunk's unit window is still open — never on
    # the non-speech tail.
    engine = FakeDuplexOmni(script=["session.closed"])
    strategy = _strategy(engine)
    chunks = _chunks(3, speech_from=0)
    chunks[2]["is_speech"] = False  # a non-speech tail chunk after the speech
    spec = _run_spec(chunks=chunks, commit=True)
    try:
        await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    assert handle.commit_log == [2]  # right after the last speech chunk, before the tail
    assert len(handle.appends) == 3  # the tail silence is still fed


@pytest.mark.asyncio
async def test_commit_covers_the_whole_speech_tail_when_all_chunks_speak():
    engine = FakeDuplexOmni(script=["session.closed"])
    strategy = _strategy(engine)
    spec = _run_spec(commit=True)
    try:
        await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    assert handle.commit_log == [3]
    assert handle.appends  # commit fired after every fed chunk


@pytest.mark.asyncio
async def test_no_speech_episode_sends_no_commit():
    engine = FakeDuplexOmni(script=["session.closed"])
    strategy = _strategy(engine)
    chunks = [{"pcm": [0.0] * 160, "is_speech": False} for _ in range(2)]
    spec = _run_spec(chunks=chunks, commit=True)
    try:
        await strategy.run_duplex_episode(spec)
    finally:
        _restore(strategy)
    assert engine.handles[0].commits == 0


@pytest.mark.asyncio
async def test_ndarray_chunks_feed_the_full_pcm_stream():
    # The dataset emits ndarray chunks; truth-testing one raises, so the
    # driver normalizes by identity and feeds the whole buffer as f32 bytes.
    engine = FakeDuplexOmni(script=["session.closed"])
    strategy = _strategy(engine)
    chunks = [{"pcm": np.zeros(16000, dtype=np.float32), "is_speech": True} for _ in range(2)]
    try:
        result = await strategy.run_duplex_episode(_run_spec(chunks=chunks))
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    assert all(len(audio) == 16000 * 4 for audio, _ in handle.appends)
    assert len(result["client_chunks"]) == 2
    assert len(result["client_chunks"][0]) == 16000


@pytest.mark.asyncio
async def test_open_session_retries_capacity_exhausted():
    engine = FakeDuplexOmni(script=["session.closed"], open_errors=[_SessionError("capacity_exhausted")])
    strategy_module_retry = strategy_module._DUPLEX_OPEN_RETRY_S
    strategy_module._DUPLEX_OPEN_RETRY_S = 0.001
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        strategy_module._DUPLEX_OPEN_RETRY_S = strategy_module_retry
        _restore(strategy)
    assert len(engine.handles) == 1  # the retried open succeeded
    assert result["session_id"] == engine.handles[0].session_id


@pytest.mark.asyncio
async def test_open_session_deadline_raises():
    engine = FakeDuplexOmni(
        script=["session.closed"],
        open_errors=[_SessionError("capacity_exhausted")] * 50,
    )
    retry_s, wait_s = strategy_module._DUPLEX_OPEN_RETRY_S, strategy_module._DUPLEX_OPEN_WAIT_S
    strategy_module._DUPLEX_OPEN_RETRY_S = 0.001
    strategy_module._DUPLEX_OPEN_WAIT_S = 0.05
    strategy = _strategy(engine)
    try:
        with pytest.raises(DuplexEpisodeError, match="capacity_exhausted"):
            await strategy.run_duplex_episode(_run_spec())
    finally:
        strategy_module._DUPLEX_OPEN_RETRY_S = retry_s
        strategy_module._DUPLEX_OPEN_WAIT_S = wait_s
        _restore(strategy)


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
    engine = FakeDuplexOmni(script=["session.closed"], terminal_wires=[])
    strategy = _strategy(engine)
    try:
        with pytest.raises(DuplexEpisodeError, match="no action-record scopes"):
            await strategy.run_duplex_episode(_run_spec())
    finally:
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
async def test_collection_drains_after_close_and_in_chronological_order():
    # speak/cancelled/done seal scopes engine-side; retention spills every
    # sealed scope into the terminal cache before the next seal, so the
    # driver collects nothing in-flight — everything drains post-close and
    # re-sorts chronologically even though the cache hands scopes back
    # most-recent-first.
    script = ["response.speak", "audio.cancelled", "response.done", "session.closed"]
    late = record_set_wire(
        session_id="duplex-0",
        response_id="resp-late",
        epoch=1,
        turn_id=2,
        status="completed",
        units=[{"action": (SPEAK_CONTENT_TOKEN_ID, "sampled", -1.0), "emission_ns": 9_000_000_000}],
    )
    early = record_set_wire(
        session_id="duplex-0",
        response_id=None,
        epoch=1,
        turn_id=1,
        status="truncated_interruption",
        units=[{"action": (LISTEN_TOKEN_ID, "sampled", -0.2), "emission_ns": 3_000_000_000}],
    )
    engine = FakeDuplexOmni(script=script, terminal_wires=[late, early])
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    handle = engine.handles[0]
    assert handle.collect_calls == 0  # no in-flight collection
    assert all(closed for _, closed in engine.terminal_collect_log)
    assert [wire["response_id"] for wire in result["record_sets"]] == [None, "resp-late"]
    assert [wire["status"] for wire in result["record_sets"]] == ["truncated_interruption", "completed"]
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_earlier_epoch_sorts_first_regardless_of_cache_order():
    first = record_set_wire(
        session_id="duplex-0",
        response_id="resp-0",
        epoch=0,
        turn_id=0,
        status="completed",
        units=[{"action": (LISTEN_TOKEN_ID, "sampled", -0.1), "emission_ns": 1_000_000_000}],
    )
    second = record_set_wire(
        session_id="duplex-0",
        response_id="resp-1",
        epoch=1,
        turn_id=1,
        status="completed",
        units=[{"action": (SPEAK_CONTENT_TOKEN_ID, "sampled", -1.0), "emission_ns": 2_000_000_000}],
    )
    engine = FakeDuplexOmni(script=["session.closed"], terminal_wires=[second, first])
    strategy = _strategy(engine)
    try:
        result = await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
    assert [wire["epoch"] for wire in result["record_sets"]] == [0, 1]


@pytest.mark.asyncio
async def test_unbounded_terminal_cache_fails_loudly():
    # A cache that never answers action_capture_not_retained must not spin
    # the drain forever; the episode fails instead of silently dropping or
    # hanging.
    wires = [record_set_wire(session_id="duplex-0", response_id=f"resp-{index}") for index in range(64)]
    engine = FakeDuplexOmni(script=["session.closed"], terminal_wires=wires + [wires[0]] * 10)
    strategy = _strategy(engine)
    try:
        with pytest.raises(DuplexEpisodeError, match="never answered"):
            await strategy.run_duplex_episode(_run_spec())
    finally:
        _restore(strategy)
