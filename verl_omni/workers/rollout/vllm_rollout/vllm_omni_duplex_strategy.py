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
"""The duplex rollout strategy: real duplex sessions in the rollout server.

Selected by ``engine_kwargs.vllm_omni.output_mode="duplex"``. Reuses the AR
strategy's config handling (the generated deploy config comes from the same
rollout-adapter stage topology) but serves no ``generate``: duplex rollout is
session-only. The server builds ``DuplexOmni`` through the same engine-args
path as ``AsyncOmni``, and :meth:`DuplexStrategy.run_duplex_episode` drives
one engine-resident session per episode next to the engine process:

- open with ``action_capture: true`` and the episode's derived seed;
- feed the user track chunk-by-chunk at the native 1 s cadence (wall-clock
  paced — every concurrent session's next submit must beat the unit deadline
  or the runtime's silence-continuation clock injects unscheduled units);
- watch ``response.speak`` to key interruption injection to the policy's own
  speak onset — the harness never cancels model output — and report the
  fed-chunk indices bracketing the interrupting speech (the trainer anchors
  the yield reward's onset and end onto them);
- make no in-flight collects (the post-fix engine spills every sealed scope
  into a terminal cache before the next seal, so mid-episode collection is
  structurally racy): after close, drain the terminal cache until the engine
  answers ``action_capture_not_retained`` and re-sort the scopes
  chronologically by ``(epoch, turn_id, emission_ns)``;
- close every session before returning, so the engine's weight-sync fence
  (``weight_sync_sessions_open``) is never tripped by this driver.

The whole episode runs server-side to keep the 1 s submits off verl's
cross-process transport; the agent loop makes one Ray call per episode batch.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from verl_omni.workers.rollout.vllm_rollout.vllm_omni_ar_strategy import ARStrategy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#: Native duplex cadence: one 1 s PCM chunk per unit at 16 kHz.
_DUPLEX_SAMPLE_RATE = 16000
_DUPLEX_CHUNK_SECONDS = 1.0
#: How long an episode may run past its last fed chunk before the driver
#: closes the session (bounded episodes, RFC #905 §3.1: segment + tail).
_DEFAULT_TAIL_HORIZON_S = 8.0
#: Bound on collect_action_records draining after close (terminal cache).
_TERMINAL_DRAIN_TIMEOUT_S = 10.0
#: Safety bound on the scopes one episode's terminal cache may hold; a longer
#: drain means the engine ignores ``action_capture_not_retained``.
_TERMINAL_DRAIN_MAX_SCOPES = 64
#: ``open_session`` slot retry: batch waves transiently exceed one engine's
#: session slots; retry ``capacity_exhausted`` opens instead of failing episodes.
_DUPLEX_OPEN_RETRY_S = 2.0
_DUPLEX_OPEN_WAIT_S = 180.0


class DuplexEpisodeError(RuntimeError):
    """A duplex episode failed; the episode is a rollout failure, not data."""


def speak_wait_seconds(run_spec: Mapping[str, Any]) -> float:
    """Wait for the first speak: the episode feed plus its tail, capped by the requested horizon."""
    requested = float(run_spec.get("speak_wait_horizon_s", 30.0))
    chunks = run_spec.get("chunks") or []
    feed_s = len(chunks) * _DUPLEX_CHUNK_SECONDS
    tail_s = float(run_spec.get("tail_horizon_s", _DEFAULT_TAIL_HORIZON_S))
    return min(requested, feed_s + tail_s)


def _pcm_f32le_bytes(samples: Any) -> bytes:
    """Little-endian float32 PCM bytes of one chunk (list or ndarray)."""
    return struct.pack(f"<{len(samples)}f", *[float(value) for value in samples])


def _silence_chunk(samples: int = _DUPLEX_SAMPLE_RATE) -> list[float]:
    return [0.0] * samples


def _record_sort_key(wire: Mapping[str, Any]) -> tuple[int, int, int]:
    """Chronological key of one record-set wire: ``(epoch, turn_id, emission_ns)``.

    ``emission_ns`` is the scope's first stamped unit emission (scopes are
    single-turn, so the first emission orders scopes within an epoch); a
    scope with no stamps sorts first.
    """
    emissions = [int(value) for value in (wire.get("emission_ns") or ()) if value is not None]
    return (int(wire.get("epoch") or 0), int(wire.get("turn_id") or 0), emissions[0] if emissions else -1)


class DuplexStrategy(ARStrategy):
    """Session-driven duplex rollout mode (see module docstring)."""

    #: The duplex engine patches nothing on the model runner.
    patches_model_runner = False

    @staticmethod
    def engine_client_cls() -> type:
        """``DuplexOmni``, imported lazily so the pin stays importable."""
        from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

        return require_duplex_vllm_omni()

    # ---- generate is not the duplex surface ------------------------------- #

    def preprocess_input(self, request, sampling_params, lora_request):
        raise RuntimeError(
            "duplex rollout is session-only: generate() is not served. The duplex "
            "agent loop drives sessions through run_duplex_episode()."
        )

    async def run_generation(self, prompt, params, request_id, lora_request, priority):
        raise RuntimeError("duplex rollout is session-only; use run_duplex_episode().")

    def process_output(self, final_res, params, sampling_params):
        raise RuntimeError("duplex rollout is session-only; use run_duplex_episode().")

    # ---- the episode driver ----------------------------------------------- #

    async def run_duplex_episode(self, run_spec: Mapping[str, Any]) -> dict[str, Any]:
        """Drive one duplex session and return its captured record sets.

        Args:
            run_spec: One episode: ``episode_id``, ``axis``, ``seed``,
                ``chunks`` (list of ``{"pcm": [floats or ndarray],
                "is_speech": bool}`` in feed order), ``event`` (the
                episode-relative annotation), optional ``commit`` (ride the
                commit on the last speech chunk's unit window), optional
                ``inject`` (``{"delay_s": float}`` — interruption episodes
                inject their speech chunks relative to the observed speak
                onset), and ``tail_horizon_s``.

        Returns:
            ``{"episode_id", "session_id", "status", "record_sets": [wire],
            "event": {...}, "client_chunks": [[float], ...],
            "speech_chunk_start_index", "speech_chunk_end_index"}`` — the drained
            scopes in chronological order plus every chunk the driver fed; the
            two indices are 1-based in fed order (interruption episodes only,
            else ``None``): the interrupting clip's first speech-carrying chunk
            and its last chunk.

        Raises:
            DuplexEpisodeError: On capture failures (``action_capture_*``),
                slot-wait deadlines, or engine errors; the session is closed
                either way.
        """
        duplex_omni = self._require_duplex_client()
        episode_id = str(run_spec["episode_id"])
        record_wires: list[dict[str, Any]] = []
        speech_chunk_span: tuple[int, int] | None = None
        session = None
        consumer: asyncio.Task | None = None
        try:
            # No ``instructions`` key: an explicit value silences the engine's
            # native duplex pipeline (zero ``response.listen`` events, empty
            # ledger). The engine's duplex policy substitutes the same default
            # system text, which the trainer's replay context carries instead.
            session = await self._open_session_with_slot_wait(
                duplex_omni,
                {
                    "modalities": ["audio", "text"],
                    "action_capture": True,
                    "seed": int(run_spec["seed"]),
                    "temperature": float(run_spec.get("temperature", 0.7)),
                    # One unit per append. Without this the feed buffers until commit.
                    "extra_body": {"auto_response": True},
                },
            )
            speak_onset: dict[str, float] = {}
            consumer = asyncio.create_task(self._consume_events(session, speak_onset, episode_id))
            fed_chunks: list[list[float]] = []
            speech_chunk_span = await self._feed_episode(session, run_spec, speak_onset, fed_chunks)
            # Let the tail play out; the consumer stops at close.
            await asyncio.sleep(float(run_spec.get("tail_horizon_s", _DEFAULT_TAIL_HORIZON_S)))
            await session.close(reason="client_close")
            await asyncio.wait_for(consumer, timeout=_TERMINAL_DRAIN_TIMEOUT_S)
            # Post-close drain: retention spills every sealed scope into the
            # terminal cache before the next seal, so consumers collect
            # nothing in-flight; the cache drains after close (most recent
            # first) and the scopes re-sort chronologically.
            record_wires.extend(await asyncio.to_thread(self._drain_terminal, duplex_omni, session.session_id))
            record_wires.sort(key=_record_sort_key)
        except Exception as error:
            raise DuplexEpisodeError(f"duplex episode {episode_id} failed: {error}") from error
        finally:
            if consumer is not None and not consumer.done():
                consumer.cancel()
            if session is not None and not getattr(session, "closed", True):
                try:
                    await session.close(reason="rollout_error")
                except Exception:  # noqa: BLE001 - the original error wins
                    logger.exception("duplex episode %s: session close failed", episode_id)
            # A failure before the success-path drain must still collect sealed
            # sets, or the engine expires them while the slot stays taken.
            if session is not None and not record_wires:
                try:
                    await asyncio.to_thread(self._drain_terminal, duplex_omni, session.session_id)
                except Exception:
                    logger.exception("duplex episode %s: terminal drain failed", episode_id)
        if not record_wires:
            raise DuplexEpisodeError(
                f"duplex episode {episode_id} produced no action-record scopes; capture must be on"
            )
        return {
            "episode_id": episode_id,
            "session_id": session.session_id,
            "status": record_wires[-1].get("status", "unknown"),
            "record_sets": record_wires,
            "event": dict(run_spec.get("event") or {}),
            "client_chunks": fed_chunks,
            "speech_chunk_start_index": speech_chunk_span[0] if speech_chunk_span else None,
            "speech_chunk_end_index": speech_chunk_span[1] if speech_chunk_span else None,
        }

    def _require_duplex_client(self) -> Any:
        """The server's engine client, checked for the session surface it serves."""
        engine = getattr(self.server, "engine", None)
        if engine is None or not callable(getattr(engine, "open_session", None)):
            raise DuplexEpisodeError(
                "the rollout server did not build a duplex engine client; duplex rollout "
                "requires output_mode=duplex from engine start"
            )
        return engine

    async def _open_session_with_slot_wait(self, duplex_omni: Any, config: Mapping[str, Any]) -> Any:
        """Open a session, retrying ``capacity_exhausted`` until the deadline.

        Batch waves transiently exceed one engine's session slots; the open
        is retried every :data:`_DUPLEX_OPEN_RETRY_S` up to
        :data:`_DUPLEX_OPEN_WAIT_S` and any other error (or the deadline)
        raises into the episode failure path.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _DUPLEX_OPEN_WAIT_S
        while True:
            try:
                return await duplex_omni.open_session(dict(config))
            except Exception as error:  # noqa: BLE001 - re-raised below
                code = str(getattr(error, "code", "") or "")
                if "capacity_exhausted" not in code and "capacity_exhausted" not in str(error):
                    raise
                if loop.time() >= deadline:
                    raise
                logger.info("duplex open: engine session slots exhausted; retrying in %.0fs", _DUPLEX_OPEN_RETRY_S)
                await asyncio.sleep(_DUPLEX_OPEN_RETRY_S)

    async def _consume_events(
        self,
        session: Any,
        speak_onset: dict[str, float],
        episode_id: str,
    ) -> None:
        """Watch the session's events: the speak onset + engine errors.

        No collection happens here: retention spills every sealed scope into
        the engine's terminal cache before the next seal, so any
        in-flight collect is structurally racy and the driver drains
        post-close instead (``run_duplex_episode``). The onset is stored in
        absolute event-loop time — the same clock the ``_feed_episode``
        schedules run on.
        """
        loop = asyncio.get_running_loop()
        try:
            async for event in session.events():
                event_type = getattr(event, "type", "") or ""
                if event_type == "response.speak" and "at" not in speak_onset:
                    speak_onset["at"] = loop.time()
                if event_type == "error":
                    raise DuplexEpisodeError(
                        f"duplex episode {episode_id} engine error: {getattr(event, 'code', event)}"
                    )
        except DuplexEpisodeError:
            raise
        except Exception as error:  # noqa: BLE001 - surfaced by the driver
            raise DuplexEpisodeError(f"duplex episode {episode_id} event stream failed: {error}") from error

    async def _feed_episode(
        self,
        session: Any,
        run_spec: Mapping[str, Any],
        speak_onset: dict[str, float],
        fed_chunks: list[list[float]],
    ) -> tuple[int, int] | None:
        """Feed the episode's chunks at the native 1 s cadence.

        Pause/turn episodes feed their user-track window straight through.
        Interruption episodes feed driver-generated silence until the policy
        speaks (the observed onset plus ``inject.delay_s``), then the
        interrupting speech, then ``inject.tail_silence_chunks`` of silence —
        the harness never cancels model output, so yielding stays the model's
        own decision. A model that never speaks within the horizon still gets
        the speech fed (the reward's silence floor takes care of it).

        Returns:
            For interruption episodes, ``(start, end)``: the 1-based indices,
            in fed order, of the interrupting clip's first speech-carrying
            chunk and of its last chunk — the i-th index maps onto the i-th
            client-origin unit decision, where the trainer re-anchors the
            yield reward's onset and end. ``None`` otherwise.
        """
        chunks = list(run_spec.get("chunks") or [])
        inject = run_spec.get("inject")
        loop = asyncio.get_running_loop()
        schedule = loop.time() + _DUPLEX_CHUNK_SECONDS
        speech_chunk_span: tuple[int, int] | None = None
        # Each paced append is its own input unit, so a commit sent after the
        # whole feed is dropped as silence: it rides the last speech chunk.
        commit = bool(run_spec.get("commit", False))
        speech_indices = [index for index, chunk in enumerate(chunks) if bool(chunk.get("is_speech", True))]
        last_speech_index = speech_indices[-1] if speech_indices else None

        async def _feed_chunks(feed_chunks: list[dict[str, Any]]) -> None:
            nonlocal schedule
            for index, chunk in enumerate(feed_chunks):
                await self._submit_at(session, chunk, schedule, run_spec, fed_chunks)
                if commit and index == last_speech_index:
                    await session.commit(final=True, create_response=True)
                schedule += _DUPLEX_CHUNK_SECONDS

        if inject is None:
            await _feed_chunks(chunks)
        else:
            if last_speech_index is None:
                raise DuplexEpisodeError(
                    f"duplex episode {run_spec['episode_id']}: the interrupting clip carries no "
                    "is_speech chunk; the mined window must overlap the user's words"
                )
            delay = float(inject.get("delay_s", 0.0))
            try:
                onset = await self._await_speak_onset(speak_onset, run_spec)
            except DuplexEpisodeError:
                logger.warning(
                    "duplex episode %s: no speak onset within the horizon; feeding the "
                    "interrupting speech without overlap",
                    run_spec["episode_id"],
                )
                onset = loop.time()
            # Pre-onset silence keeps the session's input clock at the cadence.
            while loop.time() < onset + delay - _DUPLEX_CHUNK_SECONDS / 2:
                await self._submit_at(
                    session, {"pcm": _silence_chunk(), "is_speech": False}, schedule, run_spec, fed_chunks
                )
                schedule += _DUPLEX_CHUNK_SECONDS
                schedule = max(schedule, loop.time())
            schedule = max(schedule, onset + delay)
            # Both anchors ride the fed-chunk index: the clip's first and last
            # speech-carrying chunks open and close the speech (the tail is silence).
            speech_chunk_span = (len(fed_chunks) + speech_indices[0] + 1, len(fed_chunks) + last_speech_index + 1)
            await _feed_chunks(chunks)
            for _ in range(int(inject.get("tail_silence_chunks", 2))):
                await self._submit_at(
                    session, {"pcm": _silence_chunk(), "is_speech": False}, schedule, run_spec, fed_chunks
                )
                schedule += _DUPLEX_CHUNK_SECONDS
        return speech_chunk_span

    async def _await_speak_onset(self, speak_onset: dict[str, float], run_spec: Mapping[str, Any]) -> float:
        """Wait for the policy's first speak decision, bounded by the horizon."""
        horizon = speak_wait_seconds(run_spec)
        deadline = asyncio.get_running_loop().time() + horizon
        while "at" not in speak_onset:
            if asyncio.get_running_loop().time() >= deadline:
                raise DuplexEpisodeError(
                    f"duplex episode {run_spec['episode_id']} never spoke within {horizon}s; "
                    "cannot key interruption injection to a speak onset"
                )
            await asyncio.sleep(0.05)
        return speak_onset["at"]

    async def _submit_at(
        self,
        session: Any,
        chunk: Mapping[str, Any],
        schedule: float,
        run_spec: Mapping[str, Any],
        fed_chunks: list,
    ) -> None:
        """Submit one chunk on the 1 s schedule, warning when a deadline slips.

        ``pcm`` may be a numpy array (the dataset emits ndarray chunks);
        truth-testing one is ambiguous, so it is normalized by identity.
        """
        now = asyncio.get_running_loop().time()
        if schedule > now:
            await asyncio.sleep(schedule - now)
        elif now - schedule > 0.25:
            logger.warning(
                "duplex episode %s: chunk submit %.2fs past the unit deadline",
                run_spec["episode_id"],
                now - schedule,
            )
        pcm = chunk.get("pcm")
        samples = np.asarray([] if pcm is None else pcm, dtype=np.float32)
        await session.append_audio(
            _pcm_f32le_bytes(samples),
            format="pcm_f32le",
            sample_rate_hz=_DUPLEX_SAMPLE_RATE,
            is_speech=bool(chunk.get("is_speech", True)),
        )
        fed_chunks.append(samples.tolist())

    def _drain_terminal(self, duplex_omni: Any, session_id: str) -> list[dict[str, Any]]:
        """Drain the post-close terminal cache (bounded by its own deadline).

        The cache hands scopes back most recent first and stops with the
        ``action_capture_not_retained`` status once drained; a longer drain
        means the engine never answers ``not_retained`` and the episode
        fails loudly rather than silently dropping scopes.
        """
        wires: list[dict[str, Any]] = []
        for _ in range(_TERMINAL_DRAIN_MAX_SCOPES):
            try:
                record_set = duplex_omni.collect_action_records(session_id, timeout=_TERMINAL_DRAIN_TIMEOUT_S)
            except Exception as error:  # noqa: BLE001 - drained or expired
                if "not_retained" in str(error) or getattr(error, "code", "") == "action_capture_not_retained":
                    break
                raise
            wires.append(record_set.to_wire())
        else:
            raise DuplexEpisodeError(
                f"duplex session {session_id}: the terminal cache never answered "
                f"action_capture_not_retained within {_TERMINAL_DRAIN_MAX_SCOPES} collects"
            )
        return wires


__all__ = ["DuplexEpisodeError", "DuplexStrategy"]
