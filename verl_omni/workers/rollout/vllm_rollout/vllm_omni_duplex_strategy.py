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
  speak onset — the harness never cancels model output;
- drain one sealed ``DuplexActionRecordSet`` per response while the episode
  runs (upstream retention is the current response plus one sealed set, so
  the collector must keep up) and the terminal cache after close;
- close every session before returning, so the engine's weight-sync fence
  (``weight_sync_sessions_open``) is never tripped by this driver.

The whole episode runs server-side to keep the 1 s submits off verl's
cross-process transport; the agent loop makes one Ray call per episode batch.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import Mapping, Sequence
from typing import Any

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


class DuplexEpisodeError(RuntimeError):
    """A duplex episode failed; the episode is a rollout failure, not data."""


def _pcm_f32le_bytes(samples: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(samples)}f", *[float(value) for value in samples])


def _silence_chunk(samples: int = _DUPLEX_SAMPLE_RATE) -> list[float]:
    return [0.0] * samples


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
                ``chunks`` (list of ``{"pcm": [floats], "is_speech": bool}``
                in feed order), ``event`` (the episode-relative annotation),
                ``instructions``, optional ``inject`` (``{"delay_s": float}`` —
                interruption episodes inject their speech chunks relative to
                the observed speak onset), and ``tail_horizon_s``.

        Returns:
            ``{"episode_id", "session_id", "status", "record_sets": [wire],
            "event": {...}, "client_chunks": [[float], ...]}`` — the drained
            scopes in collection order plus every chunk the driver fed.

        Raises:
            DuplexEpisodeError: On capture failures (``action_capture_*``) or
                engine errors; the session is closed either way.
        """
        duplex_omni = self._require_duplex_client()
        episode_id = str(run_spec["episode_id"])
        record_wires: list[dict[str, Any]] = []
        # 1-based index of the last fed chunk carrying the interrupting
        # speech (interruption episodes); the trainer maps it onto the unit
        # clock, because the injection point keys to the policy's own onset.
        speech_chunk_end_index: int | None = None
        session = None
        try:
            session = await duplex_omni.open_session(
                {
                    "modalities": ["audio", "text"],
                    "instructions": run_spec.get("instructions") or "Streaming Omni Conversation.",
                    "action_capture": True,
                    "seed": int(run_spec["seed"]),
                    "temperature": float(run_spec.get("temperature", 0.7)),
                }
            )
            speak_onset: dict[str, float] = {}
            consumer = asyncio.create_task(self._consume_events(session, record_wires, speak_onset, episode_id))
            fed_chunks: list[list[float]] = []
            speech_chunk_end_index = await self._feed_episode(session, run_spec, speak_onset, fed_chunks)
            # Let the tail play out; the consumer stops at close.
            await asyncio.sleep(float(run_spec.get("tail_horizon_s", _DEFAULT_TAIL_HORIZON_S)))
            await session.close(reason="client_close")
            await asyncio.wait_for(consumer, timeout=_TERMINAL_DRAIN_TIMEOUT_S)
            # Post-close drain: the terminal cache holds the last sealed set.
            record_wires.extend(await asyncio.to_thread(self._drain_terminal, duplex_omni, session.session_id))
        except Exception as error:
            if session is not None and not session.closed:
                try:
                    await session.close(reason="rollout_error")
                except Exception:  # noqa: BLE001 - the original error wins
                    logger.exception("duplex episode %s: session close failed", episode_id)
            raise DuplexEpisodeError(f"duplex episode {episode_id} failed: {error}") from error
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
            "speech_chunk_end_index": speech_chunk_end_index,
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

    async def _consume_events(
        self,
        session: Any,
        record_wires: list[dict[str, Any]],
        speak_onset: dict[str, float],
        episode_id: str,
    ) -> None:
        """Watch the session's events: speak onset + scope-transition collection.

        Collection is draining and retention holds one sealed set, so every
        seal must be collected before the next one fires
        (``action_capture_retention_overflow`` is session-fatal). The seals
        and their events: ``response.speak`` — the engine opened the
        response, which sealed the listening scope as completed (collection
        with ``None`` prefers sealed sets, so the sealed listening set is
        what returns; passing the *new* response's id here would instead
        drain the open response scope mid-flight); ``response.done`` /
        ``response.completed`` — the response boundary sealed the response
        set; ``audio.cancelled`` — a barge-in sealed it as
        ``truncated_interruption``. Whatever the close seals arrives through
        the post-close terminal drain.
        """
        loop = asyncio.get_running_loop()
        opened_at = loop.time()
        try:
            async for event in session.events():
                event_type = getattr(event, "type", "") or ""
                response_id = getattr(event, "response_id", None)
                if event_type == "response.speak":
                    if "at" not in speak_onset:
                        speak_onset["at"] = loop.time() - opened_at
                    await self._collect_sealed(session, record_wires, None)
                if event_type in ("response.done", "response.completed"):
                    await self._collect_sealed(session, record_wires, response_id)
                if event_type == "audio.cancelled":
                    await self._collect_sealed(session, record_wires, None)
                if event_type == "error":
                    raise DuplexEpisodeError(
                        f"duplex episode {episode_id} engine error: {getattr(event, 'code', event)}"
                    )
        except DuplexEpisodeError:
            raise
        except Exception as error:  # noqa: BLE001 - surfaced by the driver
            raise DuplexEpisodeError(f"duplex episode {episode_id} event stream failed: {error}") from error

    async def _collect_sealed(
        self,
        session: Any,
        record_wires: list[dict[str, Any]],
        response_id: str | None,
    ) -> None:
        """Drain one sealed scope; an already-drained scope is not an error.

        An ``in_progress`` snapshot means the open scope was drained mid-
        flight — its units left the ledger and the eventual sealed set would
        silently lose them, so the episode fails loudly instead of training
        on corrupted records.
        """
        try:
            record_set = await session.collect_action_records(response_id)
        except Exception as error:  # noqa: BLE001 - drained or not retained yet
            if "not_retained" in str(error) or getattr(error, "code", "") == "action_capture_not_retained":
                return
            raise
        if record_set.status == "in_progress":
            raise DuplexEpisodeError(
                "collected an in_progress snapshot (the open scope drained mid-flight); "
                "refusing to train on records whose sealed successor lost units"
            )
        record_wires.append(record_set.to_wire())

    async def _feed_episode(
        self,
        session: Any,
        run_spec: Mapping[str, Any],
        speak_onset: dict[str, float],
        fed_chunks: list[list[float]],
    ) -> int | None:
        """Feed the episode's chunks at the native 1 s cadence.

        Pause/turn episodes feed their user-track window straight through.
        Interruption episodes feed driver-generated silence until the policy
        speaks (the observed onset plus ``inject.delay_s``), then the
        interrupting speech, then ``inject.tail_silence_chunks`` of silence —
        the harness never cancels model output, so yielding stays the model's
        own decision. A model that never speaks within the horizon still gets
        the speech fed (the reward scores never-spoke as nothing-to-yield).

        Returns:
            For interruption episodes, the 1-based index (in fed order) of
            the last chunk carrying the interrupting speech; ``None``
            otherwise.
        """
        chunks = list(run_spec.get("chunks") or [])
        inject = run_spec.get("inject")
        loop = asyncio.get_running_loop()
        schedule = loop.time() + _DUPLEX_CHUNK_SECONDS
        speech_chunk_end_index: int | None = None
        if inject is None:
            for chunk in chunks:
                await self._submit_at(session, chunk, schedule, run_spec, fed_chunks)
                schedule += _DUPLEX_CHUNK_SECONDS
        else:
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
            for chunk in chunks:
                await self._submit_at(session, chunk, schedule, run_spec, fed_chunks)
                speech_chunk_end_index = len(fed_chunks)
                schedule += _DUPLEX_CHUNK_SECONDS
            for _ in range(int(inject.get("tail_silence_chunks", 2))):
                await self._submit_at(
                    session, {"pcm": _silence_chunk(), "is_speech": False}, schedule, run_spec, fed_chunks
                )
                schedule += _DUPLEX_CHUNK_SECONDS
        if run_spec.get("commit", False):
            await session.commit(final=True, create_response=True)
        return speech_chunk_end_index

    async def _await_speak_onset(self, speak_onset: dict[str, float], run_spec: Mapping[str, Any]) -> float:
        """Wait for the policy's first speak decision, bounded by the horizon."""
        horizon = float(run_spec.get("speak_wait_horizon_s", 30.0))
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
        """Submit one chunk on the 1 s schedule, warning when a deadline slips."""
        now = asyncio.get_running_loop().time()
        if schedule > now:
            await asyncio.sleep(schedule - now)
        elif now - schedule > 0.25:
            logger.warning(
                "duplex episode %s: chunk submit %.2fs past the unit deadline",
                run_spec["episode_id"],
                now - schedule,
            )
        await session.append_audio(
            _pcm_f32le_bytes(chunk.get("pcm") or []),
            format="pcm_f32le",
            sample_rate_hz=_DUPLEX_SAMPLE_RATE,
            is_speech=bool(chunk.get("is_speech", True)),
        )
        fed_chunks.append(list(chunk.get("pcm") or []))

    def _drain_terminal(self, duplex_omni: Any, session_id: str) -> list[dict[str, Any]]:
        """Drain the post-close terminal cache (bounded by its own deadline)."""
        wires: list[dict[str, Any]] = []
        for _ in range(4):  # one sealed scope per response; close seals at most a couple
            try:
                record_set = duplex_omni.collect_action_records(session_id, timeout=_TERMINAL_DRAIN_TIMEOUT_S)
            except Exception as error:  # noqa: BLE001 - drained or expired
                if "not_retained" in str(error) or getattr(error, "code", "") == "action_capture_not_retained":
                    break
                raise
            wires.append(record_set.to_wire())
        return wires


__all__ = ["DuplexEpisodeError", "DuplexStrategy"]
