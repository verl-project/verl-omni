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
"""The duplex trajectory: what the trainer owns on top of the engine's records.

A rollout episode is a real duplex session in vLLM-Omni. The engine seals one
:class:`DuplexActionRecordSet` per response or listening scope (RFC verl-omni
#905 §3.2); :class:`DuplexTrajectory` assembles those scopes in order and adds
the trainer's own view: the raw media it fed (one PCM chunk per client unit),
the episode's event annotation (which timing axis the segment exercises), and
the derived per-unit decision timeline the rule rewards and telemetry read.

Input embeddings are *not* captured engine-side; the replay module re-derives
them from the chunk PCM through the prompt-parity discipline. Engine-injected
silent units (``origin=engine``) are part of the replayed input sequence — the
trainer did not feed them, and only the engine knows they happened.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from verl_omni.utils.dataset.duplex.records import (
    ACTION_REEMIT,
    ACTION_SAMPLED,
    ORIGIN_CLIENT,
    STATUS_TRUNCATED_INTERRUPTION,
    DuplexActionRecordSet,
    RecordSetError,
    validate_record_sets,
)

__all__ = [
    "AXIS_INTERRUPTION",
    "AXIS_PAUSE",
    "AXIS_TURN",
    "DuplexEpisodeEvent",
    "DuplexTrajectory",
    "DuplexUnitDecision",
    "reanchor_interruption",
]

#: Event axes of the replayed two-track corpus (RFC #905 §3.1).
AXIS_PAUSE = "pause"
AXIS_TURN = "turn"
AXIS_INTERRUPTION = "interruption"
_EVENT_AXES = frozenset({AXIS_PAUSE, AXIS_TURN, AXIS_INTERRUPTION})


@dataclass(frozen=True, slots=True)
class DuplexEpisodeEvent:
    """One mined segment's annotation: the axis and its timing windows.

    Times are seconds relative to the episode start (the first fed chunk);
    windows are half-open ``[start, end)``.
    """

    axis: str
    #: Pause axis: the user's intra-utterance silence the model must sit through.
    pause_window_s: tuple[float, float] | None = None
    #: Turn axis: when the user's utterance ends (the model should take the floor).
    user_turn_end_s: float | None = None
    #: Interruption axis: when the interrupting speech starts (the model
    #: should fall silent promptly after this; the yield latency's anchor) and
    #: when it ends.
    interrupt_speech_onset_s: float | None = None
    interrupt_speech_end_s: float | None = None
    #: Free-form provenance (corpus id, dialogue index, mining parameters).
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.axis not in _EVENT_AXES:
            raise ValueError(f"unknown duplex event axis {self.axis!r}; expected one of {sorted(_EVENT_AXES)}")
        if self.axis == AXIS_PAUSE and self.pause_window_s is None:
            raise ValueError("pause episodes require pause_window_s")
        if self.axis == AXIS_TURN and self.user_turn_end_s is None:
            raise ValueError("turn episodes require user_turn_end_s")
        if self.axis == AXIS_INTERRUPTION and self.interrupt_speech_onset_s is None:
            raise ValueError("interruption episodes require interrupt_speech_onset_s")
        if self.axis == AXIS_INTERRUPTION and self.interrupt_speech_end_s is None:
            raise ValueError("interruption episodes require interrupt_speech_end_s")

    @classmethod
    def from_mapping(cls, event: Mapping[str, Any]) -> DuplexEpisodeEvent:
        """Rebuild the annotation from its stored / wire mapping.

        Accepts the episode spec's parquet ``event`` struct and the replay
        payload's ``event`` dict alike (numpy scalars included); a malformed
        annotation raises the same ``ValueError`` the constructor does.

        Args:
            event: ``{"axis", "pause_window_s", "user_turn_end_s",
            "interrupt_speech_onset_s", "interrupt_speech_end_s"}`` — the
            axis-specific anchors may be absent or ``None`` off their axis.

        Returns:
            The validated event.
        """
        pause_window = event.get("pause_window_s")
        return cls(
            axis=str(event["axis"]),
            pause_window_s=_optional_pair(pause_window),
            user_turn_end_s=_optional_float(event.get("user_turn_end_s")),
            interrupt_speech_onset_s=_optional_float(event.get("interrupt_speech_onset_s")),
            interrupt_speech_end_s=_optional_float(event.get("interrupt_speech_end_s")),
        )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_pair(value: Any) -> tuple[float, float] | None:
    if value is None:
        return None
    values = [float(item) for item in value]
    if len(values) != 2:
        raise ValueError(f"a time window must be a [start, end] pair, got {value!r}")
    return (values[0], values[1])


@dataclass(frozen=True, slots=True)
class DuplexUnitDecision:
    """One unit's policy decision on the episode timeline."""

    unit_index: int
    action_token_id: int
    action_kind: str
    #: Engine emission clock, seconds from the episode's first unit.
    time_s: float
    #: The input chunk's origin (``client`` / ``engine``-injected silence).
    origin: str
    #: True when the unit's action is a speak decision (any non-listen action).
    speaking: bool
    scope_index: int
    logprob: float | None


@dataclass(slots=True)
class DuplexTrajectory:
    """One duplex episode: sealed record scopes + trainer-owned context.

    Built by :meth:`DuplexTrajectory.assemble` at rollout time; consumed by
    the replay collator, the rule rewards, and the telemetry trip-wires.
    """

    session_id: str
    seed: int
    event: DuplexEpisodeEvent
    record_sets: tuple[DuplexActionRecordSet, ...]
    #: PCM chunks the driver fed, in feed order (float32 mono, 16 kHz).
    client_audio_chunks: tuple[np.ndarray, ...]
    #: Per-token flat action stream across scopes, in order.
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float | None, ...] = ()
    action_kinds: tuple[str, ...] = ()
    #: Validity mask over the action stream: 1 for sampled actions, 0 for
    #: forced and re-emitted ones (excluded from loss and importance ratios).
    action_mask: tuple[int, ...] = ()
    #: Which scope each action token belongs to.
    token_scope_indices: tuple[int, ...] = ()
    #: Cumulative unit index of each action token across scopes.
    token_unit_indices: tuple[int, ...] = ()

    @property
    def status(self) -> str:
        """The episode-level status: the last scope's seal status."""
        return self.record_sets[-1].status

    @property
    def barge_in(self) -> bool:
        """True when the episode ended through a barge-in truncation."""
        return any(record_set.status == STATUS_TRUNCATED_INTERRUPTION for record_set in self.record_sets)

    def unit_decisions(self, *, listen_token_id: int) -> list[DuplexUnitDecision]:
        """The per-unit decision timeline, ordered by unit.

        Re-emitted actions (terminator lookahead) are restatements of the
        previous unit's terminator, not new decisions: the engine re-feeds the
        already-generated token as the next unit's closure, so the token
        occupies one position in the context and opens no new unit window.
        They are skipped here and stay masked out of the flat stream.

        Args:
            listen_token_id: The tokenizer's ``<|listen|>`` id; a unit whose
                decision action is this token decided to stay silent.

        Raises:
            RecordSetError: If any unit carries more than one non-reemitted
                action token — the MiniCPM-o 4.5 duplex runtime emits one
                decision per unit, and a wider record means an engine
                contract change this replay does not yet implement.
        """
        decisions: list[DuplexUnitDecision] = []
        t0 = self._first_unit_time_ns()
        for scope_index, record_set in enumerate(self.record_sets):
            for record in record_set.records():
                actions = [action for action in record.actions if action.action_kind != ACTION_REEMIT]
                if not actions:
                    continue
                if len(actions) != 1:
                    raise RecordSetError(
                        f"duplex unit {record.unit_index} of session {self.session_id} carries "
                        f"{len(actions)} decision actions; this replay implements the MiniCPM-o 4.5 "
                        "one-decision-per-unit contract only"
                    )
                action = actions[0]
                emission = record.emission_ns
                time_s = (emission - t0) / 1e9 if emission is not None and t0 is not None else float(record.unit_index)
                decisions.append(
                    DuplexUnitDecision(
                        unit_index=record.unit_index,
                        action_token_id=action.token_id,
                        action_kind=action.action_kind,
                        time_s=time_s,
                        origin=record.input_chunk.origin,
                        speaking=action.token_id != listen_token_id,
                        scope_index=scope_index,
                        logprob=action.logprob,
                    )
                )
        return decisions

    def _first_unit_time_ns(self) -> int | None:
        for record_set in self.record_sets:
            for record in record_set.records():
                if record.emission_ns is not None:
                    return record.emission_ns
        return None

    @classmethod
    def assemble(
        cls,
        *,
        session_id: str,
        seed: int,
        event: DuplexEpisodeEvent,
        record_set_wires: Sequence[Mapping[str, Any]],
        client_audio_chunks: Sequence[np.ndarray],
    ) -> DuplexTrajectory:
        """Validate and assemble one episode from the engine's wire payloads.

        Args:
            session_id: The episode's duplex session id.
            seed: The per-session seed the driver opened the session with.
            event: The mined event annotation of the replayed segment.
            record_set_wires: The drained scopes' wire mappings, in collection
                order (listening scopes included — they are policy too).
            client_audio_chunks: The PCM chunks the driver fed, in feed order.

        Returns:
            The assembled trajectory with its flat action stream and mask.

        Raises:
            RecordSetError: On any wire-contract or alignment violation — a
                rollout failure to surface, never data to train on. One fed
                chunk may go unaccounted at the tail (the engine's close
                flush); it stays in ``client_audio_chunks`` unconsumed.
        """
        record_sets = tuple(DuplexActionRecordSet.from_wire(wire) for wire in record_set_wires)
        validate_record_sets(record_sets, session_id=session_id)
        chunks = tuple(np.asarray(chunk, dtype=np.float32) for chunk in client_audio_chunks)
        # Re-emit rows default to client origin in the engine's ledger but
        # consumed no fed chunk — they restate a token the engine already
        # generated, so they must not enter the client-unit count.
        client_units = sum(
            1
            for record_set in record_sets
            for record in record_set.records()
            if record.input_chunk.origin == ORIGIN_CLIENT
            and any(action.action_kind != ACTION_REEMIT for action in record.actions)
        )
        # The engine ingests each complete 1 s processor chunk as its own unit,
        # but the close flush of the final residual is nondeterministic (the
        # first-chunk window is 1035 ms): the last fed chunk may never become
        # a unit. Exactly one unaccounted chunk is that tail; anything wider is
        # a real disagreement, as is any unit without fed audio behind it.
        if client_units > len(chunks):
            raise RecordSetError(
                f"duplex episode {session_id}: the records carry {client_units} client-origin units but only "
                f"{len(chunks)} client chunks were fed; the driver and the engine disagree on ingestion"
            )
        if client_units < len(chunks) - 1:
            raise RecordSetError(
                f"duplex episode {session_id}: fed {len(chunks)} client chunks but the records carry "
                f"{client_units} client-origin units; more than the final window is missing — the driver and "
                "the engine disagree on ingestion"
            )
        for chunk in chunks:
            if chunk.ndim != 1 or chunk.size == 0:
                raise RecordSetError(f"duplex episode {session_id}: client audio chunks must be non-empty 1-D mono PCM")

        token_ids: list[int] = []
        logprobs: list[float | None] = []
        kinds: list[str] = []
        mask: list[int] = []
        scope_indices: list[int] = []
        unit_indices: list[int] = []
        unit_base = 0
        for scope_index, record_set in enumerate(record_sets):
            token_ids.extend(record_set.token_ids)
            logprobs.extend(record_set.logprobs)
            kinds.extend(record_set.action_kinds)
            mask.extend(1 if kind == ACTION_SAMPLED else 0 for kind in record_set.action_kinds)
            scope_indices.extend(scope_index for _ in record_set.token_ids)
            for record in record_set.records():
                for _ in record.actions:
                    unit_indices.append(unit_base + record.unit_index)
            unit_base += record_set.num_units()
        return cls(
            session_id=session_id,
            seed=seed,
            event=event,
            record_sets=record_sets,
            client_audio_chunks=chunks,
            token_ids=tuple(token_ids),
            logprobs=tuple(logprobs),
            action_kinds=tuple(kinds),
            action_mask=tuple(mask),
            token_scope_indices=tuple(scope_indices),
            token_unit_indices=tuple(unit_indices),
        )


def reanchor_interruption(
    trajectory: DuplexTrajectory,
    *,
    speech_chunk_start_index: Any,
    speech_chunk_end_index: Any,
    seed: int,
) -> DuplexTrajectory:
    """Rewrite an interruption episode's speech anchors onto the unit clock.

    Mining writes the corpus-relative onset and end of the interrupting
    speech, but the driver injects that speech relative to the policy's own
    speak onset. The driver reports the 1-based fed-chunk indices of the
    clip's first speech-carrying chunk and of its last chunk; the i-th fed
    chunk maps one-to-one onto the i-th client-origin unit decision, so each
    anchor becomes that decision's engine emission time. Non-interruption
    episodes and missing indices pass through unchanged.

    Raises:
        ValueError: When an index falls outside the fed chunks — the driver
            and the records disagree.
    """
    if trajectory.event.axis != AXIS_INTERRUPTION or speech_chunk_start_index is None or speech_chunk_end_index is None:
        return trajectory
    client_decisions = [
        decision for decision in trajectory.unit_decisions(listen_token_id=-1) if decision.origin == ORIGIN_CLIENT
    ]
    start_index, end_index = int(speech_chunk_start_index), int(speech_chunk_end_index)
    for index in (start_index, end_index):
        if not 1 <= index <= len(client_decisions):
            raise ValueError(
                f"duplex episode {trajectory.session_id}: speech chunk index {index} outside the "
                f"{len(client_decisions)} fed chunks; the driver and the records disagree"
            )
    anchored = replace(
        trajectory.event,
        interrupt_speech_onset_s=client_decisions[start_index - 1].time_s,
        interrupt_speech_end_s=client_decisions[end_index - 1].time_s,
    )
    return DuplexTrajectory.assemble(
        session_id=trajectory.session_id,
        seed=seed,
        event=anchored,
        record_set_wires=[record_set.to_wire() for record_set in trajectory.record_sets],
        client_audio_chunks=list(trajectory.client_audio_chunks),
    )
