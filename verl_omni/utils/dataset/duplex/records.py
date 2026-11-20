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
"""Trainer-side mirror of the duplex action-record wire contract.

The engine (vLLM-Omni's duplex action capture) seals one columnar
``DuplexActionRecordSet`` per response or listening scope and hands it to the
trainer as a JSON-safe mapping (``DuplexActionRecordSet.to_wire()``). This
module re-declares that wire contract on the trainer side so the data path —
trajectory assembly, replay collation, CPU tests — never imports
``vllm_omni``: the verl-omni pin predates the capture surface, and duplex
training must not break every other install. The contract is versioned
upstream (``schema_version``); a mismatch is a hard error here, never a
silent reinterpretation. ``tests/utils/dataset/duplex/test_contract_on_cpu.py``
cross-checks this mirror against the upstream implementation whenever the
installed vLLM-Omni provides it.

Vocabulary and column semantics follow the upstream
``vllm_omni/protocol/duplex/action_records.py`` (RFC verl-omni #905 §3.2);
the two files must stay in lockstep, guarded by that contract test.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

__all__ = [
    "ACTION_CAPTURE_RECORD_VERSION",
    "ACTION_FORCED",
    "ACTION_KINDS",
    "ACTION_REEMIT",
    "ACTION_SAMPLED",
    "INPUT_ORIGINS",
    "ORIGIN_CLIENT",
    "ORIGIN_ENGINE",
    "RECORD_STATUSES",
    "STATUS_COMPLETED",
    "STATUS_IN_PROGRESS",
    "STATUS_TRUNCATED_CANCEL",
    "STATUS_TRUNCATED_CLOSE",
    "STATUS_TRUNCATED_INTERRUPTION",
    "DuplexAction",
    "DuplexActionInputChunk",
    "DuplexActionRecord",
    "DuplexActionRecordSet",
    "DuplexActionStream",
    "RecordSetError",
]

#: Wire-record schema of the engine → trainer container this mirror reads.
ACTION_CAPTURE_RECORD_VERSION = 1

# ---- Action kinds ----
#: Drawn from the policy distribution (including post-interruption tokens).
ACTION_SAMPLED = "sampled"
#: Engine/env-resolved rather than sampled; ``reason`` names the forcing class.
ACTION_FORCED = "forced"
#: Protocol re-emission of an earlier action (terminator lookahead): no draw.
ACTION_REEMIT = "reemit"
ACTION_KINDS = frozenset({ACTION_SAMPLED, ACTION_FORCED, ACTION_REEMIT})

# ---- Record-set statuses ----
STATUS_COMPLETED = "completed"
STATUS_TRUNCATED_CLOSE = "truncated_close"
STATUS_TRUNCATED_INTERRUPTION = "truncated_interruption"
STATUS_TRUNCATED_CANCEL = "truncated_cancel"
RECORD_STATUSES = frozenset(
    {STATUS_COMPLETED, STATUS_TRUNCATED_CLOSE, STATUS_TRUNCATED_INTERRUPTION, STATUS_TRUNCATED_CANCEL}
)
#: A drained mid-flight snapshot of a still-open scope; the eventual sealed
#: set carries only the post-drain units.
STATUS_IN_PROGRESS = "in_progress"

# ---- Input origins ----
#: Fed by the consumer of the session (the trainer's rollout driver).
ORIGIN_CLIENT = "client"
#: Injected by the runtime's silence-continuation clock.
ORIGIN_ENGINE = "engine"
INPUT_ORIGINS = frozenset({ORIGIN_CLIENT, ORIGIN_ENGINE})


class RecordSetError(ValueError):
    """A record-set payload violates the wire contract.

    Maps to the engine's ``action_capture_invalid`` / ``action_capture_not_retained``
    family: a rollout failure to surface, never data to train on.
    """


@dataclass(frozen=True, slots=True)
class DuplexActionStream:
    """One named token sequence per (stage, modality); names flat-array positions."""

    name: str
    stage: int
    modality: str


@dataclass(frozen=True, slots=True)
class DuplexAction:
    """One action token with its exact-distribution log-probability and kind."""

    token_id: int
    logprob: float | None
    action_kind: str
    reason: str | None = None
    stream: str = ""


@dataclass(frozen=True, slots=True)
class DuplexActionInputChunk:
    """The unit's input chunk description (modality and origin)."""

    modality: str
    origin: str
    commit_ns: int | None = None


@dataclass(frozen=True, slots=True)
class DuplexActionRecord:
    """The derived per-unit view over a :class:`DuplexActionRecordSet`."""

    session_id: str
    response_id: str | None
    epoch: int
    turn_id: int
    unit_index: int
    stage_id: int
    input_chunk: DuplexActionInputChunk
    actions: tuple[DuplexAction, ...]
    emission_ns: int | None
    sampling: Mapping[str, Any]
    effective_seed: int | None
    model: Mapping[str, Any]


def _require_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecordSetError(f"duplex action record field {what!r} must be an integer, got {type(value).__name__}")
    return value


def _require_str(value: Any, what: str) -> str:
    if not isinstance(value, str):
        raise RecordSetError(f"duplex action record field {what!r} must be a string, got {type(value).__name__}")
    return value


@dataclass(frozen=True, slots=True)
class DuplexActionRecordSet:
    """Sealed columnar records of one capture scope: flat arrays + unit offsets.

    ``response_id`` is ``None`` for a listening scope (units sampled while no
    engine response exists — still policy, never dropped). Columns and offsets
    follow the upstream wire form exactly; see the module docstring.
    """

    session_id: str
    response_id: str | None
    epoch: int
    turn_id: int
    status: str
    schema_version: int = ACTION_CAPTURE_RECORD_VERSION
    # Token-level flat arrays (length == number of action tokens).
    token_ids: tuple[int, ...] = ()
    logprobs: tuple[float | None, ...] = ()
    action_kinds: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    stream_names: tuple[str, ...] = ()
    # Unit-level columns (length == number of units).
    unit_offsets: tuple[int, ...] = (0,)
    stage_ids: tuple[int, ...] = ()
    input_modalities: tuple[str, ...] = ()
    input_origins: tuple[str, ...] = ()
    input_commit_ns: tuple[int | None, ...] = ()
    emission_ns: tuple[int | None, ...] = ()
    sampling_params: tuple[Mapping[str, Any], ...] = ()
    # Per-unit model-sampled terminator tokens: ``{token_id, kind, reason?}``.
    terminators: tuple[tuple[Mapping[str, Any], ...], ...] = ()
    # Response-level columns.
    effective_seed: int | None = None
    streams: tuple[DuplexActionStream, ...] = ()
    # Namespaced model extension, trainer-side opaque beyond the documented keys.
    model: Mapping[str, Any] = field(default_factory=dict)

    def num_units(self) -> int:
        return len(self.stage_ids)

    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def sealed(self) -> bool:
        """True once the scope reached a terminal status."""
        return self.status != STATUS_IN_PROGRESS

    @property
    def truncated(self) -> bool:
        """True when the scope ended early (close / interruption / cancel)."""
        return self.status.startswith("truncated_")

    def records(self) -> Iterator[DuplexActionRecord]:
        """Derive the per-unit records view."""
        for unit_index in range(self.num_units()):
            start, end = self.unit_offsets[unit_index], self.unit_offsets[unit_index + 1]
            actions = tuple(
                DuplexAction(
                    token_id=self.token_ids[offset],
                    logprob=self.logprobs[offset],
                    action_kind=self.action_kinds[offset],
                    reason=self.reason_codes[offset] or None,
                    stream=self.stream_names[offset],
                )
                for offset in range(start, end)
            )
            yield DuplexActionRecord(
                session_id=self.session_id,
                response_id=self.response_id,
                epoch=self.epoch,
                turn_id=self.turn_id,
                unit_index=unit_index,
                stage_id=self.stage_ids[unit_index],
                input_chunk=DuplexActionInputChunk(
                    modality=self.input_modalities[unit_index],
                    origin=self.input_origins[unit_index],
                    commit_ns=self.input_commit_ns[unit_index],
                ),
                actions=actions,
                emission_ns=self.emission_ns[unit_index],
                sampling=self.sampling_params[unit_index],
                effective_seed=self.effective_seed,
                model=self.model,
            )

    def to_wire(self) -> dict[str, Any]:
        """Serialize to the JSON-safe wire mapping (upstream-compatible)."""
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "response_id": self.response_id,
            "epoch": self.epoch,
            "turn_id": self.turn_id,
            "status": self.status,
            "token_ids": list(self.token_ids),
            "logprobs": list(self.logprobs),
            "action_kinds": list(self.action_kinds),
            "reason_codes": list(self.reason_codes),
            "stream_names": list(self.stream_names),
            "unit_offsets": list(self.unit_offsets),
            "stage_ids": list(self.stage_ids),
            "input_modalities": list(self.input_modalities),
            "input_origins": list(self.input_origins),
            "input_commit_ns": list(self.input_commit_ns),
            "emission_ns": list(self.emission_ns),
            "sampling_params": [dict(params) for params in self.sampling_params],
            "terminators": [list(unit) for unit in self.terminators],
            "effective_seed": self.effective_seed,
            "streams": [
                {"name": stream.name, "stage": stream.stage, "modality": stream.modality} for stream in self.streams
            ],
            "model": dict(self.model),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> DuplexActionRecordSet:
        """Decode and strictly validate one wire mapping.

        Args:
            payload: The engine's ``to_wire()`` mapping (or its msgpack-decoded
                equivalent) for one scope.

        Returns:
            The validated record set.

        Raises:
            RecordSetError: On schema-version mismatch, missing or ill-typed
                universal fields, non-parallel token columns, invalid kinds /
                statuses / origins, or a non-finite logprob on a sampled row.
                Unknown keys inside the namespaced ``model`` extension pass
                through untouched, mirroring the engine.
        """
        if not isinstance(payload, Mapping):
            raise RecordSetError(f"duplex action record payload must be a mapping, got {type(payload).__name__}")
        missing = {"session_id", "epoch", "turn_id", "status"} - set(payload)
        if missing:
            raise RecordSetError(f"duplex action record payload is missing universal fields: {sorted(missing)}")
        version = payload.get("schema_version", ACTION_CAPTURE_RECORD_VERSION)
        if version != ACTION_CAPTURE_RECORD_VERSION:
            raise RecordSetError(
                f"unsupported duplex action record schema_version {version!r}; "
                f"this verl-omni build reads {ACTION_CAPTURE_RECORD_VERSION}. The installed "
                "vLLM-Omni capture surface and verl-omni are out of sync."
            )
        status = _require_str(payload["status"], "status")
        if status not in RECORD_STATUSES | {STATUS_IN_PROGRESS}:
            raise RecordSetError(f"invalid duplex action record status: {status!r}")
        session_id = _require_str(payload["session_id"], "session_id")
        response_id = payload.get("response_id")
        if response_id is not None and not isinstance(response_id, str):
            raise RecordSetError("duplex action record response_id must be a string or null")

        token_ids = tuple(_require_int(v, "token_ids[]") for v in payload.get("token_ids") or ())
        raw_logprobs = payload.get("logprobs") or ()
        logprobs: list[float | None] = []
        for value in raw_logprobs:
            if value is None:
                logprobs.append(None)
            elif isinstance(value, bool) or not isinstance(value, int | float):
                raise RecordSetError("duplex action logprobs entries must be numbers or null")
            else:
                logprobs.append(float(value))
        kinds = tuple(_require_str(v, "action_kinds[]") for v in payload.get("action_kinds") or ())
        if not (len(token_ids) == len(logprobs) == len(kinds)):
            raise RecordSetError(
                f"duplex action record token/logprob/kind columns must be parallel: "
                f"{len(token_ids)}/{len(logprobs)}/{len(kinds)}"
            )
        for kind in kinds:
            if kind not in ACTION_KINDS:
                raise RecordSetError(f"invalid duplex action kind: {kind!r}")
        for token_id, logprob, kind in zip(token_ids, logprobs, kinds, strict=True):
            if kind == ACTION_SAMPLED and (logprob is None or not math.isfinite(logprob)):
                raise RecordSetError(
                    f"sampled duplex action {token_id} carries a non-finite logprob {logprob!r}; "
                    "the capture payload is the authoritative channel — refusing to train on it"
                )
            if kind != ACTION_SAMPLED and logprob is not None and not math.isfinite(logprob):
                raise RecordSetError(f"non-sampled duplex action {token_id} carries a non-finite logprob {logprob!r}")
        for name in ("reason_codes", "stream_names"):
            column = payload.get(name) or ()
            if len(column) not in (0, len(token_ids)):
                raise RecordSetError(f"duplex action record column {name!r} length {len(column)} != {len(token_ids)}")

        unit_offsets = tuple(_require_int(v, "unit_offsets[]") for v in payload.get("unit_offsets") or ())
        if not unit_offsets or unit_offsets[0] != 0 or unit_offsets[-1] != len(token_ids):
            raise RecordSetError(
                f"duplex action record unit_offsets {unit_offsets} must start at 0 and end at the token count "
                f"{len(token_ids)}"
            )
        for left, right in pairwise(unit_offsets):
            if right <= left:
                raise RecordSetError(f"duplex action record unit_offsets must strictly increase: {unit_offsets}")
        num_units = len(unit_offsets) - 1
        for name in ("stage_ids", "input_modalities", "input_origins"):
            column = tuple(payload.get(name) or ())
            if len(column) != num_units:
                raise RecordSetError(f"duplex action record column {name!r} length {len(column)} != units {num_units}")
        origins = tuple(_require_str(v, "input_origins[]") for v in payload.get("input_origins") or ())
        for origin in origins:
            if origin not in INPUT_ORIGINS:
                raise RecordSetError(f"invalid duplex action input origin: {origin!r}")
        # Absent optional unit columns become their padded defaults so the
        # per-unit view can index them unconditionally.
        emission_ns = (
            tuple(
                None if value is None else _require_int(value, "emission_ns[]")
                for value in (payload.get("emission_ns") or ())
            )
            or (None,) * num_units
        )
        input_commit_ns = (
            tuple(
                None if value is None else _require_int(value, "input_commit_ns[]")
                for value in (payload.get("input_commit_ns") or ())
            )
            or (None,) * num_units
        )
        sampling_params = (
            tuple(
                dict(params) if isinstance(params, Mapping) else {} for params in payload.get("sampling_params") or ()
            )
            or ({},) * num_units
        )
        terminators = (
            tuple(term for term in unit if isinstance(term, Mapping)) for unit in payload.get("terminators") or ()
        ) or ((),) * num_units
        streams = tuple(
            DuplexActionStream(
                name=_require_str(stream["name"], "streams[].name"),
                stage=_require_int(stream["stage"], "streams[].stage"),
                modality=_require_str(stream["modality"], "streams[].modality"),
            )
            for stream in payload.get("streams") or ()
        )
        model = payload.get("model") or {}
        if not isinstance(model, Mapping):
            raise RecordSetError("duplex action record 'model' extension must be an object")
        effective_seed = payload.get("effective_seed")
        if effective_seed is not None and (isinstance(effective_seed, bool) or not isinstance(effective_seed, int)):
            raise RecordSetError("duplex action record effective_seed must be an integer or null")
        return cls(
            session_id=session_id,
            response_id=response_id,
            epoch=_require_int(payload["epoch"], "epoch"),
            turn_id=_require_int(payload["turn_id"], "turn_id"),
            status=status,
            schema_version=version,
            token_ids=token_ids,
            logprobs=tuple(logprobs),
            action_kinds=kinds,
            reason_codes=tuple(str(v) for v in payload.get("reason_codes") or ()) or ("",) * len(token_ids),
            stream_names=tuple(str(v) for v in payload.get("stream_names") or ()) or ("",) * len(token_ids),
            unit_offsets=unit_offsets,
            stage_ids=tuple(_require_int(v, "stage_ids[]") for v in payload.get("stage_ids") or ()),
            input_modalities=tuple(
                _require_str(v, "input_modalities[]") for v in payload.get("input_modalities") or ()
            ),
            input_origins=origins,
            input_commit_ns=input_commit_ns,
            emission_ns=emission_ns,
            sampling_params=sampling_params,
            terminators=terminators,
            effective_seed=effective_seed,
            streams=streams,
            model=dict(model),
        )


def validate_record_sets(record_sets: Sequence[DuplexActionRecordSet], *, session_id: str) -> None:
    """Check the cross-set invariants one episode's scopes must satisfy.

    Args:
        record_sets: The scopes of one session, in collection order.
        session_id: The episode's session id.

    Raises:
        RecordSetError: When the scopes disagree on session or sampling seed,
            an ``in_progress`` snapshot rides a final trajectory (the episode
            driver must finish draining before assembly), or epochs regress.
    """
    if not record_sets:
        raise RecordSetError(f"duplex episode {session_id} produced no action-record scopes")
    seeds = {record_set.effective_seed for record_set in record_sets if record_set.effective_seed is not None}
    if len(seeds) > 1:
        raise RecordSetError(f"duplex episode {session_id} carries conflicting effective seeds: {sorted(seeds)}")
    for record_set in record_sets:
        if record_set.session_id != session_id:
            raise RecordSetError(f"duplex episode {session_id} received records of session {record_set.session_id!r}")
        if record_set.status == STATUS_IN_PROGRESS:
            raise RecordSetError(
                f"duplex episode {session_id} assembled an in_progress snapshot for response "
                f"{record_set.response_id!r}; drain the scope to a sealed status before training"
            )
    epochs = [record_set.epoch for record_set in record_sets]
    if any(later < earlier for earlier, later in pairwise(epochs)):
        raise RecordSetError(f"duplex episode {session_id} scopes have non-monotonic epochs: {epochs}")
