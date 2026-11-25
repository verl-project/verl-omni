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
"""Shared builders for synthetic duplex action-record wire payloads.

The payloads mirror what the engine's capture ledger hands the trainer
(``DuplexActionRecordSet.to_wire()``); keeping one builder lets every duplex
CPU test speak the same fixture dialect.
"""

from __future__ import annotations

from typing import Any

LISTEN_TOKEN_ID = 151664
SPEAK_CONTENT_TOKEN_ID = 88
CHUNK_EOS_TOKEN_ID = 151665
TURN_EOS_TOKEN_ID = 151666


def unit_row(
    *,
    origin: str = "client",
    modality: str = "audio",
    emission_ns: int | None = 1_000_000_000,
    commit_ns: int | None = 990_000_000,
) -> dict[str, Any]:
    """One unit's input-side annotation."""
    return {"modality": modality, "origin": origin, "commit_ns": commit_ns, "emission_ns": emission_ns}


def record_set_wire(
    *,
    session_id: str = "duplex-abc",
    response_id: str | None = "resp-1",
    epoch: int = 0,
    turn_id: int = 0,
    status: str = "completed",
    units: list[dict[str, Any]] | None = None,
    seed: int | None = 7,
    model_extension: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one sealed scope's wire mapping from per-unit action descriptions.

    Each entry of ``units`` is ``{"action": (token_id, kind, logprob), "origin": ...,
    "emission_ns": ...}``; the builder flattens them into the columnar form.
    """
    units = units if units is not None else []
    token_ids: list[int] = []
    logprobs: list[float | None] = []
    kinds: list[str] = []
    reasons: list[str] = []
    streams: list[str] = []
    unit_offsets: list[int] = [0]
    origins: list[str] = []
    modalities: list[str] = []
    emissions: list[int | None] = []
    commits: list[int | None] = []
    terminators: list[list[dict[str, Any]]] = []
    for unit in units:
        token_id, kind, logprob = unit["action"]
        reason = unit.get("reason", "")
        token_ids.append(token_id)
        logprobs.append(logprob)
        kinds.append(kind)
        reasons.append(reason)
        streams.append(unit.get("stream", "content" if kind == "sampled" else "boundary"))
        unit_offsets.append(len(token_ids))
        origins.append(unit.get("origin", "client"))
        modalities.append(unit.get("modality", "audio"))
        emissions.append(unit.get("emission_ns"))
        commits.append(unit.get("commit_ns"))
        terminators.append(unit.get("terminators", []))
    return {
        "schema_version": 1,
        "session_id": session_id,
        "response_id": response_id,
        "epoch": epoch,
        "turn_id": turn_id,
        "status": status,
        "token_ids": token_ids,
        "logprobs": logprobs,
        "action_kinds": kinds,
        "reason_codes": reasons,
        "stream_names": streams,
        "unit_offsets": unit_offsets,
        "stage_ids": [0] * len(units),
        "input_modalities": modalities,
        "input_origins": origins,
        "input_commit_ns": commits,
        "emission_ns": emissions,
        "sampling_params": [{"temperature": 0.7, "top_k": 100, "top_p": 0.8}] * len(units),
        "terminators": terminators,
        "effective_seed": seed,
        "streams": [
            {"name": "boundary", "stage": 0, "modality": "audio"},
            {"name": "content", "stage": 0, "modality": "audio"},
        ],
        "model": model_extension if model_extension is not None else {"minicpmo_duplex": {}},
    }


def reemit_variant_wires(*, session_id: str = "duplex-abc", seed: int = 7) -> list[dict]:
    """A response scope whose terminator lookahead re-emits.

    Mirrors the engine's capture shape: the chunk-terminator lookahead re-
    emits the previous unit's terminator as its own record row (origin
    defaults to client in the engine's ledger). The reemit rows are separate
    units here — restatements that open no unit window and consume no fed
    chunk on the trainer side.
    """
    speaking = record_set_wire(
        session_id=session_id,
        response_id="resp-1",
        status="completed",
        units=[
            {"action": (SPEAK_CONTENT_TOKEN_ID, "sampled", -1.1)},
            {
                "action": (CHUNK_EOS_TOKEN_ID, "sampled", -0.4),
                "stream": "boundary",
                "emission_ns": 2_000_000_000,
                "terminators": [{"token_id": CHUNK_EOS_TOKEN_ID, "kind": "chunk_eos"}],
            },
            # The lookahead restatement: same terminator, reemit kind, no draw.
            {
                "action": (CHUNK_EOS_TOKEN_ID, "reemit", None),
                "reason": "terminator_reemit",
                "emission_ns": 2_050_000_000,
            },
            {"action": (SPEAK_CONTENT_TOKEN_ID, "sampled", -0.9), "emission_ns": 3_000_000_000},
            # A mixed unit: sampled decision plus its own reemit restatement.
            {"action": (CHUNK_EOS_TOKEN_ID, "sampled", -0.5), "stream": "boundary", "emission_ns": 4_000_000_000},
        ],
        seed=seed,
    )
    speaking["action_kinds"] = ["sampled", "sampled", "reemit", "sampled", "sampled", "reemit"]
    speaking["token_ids"] = [
        SPEAK_CONTENT_TOKEN_ID,
        CHUNK_EOS_TOKEN_ID,
        CHUNK_EOS_TOKEN_ID,
        SPEAK_CONTENT_TOKEN_ID,
        CHUNK_EOS_TOKEN_ID,
        CHUNK_EOS_TOKEN_ID,
    ]
    speaking["logprobs"] = [-1.1, -0.4, None, -0.9, -0.5, None]
    speaking["reason_codes"] = ["", "", "terminator_reemit", "", "", "terminator_reemit"]
    speaking["stream_names"] = ["content", "boundary", "boundary", "content", "boundary", "boundary"]
    speaking["stage_ids"] = [0] * 6
    speaking["input_modalities"] = ["audio"] * 6
    speaking["input_origins"] = ["client"] * 6
    speaking["input_commit_ns"] = [None] * 6
    speaking["emission_ns"] = [1_000_000_000, 2_000_000_000, 2_050_000_000, 3_000_000_000, 4_000_000_000, None]
    speaking["sampling_params"] = [{"temperature": 0.7, "top_k": 100, "top_p": 0.8}] * 6
    speaking["terminators"] = [[]] * 6
    speaking["unit_offsets"] = [0, 1, 2, 3, 4, 5, 6]
    return [speaking]


def listening_then_speaking_wires(
    *,
    session_id: str = "duplex-abc",
    seed: int = 7,
    listen_units: int = 2,
    speak_units: int = 2,
    barge_in: bool = False,
    engine_silence_units: int = 0,
) -> list[dict[str, Any]]:
    """A canonical episode: one listening scope, then one response scope.

    The listening units are ``<|listen|>`` draws; the response speaks content
    tokens and closes with ``<|chunk_eos|>``. ``barge_in`` seals the response
    scope as ``truncated_interruption``; ``engine_silence_units`` appends
    engine-injected silent units to the listening scope.
    """
    listening_units: list[dict[str, Any]] = [
        {"action": (LISTEN_TOKEN_ID, "sampled", -0.2), "emission_ns": 1_000_000_000 + unit * 1_000_000_000}
        for unit in range(listen_units)
    ]
    listening_units.extend(
        {
            "action": (LISTEN_TOKEN_ID, "forced", None),
            "reason": "silent_unit_listen",
            "origin": "engine",
            "emission_ns": 1_000_000_000 + (listen_units + unit) * 1_000_000_000,
        }
        for unit in range(engine_silence_units)
    )
    listening = record_set_wire(
        session_id=session_id,
        response_id=None,
        status="truncated_close",
        units=listening_units,
        seed=seed,
    )
    speaking_units: list[dict[str, Any]] = [
        {"action": (SPEAK_CONTENT_TOKEN_ID, "sampled", -1.3), "emission_ns": 3_000_000_000 + unit * 1_000_000_000}
        for unit in range(speak_units)
    ]
    speaking_units.append(
        {
            "action": (CHUNK_EOS_TOKEN_ID, "sampled", -0.4),
            "stream": "boundary",
            "emission_ns": 3_000_000_000 + speak_units * 1_000_000_000,
            "terminators": [{"token_id": CHUNK_EOS_TOKEN_ID, "kind": "chunk_eos"}],
        }
    )
    speaking = record_set_wire(
        session_id=session_id,
        response_id="resp-1",
        status="truncated_interruption" if barge_in else "completed",
        units=speaking_units,
        seed=seed,
    )
    return [listening, speaking]


def interruption_event(*, onset_s: float = 2.0, end_s: float = 5.0) -> dict[str, Any]:
    """An interruption episode's event mapping: the interrupting speech's onset and end."""
    return {"axis": "interruption", "interrupt_speech_onset_s": onset_s, "interrupt_speech_end_s": end_s}
