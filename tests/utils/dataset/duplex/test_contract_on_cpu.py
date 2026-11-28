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
"""Contract test: the trainer-side wire mirror vs the installed vLLM-Omni.

Runs only when the installed vLLM-Omni provides the duplex action-capture
surface (vllm-project/vllm-omni#10627); at the verl-omni pin it skips, and
the pin-compat story is issue #919's third PR. When it runs, it pins the
mirror to the real implementation: vocabulary, wire round-trip, and the
capture path's own validation semantics, and the engine ledger's unit folding
against the trainer's trajectory view.
"""

import numpy as np
import pytest

from verl_omni.utils.dataset.duplex import records as mirror
from verl_omni.utils.dataset.duplex.policy_measure import resolve_forbidden_token_ids
from verl_omni.utils.dataset.duplex.trajectory import AXIS_TURN, DuplexEpisodeEvent, DuplexTrajectory

pytest.importorskip(
    "vllm_omni.protocol.duplex.action_records",
    reason="installed vLLM-Omni predates the action-capture surface (see verl-omni #919)",
)

from vllm_omni.protocol.duplex import action_records as upstream  # noqa: E402


def test_vocabulary_matches_upstream():
    assert mirror.ACTION_CAPTURE_RECORD_VERSION == upstream.ACTION_CAPTURE_RECORD_VERSION
    assert mirror.ACTION_KINDS == upstream.ACTION_KINDS
    assert mirror.RECORD_STATUSES == upstream.RECORD_STATUSES
    assert mirror.STATUS_IN_PROGRESS == upstream.STATUS_IN_PROGRESS
    assert mirror.INPUT_ORIGINS == upstream.INPUT_ORIGINS


def test_mirror_decodes_upstream_wire_payloads():
    wire = upstream.DuplexActionRecordSet(
        session_id="duplex-x",
        response_id="resp-1",
        epoch=2,
        turn_id=1,
        status="truncated_interruption",
        token_ids=(5, 6),
        logprobs=(-0.5, None),
        action_kinds=("sampled", "reemit"),
        reason_codes=("", "terminator_reemit"),
        stream_names=("content", "boundary"),
        unit_offsets=(0, 1, 2),
        stage_ids=(0, 0),
        input_modalities=("audio", "audio"),
        input_origins=("client", "engine"),
        emission_ns=(1, 2),
        input_commit_ns=(None, None),
        sampling_params=({"temperature": 0.7}, {"temperature": 0.7}),
        effective_seed=99,
        model={"minicpmo_duplex": {"stage0_prob_chunk_eos": 0.25}},
    ).to_wire()
    decoded = mirror.DuplexActionRecordSet.from_wire(wire)
    assert decoded.status == "truncated_interruption"
    assert decoded.token_ids == (5, 6)
    assert decoded.logprobs == (-0.5, None)
    assert [record.input_chunk.origin for record in decoded.records()] == ["client", "engine"]
    assert decoded.model["minicpmo_duplex"]["stage0_prob_chunk_eos"] == pytest.approx(0.25)
    # And back: the mirror's wire form must be upstream-decodable.
    roundtrip = upstream.DuplexActionRecordSet.from_wire(decoded.to_wire())
    assert roundtrip.token_ids == (5, 6)
    assert roundtrip.status == decoded.status


def test_mirror_rejects_what_upstream_rejects():
    wire = upstream.DuplexActionRecordSet(
        session_id="s",
        response_id=None,
        epoch=0,
        turn_id=0,
        status="completed",
        token_ids=(1,),
        logprobs=(float("nan"),),
        action_kinds=("sampled",),
        reason_codes=("",),
        stream_names=("content",),
        unit_offsets=(0, 1),
        stage_ids=(0,),
        input_modalities=("audio",),
        input_origins=("client",),
        input_commit_ns=(None,),
        emission_ns=(1,),
        sampling_params=({"temperature": 0.7},),
    ).to_wire()
    with pytest.raises(mirror.RecordSetError):
        mirror.DuplexActionRecordSet.from_wire(wire)


def test_duplex_engine_client_guard_accepts_this_build():
    from vllm_omni.entrypoints.duplex_omni import DuplexOmni

    from verl_omni.pipelines.minicpm.duplex_rollout_adapter import require_duplex_vllm_omni

    assert require_duplex_vllm_omni() is DuplexOmni


class _ForbiddenStubTokenizer:
    """Stub tokenizer carrying the duplex special tokens and ``bad_token_ids``."""

    unk_token_id = 0
    bad_token_ids = [12, 13]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|tts_pad|>": 11, "<|chunk_eos|>": 10}.get(token, self.unk_token_id)


def test_forbidden_set_resolver_matches_the_engine_composition():
    """The trainer-side resolver composes the engine's native forbidden set
    from the tokenizer alone: ``[tts_pad, *bad_token_ids, chunk_eos]``, equal to
    ``MiniCPMO45DuplexPolicy.native_forbidden_token_ids`` on the same stub.
    """
    stub = _ForbiddenStubTokenizer()
    assert resolve_forbidden_token_ids(stub) == [11, 12, 13, 10]

    # Equality with the engine's own composition on the same stub.
    policy = pytest.importorskip(
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy",
        reason="the engine's duplex policy module is not importable",
    )
    token_ids = policy.MiniCPMO45DuplexPolicy.token_ids_from_tokenizer(stub)
    engine_set = policy.MiniCPMO45DuplexPolicy.native_forbidden_token_ids(token_ids, bad_token_ids=stub.bad_token_ids)
    assert resolve_forbidden_token_ids(stub) == engine_set


def _unit_payload(unit_index: int, seq: int, token_id: int, kind: str, logprob, *, stream: str = "content", **extra):
    from vllm_omni.protocol.duplex.action_records import build_unit_payload

    return build_unit_payload(
        session_id="duplex-fold",
        stage_id=0,
        unit_index=unit_index,
        seq=seq,
        epoch=0,
        token_ids=[token_id],
        logprobs=[logprob],
        action_kinds=[kind],
        stream_names=[stream],
        sampling_params={"temperature": 0.7},
        effective_seed=11,
        **extra,
    )


def _engine_ledger_folds_unit_steps() -> bool:
    """Whether the installed ledger folds same-``unit_index`` payloads into one record."""
    from vllm_omni.engine.duplex.session.action_ledger import SessionActionLedger

    ledger = SessionActionLedger(session_id="duplex-fold")
    ledger.bind_response("resp-1", epoch=0, turn_id=0)
    for seq in (100, 101):
        ledger.ingest(_unit_payload(0, seq, 88, "sampled", -1.0), epoch=0, turn_id=0, response_id="resp-1", stage_id=0)
    ledger.seal_active_response("resp-1")
    return ledger.take_terminal_sets()[0].num_units() == 1


ENGINE_LEDGER_FOLDS_UNIT_STEPS = _engine_ledger_folds_unit_steps()


@pytest.mark.xfail(
    not ENGINE_LEDGER_FOLDS_UNIT_STEPS,
    strict=False,
    reason="the installed vLLM-Omni ledger appends one record per decode-step payload; "
    "one record per model unit (unit_index folding) ships with the engine's capture change",
)
def test_engine_ledger_folds_a_units_decode_steps_into_one_record():
    """A spoken unit's decode steps — its restated terminator included — are one record the trainer replays once."""
    from vllm_omni.engine.duplex.session.action_ledger import SessionActionLedger

    ledger = SessionActionLedger(session_id="duplex-fold")
    ledger.bind_response("resp-1", epoch=0, turn_id=1)
    ledger.note_input_unit(0, 100, modality="audio", origin="client")
    # Unit 0: two content steps, the boundary chunk_eos, then the terminator the
    # next unit's open re-emits under the unit it closes. Unit 1: a listen decision.
    deliveries = [
        _unit_payload(0, 100, 88, "sampled", -1.3),
        _unit_payload(0, 101, 88, "sampled", -1.2),
        _unit_payload(
            0, 102, 5, "sampled", -0.4, stream="boundary", terminators=[{"token_id": 5, "kind": "chunk_eos"}]
        ),
        _unit_payload(
            0,
            103,
            5,
            "reemit",
            None,
            stream="boundary",
            reason_codes=["terminator_reemit"],
            terminators=[{"token_id": 5, "kind": "chunk_eos"}],
        ),
        _unit_payload(1, 104, 6, "sampled", -0.2),
    ]
    for payload in deliveries:
        ledger.ingest(payload, epoch=0, turn_id=1, response_id="resp-1", stage_id=0)
    ledger.seal_active_response("resp-1")
    engine_set = ledger.take_terminal_sets()[0]
    wire = engine_set.to_wire()

    decoded = mirror.DuplexActionRecordSet.from_wire(wire)
    assert decoded.num_units() == 2
    assert decoded.unit_offsets == (0, 4, 5) == tuple(engine_set.unit_offsets)
    first, second = decoded.records()
    assert [(action.token_id, action.action_kind) for action in first.actions] == [
        (88, "sampled"),
        (88, "sampled"),
        (5, "sampled"),
        (5, "reemit"),
    ]
    assert first.input_chunk.origin == "client"  # the unit's first step keeps the append's annotation
    assert decoded.terminators[0][-1] == {"token_id": 5, "kind": "chunk_eos"}
    assert [action.token_id for action in second.actions] == [6]

    trajectory = DuplexTrajectory.assemble(
        session_id="duplex-fold",
        seed=11,
        event=DuplexEpisodeEvent(axis=AXIS_TURN, user_turn_end_s=0.0),
        record_set_wires=[wire],
        client_audio_chunks=[np.zeros(16000, dtype=np.float32), np.zeros(16000, dtype=np.float32)],
    )
    decisions = trajectory.unit_decisions(listen_token_id=6)
    assert [(decision.unit_index, decision.action_token_ids) for decision in decisions] == [
        (0, (88, 88, 5)),
        (1, (6,)),
    ]
    assert decisions[0].action_kinds == ("sampled", "sampled", "sampled")  # the reemit restatement is no decision
