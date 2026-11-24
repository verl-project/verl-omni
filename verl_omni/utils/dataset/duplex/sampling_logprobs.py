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
"""Trainer-side recomputation of the exact distribution the runtime sampled.

The capture records log-probabilities of the *sampling* distribution, not of a
plain softmax: post forbidden-token mask, repetition penalty, and temperature /
top-k / top-p truncation, renormalized — and, for MiniCPM-o 4.5's two-stage
draw, a Bernoulli boundary decision on the raw softmax composed with a stage-2
content draw (RFC verl-omni #905 §3.3). Old log-probabilities and the reference
KL must therefore be recomputed under the same transform, from the recorded
distribution inputs: the effective sampling parameters, the per-stage
quantities in the namespaced ``model`` extension, and the repetition-penalty
history the engine penalized under (``state.generated_tokens``: every stage-2
draw of the session, replayed by :func:`engine_repetition_history`).

The truncation math ports vLLM-Omni's duplex sampler
(``MiniCPMO45Omni._duplex_top_k_top_p_candidates`` / ``_top_k_top_p_filter``)
step for step, so parity holds by construction; the CPU tests check the port
against an independent brute-force nucleus reference. :func:`compare_action_logprobs`
classifies each row's parity residual by nucleus membership — the M0 gate's
out-of-nucleus family is the documented bf16 tower loosening, not a bug.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.functional import softmax

from verl_omni.utils.dataset.duplex.records import (
    ACTION_FORCED,
    ACTION_REEMIT,
    ACTION_SAMPLED,
    RecordSetError,
)

__all__ = [
    "DuplexSamplingContext",
    "DuplexUnitInputs",
    "ParityReport",
    "ParityRow",
    "build_sampling_context",
    "compare_action_logprobs",
    "engine_repetition_history",
    "recompute_action_logprobs",
    "token_in_nucleus",
    "top_k_top_p_candidate_mass",
    "top_k_top_p_candidates",
    "top_k_top_p_filter",
    "unit_inputs_from_trajectory",
]

#: Identity strings the engine's ``model`` extension may declare for the
#: forbidden set (``MiniCPMO45DuplexPolicy.native_forbidden_token_ids``).
_KNOWN_FORBIDDEN_IDENTITIES = frozenset({"tts_pad+bad_token_ids+chunk_eos"})

#: The engine's repetition-penalty defaults, stamped on every captured row
#: (``_minicpmo45_duplex_build_capture_payload`` / ``MiniCPMO45DuplexPolicy.REPETITION_HISTORY_SIZE``).
_DEFAULT_REPETITION_PENALTY = 1.05
_DEFAULT_REPETITION_HISTORY_TOKENS = 512

#: The stream name the engine stamps on rows that resolved before any stage-2
#: content draw (boundary hits, forced rows); content draws carry ``"content"``.
_BOUNDARY_STREAM = "boundary"


@dataclass(frozen=True, slots=True)
class DuplexSamplingContext:
    """The distribution inputs fixed for one episode's session.

    ``forbidden_token_ids`` is resolved trainer-side from the checkpoint's
    tokenizer (``bad_token_ids``) and the framing vocabulary, and must match
    the identity the records declare.
    """

    chunk_eos_token_id: int
    tts_pad_token_id: int
    bad_token_ids: tuple[int, ...] = ()
    repetition_penalty: float = 1.05
    repetition_history_tokens: int = 512

    def forbidden_token_ids(self) -> tuple[int, ...]:
        ids = [self.tts_pad_token_id, *self.bad_token_ids, self.chunk_eos_token_id]
        return tuple(sorted({token_id for token_id in ids if token_id >= 0}))


@dataclass(frozen=True, slots=True)
class DuplexUnitInputs:
    """One action's recorded distribution inputs."""

    token_id: int
    action_kind: str
    stream: str
    temperature: float
    top_k: int
    top_p: float
    #: The stage-1 Bernoulli parameter on the raw softmax, when the row drew one.
    stage0_prob_chunk_eos: float | None = None
    #: The recorded stage-2 logprob (post-truncation renormalized).
    stage2_logprob: float | None = None
    #: The raw draw before an engine rewrite (mid-turn listen -> tts_bos).
    sampled_token: int | None = None
    forbidden_identity: str | None = None


def build_sampling_context(
    tokenizer: Any,
    vocabulary: Any,
    model_extension: Mapping[str, Any],
    *,
    minicpmo_key: str = "minicpmo_duplex",
) -> DuplexSamplingContext:
    """Resolve one episode's sampling context from the checkpoint and records.

    Args:
        tokenizer: The checkpoint's tokenizer (source of ``bad_token_ids``).
        vocabulary: The resolved :class:`DuplexTokenVocabulary`.
        model_extension: The record set's namespaced ``model`` mapping.
        minicpmo_key: The model-extension namespace to read.

    Raises:
        RecordSetError: When the records declare an unknown forbidden-set
            identity — recomputing against a different mask would silently
            corrupt every log-probability.
    """
    extension = model_extension.get(minicpmo_key) if isinstance(model_extension, Mapping) else None
    extension = extension if isinstance(extension, Mapping) else {}
    identity = extension.get("forbidden_tokens")
    if identity is not None and identity not in _KNOWN_FORBIDDEN_IDENTITIES:
        raise RecordSetError(
            f"the engine records declare unknown forbidden-token identity {identity!r}; this recompute "
            f"implements {sorted(_KNOWN_FORBIDDEN_IDENTITIES)} only"
        )
    bad_token_ids = tuple(int(value) for value in getattr(tokenizer, "bad_token_ids", None) or ())
    # The engine stamps ``repetition_penalty`` / ``repetition_history_tokens`` on every
    # captured row; an extension carrying neither is a key-name drift, and defaulting
    # it would silently recompute under a penalty the engine never applied.
    penalty = extension.get("repetition_penalty")
    history = extension.get("repetition_history_tokens")
    if extension and penalty is None and history is None:
        raise RecordSetError(
            "the engine records' model extension declares neither repetition_penalty nor "
            f"repetition_history_tokens (keys: {sorted(extension)}); this recompute reads the engine's "
            "stamped keys only"
        )
    return DuplexSamplingContext(
        chunk_eos_token_id=_chunk_eos_id(tokenizer),
        tts_pad_token_id=_resolve_special(tokenizer, "<|tts_pad|>"),
        bad_token_ids=bad_token_ids,
        repetition_penalty=float(penalty if penalty is not None else _DEFAULT_REPETITION_PENALTY),
        repetition_history_tokens=int(history if history is not None else _DEFAULT_REPETITION_HISTORY_TOKENS),
    )


def _chunk_eos_id(tokenizer: Any) -> int:
    token_id = _resolve_special(tokenizer, "<|chunk_eos|>")
    if token_id < 0:
        raise RecordSetError("the checkpoint tokenizer has no <|chunk_eos|> id; cannot recompute duplex logprobs")
    return token_id


def _resolve_special(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else None
    unk_id = getattr(tokenizer, "unk_token_id", None)
    try:
        token_id = int(value) if value is not None else -1
    except (TypeError, ValueError):
        token_id = -1
    if token_id < 0 or token_id == unk_id:
        ids = tokenizer.encode(token, add_special_tokens=False)
        token_id = int(ids[0]) if len(ids) == 1 and int(ids[0]) != unk_id else -1
    return token_id


def top_k_top_p_filter(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    """Port of the engine's greedy truncation filter (ties at the k-th value stay)."""
    if top_k > 0 and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1, stable=True)
        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = torch.zeros_like(logits, dtype=torch.bool)
        remove.scatter_(dim=-1, index=sorted_indices, src=sorted_remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def top_k_top_p_candidates(
    logits: torch.Tensor,
    *,
    top_k: int,
    top_p: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Port of the engine's ``_duplex_top_k_top_p_candidates`` on one ``[V]`` logit row.

    Returns:
        ``(probs, indices, kth, ties)``: the ordered slots' renormalized
        probabilities over a leading batch axis (zero where the nucleus cut
        removed the token), their token ids (``-1`` for the trailing tie slot
        shared by the k-th-value ties), the k-th value, and how many of those
        ties ride that slot.
    """
    logits = logits.reshape(1, -1).float()
    vocab = logits.shape[-1]
    k = int(top_k) if top_k else 0
    if 0 < k < vocab:
        values, indices = torch.topk(logits, k, dim=-1)
        kth = values[..., -1:]
        values = values.masked_fill(values <= kth, float("-inf"))
        perm = torch.argsort(indices, dim=-1)
        values, order = torch.sort(values.gather(-1, perm), descending=True, dim=-1, stable=True)
        indices = indices.gather(-1, perm).gather(-1, order)
        tied = (logits == kth).sum(dim=-1, keepdim=True)
    else:
        values, indices = torch.sort(logits, descending=True, dim=-1, stable=True)
        kth = torch.full_like(values[..., :1], float("-inf"))
        tied = torch.zeros_like(indices[..., :1])
    values, kth_value = values.float(), kth.float()
    top = torch.maximum(values[..., :1], kth_value)
    weights, tie_weight = (values - top).exp(), torch.where(tied > 0, (kth_value - top).exp(), 0.0)
    total = weights.sum(dim=-1, keepdim=True) + tied * tie_weight
    probs, tie_prob = weights / total, tie_weight / total
    ties = tied
    if 0.0 < float(top_p) < 1.0:
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        probs = probs.masked_fill(remove, 0.0)
        before = cumulative[..., -1:]
        last = ((float(top_p) - before) / tie_prob.clamp(min=torch.finfo(torch.float32).tiny)).floor()
        last = torch.where(before + (last + 1) * tie_prob <= float(top_p), last + 1, last)
        last = torch.where(before + last * tie_prob > float(top_p), last - 1, last)
        kept = torch.where(tie_prob > 0, last + 1, torch.zeros_like(last))
        ties = torch.minimum(kept.clamp(min=0), tied.float()).long()
    probs = torch.cat([probs, ties * tie_prob], dim=-1)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    indices = torch.cat([indices, torch.full_like(indices[..., :1], -1)], dim=-1)
    return probs, indices, kth, ties


def _drawn_mass(logits: torch.Tensor, candidates: tuple[torch.Tensor, ...], token_id: int) -> tuple[torch.Tensor, bool]:
    """``(mass, in_nucleus)`` of ``token_id``: its regular slot, or one tie share at the k-th value."""
    probs, indices, kth, ties = candidates
    drawn = torch.tensor([[token_id]], dtype=torch.long)
    regular = (indices == drawn).float() * probs
    # Only the ``ties`` lowest-index tokens at the k-th value ride the tie
    # slot; a tied token outside that subset is never drawn.
    if kth.numel() and float(ties.reshape(-1)[0]) > 0:
        tied_ids = torch.nonzero(logits.reshape(-1) == kth.reshape(-1)[0]).reshape(-1)
        kept_ties = tied_ids[: int(ties.reshape(-1)[0])]
        in_kept_ties = float((kept_ties == drawn.reshape(-1)[0]).sum()) > 0
    else:
        in_kept_ties = False
    per_tie = probs[..., -1:] / ties.clamp(min=1).to(probs.dtype)
    tied_hit = (logits.gather(1, drawn) == kth).to(probs.dtype) * float(in_kept_ties)
    mass = regular.sum(dim=-1) + tied_hit * per_tie
    return mass, bool(float(regular.sum(dim=-1)) > 0.0 or float(tied_hit) > 0.0)


def top_k_top_p_candidate_mass(
    logits: torch.Tensor,
    token_id: int,
    *,
    top_k: int,
    top_p: float,
) -> float:
    """Probability mass of ``token_id`` under the engine's candidate distribution.

    Ports ``_duplex_top_k_top_p_candidates`` and evaluates the drawn token's
    renormalized mass exactly as the engine's capture does: its regular slot,
    or one tie share when it sits at the k-th value.
    """
    logits = logits.reshape(1, -1).float()
    mass, _ = _drawn_mass(logits, top_k_top_p_candidates(logits, top_k=top_k, top_p=top_p), token_id)
    return float(mass.clamp(min=torch.finfo(torch.float32).tiny).log().item())


def token_in_nucleus(
    logits: torch.Tensor,
    token_id: int,
    *,
    top_k: int,
    top_p: float,
) -> bool:
    """Whether ``token_id`` survives the engine's top-k / top-p candidate cut.

    The engine draws from the candidate support, so a captured token sits inside
    the ENGINE's nucleus by construction; under the replay's logits it may fall
    out (bf16 tower numerics moving it across the top-p boundary), where its
    renormalized mass collapses to the float32 floor.
    """
    logits = logits.reshape(1, -1).float()
    return _drawn_mass(logits, top_k_top_p_candidates(logits, top_k=top_k, top_p=top_p), token_id)[1]


def engine_repetition_history(unit_inputs: Sequence[DuplexUnitInputs]) -> list[list[int]]:
    """Per action, the token history the engine's repetition penalty divided.

    Replays ``state.generated_tokens`` (``_record_minicpmo45_duplex_generation_token``):
    every stage-2 content draw of the session in emission order — the sampled
    ``content`` rows, and the raw draw behind a mid-turn listen rewrite
    (``sampled_token``, recorded before the rewrite) — trimmed to
    ``repetition_history_tokens`` by the recompute. Rows that resolve before
    the stage-2 draw never enter it: boundary-hit ``chunk_eos``, cap- and
    silence-forced rows (all stamped ``boundary``), and re-emits. While the
    history is still empty the engine penalizes over the request's own output
    tokens so far (``recent``); the records carry those only as the prior
    rows, so the prior recorded action tokens stand in.

    Args:
        unit_inputs: The recorded per-action inputs, in action order
            (:func:`unit_inputs_from_trajectory`).

    Returns:
        One history per action, in action order.
    """
    generated: list[int] = []
    prior: list[int] = []
    history: list[list[int]] = []
    for unit in unit_inputs:
        history.append(list(generated) if generated else list(prior))
        prior.append(unit.token_id)
        if unit.stream == _BOUNDARY_STREAM or unit.action_kind == ACTION_REEMIT:
            continue
        if unit.action_kind not in (ACTION_SAMPLED, ACTION_FORCED):
            raise RecordSetError(f"unknown action kind {unit.action_kind!r}")
        generated.append(unit.sampled_token if unit.sampled_token is not None else unit.token_id)
    return history


def recompute_action_logprobs(
    logits: torch.Tensor,
    unit_inputs: Sequence[DuplexUnitInputs],
    context: DuplexSamplingContext,
    *,
    action_history: Sequence[Sequence[int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the captured sampling log-probability of every action.

    Args:
        logits: ``[T, V]`` actor logits at the episode's action positions, in
            action order.
        unit_inputs: The recorded distribution inputs per action.
        context: The episode's sampling context.
        action_history: Per action, the token history the engine's repetition
            penalty divided (:func:`engine_repetition_history` over ``unit_inputs``).
            ``None`` falls back to the prefix of recorded action tokens.

    Returns:
        ``(values, valid)``: recomputed log-probabilities (``NaN`` where not
        applicable) and a mask that is true exactly on sampled rows.

    Raises:
        RecordSetError: On shape mismatches or a forbidden identity drift.
    """
    if logits.ndim != 2 or logits.shape[0] != len(unit_inputs):
        raise RecordSetError(
            f"duplicate logprob recompute got logits for {tuple(logits.shape)} but {len(unit_inputs)} actions"
        )
    if action_history is None:
        history: list[Sequence[int]] = [
            [unit.token_id for unit in unit_inputs[:index]] for index in range(len(unit_inputs))
        ]
    else:
        history = list(action_history)
    values = torch.full((len(unit_inputs),), float("nan"))
    valid = torch.zeros(len(unit_inputs), dtype=torch.bool)
    forbidden_index = torch.tensor(context.forbidden_token_ids(), dtype=torch.long)
    vocab = logits.shape[-1]

    for index, unit in enumerate(unit_inputs):
        if unit.action_kind != ACTION_SAMPLED:
            valid[index] = False
            continue
        if unit.forbidden_identity is not None and unit.forbidden_identity not in _KNOWN_FORBIDDEN_IDENTITIES:
            raise RecordSetError(f"action {index} declares unknown forbidden identity {unit.forbidden_identity!r}")
        row = logits[index : index + 1].float()
        if unit.token_id == context.chunk_eos_token_id and unit.stage0_prob_chunk_eos is not None:
            # Boundary hit: the joint is the raw-softmax Bernoulli parameter.
            values[index] = torch.log_softmax(row, dim=-1)[0, context.chunk_eos_token_id]
            valid[index] = True
            continue
        prob_eos = softmax(row, dim=-1)[0, context.chunk_eos_token_id]
        stage2 = row.clone()
        if forbidden_index.numel():
            stage2.index_fill_(1, forbidden_index, float("-inf"))
        penalized = [
            token_id for token_id in set(history[index][-context.repetition_history_tokens :]) if 0 <= token_id < vocab
        ]
        if penalized and context.repetition_penalty != 1.0:
            cols = torch.tensor(penalized, dtype=torch.long)
            stage2[0, cols] = stage2[0, cols] / context.repetition_penalty
        target = unit.sampled_token if unit.sampled_token is not None else unit.token_id
        if unit.temperature <= 0:
            filtered = top_k_top_p_filter(stage2, top_k=unit.top_k, top_p=unit.top_p)
            stage2_logprob = float(torch.log_softmax(filtered.float(), dim=-1)[0, target].item())
        else:
            stage2 = stage2 / unit.temperature
            stage2_logprob = top_k_top_p_candidate_mass(stage2, target, top_k=unit.top_k, top_p=unit.top_p)
        keep_eos = max(1.0 - float(prob_eos), torch.finfo(torch.float32).tiny)
        values[index] = torch.log(torch.tensor(keep_eos)).item() + stage2_logprob
        valid[index] = True
    return values, valid


def unit_inputs_from_trajectory(
    trajectory: Any,
    *,
    minicpmo_key: str = "minicpmo_duplex",
) -> list[DuplexUnitInputs]:
    """Collect the recorded per-action distribution inputs from a trajectory.

    Args:
        trajectory: The assembled :class:`DuplexTrajectory`; per-action inputs
            come from its scopes' stream/kind columns, per-unit sampling
            parameters, and per-scope ``model`` extensions.

    Returns:
        One :class:`DuplexUnitInputs` per action, in action order.

        The stage quantities are scope-level in the engine's records (its
        ledger merges per-unit ``model`` extensions last-write-wins), so they
        reflect the scope's last unit; the boundary gate uses them only as a
        presence check alongside the token id, and ``stage2_logprob`` rides
        through unused by the recompute.
    """
    inputs: list[DuplexUnitInputs] = []
    for record_set in trajectory.record_sets:
        scope_extension = record_set.model.get(minicpmo_key) if isinstance(record_set.model, Mapping) else None
        scope_extension = scope_extension if isinstance(scope_extension, Mapping) else {}
        for record in record_set.records():
            params = record.sampling or {}
            for action in record.actions:
                if action.action_kind not in (ACTION_SAMPLED, ACTION_FORCED, ACTION_REEMIT):
                    raise RecordSetError(f"unknown action kind {action.action_kind!r}")
                inputs.append(
                    DuplexUnitInputs(
                        token_id=action.token_id,
                        action_kind=action.action_kind,
                        stream=action.stream,
                        temperature=float(params.get("temperature", 0.7) or 0.0),
                        top_k=int(params.get("top_k", 100) or 0),
                        top_p=float(params.get("top_p", 0.8) or 1.0),
                        stage0_prob_chunk_eos=(
                            float(scope_extension["stage0_prob_chunk_eos"])
                            if action.action_kind == ACTION_SAMPLED and "stage0_prob_chunk_eos" in scope_extension
                            else None
                        ),
                        stage2_logprob=(
                            float(scope_extension["stage2_logprob"])
                            if action.action_kind == ACTION_SAMPLED and "stage2_logprob" in scope_extension
                            else None
                        ),
                        sampled_token=(
                            int(scope_extension["sampled_token"])
                            if action.action_kind == ACTION_FORCED and "sampled_token" in scope_extension
                            else None
                        ),
                        forbidden_identity=scope_extension.get("forbidden_tokens"),
                    )
                )
    return inputs


@dataclass(frozen=True, slots=True)
class ParityRow:
    """One sampled action's parity residual, classified by nucleus membership."""

    index: int
    token_id: int
    captured: float
    recomputed: float
    #: ``recomputed - captured``.
    delta: float
    #: False when the replay's logits cut the engine-drawn token out of the
    #: recorded top-k / top-p candidate support.
    in_nucleus: bool

    @property
    def abs_delta(self) -> float:
        return abs(self.delta)


def _report_max(rows: Sequence[ParityRow]) -> float:
    return max((row.abs_delta for row in rows), default=float("nan"))


@dataclass(frozen=True, slots=True)
class ParityReport:
    """The M0 parity verdict over one episode's sampled actions.

    The gate passes on in-nucleus rows; out-of-nucleus rows are the documented
    loosening family (the engine drew the token inside its nucleus, the replay's
    bf16 tower logits moved it across the top-p boundary), reported
    separately and optionally bounded by ``max_out_of_nucleus``.
    """

    rows: tuple[ParityRow, ...]
    rel_tol: float
    abs_tol: float
    max_out_of_nucleus: int | None
    in_nucleus_max_abs_delta: float
    out_of_nucleus_max_abs_delta: float
    breaches: tuple[ParityRow, ...]
    documented_loosening: tuple[ParityRow, ...]

    @property
    def matches(self) -> bool:
        """True when sampled rows exist, none in-nucleus row breaches, and the loosening family stays bounded."""
        if not self.rows or self.breaches:
            return False
        return self.max_out_of_nucleus is None or len(self.documented_loosening) <= self.max_out_of_nucleus

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary for the parity report."""
        return {
            "num_sampled": len(self.rows),
            "num_in_nucleus": len(self.rows) - len(self.documented_loosening),
            "num_out_of_nucleus": len(self.documented_loosening),
            "in_nucleus_max_abs_delta": self.in_nucleus_max_abs_delta,
            "out_of_nucleus_max_abs_delta": self.out_of_nucleus_max_abs_delta,
            "documented_loosening": [_row_dict(row) for row in self.documented_loosening],
            "breaches": [_row_dict(row) for row in self.breaches],
            "matches": self.matches,
        }


def _row_dict(row: ParityRow) -> dict[str, Any]:
    return {
        "index": row.index,
        "token_id": row.token_id,
        "captured": row.captured,
        "recomputed": row.recomputed,
        "delta": row.delta,
    }


def compare_action_logprobs(
    logits: torch.Tensor,
    unit_inputs: Sequence[DuplexUnitInputs],
    context: DuplexSamplingContext,
    recomputed: torch.Tensor,
    valid: torch.Tensor,
    captured: Sequence[float | None],
    *,
    rel_tol: float,
    abs_tol: float,
    max_out_of_nucleus: int | None = None,
) -> ParityReport:
    """Compare recomputed against engine-captured logprobs, nucleus-cut aware.

    Args:
        logits: ``[T, V]`` actor logits at the action positions — the
            recompute's own input, classified row by row.
        unit_inputs: The recorded distribution inputs per action.
        context: The episode's sampling context (names the boundary token).
        recomputed: ``[T]`` values from :func:`recompute_action_logprobs`.
        valid: ``[T]`` validity mask from the recompute.
        captured: ``[T]`` engine-captured logprob per action, aligned with
            ``unit_inputs`` (``None`` where no draw happened).
        rel_tol: Relative tolerance on ``|captured|``.
        abs_tol: Absolute tolerance.
        max_out_of_nucleus: Loose rows allowed before the gate fails;
            ``None`` reports the family without bounding it.

    Returns:
        The per-row report; ``matches`` is the gate verdict.

    Raises:
        RecordSetError: On column misalignment.
    """
    count = len(unit_inputs)
    if logits.ndim != 2 or logits.shape[0] != count or recomputed.numel() != count or valid.numel() != count:
        raise RecordSetError(
            f"parity comparison got logits {tuple(logits.shape)}, {recomputed.numel()} recomputed values and "
            f"{valid.numel()} validity bits for {count} actions"
        )
    if len(captured) != count:
        raise RecordSetError(f"parity comparison got {len(captured)} captured logprobs for {count} actions")
    values, flags = recomputed.reshape(-1).tolist(), valid.reshape(-1).tolist()
    rows: list[ParityRow] = []
    for index, unit in enumerate(unit_inputs):
        captured_logprob = captured[index]
        if not flags[index] or captured_logprob is None or not math.isfinite(captured_logprob):
            continue
        if not math.isfinite(values[index]):
            raise RecordSetError(
                f"parity comparison: action {index} recomputed to a non-finite logprob {values[index]!r}"
            )
        # A boundary hit is the raw-softmax Bernoulli joint: no truncation applies.
        boundary_hit = unit.token_id == context.chunk_eos_token_id and unit.stage0_prob_chunk_eos is not None
        in_nucleus = True
        if unit.temperature > 0 and not boundary_hit:
            row = logits[index : index + 1].float() / unit.temperature
            target = unit.sampled_token if unit.sampled_token is not None else unit.token_id
            in_nucleus = token_in_nucleus(row, target, top_k=unit.top_k, top_p=unit.top_p)
        rows.append(
            ParityRow(
                index=index,
                token_id=unit.token_id,
                captured=float(captured_logprob),
                recomputed=float(values[index]),
                delta=float(values[index]) - float(captured_logprob),
                in_nucleus=in_nucleus,
            )
        )
    loosened = tuple(row for row in rows if not row.in_nucleus)
    in_nucleus_rows = tuple(row for row in rows if row.in_nucleus)
    breaches = tuple(row for row in in_nucleus_rows if row.abs_delta > abs_tol + rel_tol * abs(row.captured))
    return ParityReport(
        rows=tuple(rows),
        rel_tol=rel_tol,
        abs_tol=abs_tol,
        max_out_of_nucleus=max_out_of_nucleus,
        in_nucleus_max_abs_delta=_report_max(in_nucleus_rows),
        out_of_nucleus_max_abs_delta=_report_max(loosened),
        breaches=breaches,
        documented_loosening=loosened,
    )
