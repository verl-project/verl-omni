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
history reconstructible from the trajectory's token ids.

The truncation math ports vLLM-Omni's duplex sampler
(``MiniCPMO45Omni._duplex_top_k_top_p_candidates`` / ``_top_k_top_p_filter``)
step for step, so parity holds by construction; the CPU tests check the port
against an independent brute-force nucleus reference.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.functional import softmax

from verl_omni.utils.dataset.duplex.records import ACTION_SAMPLED, RecordSetError

__all__ = [
    "DuplexSamplingContext",
    "DuplexUnitInputs",
    "build_sampling_context",
    "recompute_action_logprobs",
    "top_k_top_p_candidate_mass",
    "top_k_top_p_filter",
    "unit_inputs_from_trajectory",
]

#: Identity strings the engine's ``model`` extension may declare for the
#: forbidden set (``MiniCPMO45DuplexPolicy.native_forbidden_token_ids``).
_KNOWN_FORBIDDEN_IDENTITIES = frozenset({"tts_pad+bad_token_ids+chunk_eos"})


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
    penalty = float(extension.get("repetition_penalty", 1.05))
    history = int(extension.get("repetition_history_tokens", 512))
    return DuplexSamplingContext(
        chunk_eos_token_id=_chunk_eos_id(tokenizer),
        tts_pad_token_id=_resolve_special(tokenizer, "<|tts_pad|>"),
        bad_token_ids=bad_token_ids,
        repetition_penalty=penalty,
        repetition_history_tokens=history,
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
    return float(mass.clamp(min=torch.finfo(torch.float32).tiny).log().item())


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
        action_history: Per action, the session's generated tokens before it
            (the repetition-penalty history). Defaults to the prefix of
            recorded action tokens.

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
    from verl_omni.utils.dataset.duplex.records import ACTION_FORCED, ACTION_REEMIT

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
