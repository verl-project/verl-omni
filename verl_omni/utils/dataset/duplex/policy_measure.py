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
"""The two-stage duplex policy's log-probability measure.

The engine draws a boundary first — ``chunk_eos`` with probability
``p0 = softmax(z)[chunk_eos]`` on the raw logits — and otherwise a content
token from ``softmax(mask(z) / T)``, where ``mask`` sends the forbidden set
``{tts_pad} ∪ bad_token_ids ∪ {chunk_eos}`` to ``-inf``. Every log-probability
it records is the pre-truncation joint of that draw: ``log p0`` for a boundary
hit, ``log(1 − p0) + log_softmax(mask(z) / T)[a]`` for a content token; greedy
rows (``T < 1e-5``) compare at ``T = 1``. The M0 parity gate and the actor's
training forwards evaluate this one definition on the trainer-side logits.
"""

from __future__ import annotations

import torch
from torch.nn.functional import log_softmax, softmax

from verl_omni.utils.dataset.duplex.records import RecordSetError

__all__ = ["resolve_forbidden_token_ids", "two_stage_logprob", "two_stage_logprobs"]

# The engine samples greedily below this temperature and reports the unscaled logits' log-probabilities.
_GREEDY_TEMPERATURE = 1e-5
# Keep log(p0) and log1p(-p0) finite where the fp32 softmax saturates on either side.
_P0_FLOOR = torch.finfo(torch.float32).tiny
_P0_CEILING = 1.0 - 1e-7


def _special_token_id(tokenizer, token: str) -> int:
    """Resolve one special token to its id (``-1`` when unresolvable)."""
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


def resolve_forbidden_token_ids(tokenizer) -> list[int]:
    """The engine's native forbidden set: ``[tts_pad, *bad_token_ids, chunk_eos]``.

    Resolved from the checkpoint's tokenizer — the same composition as
    ``MiniCPMO45DuplexPolicy.native_forbidden_token_ids``, without importing
    ``vllm_omni`` on the data path.

    Args:
        tokenizer: The checkpoint tokenizer carrying ``bad_token_ids``.

    Returns:
        The forbidden ids; ``chunk_eos`` is the last element by construction.

    Raises:
        RecordSetError: When ``<|tts_pad|>`` / ``<|chunk_eos|>`` do not resolve
            to a single id — masking an unresolved id would silently corrupt
            stage 2.
    """
    tts_pad = _special_token_id(tokenizer, "<|tts_pad|>")
    chunk_eos = _special_token_id(tokenizer, "<|chunk_eos|>")
    if tts_pad < 0 or chunk_eos < 0:
        raise RecordSetError(
            f"the checkpoint tokenizer has no <|tts_pad|> ({tts_pad}) / <|chunk_eos|> ({chunk_eos}) id; "
            "cannot resolve the engine's forbidden set"
        )
    bad_token_ids = [int(value) for value in getattr(tokenizer, "bad_token_ids", None) or ()]
    return [tts_pad, *bad_token_ids, chunk_eos]


def _boundary_probability(logits: torch.Tensor, chunk_eos_id: int) -> torch.Tensor:
    """``p0 = softmax(z)[chunk_eos]`` on the raw logits, clamped away from 0 and 1."""
    return softmax(logits, dim=-1)[..., chunk_eos_id].clamp(min=_P0_FLOOR, max=_P0_CEILING)


def two_stage_logprob(logits_row, token_id: int, *, chunk_eos_id: int, forbidden_index, temperature: float) -> float:
    """The two-stage joint's pre-truncation log-probability of ``token_id``.

    Stage 1 draws the boundary: ``log P(chunk_eos)`` on the raw softmax of the
    actor logits row. Stage 2 draws content among the remaining tokens:
    ``log(1 - P) + log_softmax(mask(z) / T)[a]`` with the forbidden set masked
    to ``-inf`` (greedy rows, ``T < 1e-5``, compare at ``T = 1``). No repetition
    penalty, top-k, or top-p applies — the engine's capture is the
    pre-truncation joint.

    Args:
        logits_row: ``[V]`` actor logits at the action's predicting position.
        token_id: The drawn token.
        chunk_eos_id: The boundary token's id.
        forbidden_index: Indices of ``[tts_pad, *bad_token_ids, chunk_eos]``
            (a sequence or a long tensor).
        temperature: The row's sampling temperature.

    Returns:
        The log-probability in nats; ``-inf`` only for a forbidden token drawn
        through stage 2, whose mass under the engine is zero.
    """
    z = logits_row.float()
    p0 = _boundary_probability(z, chunk_eos_id)
    if token_id == chunk_eos_id:
        return float(torch.log(p0))
    masked = z.index_fill(0, torch.as_tensor(forbidden_index, dtype=torch.long, device=z.device), float("-inf"))
    effective_temperature = temperature if temperature >= _GREEDY_TEMPERATURE else 1.0
    stage2 = log_softmax(masked / effective_temperature, dim=-1)[token_id]
    return float(torch.log1p(-p0)) + float(stage2)


def two_stage_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    chunk_eos_id: int,
    forbidden_index: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """Vectorized, differentiable :func:`two_stage_logprob` over logit rows.

    Args:
        logits: ``[N, V]`` fp32 logits, one row per action's predicting position.
        token_ids: ``[N]`` long ids of the drawn tokens.
        chunk_eos_id: The boundary token's id.
        forbidden_index: ``[F]`` long ids masked out of stage 2, on the logits'
            device (``chunk_eos_id`` among them).
        temperature: ``[N]`` or single-element fp32 sampling temperatures.

    Returns:
        ``[N]`` fp32 joint log-probabilities on the autograd graph of ``logits``:
        the stage-2 term has exactly zero gradient on the forbidden logits, the
        stage-0 term (a raw softmax) depends on every logit, and a content
        row's gradient reaches its non-drawn content logits only through their
        aggregate mass.

    Raises:
        ValueError: When ``token_ids`` or ``temperature`` do not line up with
            the logit rows.
    """
    rows = logits.shape[0]
    if logits.ndim != 2 or token_ids.shape != (rows,):
        raise ValueError(
            f"two_stage_logprobs expects logits [N, V] and token_ids [N]; "
            f"got {tuple(logits.shape)} and {tuple(token_ids.shape)}"
        )
    temperature = torch.as_tensor(temperature, dtype=torch.float32, device=logits.device).reshape(-1)
    if temperature.numel() == 1:
        temperature = temperature.expand(rows)
    if temperature.shape != (rows,):
        raise ValueError(
            f"temperature must hold one value per logit row ({rows}) or a single scalar; got {tuple(temperature.shape)}"
        )

    p0 = _boundary_probability(logits, chunk_eos_id)
    masked = logits.index_fill(-1, forbidden_index, float("-inf"))
    effective_temperature = torch.where(temperature < _GREEDY_TEMPERATURE, 1.0, temperature)
    stage2 = log_softmax(masked / effective_temperature[:, None], dim=-1).gather(1, token_ids[:, None]).squeeze(1)
    return torch.where(token_ids == chunk_eos_id, torch.log(p0), torch.log1p(-p0) + stage2)
