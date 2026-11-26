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
"""Multimodal on-policy self-distillation (SDPO / OPSD) helpers.

Provides:
- Privileged context selection per rollout group (sibling winning rollout,
  verifier text feedback, reference solution for OPSD down to ``rollout.n=1``,
  and fallback marking for GSPO/GRPO).
- Multimodal-preserving teacher reprompting that keeps image/audio/video
  content blocks, raw media payloads, and processor tensors intact while
  injecting privileged text context.
- Exact response-token alignment mapping teacher sequence logprobs and top-k
  token IDs from length ``P_t + R`` back onto student sequence length ``P_s + R``
  matching verl's ``no_padding_2_padding`` left-shift convention.
- Optional EMA shadow-weight tracker for ``teacher_source="ema"``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from verl_omni.workers.config.omni.distillation import SelfDistillationConfig

__all__ = [
    "EMATeacherState",
    "RepromptedTeacherSample",
    "TeacherContextDecision",
    "align_teacher_sequence_outputs",
    "build_multimodal_teacher_prompt",
    "build_neutral_teacher_outputs",
    "extract_sequence_rewards",
    "format_privileged_context",
    "reprompt_multimodal_messages",
    "select_teacher_contexts",
    "summarize_self_distillation_metrics",
]

MEDIA_BLOCK_TYPES = frozenset({"image", "image_url", "audio", "audio_url", "video", "video_url"})

MULTIMODAL_TENSOR_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
    "video_second_per_grid",
    "input_features",
    "feature_attention_mask",
    "audio_feature_lengths",
)


@dataclass(frozen=True)
class TeacherContextDecision:
    """Privileged context selection decision for one rollout sample."""

    has_context: bool
    source: str
    demonstration: Optional[str] = None
    feedback: Optional[str] = None
    reference_solution: Optional[str] = None
    sibling_index: Optional[int] = None
    sibling_reward: Optional[float] = None
    privileged_text: str = ""


@dataclass
class RepromptedTeacherSample:
    """Reprompted multimodal teacher input aligned with its original media payloads."""

    teacher_messages: Any
    teacher_prompt_ids: list[int]
    multi_modal_data: dict[str, Any] = field(default_factory=dict)
    mm_processor_kwargs: dict[str, Any] = field(default_factory=dict)
    multi_modal_inputs: dict[str, Any] = field(default_factory=dict)
    added_prompt_tokens: int = 0
    trimmed_context: bool = False
    context_retained: bool = True


def _unwrap_scalar(value: Any) -> Any:
    """Unwrap 0-d numpy/torch scalars and NonTensorData wrappers."""
    if hasattr(value, "data") and type(value).__name__ == "NonTensorData":
        value = value.data
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0:
        return value.item()
    return value


def _extract_non_empty_text(value: Any) -> Optional[str]:
    """Normalize arbitrary feedback/reference metadata into a stripped non-empty string."""
    value = _unwrap_scalar(value)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, Mapping):
        for candidate_key in ("solution", "reference", "ground_truth", "answer", "feedback", "text"):
            if candidate_key in value:
                nested = _extract_non_empty_text(value[candidate_key])
                if nested is not None:
                    return nested
        return None
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        parts = [text for item in value if (text := _extract_non_empty_text(item)) is not None]
        return "\n".join(parts) if parts else None
    text = str(value).strip()
    return text if text else None


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to ``max_chars`` characters cleanly."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def extract_sequence_rewards(rm_scores: Any) -> list[float]:
    """Convert sequence-level or token-level ``rm_scores`` into per-sample floats."""
    if isinstance(rm_scores, torch.Tensor):
        if rm_scores.is_nested:
            return [float(row.sum().item()) for row in rm_scores.unbind()]
        if rm_scores.ndim == 1:
            return [float(x) for x in rm_scores.detach().cpu().tolist()]
        if rm_scores.ndim >= 2:
            return [float(x) for x in rm_scores.detach().float().sum(dim=-1).cpu().tolist()]
        return [float(rm_scores.item())]
    if isinstance(rm_scores, Sequence):
        rewards: list[float] = []
        for item in rm_scores:
            item = _unwrap_scalar(item)
            if isinstance(item, torch.Tensor):
                rewards.append(float(item.detach().float().sum().item()))
            else:
                rewards.append(float(item))
        return rewards
    raise TypeError(f"Unsupported rm_scores type: {type(rm_scores).__name__}")


def format_privileged_context(
    *,
    demonstration: Optional[str] = None,
    feedback: Optional[str] = None,
    reference_solution: Optional[str] = None,
    max_demonstration_chars: int = 4096,
    max_feedback_chars: int = 2048,
) -> str:
    """Format privileged sibling/feedback/reference text for teacher reprompting."""
    sections: list[str] = []
    if demonstration:
        clipped = _truncate_text(demonstration.strip(), max_demonstration_chars)
        sections.append(f"Correct example solution:\n{clipped}")
    if reference_solution:
        clipped = _truncate_text(reference_solution.strip(), max_demonstration_chars)
        sections.append(f"Reference solution:\n{clipped}")
    if feedback:
        clipped = _truncate_text(feedback.strip(), max_feedback_chars)
        sections.append(f"Verifier feedback:\n{clipped}")
    return "\n\n".join(sections)


def _lookup_row_field(
    index: int,
    key: str,
    sources: Sequence[Optional[Sequence[Any]]],
) -> Optional[str]:
    """Find a non-empty text value for sample ``index`` across candidate metadata tables."""
    for table in sources:
        if table is None or index >= len(table):
            continue
        row = _unwrap_scalar(table[index])
        if isinstance(row, Mapping):
            if key in row:
                found = _extract_non_empty_text(row[key])
                if found is not None:
                    return found
        elif key == "":
            found = _extract_non_empty_text(row)
            if found is not None:
                return found
    return None


def select_teacher_contexts(
    *,
    uids: Sequence[Any],
    response_texts: Sequence[str],
    rewards: Sequence[float],
    config: SelfDistillationConfig,
    extra_infos: Optional[Sequence[Any]] = None,
    reward_extra_infos: Optional[Sequence[Any]] = None,
    reward_models: Optional[Sequence[Any]] = None,
    feedbacks: Optional[Sequence[Any]] = None,
    reference_solutions: Optional[Sequence[Any]] = None,
) -> list[TeacherContextDecision]:
    """Select per-sample privileged context for SDPO, OPSD, or auto self-distillation.

    Args:
        uids: Group identifiers aligning rollouts generated from the same prompt.
        response_texts: Decoded student responses, one per sample.
        rewards: Sequence-level scalar rewards, one per sample.
        config: ``SelfDistillationConfig`` controlling mode and thresholds.
        extra_infos: Optional per-sample ``extra_info`` dicts.
        reward_extra_infos: Optional per-sample ``reward_extra_info`` dicts.
        reward_models: Optional per-sample ``reward_model`` dicts (containing e.g.
            ``ground_truth``).
        feedbacks: Optional explicit per-sample verifier feedback strings.
        reference_solutions: Optional explicit per-sample reference solutions.

    Returns:
        List of ``TeacherContextDecision`` objects aligned 1-to-1 with inputs.
    """
    n = len(uids)
    if len(response_texts) != n or len(rewards) != n:
        raise ValueError(
            f"Length mismatch: len(uids)={n}, len(response_texts)={len(response_texts)}, len(rewards)={len(rewards)}."
        )

    groups: dict[Any, list[int]] = defaultdict(list)
    for idx, uid in enumerate(uids):
        groups[_unwrap_scalar(uid)].append(idx)

    decisions: list[TeacherContextDecision] = []
    for idx in range(n):
        uid = _unwrap_scalar(uids[idx])
        group_indices = groups[uid]

        # 1. Candidate sibling demonstration (SDPO).
        sibling_idx: Optional[int] = None
        sibling_reward: Optional[float] = None
        sibling_demo: Optional[str] = None
        if config.mode in ("sdpo", "auto"):
            best_key: Optional[tuple[float, int, int]] = None
            for cand_idx in group_indices:
                if config.dont_reprompt_on_self_success and cand_idx == idx:
                    continue
                cand_reward = float(rewards[cand_idx])
                if cand_reward < float(config.reward_threshold):
                    continue
                cand_text = response_texts[cand_idx].strip()
                if not cand_text:
                    continue
                # Highest reward first, then shorter response, then earlier index.
                sort_key = (cand_reward, -len(cand_text), -cand_idx)
                if best_key is None or sort_key > best_key:
                    best_key = sort_key
                    sibling_idx = cand_idx
                    sibling_reward = cand_reward
                    sibling_demo = cand_text

        # 2. Candidate environment / verifier feedback (SDPO).
        feedback_text: Optional[str] = None
        if config.mode in ("sdpo", "auto") and config.include_environment_feedback:
            if feedbacks is not None and idx < len(feedbacks):
                feedback_text = _extract_non_empty_text(feedbacks[idx])
            if feedback_text is None:
                feedback_text = _lookup_row_field(
                    idx,
                    config.environment_feedback_key,
                    (reward_extra_infos, extra_infos),
                )

        # 3. Candidate ground-truth reference solution (OPSD / auto fallback).
        reference_text: Optional[str] = None
        if config.mode in ("opsd", "auto"):
            if reference_solutions is not None and idx < len(reference_solutions):
                reference_text = _extract_non_empty_text(reference_solutions[idx])
            if reference_text is None:
                reference_text = _lookup_row_field(
                    idx,
                    config.reference_solution_key,
                    (reward_models, extra_infos, reward_extra_infos),
                )

        if config.mode == "sdpo":
            use_demo = sibling_demo
            use_fb = feedback_text
            use_ref = None
        elif config.mode == "opsd":
            use_demo = None
            use_fb = None
            use_ref = reference_text
        else:
            # auto: prefer SDPO sibling/feedback, fall back to OPSD reference.
            if sibling_demo is not None or feedback_text is not None:
                use_demo = sibling_demo
                use_fb = feedback_text
                use_ref = None
            else:
                use_demo = None
                use_fb = None
                use_ref = reference_text

        privileged = format_privileged_context(
            demonstration=use_demo,
            feedback=use_fb,
            reference_solution=use_ref,
            max_demonstration_chars=config.max_demonstration_chars,
            max_feedback_chars=config.max_feedback_chars,
        )
        if not privileged:
            decisions.append(TeacherContextDecision(has_context=False, source="none"))
            continue

        source_parts: list[str] = []
        if use_demo is not None:
            source_parts.append("sibling")
        if use_ref is not None:
            source_parts.append("reference")
        if use_fb is not None:
            source_parts.append("feedback")
        decisions.append(
            TeacherContextDecision(
                has_context=True,
                source="+".join(source_parts),
                demonstration=use_demo,
                feedback=use_fb,
                reference_solution=use_ref,
                sibling_index=sibling_idx if use_demo is not None else None,
                sibling_reward=sibling_reward if use_demo is not None else None,
                privileged_text=privileged,
            )
        )

    return decisions


def _shallow_copy_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Copy message containers without deep-copying heavy image/audio objects."""
    copied: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, list):
            new_msg["content"] = [dict(block) if isinstance(block, Mapping) else block for block in content]
        copied.append(new_msg)
    return copied


def reprompt_multimodal_messages(
    raw_prompt: Any,
    privileged_text: str,
    *,
    reprompt_template: str = "{context}",
) -> Any:
    """Inject privileged text into a chat message list or prompt string without touching media blocks.

    For structured multimodal chat messages (e.g. Qwen3-Omni / Qwen2.5-VL where
    ``content`` is a list of ``{"type": "image"|"audio"|"video"|"text", ...}``
    blocks), all media blocks are preserved in their exact original order and
    the privileged text is appended as a trailing text block on the last user
    turn.
    """
    if not privileged_text:
        return raw_prompt

    injected = reprompt_template.format(context=privileged_text)

    if isinstance(raw_prompt, str):
        return f"{raw_prompt}{injected}"

    if not isinstance(raw_prompt, Sequence) or isinstance(raw_prompt, bytes | bytearray):
        raise TypeError(f"Unsupported raw_prompt type for reprompting: {type(raw_prompt).__name__}")

    messages = _shallow_copy_messages(raw_prompt)
    if not messages:
        return [{"role": "user", "content": injected.lstrip()}]

    target_idx: Optional[int] = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            target_idx = idx
            break

    if target_idx is None:
        messages.append({"role": "user", "content": injected.lstrip()})
        return messages

    target = messages[target_idx]
    content = target.get("content")
    if isinstance(content, str):
        target["content"] = f"{content}{injected}"
    elif isinstance(content, list):
        # Append strictly after existing media/text blocks so media placeholder order never shifts.
        content.append({"type": "text", "text": injected})
    elif content is None:
        target["content"] = injected.lstrip()
    else:
        raise TypeError(f"Unsupported message content type: {type(content).__name__}")

    return messages


def _collect_media_token_ids(tokenizer: Any, processor: Any = None) -> tuple[int, ...]:
    """Discover image/audio/video placeholder token IDs configured on processor/tokenizer."""
    ids: list[int] = []
    for obj in (processor, tokenizer):
        if obj is None:
            continue
        for attr in ("image_token_id", "video_token_id", "audio_token_id"):
            val = getattr(obj, attr, None)
            if isinstance(val, int) and val >= 0 and val not in ids:
                ids.append(val)
        for token_attr in ("image_token", "video_token", "audio_token"):
            tok = getattr(obj, token_attr, None)
            if isinstance(tok, str) and tok and hasattr(tokenizer, "convert_tokens_to_ids"):
                tid = tokenizer.convert_tokens_to_ids(tok)
                if isinstance(tid, int) and tid >= 0 and tid not in ids:
                    ids.append(tid)
    return tuple(ids)


def _count_media_tokens(token_ids: Sequence[int], media_token_ids: Sequence[int]) -> dict[int, int]:
    """Count occurrences of each multimodal placeholder token ID."""
    if not media_token_ids:
        return {}
    tracked = set(media_token_ids)
    counts = {tid: 0 for tid in tracked}
    for tid in token_ids:
        if tid in tracked:
            counts[tid] += 1
    return counts


def _tokenize_teacher_prompt(
    teacher_messages: Any,
    *,
    tokenizer: Any,
    processor: Any = None,
    multi_modal_data: Optional[Mapping[str, Any]] = None,
    mm_processor_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[list[int], dict[str, Any]]:
    """Render and tokenize a reprompted teacher prompt, preserving processor tensors when available."""
    mm_data = dict(multi_modal_data or {})
    proc_kwargs = dict(mm_processor_kwargs or {})
    rendered_text: str
    if isinstance(teacher_messages, str):
        rendered_text = teacher_messages
    elif processor is not None and hasattr(processor, "apply_chat_template"):
        rendered_text = processor.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    elif hasattr(tokenizer, "apply_chat_template"):
        rendered_text = tokenizer.apply_chat_template(
            teacher_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        rendered_text = "\n".join(str(m.get("content", "")) for m in teacher_messages)

    if processor is not None and callable(processor):
        call_kwargs: dict[str, Any] = {"text": [rendered_text], "return_tensors": "pt", **proc_kwargs}
        if mm_data.get("images") is not None:
            call_kwargs["images"] = mm_data["images"]
        if mm_data.get("videos") is not None:
            call_kwargs["videos"] = mm_data["videos"]
        if mm_data.get("audios") is not None:
            call_kwargs["audio"] = mm_data["audios"]
        encoded = processor(**call_kwargs)
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            prompt_ids = [int(x) for x in input_ids[0].tolist()]
        else:
            prompt_ids = [int(x) for x in input_ids[0]]
        mm_inputs = {key: encoded[key] for key in MULTIMODAL_TENSOR_KEYS if key in encoded and encoded[key] is not None}
        return prompt_ids, mm_inputs

    if hasattr(tokenizer, "encode"):
        prompt_ids = [int(x) for x in tokenizer.encode(rendered_text, add_special_tokens=False)]
    else:
        encoded = tokenizer(rendered_text, add_special_tokens=False)
        prompt_ids = [int(x) for x in encoded["input_ids"]]
    return prompt_ids, {}


_MEDIA_BLOCK_TYPES = frozenset({"image", "image_url", "video", "video_url", "audio", "audio_url"})


def _strip_media_blocks(raw_prompt: Any) -> Any:
    """Remove multimodal content blocks from structured messages while preserving text."""
    if not isinstance(raw_prompt, Sequence) or isinstance(raw_prompt, str | bytes | bytearray):
        return raw_prompt
    stripped: list[dict[str, Any]] = []
    for msg in raw_prompt:
        if not isinstance(msg, Mapping):
            continue
        new_msg = dict(msg)
        content = new_msg.get("content")
        if isinstance(content, list):
            kept = [
                dict(block) if isinstance(block, Mapping) else block
                for block in content
                if not (isinstance(block, Mapping) and block.get("type") in _MEDIA_BLOCK_TYPES)
            ]
            new_msg["content"] = kept
        stripped.append(new_msg)
    return stripped


def build_multimodal_teacher_prompt(
    *,
    raw_prompt: Any,
    student_prompt_ids: Sequence[int],
    privileged_text: str,
    tokenizer: Any,
    processor: Any = None,
    multi_modal_data: Optional[Mapping[str, Any]] = None,
    mm_processor_kwargs: Optional[Mapping[str, Any]] = None,
    existing_multi_modal_inputs: Optional[Mapping[str, Any]] = None,
    config: Optional[SelfDistillationConfig] = None,
    max_prompt_tokens: Optional[int] = None,
) -> RepromptedTeacherSample:
    """Build a tokenized multimodal teacher prompt while preserving media blocks and tensors.

    When ``config.max_reprompt_tokens`` or ``max_prompt_tokens`` is set and the
    reprompted sequence exceeds the effective budget, only the injected
    ``privileged_text`` is progressively trimmed — original user media tokens
    are never dropped or truncated. If no nonempty prefix of ``privileged_text``
    fits within the budget, ``context_retained`` is set to ``False`` so the
    caller can route the sample to fallback.
    """
    cfg = config or SelfDistillationConfig(enabled=True)
    if cfg.preserve_multimodal_inputs:
        effective_raw_prompt = raw_prompt
        mm_data = dict(multi_modal_data or {})
        proc_kwargs = dict(mm_processor_kwargs or {})
        preserved_mm_inputs = dict(existing_multi_modal_inputs or {})
    else:
        effective_raw_prompt = _strip_media_blocks(raw_prompt)
        mm_data = {}
        proc_kwargs = {}
        preserved_mm_inputs = {}

    budget_candidates = [b for b in (cfg.max_reprompt_tokens, max_prompt_tokens) if b is not None]
    effective_budget: Optional[int] = min(budget_candidates) if budget_candidates else None

    media_token_ids = _collect_media_token_ids(tokenizer, processor)
    expected_media_counts = _count_media_tokens(student_prompt_ids, media_token_ids)

    current_text = privileged_text
    trimmed = False
    context_retained = bool(current_text.strip())
    teacher_messages = reprompt_multimodal_messages(
        effective_raw_prompt,
        current_text,
        reprompt_template=cfg.reprompt_template,
    )
    teacher_prompt_ids, computed_mm_inputs = _tokenize_teacher_prompt(
        teacher_messages,
        tokenizer=tokenizer,
        processor=processor,
        multi_modal_data=mm_data,
        mm_processor_kwargs=proc_kwargs,
    )

    if effective_budget is not None and len(teacher_prompt_ids) > effective_budget:
        if effective_budget <= 0 or not current_text:
            context_retained = False
            trimmed = bool(current_text)
        else:
            # Binary-search the longest nonempty prefix of privileged_text that fits.
            lo, hi = 1, len(current_text)
            best_result: Optional[tuple[Any, list[int], dict[str, Any]]] = None
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate_text = current_text[:mid].rstrip()
                if not candidate_text:
                    lo = mid + 1
                    continue
                cand_messages = reprompt_multimodal_messages(
                    effective_raw_prompt,
                    candidate_text,
                    reprompt_template=cfg.reprompt_template,
                )
                cand_ids, cand_mm = _tokenize_teacher_prompt(
                    cand_messages,
                    tokenizer=tokenizer,
                    processor=processor,
                    multi_modal_data=mm_data,
                    mm_processor_kwargs=proc_kwargs,
                )
                if len(cand_ids) <= effective_budget:
                    best_result = (cand_messages, cand_ids, cand_mm)
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best_result is not None:
                teacher_messages, teacher_prompt_ids, computed_mm_inputs = best_result
                trimmed = True
                context_retained = True
            else:
                context_retained = False
                trimmed = True

    if context_retained and cfg.preserve_multimodal_inputs and media_token_ids:
        actual_media_counts = _count_media_tokens(teacher_prompt_ids, media_token_ids)
        if actual_media_counts != expected_media_counts:
            raise ValueError(
                "Teacher reprompting altered multimodal placeholder token counts: "
                f"expected={expected_media_counts}, got={actual_media_counts}."
            )

    final_mm_inputs = computed_mm_inputs if computed_mm_inputs else preserved_mm_inputs
    return RepromptedTeacherSample(
        teacher_messages=teacher_messages,
        teacher_prompt_ids=teacher_prompt_ids,
        multi_modal_data=mm_data,
        mm_processor_kwargs=proc_kwargs,
        multi_modal_inputs=final_mm_inputs,
        added_prompt_tokens=len(teacher_prompt_ids) - len(student_prompt_ids),
        trimmed_context=trimmed,
        context_retained=context_retained,
    )


def align_teacher_sequence_outputs(
    *,
    teacher_logprobs: torch.Tensor,
    teacher_ids: Optional[torch.Tensor],
    teacher_prompt_len: int,
    student_prompt_len: int,
    response_len: int,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align reprompted teacher outputs onto student sequence positions.

    Matches verl's ``extract_prompt_logprobs`` / ``no_padding_2_padding``
    left-shifted layout: for a sequence of prompt length ``P`` and response
    length ``R``, predictions for ``response[0..R-1]`` sit at indices
    ``[P - 1 : P + R - 1]`` inside a tensor of shape ``(P + R, K)``.

    Also accepts response-only tensors of shape ``(R, K)`` (or ``(R,)``) when
    teacher logprobs were already sliced to response tokens.

    Args:
        teacher_logprobs: Teacher logprobs of shape ``(P_t + R, K)``, ``(P_t + R,)``,
            ``(R, K)``, or ``(R,)``.
        teacher_ids: Optional teacher token IDs matching ``teacher_logprobs`` shape.
        teacher_prompt_len: Token length ``P_t >= 1`` of the reprompted teacher prompt.
        student_prompt_len: Token length ``P_s >= 1`` of the original student prompt.
        response_len: Number of student response tokens ``R >= 0``.
        pad_token_id: Fill value for non-response prompt/terminal slots in ``aligned_ids``.

    Returns:
        Tuple ``(aligned_ids, aligned_logprobs)`` each of shape ``(P_s + R, K)``.
    """
    if student_prompt_len <= 0:
        raise ValueError(f"student_prompt_len must be >= 1, got {student_prompt_len}.")
    if teacher_prompt_len <= 0:
        raise ValueError(f"teacher_prompt_len must be >= 1, got {teacher_prompt_len}.")
    if response_len < 0:
        raise ValueError(f"response_len must be >= 0, got {response_len}.")

    lp = teacher_logprobs.unsqueeze(-1) if teacher_logprobs.ndim == 1 else teacher_logprobs
    if lp.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D teacher_logprobs, got shape {tuple(teacher_logprobs.shape)}.")

    topk = lp.shape[1]
    if teacher_ids is None:
        tids = torch.full(lp.shape, pad_token_id, dtype=torch.int32, device=lp.device)
    else:
        tids = teacher_ids.unsqueeze(-1) if teacher_ids.ndim == 1 else teacher_ids
        if tids.shape != lp.shape:
            raise ValueError(
                f"teacher_ids shape {tuple(tids.shape)} does not match teacher_logprobs shape {tuple(lp.shape)}."
            )

    expected_full_len = teacher_prompt_len + response_len
    if lp.shape[0] == expected_full_len:
        resp_lp = lp[teacher_prompt_len - 1 : teacher_prompt_len + response_len - 1]
        resp_ids = tids[teacher_prompt_len - 1 : teacher_prompt_len + response_len - 1]
    elif lp.shape[0] == response_len:
        resp_lp = lp
        resp_ids = tids
    else:
        raise ValueError(
            f"teacher_logprobs length {lp.shape[0]} matches neither teacher sequence length "
            f"({expected_full_len}) nor response_len ({response_len})."
        )

    student_seq_len = student_prompt_len + response_len
    aligned_lp = torch.zeros((student_seq_len, topk), dtype=lp.dtype, device=lp.device)
    aligned_ids = torch.full((student_seq_len, topk), pad_token_id, dtype=tids.dtype, device=tids.device)
    if response_len > 0:
        aligned_lp[student_prompt_len - 1 : student_prompt_len + response_len - 1] = resp_lp
        aligned_ids[student_prompt_len - 1 : student_prompt_len + response_len - 1] = resp_ids

    return aligned_ids, aligned_lp


def build_neutral_teacher_outputs(
    *,
    student_prompt_len: int,
    response_len: int,
    topk: int = 1,
    pad_token_id: int = 0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build zero/pad teacher tensors of shape ``(P_s + R, topk)`` for fallback samples."""
    seq_len = student_prompt_len + response_len
    neutral_ids = torch.full((seq_len, topk), pad_token_id, dtype=torch.int32)
    neutral_logprobs = torch.zeros((seq_len, topk), dtype=dtype)
    return neutral_ids, neutral_logprobs


def summarize_self_distillation_metrics(
    decisions: Sequence[TeacherContextDecision],
    *,
    added_prompt_tokens: Optional[Sequence[int]] = None,
    fallback_to_pg: bool = True,
) -> dict[str, float]:
    """Compute scalar logging metrics summarizing self-distillation coverage for one batch."""
    total = len(decisions)
    if total == 0:
        return {}
    valid = sum(1 for d in decisions if d.has_context)
    fallback = total - valid if fallback_to_pg else 0
    sibling = sum(1 for d in decisions if d.has_context and "sibling" in d.source)
    feedback = sum(1 for d in decisions if d.has_context and "feedback" in d.source)
    reference = sum(1 for d in decisions if d.has_context and "reference" in d.source)
    metrics: dict[str, float] = {
        "self_distillation/valid_ratio": valid / total,
        "self_distillation/fallback_ratio": fallback / total,
        "self_distillation/source_sibling_ratio": sibling / total,
        "self_distillation/source_feedback_ratio": feedback / total,
        "self_distillation/source_reference_ratio": reference / total,
    }
    if added_prompt_tokens:
        metrics["self_distillation/mean_reprompt_delta_tokens"] = float(
            sum(added_prompt_tokens) / len(added_prompt_tokens)
        )
    return metrics


class EMATeacherState:
    """CPU-offloaded exponential moving average shadow weights for self-teacher scoring."""

    def __init__(self, decay: float = 0.99) -> None:
        if not (0.0 < float(decay) < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {decay!r}.")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        self.num_updates: int = 0

    @torch.no_grad()
    def initialize(self, model: nn.Module) -> None:
        """Snapshot all floating-point parameters of ``model`` onto CPU."""
        self.shadow = {
            name: param.detach().to(device="cpu", dtype=torch.float32).clone()
            for name, param in model.named_parameters()
            if param.is_floating_point()
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> int:
        """Update EMA shadow parameters in place from ``model``."""
        if not self.shadow:
            self.initialize(model)
            return len(self.shadow)
        updated = 0
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            src = param.detach().to(device="cpu", dtype=torch.float32)
            if name not in self.shadow:
                self.shadow[name] = src.clone()
            else:
                self.shadow[name].mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            updated += 1
        self.num_updates += 1
        return updated

    @contextmanager
    def swap_into(self, model: nn.Module) -> Iterator[nn.Module]:
        """Temporarily copy EMA weights into ``model``, always restoring live weights on exit."""
        if not self.shadow:
            yield model
            return
        backups: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backups[name] = param.data.detach().to(device="cpu").clone()
                    param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backups:
                        param.data.copy_(backups[name].to(device=param.device, dtype=param.dtype))
