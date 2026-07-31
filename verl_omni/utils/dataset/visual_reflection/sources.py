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
"""Public pair and prompt-only source adapters plus narrow synthesis I/O."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import (
    DraftGenerationRequest,
    EditSynthesisRequest,
    EditVerificationRequest,
    ImagePairSource,
    ImageSynthesisRequest,
    ImageSynthesisResult,
    PairSynthesisRequest,
    PairVerificationRequest,
    PromptKTurnSynthesisRequest,
    PromptOnlySource,
    ReflectionSynthesisRequest,
    ReflectionSynthesisResult,
    RejectionReason,
    SynthesisResult,
    TerminalVerificationRequest,
    VerificationResult,
    VisualReflectionDataError,
    canonical_json,
    derive_pair_source_dedup_key,
    derive_prompt_source_dedup_key,
    require_nonempty_text,
    stable_digest,
    validate_image_ref,
    validate_reflection_step,
    validate_trajectory,
)
from .images import ImageResolver
from .partition import validate_source_split_assignment, validate_split_provenance
from .protocol import format_reflection_response, parse_reflection_response

ECHO_DATASET_ID = "Yejy53/Echo-4o-Image"
ECHO_DATASET_CONFIG = "Instruction-Following-Image"
ECHO_IMAGE_PATH_PREFIX = "/Echo-4o-Image/Instruction-Following-Image/"
MIDJOURNEY_DATASET_ID = "brivangl/midjourney-v6-llava"


def adapt_echo_pair_record(
    record: Mapping[str, Any],
    *,
    image_resolver: ImageResolver,
    source_record_id: str | None = None,
    prompt_field: str = "instruction",
    image_field: str = "output_image",
) -> ImagePairSource:
    """Normalize one Echo instruction-following T2I pair without teacher inference."""
    if not isinstance(record, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "Echo record must be a mapping",
        )
    task_type = _required_value(record, "task_type", source_record_id)
    if task_type != "t2i":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"Echo task_type must be 't2i', got {task_type!r}",
            field="task_type",
            source_record_id=source_record_id,
        )
    prompt = require_nonempty_text(
        _required_value(record, prompt_field, source_record_id),
        field=prompt_field,
        reason=RejectionReason.EMPTY_PROMPT,
        source_record_id=source_record_id,
    )
    raw_image_value = _required_value(record, image_field, source_record_id)
    if isinstance(raw_image_value, str):
        public_id = require_nonempty_text(raw_image_value, field=image_field)
        if source_record_id is not None:
            override = require_nonempty_text(source_record_id, field="source_record_id")
            if override != public_id:
                raise VisualReflectionDataError(
                    RejectionReason.DUPLICATE_SOURCE_RECORD,
                    "source_record_id override does not match the public Echo output image path",
                    field="source_record_id",
                    source_record_id=override,
                )
        record_id = public_id
    else:
        record_id = require_nonempty_text(
            source_record_id,
            field="source_record_id",
            source_record_id=source_record_id,
        )
    image_value = _normalize_echo_image_path(raw_image_value, source_record_id=record_id)
    reference_image = image_resolver(
        image_value,
        field=image_field,
        index=0,
        source_record_id=record_id,
    )
    dedup_key = derive_pair_source_dedup_key(prompt, reference_image)
    return {
        "source_dataset": ECHO_DATASET_ID,
        "source_record_id": record_id,
        "pipeline_variant": "pair_0_1_turn",
        "prompt": prompt,
        "reference_image": reference_image,
        "dedup_key": dedup_key,
    }


def adapt_midjourney_prompt_record(
    record: Mapping[str, Any],
    *,
    source_record_id: str | None = None,
    prompt_field: str = "prompt",
    id_field: str = "id",
) -> PromptOnlySource:
    """Normalize only the Midjourney prompt and never touch image or LLaVA fields."""
    if not isinstance(record, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "Midjourney record must be a mapping",
        )
    prompt = require_nonempty_text(
        _required_value(record, prompt_field, source_record_id),
        field=prompt_field,
        reason=RejectionReason.EMPTY_PROMPT,
        source_record_id=source_record_id,
    )
    public_id = require_nonempty_text(_required_value(record, id_field, source_record_id), field=id_field)
    if source_record_id is not None:
        override = require_nonempty_text(source_record_id, field="source_record_id")
        if override != public_id:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "source_record_id override does not match the public Midjourney id",
                field="source_record_id",
                source_record_id=override,
            )
    record_id = public_id
    dedup_key = derive_prompt_source_dedup_key(prompt)
    return {
        "source_dataset": MIDJOURNEY_DATASET_ID,
        "source_record_id": record_id,
        "pipeline_variant": "prompt_k_turn",
        "prompt": prompt,
        "dedup_key": dedup_key,
    }


def build_pair_synthesis_request(
    source: Mapping[str, Any],
    *,
    manifest_id: str,
    source_split: Mapping[str, Any],
    partition: Mapping[str, Any],
    seed: int,
    max_attempts: int,
) -> PairSynthesisRequest:
    """Attach partition and retry state to a normalized Echo pair source."""
    normalized = _validate_echo_source(source)
    manifest = require_nonempty_text(manifest_id, field="manifest_id")
    normalized_partition = validate_split_provenance(partition)
    assignment = validate_source_split_assignment(normalized, source_split, partition=normalized_partition)
    _require_nonnegative_int(seed, field="seed")
    _require_positive_int(max_attempts, field="max_attempts")
    request_payload = {
        "manifest_id": manifest,
        "source": normalized,
        "source_split": assignment,
        "partition": normalized_partition,
        "seed": seed,
        "max_attempts": max_attempts,
    }
    return {
        **normalized,
        "request_id": f"request_{stable_digest('pair_0_1_turn', request_payload)}",
        "manifest_id": manifest,
        "partition": normalized_partition,
        "partition_id": assignment["partition_id"],
        "split": assignment["split"],
        "seed": seed,
        "max_attempts": max_attempts,
    }


def build_prompt_synthesis_request(
    source: Mapping[str, Any],
    *,
    manifest_id: str,
    source_split: Mapping[str, Any],
    partition: Mapping[str, Any],
    seed: int,
    max_attempts: int,
    max_edit_turns: int,
) -> PromptKTurnSynthesisRequest:
    """Attach partition and bounded retry/turn state to a prompt-only source."""
    normalized = _validate_prompt_source(source)
    manifest = require_nonempty_text(manifest_id, field="manifest_id")
    normalized_partition = validate_split_provenance(partition)
    assignment = validate_source_split_assignment(normalized, source_split, partition=normalized_partition)
    _require_nonnegative_int(seed, field="seed")
    _require_positive_int(max_attempts, field="max_attempts")
    _require_positive_int(max_edit_turns, field="max_edit_turns")
    request_payload = {
        "manifest_id": manifest,
        "source": normalized,
        "source_split": assignment,
        "partition": normalized_partition,
        "seed": seed,
        "max_attempts": max_attempts,
        "max_edit_turns": max_edit_turns,
    }
    return {
        **normalized,
        "request_id": f"request_{stable_digest('prompt_k_turn', request_payload)}",
        "manifest_id": manifest,
        "partition": normalized_partition,
        "partition_id": assignment["partition_id"],
        "split": assignment["split"],
        "seed": seed,
        "max_attempts": max_attempts,
        "max_edit_turns": max_edit_turns,
    }


def synthesis_request_source(request: Mapping[str, Any]) -> ImagePairSource | PromptOnlySource:
    """Extract the exact normalized public source bound into a synthesis request."""
    normalized = _validate_synthesis_request(request)
    if normalized["pipeline_variant"] == "pair_0_1_turn":
        return {
            "source_dataset": normalized["source_dataset"],
            "source_record_id": normalized["source_record_id"],
            "pipeline_variant": "pair_0_1_turn",
            "prompt": normalized["prompt"],
            "reference_image": normalized["reference_image"],
            "dedup_key": normalized["dedup_key"],
        }
    return {
        "source_dataset": normalized["source_dataset"],
        "source_record_id": normalized["source_record_id"],
        "pipeline_variant": "prompt_k_turn",
        "prompt": normalized["prompt"],
        "dedup_key": normalized["dedup_key"],
    }


def make_draft_generation_request(
    *,
    parent_request: Mapping[str, Any],
    attempt: int,
) -> DraftGenerationRequest:
    """Build a draft image-generation request from its stable source seed."""
    parent = _validate_synthesis_request(parent_request)
    _validate_stage_attempt(parent, attempt)
    stage_payload = [parent["request_id"], parent["prompt"], parent["seed"], attempt]
    return {
        "request_id": f"stage_{stable_digest('draft_generate', stage_payload)}",
        "parent_request_id": parent["request_id"],
        "prompt": parent["prompt"],
        "seed": parent["seed"],
        "attempt": attempt,
    }


def make_reflection_synthesis_request(
    *,
    parent_request: Mapping[str, Any],
    current_image: Mapping[str, Any],
    turn_index: int,
    attempt: int,
) -> ReflectionSynthesisRequest:
    """Build a route- and turn-bounded reflector request without reference leakage."""
    parent = _validate_synthesis_request(parent_request)
    image = validate_image_ref(current_image)
    _validate_reflection_turn(parent, image=image, turn_index=turn_index)
    _validate_stage_attempt(parent, attempt)
    stage_payload = [parent["request_id"], parent["prompt"], image, turn_index, attempt]
    return {
        "request_id": f"stage_{stable_digest('reflect', stage_payload)}",
        "parent_request_id": parent["request_id"],
        "prompt": parent["prompt"],
        "current_image": image,
        "turn_index": turn_index,
        "attempt": attempt,
    }


def make_reflection_synthesis_result(
    *,
    parent_request: Mapping[str, Any],
    request: Mapping[str, Any],
    response_text: str,
) -> ReflectionSynthesisResult:
    """Parse and bind one reflector response to its narrow image-only request."""
    parent = _validate_synthesis_request(parent_request)
    if not isinstance(request, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "reflection synthesis request must be a mapping",
            field="request",
        )
    expected = make_reflection_synthesis_request(
        parent_request=parent,
        current_image=request.get("current_image"),
        turn_index=request.get("turn_index"),
        attempt=request.get("attempt"),
    )
    if canonical_json(dict(request)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "reflection synthesis request contains unknown, stale, or inconsistent fields",
            field="request",
        )
    response = parse_reflection_response(response_text)
    return {
        "request_id": expected["request_id"],
        "request": expected,
        "response_text": format_reflection_response(response),
        "response": response,
    }


def make_pair_verification_request(
    *,
    parent_request: Mapping[str, Any],
    draft_image: Mapping[str, Any],
    proposed_step: Mapping[str, Any],
    attempt: int,
) -> PairVerificationRequest:
    """Build the only pair stage allowed to see both draft and public reference."""
    parent = _validate_synthesis_request(parent_request)
    if parent["pipeline_variant"] != "pair_0_1_turn":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "pair compatibility verification requires an Echo pair parent request",
            field="parent_request.pipeline_variant",
        )
    draft = validate_image_ref(draft_image)
    reference = parent["reference_image"]
    step = validate_reflection_step(proposed_step)
    if step["action"] != "continue":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "pair compatibility verification requires a continue proposal",
            field="proposed_step.action",
        )
    _validate_stage_attempt(parent, attempt)
    stage_payload = [parent["request_id"], parent["prompt"], draft, reference, step, attempt]
    return {
        "request_id": f"stage_{stable_digest('pair_verify', stage_payload)}",
        "parent_request_id": parent["request_id"],
        "prompt": parent["prompt"],
        "draft_image": draft,
        "reference_image": reference,
        "proposed_step": step,
        "attempt": attempt,
    }


def make_terminal_verification_request(
    *,
    parent_request: Mapping[str, Any],
    current_image: Mapping[str, Any],
    proposed_step: Mapping[str, Any],
    turn_index: int,
    attempt: int,
) -> TerminalVerificationRequest:
    """Build a stop verifier request that cannot expose an unrelated reference."""
    parent = _validate_synthesis_request(parent_request)
    image = validate_image_ref(current_image)
    step = validate_reflection_step(proposed_step)
    if step["action"] != "stop":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "terminal verification requires a stop proposal",
            field="proposed_step.action",
        )
    _validate_terminal_turn(parent, image=image, turn_index=turn_index)
    _validate_stage_attempt(parent, attempt)
    stage_payload = [parent["request_id"], parent["prompt"], image, step, turn_index, attempt]
    return {
        "request_id": f"stage_{stable_digest('terminal_verify', stage_payload)}",
        "parent_request_id": parent["request_id"],
        "prompt": parent["prompt"],
        "current_image": image,
        "proposed_step": step,
        "turn_index": turn_index,
        "attempt": attempt,
    }


def make_edit_synthesis_request(
    *,
    parent_request: Mapping[str, Any],
    current_image: Mapping[str, Any],
    edit: str,
    turn_index: int,
    attempt: int,
) -> EditSynthesisRequest:
    """Build one offline editor request from clean source state and delta text."""
    parent = _validate_synthesis_request(parent_request)
    _require_prompt_parent(parent, stage="edit synthesis")
    image = validate_image_ref(current_image)
    normalized_edit = require_nonempty_text(
        edit,
        field="edit",
        reason=RejectionReason.EMPTY_CONTINUE_EDIT,
    )
    _validate_edit_turn(parent, turn_index)
    _validate_stage_attempt(parent, attempt)
    seed = _derive_edit_seed(parent["seed"], turn_index=turn_index, attempt=attempt)
    stage_payload = [parent["request_id"], parent["prompt"], image, normalized_edit, turn_index, seed, attempt]
    return {
        "request_id": f"stage_{stable_digest('edit', stage_payload)}",
        "parent_request_id": parent["request_id"],
        "prompt": parent["prompt"],
        "current_image": image,
        "edit": normalized_edit,
        "turn_index": turn_index,
        "seed": seed,
        "attempt": attempt,
    }


def make_image_synthesis_result(
    *,
    parent_request: Mapping[str, Any],
    request: Mapping[str, Any],
    output_image: Mapping[str, Any],
) -> ImageSynthesisResult:
    """Bind a materialized image to one exact, route-valid generation request."""
    parent = _validate_synthesis_request(parent_request)
    expected = _rebuild_image_synthesis_request(parent, request)
    if canonical_json(dict(request)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "image synthesis request contains unknown, stale, or inconsistent fields",
            field="request",
        )
    return {
        "request_id": expected["request_id"],
        "request": expected,
        "output_image": validate_image_ref(output_image),
    }


def make_edit_verification_request(
    *,
    parent_request: Mapping[str, Any],
    current_image: Mapping[str, Any],
    target_image: Mapping[str, Any],
    edit: str,
    turn_index: int,
    attempt: int,
) -> EditVerificationRequest:
    """Build a verifier request with explicit current and target image roles."""
    parent = _validate_synthesis_request(parent_request)
    _require_prompt_parent(parent, stage="edit verification")
    current = validate_image_ref(current_image)
    target = validate_image_ref(target_image)
    normalized_edit = require_nonempty_text(
        edit,
        field="edit",
        reason=RejectionReason.EMPTY_CONTINUE_EDIT,
    )
    _validate_edit_turn(parent, turn_index)
    _validate_stage_attempt(parent, attempt)
    stage_payload = [parent["request_id"], parent["prompt"], current, target, normalized_edit, turn_index, attempt]
    return {
        "request_id": f"stage_{stable_digest('edit_verify', stage_payload)}",
        "parent_request_id": parent["request_id"],
        "prompt": parent["prompt"],
        "current_image": current,
        "target_image": target,
        "edit": normalized_edit,
        "turn_index": turn_index,
        "attempt": attempt,
    }


def make_verification_result(
    *,
    parent_request: Mapping[str, Any],
    request: Mapping[str, Any],
    verdict: str,
    reason: str,
) -> VerificationResult:
    """Bind one strict verifier decision to an exact, route-valid stage request."""
    parent = _validate_synthesis_request(parent_request)
    expected = _rebuild_verification_request(parent, request)
    if canonical_json(dict(request)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "verification request contains unknown, stale, or inconsistent fields",
            field="request",
        )
    if verdict not in {"accept", "retry", "reject"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "verdict must be accept, retry, or reject",
            field="verdict",
        )
    normalized_reason = require_nonempty_text(reason, field="reason")
    return {
        "request_id": expected["request_id"],
        "request": expected,
        "verdict": verdict,  # type: ignore[typeddict-item]
        "reason": normalized_reason,
    }


def make_synthesis_result(
    *,
    request: Mapping[str, Any],
    status: str,
    attempt: int,
    trajectory: Mapping[str, Any] | None = None,
    rejection_reason: str | None = None,
    image_synthesis_results: Sequence[Mapping[str, Any]] = (),
    reflection_synthesis_results: Sequence[Mapping[str, Any]] = (),
    verification_results: Sequence[Mapping[str, Any]] = (),
) -> SynthesisResult:
    """Validate the accepted/retry/rejected synthesis result envelope."""
    normalized_request = _validate_synthesis_request(request)
    normalized_id = normalized_request["request_id"]
    if not isinstance(image_synthesis_results, Sequence) or isinstance(image_synthesis_results, str | bytes):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "image_synthesis_results must be a sequence",
            field="image_synthesis_results",
        )
    normalized_image_synthesis_results = sorted(
        [_validate_image_synthesis_result(normalized_request, result) for result in image_synthesis_results],
        key=lambda result: result["request_id"],
    )
    if not isinstance(reflection_synthesis_results, Sequence) or isinstance(reflection_synthesis_results, str | bytes):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "reflection_synthesis_results must be a sequence",
            field="reflection_synthesis_results",
        )
    normalized_reflection_synthesis_results = sorted(
        [_validate_reflection_synthesis_result(normalized_request, result) for result in reflection_synthesis_results],
        key=lambda result: result["request_id"],
    )
    if not isinstance(verification_results, Sequence) or isinstance(verification_results, str | bytes):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "verification_results must be a sequence",
            field="verification_results",
        )
    normalized_verification_results = sorted(
        [_validate_verification_result(normalized_request, result) for result in verification_results],
        key=lambda result: result["request_id"],
    )
    source_split = {
        "source_dataset": normalized_request["source_dataset"],
        "source_record_id": normalized_request["source_record_id"],
        "dedup_key": normalized_request["dedup_key"],
        "partition_id": normalized_request["partition_id"],
        "split": normalized_request["split"],
    }
    _require_nonnegative_int(attempt, field="attempt")
    if attempt >= normalized_request["max_attempts"]:
        raise VisualReflectionDataError(
            RejectionReason.TURN_LIMIT_EXHAUSTED,
            "attempt is outside the request's bounded max_attempts",
            field="attempt",
            source_record_id=normalized_request["source_record_id"],
        )
    _validate_nested_result_attempts(
        attempt=attempt,
        source_record_id=normalized_request["source_record_id"],
        image_synthesis_results=normalized_image_synthesis_results,
        reflection_synthesis_results=normalized_reflection_synthesis_results,
        verification_results=normalized_verification_results,
    )
    if status not in {"accepted", "retry", "rejected"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "status must be accepted, retry, or rejected",
            field="status",
        )
    if status == "accepted":
        if trajectory is None or rejection_reason is not None:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "accepted synthesis requires a trajectory and no rejection_reason",
            )
        normalized_trajectory = validate_trajectory(trajectory)
        for field in ("manifest_id", "source_dataset", "source_record_id", "pipeline_variant", "prompt"):
            if normalized_trajectory[field] != normalized_request[field]:
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_FIELD_TYPE,
                    f"accepted trajectory {field} does not match its synthesis request",
                    field=f"trajectory.{field}",
                    source_record_id=normalized_request["source_record_id"],
                )
        edit_turns = len(normalized_trajectory["images"]) - 1
        max_edit_turns = (
            1 if normalized_request["pipeline_variant"] == "pair_0_1_turn" else normalized_request["max_edit_turns"]
        )
        if edit_turns > max_edit_turns:
            raise VisualReflectionDataError(
                RejectionReason.TURN_LIMIT_EXHAUSTED,
                f"accepted trajectory has {edit_turns} edits but request allows {max_edit_turns}",
                source_record_id=normalized_request["source_record_id"],
            )
        if (
            normalized_request["pipeline_variant"] == "pair_0_1_turn"
            and edit_turns == 1
            and normalized_trajectory["images"][-1]["sha256"] != normalized_request["reference_image"]["sha256"]
        ):
            raise VisualReflectionDataError(
                RejectionReason.INCOMPATIBLE_EDIT_TARGET,
                "one-edit pair trajectory must terminate at the public reference image",
                field="trajectory.images[-1]",
                source_record_id=normalized_request["source_record_id"],
            )
        _validate_acceptance_verifications(
            normalized_request,
            normalized_trajectory,
            attempt=attempt,
            verification_results=normalized_verification_results,
        )
        _validate_acceptance_image_lineage(
            normalized_request,
            normalized_trajectory,
            attempt=attempt,
            image_synthesis_results=normalized_image_synthesis_results,
        )
        _validate_acceptance_reflection_lineage(
            normalized_request,
            normalized_trajectory,
            attempt=attempt,
            reflection_synthesis_results=normalized_reflection_synthesis_results,
        )
        return {
            "request_id": normalized_id,
            "request": normalized_request,
            "status": "accepted",
            "attempt": attempt,
            "source_split": source_split,
            "image_synthesis_results": normalized_image_synthesis_results,
            "reflection_synthesis_results": normalized_reflection_synthesis_results,
            "verification_results": normalized_verification_results,
            "trajectory": normalized_trajectory,
            "rejection_reason": None,
        }
    if trajectory is not None:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{status} synthesis cannot contain a trajectory",
        )
    if status == "retry" and attempt + 1 >= normalized_request["max_attempts"]:
        raise VisualReflectionDataError(
            RejectionReason.TURN_LIMIT_EXHAUSTED,
            "retry is invalid because the bounded request has no remaining attempt",
            field="status",
            source_record_id=normalized_request["source_record_id"],
        )
    normalized_reason = require_nonempty_text(rejection_reason, field="rejection_reason")
    return {
        "request_id": normalized_id,
        "request": normalized_request,
        "status": status,  # type: ignore[typeddict-item]
        "attempt": attempt,
        "source_split": source_split,
        "image_synthesis_results": normalized_image_synthesis_results,
        "reflection_synthesis_results": normalized_reflection_synthesis_results,
        "verification_results": normalized_verification_results,
        "trajectory": None,
        "rejection_reason": normalized_reason,
    }


def validate_synthesis_result(value: Mapping[str, Any]) -> SynthesisResult:
    """Rebuild and validate a complete synthesis result with its lineage evidence."""
    expected_fields = {
        "request_id",
        "request",
        "status",
        "attempt",
        "source_split",
        "image_synthesis_results",
        "reflection_synthesis_results",
        "verification_results",
        "trajectory",
        "rejection_reason",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "synthesis result has missing or unknown fields",
            field="synthesis_result",
        )
    expected = make_synthesis_result(
        request=value.get("request"),
        status=value.get("status"),
        attempt=value.get("attempt"),
        trajectory=value.get("trajectory"),
        rejection_reason=value.get("rejection_reason"),
        image_synthesis_results=value.get("image_synthesis_results"),
        reflection_synthesis_results=value.get("reflection_synthesis_results"),
        verification_results=value.get("verification_results"),
    )
    if canonical_json(dict(value)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "synthesis result contains stale or inconsistent lineage fields",
            field="synthesis_result",
        )
    return expected


def _validate_nested_result_attempts(
    *,
    attempt: int,
    source_record_id: str,
    image_synthesis_results: Sequence[ImageSynthesisResult],
    reflection_synthesis_results: Sequence[ReflectionSynthesisResult],
    verification_results: Sequence[VerificationResult],
) -> None:
    seen_request_ids: set[str] = set()
    nested_results = [*image_synthesis_results, *reflection_synthesis_results, *verification_results]
    for result in nested_results:
        if result["request"]["attempt"] != attempt:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "nested stage result attempt does not match the synthesis result attempt",
                field="synthesis_result.attempt",
                source_record_id=source_record_id,
            )
        if result["request_id"] in seen_request_ids:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "synthesis result contains a duplicate nested stage request",
                field="synthesis_result",
                source_record_id=source_record_id,
            )
        seen_request_ids.add(result["request_id"])


def _validate_image_synthesis_result(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    value: Mapping[str, Any],
) -> ImageSynthesisResult:
    if not isinstance(value, Mapping) or set(value) != {"request_id", "request", "output_image"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "image synthesis result has missing or unknown fields",
            field="image_synthesis_results",
        )
    expected = make_image_synthesis_result(
        parent_request=parent,
        request=value.get("request"),
        output_image=value.get("output_image"),
    )
    if canonical_json(dict(value)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "image synthesis result contains stale or inconsistent fields",
            field="image_synthesis_results",
        )
    return expected


def _validate_reflection_synthesis_result(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    value: Mapping[str, Any],
) -> ReflectionSynthesisResult:
    expected_fields = {"request_id", "request", "response_text", "response"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "reflection synthesis result has missing or unknown fields",
            field="reflection_synthesis_results",
        )
    expected = make_reflection_synthesis_result(
        parent_request=parent,
        request=value.get("request"),
        response_text=value.get("response_text"),
    )
    if canonical_json(dict(value)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "reflection synthesis result contains stale or inconsistent fields",
            field="reflection_synthesis_results",
        )
    return expected


def _validate_acceptance_reflection_lineage(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    trajectory: Mapping[str, Any],
    *,
    attempt: int,
    reflection_synthesis_results: Sequence[ReflectionSynthesisResult],
) -> None:
    expected: dict[str, tuple[ReflectionSynthesisRequest, Mapping[str, Any]]] = {}
    for turn_index, (image, step) in enumerate(zip(trajectory["images"], trajectory["steps"], strict=True)):
        request = make_reflection_synthesis_request(
            parent_request=parent,
            current_image=image,
            turn_index=turn_index,
            attempt=attempt,
        )
        expected[request["request_id"]] = (request, step)
    actual_by_id: dict[str, ReflectionSynthesisResult] = {}
    for result in reflection_synthesis_results:
        if result["request_id"] in actual_by_id:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "accepted synthesis contains a duplicate reflector result",
                field="reflection_synthesis_results",
                source_record_id=parent["source_record_id"],
            )
        actual_by_id[result["request_id"]] = result
    if set(actual_by_id) != set(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "accepted synthesis is missing or adding route-required reflector results",
            field="reflection_synthesis_results",
            source_record_id=parent["source_record_id"],
            details={
                "missing": sorted(set(expected) - set(actual_by_id)),
                "extra": sorted(set(actual_by_id) - set(expected)),
            },
        )
    for request_id, (expected_request, expected_step) in expected.items():
        actual = actual_by_id[request_id]
        if canonical_json(actual["request"]) != canonical_json(expected_request) or actual["response"] != expected_step:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "reflector result does not match the accepted trajectory step",
                field="reflection_synthesis_results",
                source_record_id=parent["source_record_id"],
            )


def _validate_acceptance_image_lineage(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    trajectory: Mapping[str, Any],
    *,
    attempt: int,
    image_synthesis_results: Sequence[ImageSynthesisResult],
) -> None:
    images = trajectory["images"]
    steps = trajectory["steps"]
    draft_request = make_draft_generation_request(parent_request=parent, attempt=attempt)
    expected_outputs: dict[str, tuple[ImageSynthesisRequest, Mapping[str, Any]]] = {
        draft_request["request_id"]: (draft_request, images[0])
    }
    if parent["pipeline_variant"] == "prompt_k_turn":
        for turn_index, step in enumerate(steps[:-1]):
            edit_request = make_edit_synthesis_request(
                parent_request=parent,
                current_image=images[turn_index],
                edit=step["edit"],
                turn_index=turn_index,
                attempt=attempt,
            )
            expected_outputs[edit_request["request_id"]] = (edit_request, images[turn_index + 1])

    actual_by_id: dict[str, ImageSynthesisResult] = {}
    for result in image_synthesis_results:
        if result["request_id"] in actual_by_id:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "accepted synthesis contains a duplicate image-generation result",
                field="image_synthesis_results",
                source_record_id=parent["source_record_id"],
            )
        actual_by_id[result["request_id"]] = result
    if set(actual_by_id) != set(expected_outputs):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "accepted synthesis is missing or adding route-required image-generation results",
            field="image_synthesis_results",
            source_record_id=parent["source_record_id"],
            details={
                "missing": sorted(set(expected_outputs) - set(actual_by_id)),
                "extra": sorted(set(actual_by_id) - set(expected_outputs)),
            },
        )
    for request_id, (expected_request, expected_image) in expected_outputs.items():
        actual = actual_by_id[request_id]
        if (
            canonical_json(actual["request"]) != canonical_json(expected_request)
            or actual["output_image"] != expected_image
        ):
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "image-generation result does not match the accepted trajectory state",
                field="image_synthesis_results",
                source_record_id=parent["source_record_id"],
            )


def _validate_verification_result(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    value: Mapping[str, Any],
) -> VerificationResult:
    if not isinstance(value, Mapping) or set(value) != {"request_id", "request", "verdict", "reason"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "verification result has missing or unknown fields",
            field="verification_results",
        )
    expected = make_verification_result(
        parent_request=parent,
        request=value.get("request"),
        verdict=value.get("verdict"),
        reason=value.get("reason"),
    )
    if canonical_json(dict(value)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "verification result contains stale or inconsistent fields",
            field="verification_results",
        )
    return expected


def _validate_acceptance_verifications(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    trajectory: Mapping[str, Any],
    *,
    attempt: int,
    verification_results: Sequence[VerificationResult],
) -> None:
    if any(result["verdict"] != "accept" for result in verification_results):
        raise VisualReflectionDataError(
            RejectionReason.VERIFIER_REJECTED,
            "accepted synthesis requires accept verdicts for every quality gate",
            field="verification_results.verdict",
            source_record_id=parent["source_record_id"],
        )
    images = trajectory["images"]
    steps = trajectory["steps"]
    expected_requests: list[PairVerificationRequest | TerminalVerificationRequest | EditVerificationRequest] = []
    if parent["pipeline_variant"] == "pair_0_1_turn":
        if len(images) == 2:
            expected_requests.append(
                make_pair_verification_request(
                    parent_request=parent,
                    draft_image=images[0],
                    proposed_step=steps[0],
                    attempt=attempt,
                )
            )
        expected_requests.append(
            make_terminal_verification_request(
                parent_request=parent,
                current_image=images[-1],
                proposed_step=steps[-1],
                turn_index=len(images) - 1,
                attempt=attempt,
            )
        )
    else:
        for turn_index, step in enumerate(steps[:-1]):
            expected_requests.append(
                make_edit_verification_request(
                    parent_request=parent,
                    current_image=images[turn_index],
                    target_image=images[turn_index + 1],
                    edit=step["edit"],
                    turn_index=turn_index,
                    attempt=attempt,
                )
            )
        expected_requests.append(
            make_terminal_verification_request(
                parent_request=parent,
                current_image=images[-1],
                proposed_step=steps[-1],
                turn_index=len(images) - 1,
                attempt=attempt,
            )
        )

    actual_by_id: dict[str, VerificationResult] = {}
    for result in verification_results:
        if result["request_id"] in actual_by_id:
            raise VisualReflectionDataError(
                RejectionReason.VERIFIER_REJECTED,
                "accepted synthesis contains a duplicate verifier receipt",
                field="verification_results",
                source_record_id=parent["source_record_id"],
            )
        actual_by_id[result["request_id"]] = result
    expected_by_id = {request["request_id"]: request for request in expected_requests}
    if set(actual_by_id) != set(expected_by_id):
        raise VisualReflectionDataError(
            RejectionReason.VERIFIER_REJECTED,
            "accepted synthesis is missing or adding route-required verifier receipts",
            field="verification_results",
            source_record_id=parent["source_record_id"],
            details={
                "missing": sorted(set(expected_by_id) - set(actual_by_id)),
                "extra": sorted(set(actual_by_id) - set(expected_by_id)),
            },
        )
    for request_id, expected_request in expected_by_id.items():
        if canonical_json(actual_by_id[request_id]["request"]) != canonical_json(expected_request):
            raise VisualReflectionDataError(
                RejectionReason.VERIFIER_REJECTED,
                "verifier receipt does not match the accepted trajectory transition",
                field="verification_results.request",
                source_record_id=parent["source_record_id"],
            )


def _required_value(record: Mapping[str, Any], field: str, source_record_id: str | None) -> Any:
    if field not in record:
        raise VisualReflectionDataError(
            RejectionReason.MISSING_FIELD,
            f"source record is missing {field}",
            field=field,
            source_record_id=source_record_id,
        )
    return record[field]


def _normalize_echo_image_path(value: Any, *, source_record_id: str) -> Any:
    if not isinstance(value, str):
        return value
    if value.startswith(ECHO_IMAGE_PATH_PREFIX):
        return value[len(ECHO_IMAGE_PATH_PREFIX) :]
    if value.startswith("/"):
        raise VisualReflectionDataError(
            RejectionReason.MISSING_IMAGE,
            f"Echo image path has an unsupported logical root: {value}",
            field="output_image",
            source_record_id=source_record_id,
        )
    return value


def _validate_echo_source(source: Mapping[str, Any]) -> ImagePairSource:
    if not isinstance(source, Mapping):
        raise VisualReflectionDataError(RejectionReason.INVALID_FIELD_TYPE, "Echo source must be a mapping")
    expected_fields = {
        "source_dataset",
        "source_record_id",
        "pipeline_variant",
        "prompt",
        "reference_image",
        "dedup_key",
    }
    if set(source) != expected_fields:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "normalized Echo source has missing or unknown fields",
        )
    dataset = require_nonempty_text(source.get("source_dataset"), field="source_dataset")
    if dataset != ECHO_DATASET_ID or source.get("pipeline_variant") != "pair_0_1_turn":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "source is not a normalized Echo pair",
        )
    return {
        "source_dataset": dataset,
        "source_record_id": require_nonempty_text(source.get("source_record_id"), field="source_record_id"),
        "pipeline_variant": "pair_0_1_turn",
        "prompt": require_nonempty_text(source.get("prompt"), field="prompt", reason=RejectionReason.EMPTY_PROMPT),
        "reference_image": validate_image_ref(source.get("reference_image")),
        "dedup_key": require_nonempty_text(source.get("dedup_key"), field="dedup_key"),
    }


def _validate_prompt_source(source: Mapping[str, Any]) -> PromptOnlySource:
    if not isinstance(source, Mapping):
        raise VisualReflectionDataError(RejectionReason.INVALID_FIELD_TYPE, "prompt source must be a mapping")
    expected_fields = {"source_dataset", "source_record_id", "pipeline_variant", "prompt", "dedup_key"}
    if set(source) != expected_fields:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "normalized prompt source has missing or unknown fields",
        )
    dataset = require_nonempty_text(source.get("source_dataset"), field="source_dataset")
    if dataset != MIDJOURNEY_DATASET_ID or source.get("pipeline_variant") != "prompt_k_turn":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "source is not a normalized Midjourney prompt",
        )
    return {
        "source_dataset": dataset,
        "source_record_id": require_nonempty_text(source.get("source_record_id"), field="source_record_id"),
        "pipeline_variant": "prompt_k_turn",
        "prompt": require_nonempty_text(source.get("prompt"), field="prompt", reason=RejectionReason.EMPTY_PROMPT),
        "dedup_key": require_nonempty_text(source.get("dedup_key"), field="dedup_key"),
    }


def _validate_synthesis_request(request: Mapping[str, Any]) -> PairSynthesisRequest | PromptKTurnSynthesisRequest:
    if not isinstance(request, Mapping):
        raise VisualReflectionDataError(RejectionReason.INVALID_FIELD_TYPE, "synthesis request must be a mapping")
    pipeline_variant = request.get("pipeline_variant")
    if pipeline_variant == "pair_0_1_turn":
        normalized_source = _validate_echo_source(
            {
                "source_dataset": request.get("source_dataset"),
                "source_record_id": request.get("source_record_id"),
                "pipeline_variant": request.get("pipeline_variant"),
                "prompt": request.get("prompt"),
                "reference_image": request.get("reference_image"),
                "dedup_key": request.get("dedup_key"),
            }
        )
        expected = build_pair_synthesis_request(
            normalized_source,
            manifest_id=request.get("manifest_id"),
            source_split={
                "source_dataset": request.get("source_dataset"),
                "source_record_id": request.get("source_record_id"),
                "dedup_key": request.get("dedup_key"),
                "partition_id": request.get("partition_id"),
                "split": request.get("split"),
            },
            partition=request.get("partition"),
            seed=request.get("seed"),
            max_attempts=request.get("max_attempts"),
        )
    elif pipeline_variant == "prompt_k_turn":
        normalized_source = _validate_prompt_source(
            {
                "source_dataset": request.get("source_dataset"),
                "source_record_id": request.get("source_record_id"),
                "pipeline_variant": request.get("pipeline_variant"),
                "prompt": request.get("prompt"),
                "dedup_key": request.get("dedup_key"),
            }
        )
        expected = build_prompt_synthesis_request(
            normalized_source,
            manifest_id=request.get("manifest_id"),
            source_split={
                "source_dataset": request.get("source_dataset"),
                "source_record_id": request.get("source_record_id"),
                "dedup_key": request.get("dedup_key"),
                "partition_id": request.get("partition_id"),
                "split": request.get("split"),
            },
            partition=request.get("partition"),
            seed=request.get("seed"),
            max_attempts=request.get("max_attempts"),
            max_edit_turns=request.get("max_edit_turns"),
        )
    else:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "synthesis request has an unsupported pipeline_variant",
            field="pipeline_variant",
        )
    if canonical_json(dict(request)) != canonical_json(expected):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "synthesis request contains unknown, stale, or inconsistent fields",
        )
    return expected


def _rebuild_image_synthesis_request(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    request: Mapping[str, Any],
) -> ImageSynthesisRequest:
    if not isinstance(request, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "image synthesis request must be a mapping",
            field="request",
        )
    fields = set(request)
    draft_fields = {"request_id", "parent_request_id", "prompt", "seed", "attempt"}
    edit_fields = {
        "request_id",
        "parent_request_id",
        "prompt",
        "current_image",
        "edit",
        "turn_index",
        "seed",
        "attempt",
    }
    if fields == draft_fields:
        return make_draft_generation_request(
            parent_request=parent,
            attempt=request.get("attempt"),
        )
    if fields == edit_fields:
        return make_edit_synthesis_request(
            parent_request=parent,
            current_image=request.get("current_image"),
            edit=request.get("edit"),
            turn_index=request.get("turn_index"),
            attempt=request.get("attempt"),
        )
    raise VisualReflectionDataError(
        RejectionReason.INVALID_FIELD_TYPE,
        "image synthesis request has missing or unknown fields",
        field="request",
    )


def _rebuild_verification_request(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    request: Mapping[str, Any],
) -> PairVerificationRequest | TerminalVerificationRequest | EditVerificationRequest:
    if not isinstance(request, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "verification request must be a mapping",
            field="request",
        )
    fields = set(request)
    pair_fields = {
        "request_id",
        "parent_request_id",
        "prompt",
        "draft_image",
        "reference_image",
        "proposed_step",
        "attempt",
    }
    terminal_fields = {
        "request_id",
        "parent_request_id",
        "prompt",
        "current_image",
        "proposed_step",
        "turn_index",
        "attempt",
    }
    edit_fields = {
        "request_id",
        "parent_request_id",
        "prompt",
        "current_image",
        "target_image",
        "edit",
        "turn_index",
        "attempt",
    }
    if fields == pair_fields:
        return make_pair_verification_request(
            parent_request=parent,
            draft_image=request.get("draft_image"),
            proposed_step=request.get("proposed_step"),
            attempt=request.get("attempt"),
        )
    if fields == terminal_fields:
        return make_terminal_verification_request(
            parent_request=parent,
            current_image=request.get("current_image"),
            proposed_step=request.get("proposed_step"),
            turn_index=request.get("turn_index"),
            attempt=request.get("attempt"),
        )
    if fields == edit_fields:
        return make_edit_verification_request(
            parent_request=parent,
            current_image=request.get("current_image"),
            target_image=request.get("target_image"),
            edit=request.get("edit"),
            turn_index=request.get("turn_index"),
            attempt=request.get("attempt"),
        )
    raise VisualReflectionDataError(
        RejectionReason.INVALID_FIELD_TYPE,
        "verification request has missing or unknown fields",
        field="request",
    )


def _validate_stage_attempt(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    attempt: Any,
) -> int:
    normalized = _require_nonnegative_int(attempt, field="attempt")
    if normalized >= parent["max_attempts"]:
        raise VisualReflectionDataError(
            RejectionReason.TURN_LIMIT_EXHAUSTED,
            "stage attempt is outside the parent request's bounded max_attempts",
            field="attempt",
            source_record_id=parent["source_record_id"],
        )
    return normalized


def _derive_edit_seed(parent_seed: int, *, turn_index: int, attempt: int) -> int:
    digest = stable_digest("edit_stage_seed_v1", [parent_seed, turn_index, attempt])
    return int(digest[:16], 16) & ((1 << 63) - 1)


def _validate_reflection_turn(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    *,
    image: Mapping[str, Any],
    turn_index: Any,
) -> int:
    return _validate_state_turn(parent, image=image, turn_index=turn_index, stage="reflection")


def _validate_terminal_turn(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    *,
    image: Mapping[str, Any],
    turn_index: Any,
) -> int:
    return _validate_state_turn(parent, image=image, turn_index=turn_index, stage="terminal verification")


def _validate_state_turn(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    *,
    image: Mapping[str, Any],
    turn_index: Any,
    stage: str,
) -> int:
    normalized = _require_nonnegative_int(turn_index, field="turn_index")
    max_turn = 1 if parent["pipeline_variant"] == "pair_0_1_turn" else parent["max_edit_turns"]
    if normalized > max_turn:
        raise VisualReflectionDataError(
            RejectionReason.TURN_LIMIT_EXHAUSTED,
            f"{stage} turn exceeds the parent request's bounded edit turns",
            field="turn_index",
            source_record_id=parent["source_record_id"],
        )
    if (
        parent["pipeline_variant"] == "pair_0_1_turn"
        and normalized == 1
        and image["sha256"] != parent["reference_image"]["sha256"]
    ):
        raise VisualReflectionDataError(
            RejectionReason.INCOMPATIBLE_EDIT_TARGET,
            f"pair {stage} turn 1 must judge the public reference image",
            field="current_image",
            source_record_id=parent["source_record_id"],
        )
    return normalized


def _require_prompt_parent(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    *,
    stage: str,
) -> None:
    if parent["pipeline_variant"] != "prompt_k_turn":
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{stage} is forbidden for the Echo pair route",
            field="parent_request.pipeline_variant",
            source_record_id=parent["source_record_id"],
        )


def _validate_edit_turn(
    parent: PairSynthesisRequest | PromptKTurnSynthesisRequest,
    turn_index: Any,
) -> int:
    _require_prompt_parent(parent, stage="edit")
    normalized = _require_nonnegative_int(turn_index, field="turn_index")
    max_edit_turns = parent["max_edit_turns"]  # type: ignore[typeddict-item]
    if normalized >= max_edit_turns:
        raise VisualReflectionDataError(
            RejectionReason.TURN_LIMIT_EXHAUSTED,
            "edit turn is outside the parent request's bounded max_edit_turns",
            field="turn_index",
            source_record_id=parent["source_record_id"],
        )
    return normalized


def _require_nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a non-negative integer",
            field=field,
        )
    return value


def _require_positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a positive integer",
            field=field,
        )
    return value
