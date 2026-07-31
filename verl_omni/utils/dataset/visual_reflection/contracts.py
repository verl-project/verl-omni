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
"""Tokenizer-independent contracts for multi-turn visual-reflection data."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal, TypeAlias, TypedDict

SCHEMA_VERSION = "visual_reflection_v1"
CONVERTER_VERSION = "public_adapters_v1"
FILTER_VERSION = "fail_closed_v1"
FINGERPRINT_VERSION = "nfkc_ws_casefold_v1"
SPLIT_VERSION = "sha256_threshold_v1"
T2I_TARGET_POLICY = "terminal_state_v1"
IMAGE_HASH_ALGORITHM = "sha256"

Action: TypeAlias = Literal["continue", "stop"]
PipelineVariant: TypeAlias = Literal["direct_parse", "pair_0_1_turn", "prompt_k_turn"]
AtomicTask: TypeAlias = Literal["t2i", "reflect", "edit"]
DataSplit: TypeAlias = Literal["train", "validation", "test"]
SynthesisStatus: TypeAlias = Literal["accepted", "retry", "rejected"]
VerificationVerdict: TypeAlias = Literal["accept", "retry", "reject"]
RejectionLineage: TypeAlias = Literal["direct_parse", "source_preparation", "synthesis"]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESERVED_PROTOCOL_HEADER_RE = re.compile(r"^(Reflection|Action|Edit):", re.MULTILINE)
_PIPELINE_VARIANTS = {"direct_parse", "pair_0_1_turn", "prompt_k_turn"}
_DATA_SPLITS = {"train", "validation", "test"}


class RejectionReason(str, Enum):
    """Stable rejection codes used by converters and synthesis pipelines."""

    MISSING_FIELD = "missing_field"
    INVALID_FIELD_TYPE = "invalid_field_type"
    LENGTH_MISMATCH = "length_mismatch"
    EMPTY_PROMPT = "empty_prompt"
    MISSING_IMAGE = "missing_image"
    INVALID_IMAGE_HASH = "invalid_image_hash"
    EMPTY_REFLECTION = "empty_reflection"
    EMPTY_CONTINUE_EDIT = "empty_continue_edit"
    CONTRADICTORY_CONTINUE_EDIT = "contradictory_continue_edit"
    NONEMPTY_STOP_EDIT = "nonempty_stop_edit"
    CONTRADICTORY_TERMINAL = "contradictory_terminal"
    CONTRADICTORY_TERMINAL_EDIT = "contradictory_terminal_edit"
    TRANSITION_HASH_MISMATCH = "transition_hash_mismatch"
    PROTOCOL_PARSE_ERROR = "protocol_parse_error"
    DUPLICATE_SOURCE_RECORD = "duplicate_source_record"
    DUPLICATE_CONTENT = "duplicate_content"
    INVALID_SPLIT = "invalid_split"
    INVALID_PROVENANCE = "invalid_provenance"
    INCOMPATIBLE_EDIT_TARGET = "incompatible_edit_target"
    EDIT_NON_ADHERENCE = "edit_non_adherence"
    QUALITY_REGRESSION = "quality_regression"
    MISSING_TERMINAL_STOP = "missing_terminal_stop"
    TURN_LIMIT_EXHAUSTED = "turn_limit_exhausted"
    VERIFIER_REJECTED = "verifier_rejected"


class VisualReflectionDataError(ValueError):
    """Expected fail-closed data error with a stable rejection reason."""

    def __init__(
        self,
        reason: RejectionReason,
        message: str,
        *,
        field: str | None = None,
        source_record_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason
        self.field = field
        self.source_record_id = source_record_id
        self.details = dict(details or {})


class ImageRef(TypedDict):
    """Portable image reference with an audited content digest."""

    uri: str
    sha256: str


class ReflectionStep(TypedDict):
    """Structured visual assessment and bounded next action."""

    reflection: str
    action: Action
    edit: str


class VisualReflectionTrajectory(TypedDict):
    """Canonical tokenizer-independent visual-reflection trajectory."""

    trajectory_id: str
    manifest_id: str
    source_dataset: str
    source_record_id: str
    pipeline_variant: PipelineVariant
    prompt: str
    images: list[ImageRef]
    steps: list[ReflectionStep]


class ImagePairSource(TypedDict):
    """Normalized text-to-image pair before synthesis and splitting."""

    source_dataset: str
    source_record_id: str
    pipeline_variant: Literal["pair_0_1_turn"]
    prompt: str
    reference_image: ImageRef
    dedup_key: str


class PromptOnlySource(TypedDict):
    """Normalized prompt-only source before synthesis and splitting."""

    source_dataset: str
    source_record_id: str
    pipeline_variant: Literal["prompt_k_turn"]
    prompt: str
    dedup_key: str


class PairSynthesisRequest(ImagePairSource):
    """Split-assigned request for bounded zero/one-edit pair synthesis."""

    request_id: str
    manifest_id: str
    partition: SplitProvenance
    partition_id: str
    split: DataSplit
    seed: int
    max_attempts: int


class PromptKTurnSynthesisRequest(PromptOnlySource):
    """Split-assigned request for bounded prompt-driven k-turn synthesis."""

    request_id: str
    manifest_id: str
    partition: SplitProvenance
    partition_id: str
    split: DataSplit
    seed: int
    max_attempts: int
    max_edit_turns: int


SynthesisRequest: TypeAlias = PairSynthesisRequest | PromptKTurnSynthesisRequest


class DraftGenerationRequest(TypedDict):
    """Draft image-generation input with a stable source seed."""

    request_id: str
    parent_request_id: str
    prompt: str
    seed: int
    attempt: int


class ReflectionSynthesisRequest(TypedDict):
    """Narrow reflector input that intentionally exposes no reference image."""

    request_id: str
    parent_request_id: str
    prompt: str
    current_image: ImageRef
    turn_index: int
    attempt: int


class ReflectionSynthesisResult(TypedDict):
    """Canonical parsed reflector output bound to its exact narrow request."""

    request_id: str
    request: ReflectionSynthesisRequest
    response_text: str
    response: ReflectionStep


class PairVerificationRequest(TypedDict):
    """Pair verifier input where the public reference is explicitly allowed."""

    request_id: str
    parent_request_id: str
    prompt: str
    draft_image: ImageRef
    reference_image: ImageRef
    proposed_step: ReflectionStep
    attempt: int


class TerminalVerificationRequest(TypedDict):
    """Terminal verifier input that exposes only the image being judged."""

    request_id: str
    parent_request_id: str
    prompt: str
    current_image: ImageRef
    proposed_step: ReflectionStep
    turn_index: int
    attempt: int


class EditSynthesisRequest(TypedDict):
    """Offline edit-model input for one accepted delta transition."""

    request_id: str
    parent_request_id: str
    prompt: str
    current_image: ImageRef
    edit: str
    turn_index: int
    seed: int
    attempt: int


class EditVerificationRequest(TypedDict):
    """Verifier input for edit adherence, fidelity, improvement, and preservation."""

    request_id: str
    parent_request_id: str
    prompt: str
    current_image: ImageRef
    target_image: ImageRef
    edit: str
    turn_index: int
    attempt: int


VerificationRequest: TypeAlias = PairVerificationRequest | TerminalVerificationRequest | EditVerificationRequest
ImageSynthesisRequest: TypeAlias = DraftGenerationRequest | EditSynthesisRequest


class ImageSynthesisResult(TypedDict):
    """Materialized image output bound to an exact draft or edit request."""

    request_id: str
    request: ImageSynthesisRequest
    output_image: ImageRef


class VerificationResult(TypedDict):
    """Serializable verifier decision that cannot rewrite the proposed action."""

    request_id: str
    request: VerificationRequest
    verdict: VerificationVerdict
    reason: str


class SynthesisResult(TypedDict):
    """Serializable accepted, retryable, or rejected synthesis result."""

    request_id: str
    request: SynthesisRequest
    status: SynthesisStatus
    attempt: int
    source_split: SourceSplitAssignment
    image_synthesis_results: list[ImageSynthesisResult]
    reflection_synthesis_results: list[ReflectionSynthesisResult]
    verification_results: list[VerificationResult]
    trajectory: VisualReflectionTrajectory | None
    rejection_reason: str | None


class T2IAtom(TypedDict):
    """Atomic prompt-to-terminal-image supervision entry."""

    atom_id: str
    task: Literal["t2i"]
    trajectory_id: str
    manifest_id: str
    source_dataset: str
    source_record_id: str
    pipeline_variant: PipelineVariant
    partition_id: str
    source_dedup_key: str
    split: DataSplit
    prompt: str
    target_state_index: int
    target_image: ImageRef
    t2i_target_policy: Literal["terminal_state_v1"]


class ReflectAtom(TypedDict):
    """Atomic image-and-prompt to structured reflection supervision entry."""

    atom_id: str
    task: Literal["reflect"]
    trajectory_id: str
    manifest_id: str
    source_dataset: str
    source_record_id: str
    pipeline_variant: PipelineVariant
    partition_id: str
    source_dedup_key: str
    split: DataSplit
    prompt: str
    state_index: int
    current_image: ImageRef
    response: ReflectionStep


class EditAtom(TypedDict):
    """Atomic clean-source plus delta edit to next-image supervision entry."""

    atom_id: str
    task: Literal["edit"]
    trajectory_id: str
    manifest_id: str
    source_dataset: str
    source_record_id: str
    pipeline_variant: PipelineVariant
    partition_id: str
    source_dedup_key: str
    split: DataSplit
    prompt: str
    state_index: int
    current_image: ImageRef
    edit: str
    target_image: ImageRef


AtomicSFTExample: TypeAlias = T2IAtom | ReflectAtom | EditAtom


class SourceDatasetProvenance(TypedDict):
    """Pinned public-source provenance recorded in a release manifest."""

    dataset_id: str
    config: str
    split: str
    revision: str
    license: str
    pipeline_variant: PipelineVariant


class ModelProvenance(TypedDict):
    """Pinned offline generator, reflector, verifier, or editor revision."""

    model_id: str
    revision: str


class SplitProvenance(TypedDict):
    """Deterministic source-level partition configuration."""

    algorithm: str
    fingerprint_version: str
    partition_id: str
    seed: str
    ratios: dict[str, float]


class ManifestCounts(TypedDict):
    """Accepted and rejected release counts."""

    accepted: int
    accepted_by_split: dict[str, int]
    rejected: int
    rejected_by_reason: dict[str, int]


class DataManifest(TypedDict):
    """Versioned provenance manifest for one converted data release."""

    manifest_id: str
    schema_version: str
    converter_version: str
    filter_version: str
    code_revision: str
    t2i_target_policy: Literal["terminal_state_v1"]
    image_hash_algorithm: Literal["sha256"]
    sources: list[SourceDatasetProvenance]
    models: dict[str, ModelProvenance | None]
    prompt_template_hashes: dict[str, str]
    seeds: dict[str, int]
    converter_config: dict[str, Any]
    partition: SplitProvenance
    image_hashes: list[str]
    counts: ManifestCounts


class RejectionRecord(TypedDict):
    """One final source-record rejection after any bounded retries are exhausted."""

    manifest_id: str
    source_dataset: str
    source_record_id: str
    pipeline_variant: PipelineVariant
    lineage_kind: RejectionLineage
    stage: str
    reason: str
    message: str
    attempt: int
    request_id: str | None
    source_split: SourceSplitAssignment | None
    synthesis_result: SynthesisResult | None
    details: dict[str, Any]


class DuplicateRecord(TypedDict):
    """Stable exact-duplicate decision made before splitting."""

    source_dataset: str
    source_record_id: str
    retained_source_dataset: str
    retained_source_record_id: str
    dedup_key: str


class SourceSplitAssignment(TypedDict):
    """Source identity, canonical dedup group, and pre-expansion split."""

    source_dataset: str
    source_record_id: str
    dedup_key: str
    partition_id: str
    split: DataSplit


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically for IDs and artifacts."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"value is not canonical JSON: {exc}",
        ) from exc


def stable_digest(namespace: str, value: Any) -> str:
    """Return a namespaced SHA-256 digest of canonical JSON data."""
    payload = f"{namespace}\0{canonical_json(value)}".encode()
    return hashlib.sha256(payload).hexdigest()


def normalize_text_for_fingerprint(value: str) -> str:
    """Normalize text for exact, versioned source deduplication."""
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def derive_pair_source_dedup_key(prompt: str, reference_image: Mapping[str, Any]) -> str:
    """Derive the exact prompt-plus-reference fingerprint for pair synthesis."""
    normalized_prompt = require_nonempty_text(prompt, field="prompt", reason=RejectionReason.EMPTY_PROMPT)
    image = validate_image_ref(reference_image)
    return stable_digest(
        "image_pair_source_dedup_v1",
        {
            "version": FINGERPRINT_VERSION,
            "prompt": normalize_text_for_fingerprint(normalized_prompt),
            "reference_sha256": image["sha256"],
        },
    )


def derive_prompt_source_dedup_key(prompt: str) -> str:
    """Derive the exact normalized-prompt fingerprint for prompt-only synthesis."""
    normalized_prompt = require_nonempty_text(prompt, field="prompt", reason=RejectionReason.EMPTY_PROMPT)
    return stable_digest(
        "prompt_only_source_dedup_v1",
        {"version": FINGERPRINT_VERSION, "prompt": normalize_text_for_fingerprint(normalized_prompt)},
    )


def require_nonempty_text(
    value: Any,
    *,
    field: str,
    reason: RejectionReason = RejectionReason.INVALID_FIELD_TYPE,
    source_record_id: str | None = None,
) -> str:
    """Validate a string field and trim only its outer whitespace."""
    if not isinstance(value, str):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a string",
            field=field,
            source_record_id=source_record_id,
        )
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise VisualReflectionDataError(
            reason,
            f"{field} must not be empty",
            field=field,
            source_record_id=source_record_id,
        )
    return text


def validate_data_split(split: Any) -> DataSplit:
    """Validate a canonical train, validation, or test split name."""
    if not isinstance(split, str) or split not in _DATA_SPLITS:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            f"split must be one of {sorted(_DATA_SPLITS)}, got {split!r}",
            field="split",
        )
    return split  # type: ignore[return-value]


def validate_image_ref(value: Mapping[str, Any]) -> ImageRef:
    """Validate and normalize a canonical image reference."""
    mapping = _require_mapping(value, field="image")
    _require_exact_keys(mapping, {"uri", "sha256"}, field="image")
    uri = require_nonempty_text(mapping["uri"], field="image.uri")
    digest = mapping["sha256"]
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_IMAGE_HASH,
            "image.sha256 must be 64 lowercase hexadecimal characters",
            field="image.sha256",
        )
    return {"uri": uri, "sha256": digest}


def validate_reflection_step(value: Mapping[str, Any]) -> ReflectionStep:
    """Validate and normalize one Reflection/Action/Edit step."""
    mapping = _require_mapping(value, field="step")
    _require_exact_keys(mapping, {"reflection", "action", "edit"}, field="step")
    reflection = require_nonempty_text(
        mapping["reflection"], field="step.reflection", reason=RejectionReason.EMPTY_REFLECTION
    )
    action = mapping["action"]
    if not isinstance(action, str) or action not in {"continue", "stop"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "step.action must be exactly 'continue' or 'stop'",
            field="step.action",
        )
    edit_value = mapping["edit"]
    if not isinstance(edit_value, str):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "step.edit must be a string",
            field="step.edit",
        )
    edit = edit_value.replace("\r\n", "\n").replace("\r", "\n").strip()
    if action == "continue" and not edit:
        raise VisualReflectionDataError(
            RejectionReason.EMPTY_CONTINUE_EDIT,
            "continue requires a non-empty Edit",
            field="step.edit",
        )
    if action == "stop" and edit:
        raise VisualReflectionDataError(
            RejectionReason.NONEMPTY_STOP_EDIT,
            "stop requires an empty Edit",
            field="step.edit",
        )
    for field, text in (("reflection", reflection), ("edit", edit)):
        if _RESERVED_PROTOCOL_HEADER_RE.search(text):
            raise VisualReflectionDataError(
                RejectionReason.PROTOCOL_PARSE_ERROR,
                f"step.{field} contains an ambiguous reserved protocol header line",
                field=f"step.{field}",
            )
    return {"reflection": reflection, "action": action, "edit": edit}  # type: ignore[typeddict-item]


def validate_trajectory(value: Mapping[str, Any]) -> VisualReflectionTrajectory:
    """Validate the canonical trajectory shape and all transition invariants."""
    mapping = _require_mapping(value, field="trajectory")
    expected = {
        "trajectory_id",
        "manifest_id",
        "source_dataset",
        "source_record_id",
        "pipeline_variant",
        "prompt",
        "images",
        "steps",
    }
    _require_exact_keys(mapping, expected, field="trajectory")
    trajectory_id = require_nonempty_text(mapping["trajectory_id"], field="trajectory.trajectory_id")
    manifest_id = require_nonempty_text(mapping["manifest_id"], field="trajectory.manifest_id")
    source_dataset = require_nonempty_text(mapping["source_dataset"], field="trajectory.source_dataset")
    source_record_id = require_nonempty_text(mapping["source_record_id"], field="trajectory.source_record_id")
    pipeline_variant = mapping["pipeline_variant"]
    if not isinstance(pipeline_variant, str) or pipeline_variant not in _PIPELINE_VARIANTS:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"trajectory.pipeline_variant must be one of {sorted(_PIPELINE_VARIANTS)}",
            field="trajectory.pipeline_variant",
            source_record_id=source_record_id,
        )
    prompt = require_nonempty_text(
        mapping["prompt"],
        field="trajectory.prompt",
        reason=RejectionReason.EMPTY_PROMPT,
        source_record_id=source_record_id,
    )
    images_value = mapping["images"]
    steps_value = mapping["steps"]
    if not isinstance(images_value, list) or not isinstance(steps_value, list):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "trajectory.images and trajectory.steps must be lists",
            source_record_id=source_record_id,
        )
    if not images_value:
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            "trajectory must contain at least one image and one step",
            source_record_id=source_record_id,
        )
    if len(images_value) != len(steps_value):
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            f"trajectory has {len(images_value)} images but {len(steps_value)} steps",
            source_record_id=source_record_id,
        )
    images = [validate_image_ref(image) for image in images_value]
    steps = [validate_reflection_step(step) for step in steps_value]
    for index, step in enumerate(steps[:-1]):
        if step["action"] != "continue":
            raise VisualReflectionDataError(
                RejectionReason.CONTRADICTORY_TERMINAL,
                f"step {index} stops before the final state",
                field=f"trajectory.steps[{index}].action",
                source_record_id=source_record_id,
            )
    if steps[-1]["action"] != "stop":
        raise VisualReflectionDataError(
            RejectionReason.MISSING_TERMINAL_STOP,
            "the final trajectory step must stop",
            field=f"trajectory.steps[{len(steps) - 1}].action",
            source_record_id=source_record_id,
        )
    expected_trajectory_id = derive_trajectory_id(
        manifest_id=manifest_id,
        source_dataset=source_dataset,
        source_record_id=source_record_id,
        pipeline_variant=pipeline_variant,  # type: ignore[arg-type]
        prompt=prompt,
        images=images,
        steps=steps,
    )
    if trajectory_id != expected_trajectory_id:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "trajectory_id does not match the canonical trajectory content",
            field="trajectory.trajectory_id",
            source_record_id=source_record_id,
        )
    return {
        "trajectory_id": trajectory_id,
        "manifest_id": manifest_id,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "pipeline_variant": pipeline_variant,  # type: ignore[typeddict-item]
        "prompt": prompt,
        "images": images,
        "steps": steps,
    }


def derive_trajectory_id(
    *,
    manifest_id: str,
    source_dataset: str,
    source_record_id: str,
    pipeline_variant: PipelineVariant,
    prompt: str,
    images: list[ImageRef],
    steps: list[ReflectionStep],
) -> str:
    """Derive a stable trajectory ID from its immutable source and content."""
    payload = {
        "manifest_id": manifest_id,
        "source_dataset": source_dataset,
        "source_record_id": source_record_id,
        "pipeline_variant": pipeline_variant,
        "prompt": prompt,
        "images": images,
        "steps": steps,
    }
    return f"traj_{stable_digest(SCHEMA_VERSION, payload)}"


def trajectory_dedup_key(trajectory: Mapping[str, Any]) -> str:
    """Compute the exact source-level dedup key for a direct-parsed trajectory."""
    normalized = validate_trajectory(trajectory)
    payload = {
        "version": FINGERPRINT_VERSION,
        "prompt": normalize_text_for_fingerprint(normalized["prompt"]),
        "image_sha256": [image["sha256"] for image in normalized["images"]],
        "steps": normalized["steps"],
    }
    return stable_digest("trajectory_dedup", payload)


def _require_mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a mapping",
            field=field,
        )
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    missing = expected - actual
    extra = actual - expected
    if missing:
        raise VisualReflectionDataError(
            RejectionReason.MISSING_FIELD,
            f"{field} is missing required fields: {sorted(missing)}",
            field=field,
        )
    if extra:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} contains unknown fields: {sorted(extra)}",
            field=field,
        )
