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
"""Exact source-level deduplication and deterministic split assignment."""

from __future__ import annotations

import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

from .contracts import (
    FINGERPRINT_VERSION,
    SPLIT_VERSION,
    DataSplit,
    DuplicateRecord,
    RejectionReason,
    SourceSplitAssignment,
    SplitProvenance,
    VisualReflectionDataError,
    canonical_json,
    derive_pair_source_dedup_key,
    derive_prompt_source_dedup_key,
    require_nonempty_text,
    stable_digest,
    trajectory_dedup_key,
)

T = TypeVar("T", bound=Mapping[str, Any])
SourceIdentity = tuple[str, str]
_PARTITION_ID_RE = re.compile(r"^partition_[0-9a-f]{64}$")


def deduplicate_source_records(records: Sequence[T]) -> tuple[list[T], list[DuplicateRecord]]:
    """Keep a stable survivor for every exact dedup group before splitting."""
    unique_by_identity: dict[SourceIdentity, T] = {}
    signature_by_identity: dict[SourceIdentity, str] = {}
    repeated_identities: list[SourceIdentity] = []
    for record in records:
        identity = source_identity(record)
        signature = canonical_json(record)
        previous_signature = signature_by_identity.get(identity)
        if previous_signature is not None and previous_signature != signature:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                f"source identity {identity!r} has conflicting content",
                source_record_id=identity[1],
            )
        if previous_signature is not None:
            repeated_identities.append(identity)
            continue
        signature_by_identity[identity] = signature
        unique_by_identity[identity] = record

    groups: dict[str, list[SourceIdentity]] = defaultdict(list)
    for identity, record in unique_by_identity.items():
        groups[source_dedup_key(record)].append(identity)

    retained: list[T] = []
    duplicates: list[DuplicateRecord] = []
    retained_by_identity: dict[SourceIdentity, SourceIdentity] = {}
    for dedup_key in sorted(groups):
        identities = sorted(groups[dedup_key])
        survivor = identities[0]
        retained.append(unique_by_identity[survivor])
        for identity in identities:
            retained_by_identity[identity] = survivor
        for identity in identities[1:]:
            duplicates.append(_duplicate_record(identity, survivor, dedup_key))

    for identity in sorted(repeated_identities):
        survivor = retained_by_identity[identity]
        duplicates.append(_duplicate_record(identity, survivor, source_dedup_key(unique_by_identity[identity])))
    retained.sort(key=source_identity)
    duplicates.sort(key=lambda row: (row["source_dataset"], row["source_record_id"], row["dedup_key"]))
    return retained, duplicates


def assign_source_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    ratios: Mapping[str, float],
    seed: str | int,
) -> dict[SourceIdentity, SourceSplitAssignment]:
    """Assign whole source/dedup groups to stable SHA-256 threshold splits."""
    partition = make_split_provenance(ratios=ratios, seed=seed)
    assignments: dict[SourceIdentity, SourceSplitAssignment] = {}
    seen_identity_keys: dict[SourceIdentity, str] = {}
    for record in records:
        identity = source_identity(record)
        dedup_key = source_dedup_key(record)
        previous = seen_identity_keys.get(identity)
        if previous is not None and previous != dedup_key:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                f"source identity {identity!r} maps to multiple dedup keys",
                source_record_id=identity[1],
            )
        seen_identity_keys[identity] = dedup_key
        split = _split_for_dedup_key(dedup_key, partition)
        assignments[identity] = {
            "source_dataset": identity[0],
            "source_record_id": identity[1],
            "dedup_key": dedup_key,
            "partition_id": partition["partition_id"],
            "split": split,
        }
    return assignments


def derive_source_seed(root_seed: int, record: Mapping[str, Any]) -> int:
    """Derive an order-independent non-negative 63-bit seed for one source record."""
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "root_seed must be a non-negative integer",
            field="root_seed",
        )
    dataset, record_id = source_identity(record)
    digest = hashlib.sha256(f"source_seed_v1\0{root_seed}\0{dataset}\0{record_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def make_split_provenance(*, ratios: Mapping[str, float], seed: str | int) -> SplitProvenance:
    """Normalize split settings and derive their immutable partition ID."""
    normalized_ratios = _validate_ratios(ratios)
    seed_text = _normalize_split_seed(seed)
    payload = {
        "algorithm": SPLIT_VERSION,
        "fingerprint_version": FINGERPRINT_VERSION,
        "seed": seed_text,
        "ratios": normalized_ratios,
    }
    return {**payload, "partition_id": f"partition_{stable_digest('source_partition_v1', payload)}"}


def validate_split_provenance(value: Mapping[str, Any]) -> SplitProvenance:
    """Validate that a partition ID is exactly derived from its split settings."""
    expected_fields = {"algorithm", "fingerprint_version", "partition_id", "seed", "ratios"}
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "partition has missing or unknown fields",
            field="partition",
        )
    expected = make_split_provenance(ratios=value["ratios"], seed=value["seed"])
    if any(value[field] != expected[field] for field in ("algorithm", "fingerprint_version", "partition_id")):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "partition ID or version does not match its canonical seed and ratios",
            field="partition",
        )
    return expected


def derive_split_for_dedup_key(dedup_key: str, *, partition: Mapping[str, Any]) -> DataSplit:
    """Recompute the deterministic split for one canonical dedup group."""
    normalized_partition = validate_split_provenance(partition)
    return _split_for_dedup_key(dedup_key, normalized_partition)


def source_identity(record: Mapping[str, Any]) -> SourceIdentity:
    """Return the stable public dataset and source record identity."""
    if not isinstance(record, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "source record must be a mapping",
        )
    dataset = require_nonempty_text(record.get("source_dataset"), field="source_dataset")
    record_id = require_nonempty_text(record.get("source_record_id"), field="source_record_id")
    return dataset, record_id


def source_dedup_key(record: Mapping[str, Any]) -> str:
    """Read or derive a versioned exact dedup key for a normalized source."""
    pipeline_variant = record.get("pipeline_variant")
    if pipeline_variant == "direct_parse":
        return trajectory_dedup_key(record)
    value = require_nonempty_text(record.get("dedup_key"), field="dedup_key")
    if pipeline_variant == "pair_0_1_turn":
        expected_fields = {
            "source_dataset",
            "source_record_id",
            "pipeline_variant",
            "prompt",
            "reference_image",
            "dedup_key",
        }
        if set(record) != expected_fields:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "normalized pair source has missing or unknown fields",
            )
        expected = derive_pair_source_dedup_key(record.get("prompt"), record.get("reference_image"))
    elif pipeline_variant == "prompt_k_turn":
        expected_fields = {"source_dataset", "source_record_id", "pipeline_variant", "prompt", "dedup_key"}
        if set(record) != expected_fields:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "normalized prompt source has missing or unknown fields",
            )
        expected = derive_prompt_source_dedup_key(record.get("prompt"))
    else:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"unsupported pipeline_variant: {pipeline_variant!r}",
            field="pipeline_variant",
        )
    if value != expected:
        raise VisualReflectionDataError(
            RejectionReason.DUPLICATE_CONTENT,
            "source dedup_key does not match its canonical content fingerprint",
            field="dedup_key",
            source_record_id=str(record.get("source_record_id") or "") or None,
        )
    return expected


def validate_source_split_assignment(
    record: Mapping[str, Any],
    assignment: Mapping[str, Any],
    *,
    partition: Mapping[str, Any] | None = None,
) -> SourceSplitAssignment:
    """Bind a normalized source to its precomputed dedup group and split."""
    if not isinstance(assignment, Mapping) or set(assignment) != {
        "source_dataset",
        "source_record_id",
        "dedup_key",
        "partition_id",
        "split",
    }:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "source split assignment has missing or unknown fields",
            field="source_split",
        )
    dataset, record_id = source_identity(record)
    if assignment.get("source_dataset") != dataset or assignment.get("source_record_id") != record_id:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "source split assignment identity does not match the record",
            field="source_split",
            source_record_id=record_id,
        )
    assigned_dedup_key = require_nonempty_text(assignment.get("dedup_key"), field="source_split.dedup_key")
    partition_id = require_nonempty_text(assignment.get("partition_id"), field="source_split.partition_id")
    if not _PARTITION_ID_RE.fullmatch(partition_id):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "source split assignment partition_id is not canonical",
            field="source_split.partition_id",
            source_record_id=record_id,
        )
    can_recompute_dedup = record.get("pipeline_variant") == "direct_parse" or "dedup_key" in record
    if can_recompute_dedup and assigned_dedup_key != source_dedup_key(record):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "source split assignment dedup_key does not match the record",
            field="source_split.dedup_key",
            source_record_id=record_id,
        )
    split = assignment.get("split")
    if split not in {"train", "validation", "test"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "source split assignment has an invalid split",
            field="source_split.split",
            source_record_id=record_id,
        )
    if partition is not None:
        normalized_partition = validate_split_provenance(partition)
        if partition_id != normalized_partition["partition_id"]:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_SPLIT,
                "source split assignment belongs to a different partition",
                field="source_split.partition_id",
                source_record_id=record_id,
            )
        expected_split = _split_for_dedup_key(assigned_dedup_key, normalized_partition)
        if split != expected_split:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_SPLIT,
                "source split assignment does not match its partition thresholds",
                field="source_split.split",
                source_record_id=record_id,
            )
    return {
        "source_dataset": dataset,
        "source_record_id": record_id,
        "dedup_key": assigned_dedup_key,
        "partition_id": partition_id,
        "split": split,  # type: ignore[typeddict-item]
    }


def _split_for_dedup_key(dedup_key: str, partition: SplitProvenance) -> DataSplit:
    normalized_key = require_nonempty_text(dedup_key, field="dedup_key")
    digest = hashlib.sha256(f"{SPLIT_VERSION}\0{partition['seed']}\0{normalized_key}".encode()).digest()
    bucket = int.from_bytes(digest, "big") / 2 ** (8 * len(digest))
    train_boundary = partition["ratios"]["train"]
    validation_boundary = train_boundary + partition["ratios"]["validation"]
    if bucket < train_boundary:
        return "train"
    if bucket < validation_boundary:
        return "validation"
    return "test"


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(ratios, Mapping) or set(ratios) != {"train", "validation", "test"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "ratios must contain exactly train, validation, and test",
            field="ratios",
        )
    normalized: dict[str, float] = {}
    for split in ("train", "validation", "test"):
        value = ratios[split]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_SPLIT,
                f"ratio for {split} must be a finite non-negative number",
                field=f"ratios.{split}",
            )
        normalized[split] = float(value)
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "split ratios must sum to 1.0",
            field="ratios",
        )
    return normalized


def _normalize_split_seed(seed: str | int) -> str:
    if isinstance(seed, bool) or not isinstance(seed, str | int) or (isinstance(seed, int) and seed < 0):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "split seed must be a non-empty string or non-negative integer",
            field="split_seed",
        )
    return require_nonempty_text(str(seed), field="split_seed")


def _duplicate_record(identity: SourceIdentity, survivor: SourceIdentity, dedup_key: str) -> DuplicateRecord:
    return {
        "source_dataset": identity[0],
        "source_record_id": identity[1],
        "retained_source_dataset": survivor[0],
        "retained_source_record_id": survivor[1],
        "dedup_key": dedup_key,
    }
