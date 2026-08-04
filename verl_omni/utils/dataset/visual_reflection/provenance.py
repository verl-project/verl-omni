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
"""Reproducible provenance manifests and fail-closed rejection ledgers."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    CONVERTER_VERSION,
    FILTER_VERSION,
    IMAGE_HASH_ALGORITHM,
    SCHEMA_VERSION,
    T2I_TARGET_POLICY,
    DataManifest,
    ModelProvenance,
    RejectionReason,
    RejectionRecord,
    SourceDatasetProvenance,
    SplitProvenance,
    VisualReflectionDataError,
    canonical_json,
    require_nonempty_text,
    stable_digest,
    validate_trajectory,
)
from .images import validate_image_asset
from .partition import (
    derive_source_seed,
    source_identity,
    validate_source_split_assignment,
    validate_split_provenance,
)
from .sources import validate_synthesis_result

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_FLOATING_REVISIONS = {"head", "latest", "main", "master"}
_PIPELINE_VARIANTS = {"direct_parse", "pair_0_1_turn", "prompt_k_turn"}
_SOURCE_PREPARATION_STAGES = {"source_adapter", "source_dedup", "source_filter"}
_MANIFEST_DIGEST_DOMAIN = "visual_reflection_manifest_v1"


def _rejection_sort_key(entry: RejectionRecord) -> tuple[Any, ...]:
    return (
        entry["source_dataset"],
        entry["source_record_id"],
        entry["pipeline_variant"],
        entry["lineage_kind"],
        entry["stage"],
        entry["attempt"],
        entry["reason"],
    )


class RejectionLedger:
    """Final source rejections with one deterministic JSONL row per source.

    Retryable intermediate attempts are synthesis events, not release rejections. A
    row is appended only after a source is finally rejected; ``attempt`` records
    which bounded attempt produced that final outcome.
    """

    def __init__(self, manifest_id: str) -> None:
        self.manifest_id = require_nonempty_text(manifest_id, field="manifest_id")
        self._entries: list[RejectionRecord] = []
        self._source_keys: set[tuple[str, str]] = set()

    @property
    def entries(self) -> list[RejectionRecord]:
        """Return defensive copies sorted independently of completion order."""
        return [json.loads(canonical_json(entry)) for entry in sorted(self._entries, key=_rejection_sort_key)]

    def append(
        self,
        *,
        source_dataset: str,
        source_record_id: str,
        pipeline_variant: str,
        stage: str,
        reason: RejectionReason | str,
        message: str,
        attempt: int = 0,
        details: Mapping[str, Any] | None = None,
    ) -> RejectionRecord:
        """Append one final direct-parse rejection and return a defensive copy."""
        dataset = require_nonempty_text(source_dataset, field="source_dataset")
        record_id = require_nonempty_text(source_record_id, field="source_record_id")
        if pipeline_variant != "direct_parse":
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "synthesis rejections must use append_synthesis_result",
                field="pipeline_variant",
            )
        try:
            normalized_reason = reason if isinstance(reason, RejectionReason) else RejectionReason(reason)
        except ValueError as exc:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"unknown rejection reason: {reason!r}",
                field="reason",
            ) from exc
        if attempt != 0:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "direct-parse rejections must use attempt 0",
                field="attempt",
            )
        if stage != "direct_parse":
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "direct-parse rejection stage must be direct_parse",
                field="stage",
            )
        return self._append_entry(
            dataset=dataset,
            record_id=record_id,
            pipeline_variant="direct_parse",
            lineage_kind="direct_parse",
            stage=stage,
            reason=normalized_reason,
            message=message,
            attempt=0,
            request_id=None,
            source_split=None,
            synthesis_result=None,
            details=details,
        )

    def append_source_rejection(
        self,
        *,
        source_dataset: str,
        source_record_id: str,
        pipeline_variant: str,
        stage: str,
        reason: RejectionReason | str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> RejectionRecord:
        """Append a pre-request adapter, dedup, or filter rejection for any route."""
        dataset = require_nonempty_text(source_dataset, field="source_dataset")
        record_id = require_nonempty_text(source_record_id, field="source_record_id")
        if pipeline_variant not in _PIPELINE_VARIANTS:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"invalid pipeline_variant: {pipeline_variant!r}",
                field="pipeline_variant",
            )
        if stage not in _SOURCE_PREPARATION_STAGES:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"source preparation stage must be one of {sorted(_SOURCE_PREPARATION_STAGES)}",
                field="stage",
            )
        try:
            normalized_reason = reason if isinstance(reason, RejectionReason) else RejectionReason(reason)
        except ValueError as exc:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"unknown rejection reason: {reason!r}",
                field="reason",
            ) from exc
        return self._append_entry(
            dataset=dataset,
            record_id=record_id,
            pipeline_variant=pipeline_variant,
            lineage_kind="source_preparation",
            stage=stage,
            reason=normalized_reason,
            message=message,
            attempt=0,
            request_id=None,
            source_split=None,
            synthesis_result=None,
            details=details,
        )

    def append_synthesis_result(
        self,
        result: Mapping[str, Any],
        *,
        stage: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> RejectionRecord:
        """Append one validated final synthesis rejection with full request lineage."""
        normalized = validate_synthesis_result(result)
        if normalized["status"] != "rejected" or normalized["trajectory"] is not None:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "rejection ledger requires a final rejected synthesis result",
                field="synthesis_result.status",
            )
        if normalized["request"]["manifest_id"] != self.manifest_id:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "synthesis rejection manifest_id does not match the ledger",
                field="synthesis_result.request.manifest_id",
            )
        try:
            reason = RejectionReason(normalized["rejection_reason"])
        except ValueError as exc:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "final synthesis rejection_reason must be a stable rejection code",
                field="synthesis_result.rejection_reason",
            ) from exc
        request = normalized["request"]
        return self._append_entry(
            dataset=request["source_dataset"],
            record_id=request["source_record_id"],
            pipeline_variant=request["pipeline_variant"],
            lineage_kind="synthesis",
            stage=stage,
            reason=reason,
            message=message,
            attempt=normalized["attempt"],
            request_id=normalized["request_id"],
            source_split=normalized["source_split"],
            synthesis_result=normalized,
            details=details,
        )

    def _append_entry(
        self,
        *,
        dataset: str,
        record_id: str,
        pipeline_variant: str,
        lineage_kind: str,
        stage: str,
        reason: RejectionReason,
        message: str,
        attempt: int,
        request_id: str | None,
        source_split: Mapping[str, Any] | None,
        synthesis_result: Mapping[str, Any] | None,
        details: Mapping[str, Any] | None,
    ) -> RejectionRecord:
        normalized_stage = require_nonempty_text(stage, field="stage")
        normalized_message = require_nonempty_text(message, field="message")
        source_key = (dataset, record_id)
        if source_key in self._source_keys:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "rejection ledger already contains a final row for this source",
                source_record_id=record_id,
            )
        normalized_details = json.loads(canonical_json(dict(details or {})))
        entry: RejectionRecord = {
            "manifest_id": self.manifest_id,
            "source_dataset": dataset,
            "source_record_id": record_id,
            "pipeline_variant": pipeline_variant,  # type: ignore[typeddict-item]
            "lineage_kind": lineage_kind,  # type: ignore[typeddict-item]
            "stage": normalized_stage,
            "reason": reason.value,
            "message": normalized_message,
            "attempt": attempt,
            "request_id": request_id,
            "source_split": None if source_split is None else json.loads(canonical_json(source_split)),
            "synthesis_result": None if synthesis_result is None else json.loads(canonical_json(synthesis_result)),
            "details": normalized_details,
        }
        stored_entry = json.loads(canonical_json(entry))
        self._entries.append(stored_entry)
        self._source_keys.add(source_key)
        return json.loads(canonical_json(stored_entry))

    def append_error(
        self,
        error: VisualReflectionDataError,
        *,
        source_dataset: str,
        source_record_id: str,
    ) -> RejectionRecord:
        """Append a classified direct-parse error without swallowing unknown bugs."""
        details = dict(error.details)
        if error.field is not None:
            details["field"] = error.field
        return self.append(
            source_dataset=source_dataset,
            source_record_id=source_record_id,
            pipeline_variant="direct_parse",
            stage="direct_parse",
            reason=error.reason,
            message=str(error),
            attempt=0,
            details=details,
        )

    def append_source_error(
        self,
        error: VisualReflectionDataError,
        *,
        source_dataset: str,
        source_record_id: str,
        pipeline_variant: str,
        stage: str,
    ) -> RejectionRecord:
        """Append a classified pre-request adapter, dedup, or filter error."""
        details = dict(error.details)
        if error.field is not None:
            details["field"] = error.field
        return self.append_source_rejection(
            source_dataset=source_dataset,
            source_record_id=source_record_id,
            pipeline_variant=pipeline_variant,
            stage=stage,
            reason=error.reason,
            message=str(error),
            details=details,
        )

    def counts_by_reason(self) -> dict[str, int]:
        """Return stable rejection counts keyed by reason code."""
        counts = Counter(entry["reason"] for entry in self._entries)
        return dict(sorted(counts.items()))

    def to_jsonl(self) -> str:
        """Serialize rows in stable source order with a trailing newline when non-empty."""
        if not self._entries:
            return ""
        return "\n".join(canonical_json(entry) for entry in self.entries) + "\n"

    def write_jsonl(self, path: str | Path) -> None:
        """Write the deterministic rejection ledger to disk."""
        Path(path).write_text(self.to_jsonl(), encoding="utf-8")


def derive_manifest_id(
    *,
    sources: Sequence[Mapping[str, Any]],
    code_revision: str,
    models: Mapping[str, Mapping[str, Any] | None],
    prompt_template_hashes: Mapping[str, str],
    seeds: Mapping[str, int],
    converter_config: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> str:
    """Derive a manifest ID from immutable recipe inputs, excluding result counts."""
    recipe = _normalized_recipe(
        sources=sources,
        code_revision=code_revision,
        models=models,
        prompt_template_hashes=prompt_template_hashes,
        seeds=seeds,
        converter_config=converter_config,
        partition=partition,
    )
    return f"manifest_{stable_digest(_MANIFEST_DIGEST_DOMAIN, recipe)}"


def build_data_manifest(
    *,
    sources: Sequence[Mapping[str, Any]],
    code_revision: str,
    models: Mapping[str, Mapping[str, Any] | None],
    prompt_template_hashes: Mapping[str, str],
    seeds: Mapping[str, int],
    converter_config: Mapping[str, Any],
    partition: Mapping[str, Any],
    accepted_trajectories: Sequence[Mapping[str, Any]],
    accepted_source_splits: Sequence[Mapping[str, Any]],
    accepted_synthesis_results: Sequence[Mapping[str, Any]],
    image_asset_roots: Mapping[str, str | Path],
    rejection_ledger: RejectionLedger,
) -> DataManifest:
    """Finalize a release manifest after auditing splits, assets, and outcomes."""
    recipe = _normalized_recipe(
        sources=sources,
        code_revision=code_revision,
        models=models,
        prompt_template_hashes=prompt_template_hashes,
        seeds=seeds,
        converter_config=converter_config,
        partition=partition,
    )
    manifest_id = f"manifest_{stable_digest(_MANIFEST_DIGEST_DOMAIN, recipe)}"
    if rejection_ledger.manifest_id != manifest_id:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "rejection ledger manifest_id does not match recipe-derived manifest_id",
            field="manifest_id",
        )
    if not isinstance(image_asset_roots, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "image_asset_roots must map source datasets to local asset roots",
            field="image_asset_roots",
        )
    source_routes = {(source["dataset_id"], source["pipeline_variant"]) for source in recipe["sources"]}
    direct_by_identity: dict[tuple[str, str], Any] = {}
    for raw_trajectory in accepted_trajectories:
        trajectory = validate_trajectory(raw_trajectory)
        if trajectory["pipeline_variant"] != "direct_parse":
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "synthesized trajectories must enter through accepted_synthesis_results",
                field="accepted_trajectories.pipeline_variant",
                source_record_id=trajectory["source_record_id"],
            )
        source_key = source_identity(trajectory)
        if source_key in direct_by_identity:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "a source record produced more than one accepted trajectory",
                source_record_id=trajectory["source_record_id"],
            )
        direct_by_identity[source_key] = trajectory

    assignments_by_identity: dict[tuple[str, str], Mapping[str, Any]] = {}
    for raw_assignment in accepted_source_splits:
        assignment_identity = source_identity(raw_assignment)
        if assignment_identity in assignments_by_identity:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "accepted source has more than one split assignment",
                field="accepted_source_splits",
                source_record_id=assignment_identity[1],
            )
        assignments_by_identity[assignment_identity] = raw_assignment
    direct_source_keys = set(direct_by_identity)
    assignment_source_keys = set(assignments_by_identity)
    if direct_source_keys != assignment_source_keys:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "direct trajectories and source split assignments must have identical source identities",
            field="accepted_source_splits",
            details={
                "missing": sorted(direct_source_keys - assignment_source_keys),
                "extra": sorted(assignment_source_keys - direct_source_keys),
            },
        )

    accepted_records: list[tuple[Any, Mapping[str, Any]]] = []
    supplemental_images_by_identity: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for source_key, trajectory in direct_by_identity.items():
        assignment = validate_source_split_assignment(
            trajectory,
            assignments_by_identity[source_key],
            partition=recipe["partition"],
        )
        accepted_records.append((trajectory, assignment))

    for raw_result in accepted_synthesis_results:
        result = validate_synthesis_result(raw_result)
        if result["status"] != "accepted" or result["trajectory"] is None:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "accepted_synthesis_results may contain only accepted results",
                field="accepted_synthesis_results.status",
            )
        _validate_synthesis_recipe_lineage(result["request"], recipe=recipe, manifest_id=manifest_id)
        accepted_records.append((result["trajectory"], result["source_split"]))
        if result["request"]["pipeline_variant"] == "pair_0_1_turn":
            request_identity = source_identity(result["request"])
            supplemental_images_by_identity[request_identity] = [result["request"]["reference_image"]]

    accepted_by_identity: dict[tuple[str, str], Any] = {}
    accepted_by_split = {"train": 0, "validation": 0, "test": 0}
    dedup_owners: dict[str, tuple[str, str]] = {}
    normalized_hashes: set[str] = set()
    for trajectory, raw_assignment in accepted_records:
        source_key = source_identity(trajectory)
        if source_key in accepted_by_identity:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "a source record produced more than one accepted trajectory",
                source_record_id=source_key[1],
            )
        if trajectory["manifest_id"] != manifest_id:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "accepted trajectory manifest_id does not match the release manifest",
                field="trajectory.manifest_id",
                source_record_id=trajectory["source_record_id"],
            )
        if (trajectory["source_dataset"], trajectory["pipeline_variant"]) not in source_routes:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "accepted trajectory source route is absent from manifest sources",
                source_record_id=trajectory["source_record_id"],
            )
        assignment = validate_source_split_assignment(
            trajectory,
            raw_assignment,
            partition=recipe["partition"],
        )
        previous_owner = dedup_owners.get(assignment["dedup_key"])
        if previous_owner is not None:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_CONTENT,
                f"accepted sources {previous_owner!r} and {source_key!r} share one dedup group",
                field="accepted_source_splits.dedup_key",
                source_record_id=source_key[1],
            )
        dedup_owners[assignment["dedup_key"]] = source_key
        accepted_by_split[assignment["split"]] += 1
        asset_root = image_asset_roots.get(trajectory["source_dataset"])
        if asset_root is None:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "accepted trajectory source has no configured image asset root",
                field=f"image_asset_roots.{trajectory['source_dataset']}",
                source_record_id=trajectory["source_record_id"],
            )
        audited_images = [*trajectory["images"], *supplemental_images_by_identity.get(source_key, [])]
        for image in audited_images:
            audited = validate_image_asset(image, asset_root=asset_root)
            if audited != image:
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_PROVENANCE,
                    "accepted trajectory contains a non-canonical image URI",
                    field="trajectory.images",
                    source_record_id=trajectory["source_record_id"],
                )
            normalized_hashes.add(audited["sha256"])
        accepted_by_identity[source_key] = trajectory

    accepted_source_keys = set(accepted_by_identity)

    ledger_entries = rejection_ledger.entries
    for entry in ledger_entries:
        if (entry["source_dataset"], entry["pipeline_variant"]) not in source_routes:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "rejection ledger source route is absent from manifest sources",
                source_record_id=entry["source_record_id"],
            )
        lineage_kind = entry["lineage_kind"]
        if lineage_kind == "direct_parse":
            if (
                entry["pipeline_variant"] != "direct_parse"
                or entry["stage"] != "direct_parse"
                or entry["attempt"] != 0
                or entry["request_id"] is not None
                or entry["source_split"] is not None
                or entry["synthesis_result"] is not None
            ):
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_PROVENANCE,
                    "direct-parse rejection contains synthesis lineage",
                    source_record_id=entry["source_record_id"],
                )
            continue
        if lineage_kind == "source_preparation":
            if (
                entry["stage"] not in _SOURCE_PREPARATION_STAGES
                or entry["attempt"] != 0
                or entry["request_id"] is not None
                or entry["source_split"] is not None
                or entry["synthesis_result"] is not None
            ):
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_PROVENANCE,
                    "source preparation rejection contains request-stage lineage",
                    source_record_id=entry["source_record_id"],
                )
            continue
        if lineage_kind != "synthesis" or entry["synthesis_result"] is None:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "synthesis rejection is missing its validated result lineage",
                source_record_id=entry["source_record_id"],
            )
        result = validate_synthesis_result(entry["synthesis_result"])
        if result["status"] != "rejected" or result["trajectory"] is not None:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "rejection ledger synthesis result is not final rejected",
                source_record_id=entry["source_record_id"],
            )
        _validate_synthesis_recipe_lineage(result["request"], recipe=recipe, manifest_id=manifest_id)
        request = result["request"]
        if (
            entry["request_id"] != result["request_id"]
            or entry["source_split"] != result["source_split"]
            or entry["attempt"] != result["attempt"]
            or entry["reason"] != result["rejection_reason"]
            or entry["source_dataset"] != request["source_dataset"]
            or entry["source_record_id"] != request["source_record_id"]
            or entry["pipeline_variant"] != request["pipeline_variant"]
        ):
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "rejection ledger row does not match its synthesis result",
                source_record_id=entry["source_record_id"],
            )
    rejected_source_keys = {(entry["source_dataset"], entry["source_record_id"]) for entry in ledger_entries}
    overlap = accepted_source_keys & rejected_source_keys
    if overlap:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"source records cannot be both accepted and rejected: {sorted(overlap)}",
        )
    counts_by_reason = rejection_ledger.counts_by_reason()
    rejected = len(ledger_entries)
    if rejected != sum(counts_by_reason.values()):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "rejection counts do not match ledger rows",
        )
    manifest: DataManifest = {
        "manifest_id": manifest_id,
        **recipe,
        "image_hashes": sorted(normalized_hashes),
        "counts": {
            "accepted": len(accepted_source_keys),
            "accepted_by_split": accepted_by_split,
            "rejected": rejected,
            "rejected_by_reason": counts_by_reason,
        },
    }
    canonical_json(manifest)
    return manifest


def _validate_synthesis_recipe_lineage(
    request: Mapping[str, Any],
    *,
    recipe: Mapping[str, Any],
    manifest_id: str,
) -> None:
    record_id = str(request["source_record_id"])
    if request["manifest_id"] != manifest_id:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "synthesis request manifest_id does not match the release recipe",
            field="synthesis_result.request.manifest_id",
            source_record_id=record_id,
        )
    if canonical_json(request["partition"]) != canonical_json(recipe["partition"]):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "synthesis request partition does not match the release recipe",
            field="synthesis_result.request.partition",
            source_record_id=record_id,
        )
    root_seed = recipe["seeds"].get("root")
    if root_seed is None or request["seed"] != derive_source_seed(root_seed, request):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "synthesis request seed is not the recipe-derived per-source seed",
            field="synthesis_result.request.seed",
            source_record_id=record_id,
        )
    converter_config = recipe["converter_config"]
    if request["pipeline_variant"] == "pair_0_1_turn":
        if request["max_attempts"] != converter_config["pair_max_attempts"]:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                "pair synthesis max_attempts does not match converter_config",
                field="synthesis_result.request.max_attempts",
                source_record_id=record_id,
            )
    elif (
        request["max_attempts"] != converter_config["prompt_max_attempts"]
        or request["max_edit_turns"] != converter_config["prompt_max_edit_turns"]
    ):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "prompt synthesis bounds do not match converter_config",
            field="synthesis_result.request",
            source_record_id=record_id,
        )


def _normalized_recipe(
    *,
    sources: Sequence[Mapping[str, Any]],
    code_revision: str,
    models: Mapping[str, Mapping[str, Any] | None],
    prompt_template_hashes: Mapping[str, str],
    seeds: Mapping[str, int],
    converter_config: Mapping[str, Any],
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_sources = [_validate_source_provenance(source) for source in sources]
    normalized_sources.sort(key=canonical_json)
    if not normalized_sources:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "manifest requires at least one pinned source",
            field="sources",
        )
    source_keys: set[tuple[str, str, str]] = set()
    for source in normalized_sources:
        source_key = (source["dataset_id"], source["config"], source["split"])
        if source_key in source_keys:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                f"manifest contains duplicate source provenance for {source_key}",
                field="sources",
            )
        source_keys.add(source_key)
    normalized_models: dict[str, ModelProvenance | None] = {}
    for role, model in sorted(models.items()):
        normalized_role = require_nonempty_text(role, field="models.role")
        normalized_models[normalized_role] = None if model is None else _validate_model_provenance(model)
    normalized_templates = {
        require_nonempty_text(name, field="prompt_template_hashes.name"): _validate_sha256(
            digest, field=f"prompt_template_hashes.{name}"
        )
        for name, digest in sorted(prompt_template_hashes.items())
    }
    normalized_seeds: dict[str, int] = {}
    for name, seed in sorted(seeds.items()):
        normalized_name = require_nonempty_text(name, field="seeds.name")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_PROVENANCE,
                f"seed {normalized_name} must be a non-negative integer",
                field=f"seeds.{normalized_name}",
            )
        normalized_seeds[normalized_name] = seed
    if not isinstance(converter_config, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "converter_config must be a mapping",
            field="converter_config",
        )
    variants = {source["pipeline_variant"] for source in normalized_sources}
    normalized_converter_config = _validate_converter_config(
        json.loads(canonical_json(dict(converter_config))), variants
    )
    required_model_roles: set[str] = set()
    required_template_roles: set[str] = set()
    if "pair_0_1_turn" in variants:
        required_model_roles.update({"generator", "reflector", "verifier"})
        required_template_roles.update({"reflector", "verifier"})
    if "prompt_k_turn" in variants:
        required_model_roles.update({"generator", "reflector", "verifier", "editor"})
        required_template_roles.update({"reflector", "verifier", "editor"})
    missing_models = sorted(role for role in required_model_roles if normalized_models.get(role) is None)
    if missing_models:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"synthesis routes require pinned model roles: {missing_models}",
            field="models",
        )
    missing_templates = sorted(required_template_roles - set(normalized_templates))
    if missing_templates:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"synthesis routes require prompt template hashes: {missing_templates}",
            field="prompt_template_hashes",
        )
    if required_model_roles and "root" not in normalized_seeds:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "synthesis routes require a root seed",
            field="seeds.root",
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "filter_version": FILTER_VERSION,
        "code_revision": _validate_revision(code_revision, field="code_revision"),
        "t2i_target_policy": T2I_TARGET_POLICY,
        "image_hash_algorithm": IMAGE_HASH_ALGORITHM,
        "sources": normalized_sources,
        "models": normalized_models,
        "prompt_template_hashes": normalized_templates,
        "seeds": normalized_seeds,
        "converter_config": normalized_converter_config,
        "partition": _validate_partition_provenance(partition),
    }


def _validate_source_provenance(value: Mapping[str, Any]) -> SourceDatasetProvenance:
    if not isinstance(value, Mapping):
        raise VisualReflectionDataError(RejectionReason.INVALID_PROVENANCE, "source provenance must be a mapping")
    if set(value) != {"dataset_id", "config", "split", "revision", "license", "pipeline_variant"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "source provenance has missing or unknown fields",
            field="sources",
        )
    pipeline_variant = value.get("pipeline_variant")
    if pipeline_variant not in _PIPELINE_VARIANTS:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"invalid source pipeline_variant: {pipeline_variant!r}",
            field="sources.pipeline_variant",
        )
    dataset_id = require_nonempty_text(value.get("dataset_id"), field="sources.dataset_id")
    config = require_nonempty_text(value.get("config"), field="sources.config")
    split = require_nonempty_text(value.get("split"), field="sources.split")
    license_declaration = require_nonempty_text(value.get("license"), field="sources.license")
    return {
        "dataset_id": dataset_id,
        "config": config,
        "split": split,
        "revision": _validate_revision(value.get("revision"), field="sources.revision"),
        "license": license_declaration,
        "pipeline_variant": pipeline_variant,  # type: ignore[typeddict-item]
    }


def _validate_model_provenance(value: Mapping[str, Any]) -> ModelProvenance:
    if not isinstance(value, Mapping):
        raise VisualReflectionDataError(RejectionReason.INVALID_PROVENANCE, "model provenance must be a mapping")
    if set(value) != {"model_id", "revision"}:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "model provenance has missing or unknown fields",
            field="models",
        )
    return {
        "model_id": require_nonempty_text(value.get("model_id"), field="models.model_id"),
        "revision": _validate_revision(value.get("revision"), field="models.revision"),
    }


def _validate_partition_provenance(value: Mapping[str, Any]) -> SplitProvenance:
    try:
        return validate_split_provenance(value)
    except VisualReflectionDataError as exc:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            "partition settings are invalid",
            field="partition",
        ) from exc


def _validate_revision(value: Any, *, field: str) -> str:
    revision = require_nonempty_text(value, field=field)
    if revision.casefold() in _FLOATING_REVISIONS or not _COMMIT_SHA_RE.fullmatch(revision):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"{field} must be an immutable 40-character lowercase commit SHA, not {revision!r}",
            field=field,
        )
    return revision


def _validate_converter_config(value: dict[str, Any], variants: set[str]) -> dict[str, Any]:
    if "pair_0_1_turn" in variants:
        _require_positive_config_int(value, "pair_max_attempts")
    if "prompt_k_turn" in variants:
        _require_positive_config_int(value, "prompt_max_attempts")
        _require_positive_config_int(value, "prompt_max_edit_turns")
    return value


def _require_positive_config_int(value: Mapping[str, Any], field: str) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"converter_config.{field} must be a positive integer",
            field=f"converter_config.{field}",
        )
    return item


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_PROVENANCE,
            f"{field} must be a lowercase SHA-256 digest",
            field=field,
        )
    return value
