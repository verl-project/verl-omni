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
"""Deterministic expansion from canonical trajectories to atomic SFT entries."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .contracts import (
    T2I_TARGET_POLICY,
    AtomicSFTExample,
    EditAtom,
    ReflectAtom,
    RejectionReason,
    T2IAtom,
    VisualReflectionDataError,
    require_nonempty_text,
    stable_digest,
    validate_data_split,
    validate_image_ref,
    validate_reflection_step,
    validate_trajectory,
)
from .partition import (
    derive_split_for_dedup_key,
    source_dedup_key,
    source_identity,
    validate_source_split_assignment,
    validate_split_provenance,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PIPELINE_VARIANTS = {"direct_parse", "pair_0_1_turn", "prompt_k_turn"}
_COMMON_ATOM_FIELDS = {
    "atom_id",
    "task",
    "trajectory_id",
    "manifest_id",
    "source_dataset",
    "source_record_id",
    "pipeline_variant",
    "partition_id",
    "source_dedup_key",
    "split",
    "prompt",
}


def plan_atomic_examples(
    trajectory: Mapping[str, Any],
    *,
    source_split: Mapping[str, Any],
    partition: Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
    t2i_target_policy: str = T2I_TARGET_POLICY,
) -> list[AtomicSFTExample]:
    """Emit one terminal T2I, all reflections, then all edit transitions."""
    normalized = validate_trajectory(trajectory)
    split_record = _planner_source_record(normalized, source_record)
    assignment = validate_source_split_assignment(split_record, source_split, partition=partition)
    if t2i_target_policy != T2I_TARGET_POLICY:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"unsupported t2i_target_policy: {t2i_target_policy!r}",
            field="t2i_target_policy",
            source_record_id=normalized["source_record_id"],
        )
    common = {
        "trajectory_id": normalized["trajectory_id"],
        "manifest_id": normalized["manifest_id"],
        "source_dataset": normalized["source_dataset"],
        "source_record_id": normalized["source_record_id"],
        "pipeline_variant": normalized["pipeline_variant"],
        "partition_id": assignment["partition_id"],
        "source_dedup_key": assignment["dedup_key"],
        "split": assignment["split"],
        "prompt": normalized["prompt"],
    }
    final_index = len(normalized["images"]) - 1
    t2i_payload = {
        **common,
        "task": "t2i",
        "target_state_index": final_index,
        "target_image": dict(normalized["images"][-1]),
        "t2i_target_policy": "terminal_state_v1",
    }
    t2i: T2IAtom = {"atom_id": _atom_id(t2i_payload), **t2i_payload}  # type: ignore[typeddict-item]
    reflections: list[ReflectAtom] = []
    for state_index, (image, step) in enumerate(zip(normalized["images"], normalized["steps"], strict=True)):
        payload = {
            **common,
            "task": "reflect",
            "state_index": state_index,
            "current_image": dict(image),
            "response": dict(step),
        }
        reflections.append({"atom_id": _atom_id(payload), **payload})  # type: ignore[arg-type]
    edits: list[EditAtom] = []
    for state_index, step in enumerate(normalized["steps"][:-1]):
        payload = {
            **common,
            "task": "edit",
            "state_index": state_index,
            "current_image": dict(normalized["images"][state_index]),
            "edit": step["edit"],
            "target_image": dict(normalized["images"][state_index + 1]),
        }
        edits.append({"atom_id": _atom_id(payload), **payload})  # type: ignore[arg-type]
    return [t2i, *reflections, *edits]


def validate_atomic_example(
    value: Mapping[str, Any],
    *,
    partition: Mapping[str, Any],
    source_record: Mapping[str, Any],
) -> AtomicSFTExample:
    """Validate one persisted atom, including its partition and content-derived ID."""
    if not isinstance(value, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "atomic example must be a mapping",
            field="atom",
        )
    task = value.get("task")
    if task == "t2i":
        expected_fields = _COMMON_ATOM_FIELDS | {"target_state_index", "target_image", "t2i_target_policy"}
    elif task == "reflect":
        expected_fields = _COMMON_ATOM_FIELDS | {"state_index", "current_image", "response"}
    elif task == "edit":
        expected_fields = _COMMON_ATOM_FIELDS | {"state_index", "current_image", "edit", "target_image"}
    else:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"unsupported atomic task: {task!r}",
            field="atom.task",
        )
    if set(value) != expected_fields:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "atomic example has missing or unknown fields",
            field="atom",
        )
    normalized_partition = validate_split_provenance(partition)
    partition_id = require_nonempty_text(value.get("partition_id"), field="atom.partition_id")
    if partition_id != normalized_partition["partition_id"]:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "atomic example belongs to a different partition",
            field="atom.partition_id",
        )
    dedup_key = require_nonempty_text(value.get("source_dedup_key"), field="atom.source_dedup_key")
    if not _DIGEST_RE.fullmatch(dedup_key):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "atom.source_dedup_key must be a lowercase SHA-256 digest",
            field="atom.source_dedup_key",
        )
    source_dataset, source_record_id = source_identity(source_record)
    if (
        source_dataset != value.get("source_dataset")
        or source_record_id != value.get("source_record_id")
        or source_record.get("pipeline_variant") != value.get("pipeline_variant")
        or source_record.get("prompt") != value.get("prompt")
        or source_dedup_key(source_record) != dedup_key
    ):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "atomic example source provenance does not match its normalized source record",
            field="source_record",
            source_record_id=source_record_id,
        )
    split = validate_data_split(value.get("split"))
    if split != derive_split_for_dedup_key(dedup_key, partition=normalized_partition):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "atomic example split does not match its partition thresholds",
            field="atom.split",
        )
    pipeline_variant = value.get("pipeline_variant")
    if pipeline_variant not in _PIPELINE_VARIANTS:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "atomic example has an invalid pipeline_variant",
            field="atom.pipeline_variant",
        )
    payload: dict[str, Any] = {
        "trajectory_id": require_nonempty_text(value.get("trajectory_id"), field="atom.trajectory_id"),
        "manifest_id": require_nonempty_text(value.get("manifest_id"), field="atom.manifest_id"),
        "source_dataset": require_nonempty_text(value.get("source_dataset"), field="atom.source_dataset"),
        "source_record_id": require_nonempty_text(value.get("source_record_id"), field="atom.source_record_id"),
        "pipeline_variant": pipeline_variant,
        "partition_id": partition_id,
        "source_dedup_key": dedup_key,
        "split": split,
        "prompt": require_nonempty_text(value.get("prompt"), field="atom.prompt", reason=RejectionReason.EMPTY_PROMPT),
        "task": task,
    }
    if task == "t2i":
        target_index = _require_state_index(value.get("target_state_index"), field="atom.target_state_index")
        if value.get("t2i_target_policy") != T2I_TARGET_POLICY:
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                "atomic T2I target policy is unsupported",
                field="atom.t2i_target_policy",
            )
        payload.update(
            {
                "target_state_index": target_index,
                "target_image": validate_image_ref(value.get("target_image")),
                "t2i_target_policy": T2I_TARGET_POLICY,
            }
        )
    elif task == "reflect":
        payload.update(
            {
                "state_index": _require_state_index(value.get("state_index"), field="atom.state_index"),
                "current_image": validate_image_ref(value.get("current_image")),
                "response": validate_reflection_step(value.get("response")),
            }
        )
    else:
        payload.update(
            {
                "state_index": _require_state_index(value.get("state_index"), field="atom.state_index"),
                "current_image": validate_image_ref(value.get("current_image")),
                "edit": require_nonempty_text(
                    value.get("edit"), field="atom.edit", reason=RejectionReason.EMPTY_CONTINUE_EDIT
                ),
                "target_image": validate_image_ref(value.get("target_image")),
            }
        )
    atom_id = require_nonempty_text(value.get("atom_id"), field="atom.atom_id")
    if atom_id != _atom_id(payload):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "atom_id does not match the canonical atomic example content",
            field="atom.atom_id",
        )
    return {"atom_id": atom_id, **payload}  # type: ignore[return-value]


def _require_state_index(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a non-negative integer",
            field=field,
        )
    return value


def _planner_source_record(
    trajectory: Mapping[str, Any],
    source_record: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if trajectory["pipeline_variant"] == "direct_parse":
        if source_record is not None and source_record is not trajectory:
            normalized_identity = source_identity(source_record)
            if normalized_identity != source_identity(trajectory) or source_dedup_key(
                source_record
            ) != source_dedup_key(trajectory):
                raise VisualReflectionDataError(
                    RejectionReason.INVALID_SPLIT,
                    "direct trajectory source_record does not match the trajectory",
                    field="source_record",
                    source_record_id=trajectory["source_record_id"],
                )
        return trajectory
    if source_record is None:
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "synthesized trajectories require their normalized source record before atomic expansion",
            field="source_record",
            source_record_id=trajectory["source_record_id"],
        )
    dataset, record_id = source_identity(source_record)
    if (
        dataset != trajectory["source_dataset"]
        or record_id != trajectory["source_record_id"]
        or source_record.get("pipeline_variant") != trajectory["pipeline_variant"]
        or source_record.get("prompt") != trajectory["prompt"]
    ):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_SPLIT,
            "normalized synthesis source does not match the trajectory",
            field="source_record",
            source_record_id=trajectory["source_record_id"],
        )
    source_dedup_key(source_record)
    if trajectory["pipeline_variant"] == "pair_0_1_turn":
        if len(trajectory["images"]) > 2:
            raise VisualReflectionDataError(
                RejectionReason.TURN_LIMIT_EXHAUSTED,
                "Pair-source trajectories may contain at most one edit",
                source_record_id=trajectory["source_record_id"],
            )
        if (
            len(trajectory["images"]) == 2
            and trajectory["images"][-1]["sha256"] != source_record["reference_image"]["sha256"]
        ):
            raise VisualReflectionDataError(
                RejectionReason.INCOMPATIBLE_EDIT_TARGET,
                "One-edit pair-source trajectory must terminate at the reference image",
                source_record_id=trajectory["source_record_id"],
            )
    return source_record


def _atom_id(payload: Mapping[str, Any]) -> str:
    return f"atom_{stable_digest('atomic_sft_v1', payload)}"
