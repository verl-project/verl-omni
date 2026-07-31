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
"""Fail-closed direct parser for public UniCoT self-reflection rows."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ImageRef,
    ReflectionStep,
    RejectionReason,
    VisualReflectionDataError,
    VisualReflectionTrajectory,
    derive_trajectory_id,
    require_nonempty_text,
    validate_trajectory,
)
from .images import ImageResolver

UNICOT_DATASET_ID = "Fr0zencr4nE/UniCoT-Self-Reflection-6K"
UNICOT_REFLECTION_CLEANER_VERSION = "summary_strip_eval_nfkc_ws_char_limit_v1"
UNICOT_TERMINAL_EDIT_POLICY_VERSION = "exact_sentinel_v1"
DEFAULT_TERMINAL_EDIT_SENTINEL = "Everything is good. No editing needed."


@dataclass(frozen=True)
class UniCoTParserConfig:
    """Versioned tokenizer-independent controls for UniCoT direct parsing."""

    fallback_max_chars: int = 2048

    def __post_init__(self) -> None:
        if isinstance(self.fallback_max_chars, bool) or not isinstance(self.fallback_max_chars, int):
            raise TypeError("fallback_max_chars must be an integer")
        if self.fallback_max_chars <= 0:
            raise ValueError("fallback_max_chars must be positive")


def unicot_converter_config(config: UniCoTParserConfig | None = None) -> dict[str, Any]:
    """Return the exact UniCoT filter settings that must be stored in a manifest."""
    parser_config = config or UniCoTParserConfig()
    return {
        "unicot_fallback_max_chars": parser_config.fallback_max_chars,
        "unicot_reflection_cleaner": UNICOT_REFLECTION_CLEANER_VERSION,
        "unicot_terminal_edit_policy": UNICOT_TERMINAL_EDIT_POLICY_VERSION,
        "unicot_terminal_edit_sentinels": [_normalize_terminal_edit(DEFAULT_TERMINAL_EDIT_SENTINEL)],
    }


def parse_unicot_record(
    record: Mapping[str, Any],
    *,
    manifest_id: str,
    image_resolver: ImageResolver,
    source_record_id: str | None = None,
    config: UniCoTParserConfig | None = None,
) -> VisualReflectionTrajectory:
    """Parse one source row into a validated canonical trajectory.

    Actions are derived only from the ``output_image`` transition structure.
    Source terminal edit text must match the versioned exact no-edit sentinel
    policy, after which it is canonicalized to ``Edit:``.
    """
    if not isinstance(record, Mapping):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            "UniCoT record must be a mapping",
        )
    parser_config = config or UniCoTParserConfig()
    dataset_record_id = require_nonempty_text(
        _required_field(record, "data_id", source_record_id or "<unknown>"),
        field="source_record_id",
        source_record_id=source_record_id,
    )
    if source_record_id is not None:
        override_record_id = require_nonempty_text(source_record_id, field="source_record_id")
        if override_record_id != dataset_record_id:
            raise VisualReflectionDataError(
                RejectionReason.DUPLICATE_SOURCE_RECORD,
                "source_record_id override does not match UniCoT data_id",
                field="source_record_id",
                source_record_id=override_record_id,
            )
    record_id = dataset_record_id
    manifest = require_nonempty_text(manifest_id, field="manifest_id", source_record_id=record_id)
    prompt = require_nonempty_text(
        _required_field(record, "prompt", record_id),
        field="prompt",
        reason=RejectionReason.EMPTY_PROMPT,
        source_record_id=record_id,
    )

    inputs = _required_list(record, "input_image", record_id)
    outputs = _required_list(record, "output_image", record_id)
    edits = _required_list(record, "edit", record_id)
    if not inputs:
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            "input_image must contain at least one state",
            field="input_image",
            source_record_id=record_id,
        )
    state_count = len(inputs)
    _require_length(outputs, state_count, "output_image", record_id)
    _require_length(edits, state_count, "edit", record_id)
    evaluations = _required_list(record, "eval", record_id)
    summaries = _required_list(record, "eval_summary", record_id)
    _require_length(evaluations, state_count, "eval", record_id)
    _require_length(summaries, state_count, "eval_summary", record_id)

    for index, output in enumerate(outputs[:-1]):
        if output is None:
            raise VisualReflectionDataError(
                RejectionReason.CONTRADICTORY_TERMINAL,
                f"output_image[{index}] is terminal but later states exist",
                field=f"output_image[{index}]",
                source_record_id=record_id,
            )
    if outputs[-1] is not None:
        raise VisualReflectionDataError(
            RejectionReason.MISSING_TERMINAL_STOP,
            "the final output_image must be null",
            field=f"output_image[{state_count - 1}]",
            source_record_id=record_id,
        )

    terminal_sentinels = {_normalize_terminal_edit(DEFAULT_TERMINAL_EDIT_SENTINEL)}
    steps: list[ReflectionStep] = []
    for index in range(state_count):
        reflection = _reflection_for_state(
            summaries[index],
            evaluations[index],
            index=index,
            source_record_id=record_id,
            fallback_max_chars=parser_config.fallback_max_chars,
        )
        source_edit = edits[index]
        if not isinstance(source_edit, str):
            raise VisualReflectionDataError(
                RejectionReason.INVALID_FIELD_TYPE,
                f"edit[{index}] must be a string",
                field=f"edit[{index}]",
                source_record_id=record_id,
            )
        edit = source_edit.replace("\r\n", "\n").replace("\r", "\n").strip()
        if index < state_count - 1:
            if not edit:
                raise VisualReflectionDataError(
                    RejectionReason.EMPTY_CONTINUE_EDIT,
                    f"edit[{index}] is empty for a continue transition",
                    field=f"edit[{index}]",
                    source_record_id=record_id,
                )
            if _normalize_terminal_edit(edit) in terminal_sentinels:
                raise VisualReflectionDataError(
                    RejectionReason.CONTRADICTORY_CONTINUE_EDIT,
                    f"edit[{index}] is the terminal no-edit sentinel on a continue transition",
                    field=f"edit[{index}]",
                    source_record_id=record_id,
                )
            steps.append({"reflection": reflection, "action": "continue", "edit": edit})
            continue

        normalized_terminal_edit = _normalize_terminal_edit(edit)
        if normalized_terminal_edit not in terminal_sentinels:
            raise VisualReflectionDataError(
                RejectionReason.CONTRADICTORY_TERMINAL_EDIT,
                "terminal source edit is not an allowed exact no-edit sentinel",
                field=f"edit[{index}]",
                source_record_id=record_id,
            )
        steps.append({"reflection": reflection, "action": "stop", "edit": ""})

    input_refs = [
        image_resolver(value, field="input_image", index=index, source_record_id=record_id)
        for index, value in enumerate(inputs)
    ]
    # Preserve the generated output URI as the supervised next state while
    # auditing its content continuity against the next row's current input.
    images: list[ImageRef] = [input_refs[0]]
    for index, output in enumerate(outputs[:-1]):
        output_ref = image_resolver(
            output,
            field="output_image",
            index=index,
            source_record_id=record_id,
        )
        next_input = input_refs[index + 1]
        if output_ref["sha256"] != next_input["sha256"]:
            raise VisualReflectionDataError(
                RejectionReason.TRANSITION_HASH_MISMATCH,
                f"output_image[{index}] does not hash-match input_image[{index + 1}]",
                field=f"output_image[{index}]",
                source_record_id=record_id,
                details={"output_sha256": output_ref["sha256"], "input_sha256": next_input["sha256"]},
            )
        images.append(output_ref)

    trajectory_id = derive_trajectory_id(
        manifest_id=manifest,
        source_dataset=UNICOT_DATASET_ID,
        source_record_id=record_id,
        pipeline_variant="direct_parse",
        prompt=prompt,
        images=images,
        steps=steps,
    )
    return validate_trajectory(
        {
            "trajectory_id": trajectory_id,
            "manifest_id": manifest,
            "source_dataset": UNICOT_DATASET_ID,
            "source_record_id": record_id,
            "pipeline_variant": "direct_parse",
            "prompt": prompt,
            "images": images,
            "steps": steps,
        }
    )


def _required_field(record: Mapping[str, Any], field: str, source_record_id: str) -> Any:
    if field not in record:
        raise VisualReflectionDataError(
            RejectionReason.MISSING_FIELD,
            f"UniCoT record is missing {field}",
            field=field,
            source_record_id=source_record_id,
        )
    return record[field]


def _required_list(record: Mapping[str, Any], field: str, source_record_id: str) -> list[Any]:
    value = _required_field(record, field, source_record_id)
    if not isinstance(value, list):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a list",
            field=field,
            source_record_id=source_record_id,
        )
    return value


def _require_length(value: Sequence[Any], expected: int, field: str, source_record_id: str) -> None:
    if len(value) != expected:
        raise VisualReflectionDataError(
            RejectionReason.LENGTH_MISMATCH,
            f"{field} has length {len(value)} but expected {expected}",
            field=field,
            source_record_id=source_record_id,
        )


def _reflection_for_state(
    summary_value: Any,
    evaluation_value: Any,
    *,
    index: int,
    source_record_id: str,
    fallback_max_chars: int,
) -> str:
    summary = _optional_text(summary_value, field=f"eval_summary[{index}]", source_record_id=source_record_id)
    evaluation = _optional_text(evaluation_value, field=f"eval[{index}]", source_record_id=source_record_id)
    if summary:
        return summary
    cleaned = " ".join(unicodedata.normalize("NFKC", evaluation).split())[:fallback_max_chars].rstrip()
    if not cleaned:
        raise VisualReflectionDataError(
            RejectionReason.EMPTY_REFLECTION,
            f"state {index} has neither a non-empty eval_summary nor eval fallback",
            field=f"eval_summary[{index}]",
            source_record_id=source_record_id,
        )
    return cleaned


def _optional_text(value: Any, *, field: str, source_record_id: str) -> str:
    if not isinstance(value, str):
        raise VisualReflectionDataError(
            RejectionReason.INVALID_FIELD_TYPE,
            f"{field} must be a string",
            field=field,
            source_record_id=source_record_id,
        )
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_terminal_edit(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()
