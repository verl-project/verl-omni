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
"""CPU tests for fail-closed UniCoT-Breakdown parsing."""

import copy

import pytest

from verl_omni.utils.dataset.visual_reflection import RejectionReason, VisualReflectionDataError
from verl_omni.utils.dataset.visual_reflection.unicot_breakdown import (
    breakdown_converter_config,
    parse_unicot_breakdown_record,
)


def _plan_row(count: int = 2, *, data_id: str = "fixture") -> dict:
    subtasks: list[str | None] = [f"Subtask {index}." for index in range(count)]
    subtasks.extend([None] * (3 - count))
    images: list[str | None] = [f"./images/{data_id}_{index}.png" for index in range(count)]
    images.extend([None] * (3 - count))
    return {
        "data_id": data_id,
        "prompt": "A complex visual request.",
        "subtasks": subtasks,
        "subtask_images": images,
    }


def _reflect_row() -> dict:
    return {
        "data_id": "simple",
        "prompt": "A simple visual request.",
        "subtasks": ["No breakdown needed.", None, None],
        "subtask_images": [None, None, None],
    }


@pytest.mark.parametrize("count", [1, 2, 3])
def test_valid_plan_prefixes(count):
    parsed = parse_unicot_breakdown_record(_plan_row(count), manifest_id="manifest")

    assert parsed.task_type == "plan"
    assert parsed.plan_expected is True
    assert parsed.expected_num_images == count
    assert len(parsed.subtasks) == count
    assert len(parsed.subtask_images) == count


def test_no_breakdown_normalizes_to_reflect():
    row = _reflect_row()
    row["subtasks"][0] = "  NO breakdown\nneeded. "
    parsed = parse_unicot_breakdown_record(row, manifest_id="manifest")

    assert parsed.task_type == "reflect"
    assert parsed.plan_expected is False
    assert parsed.expected_num_images == 1
    assert parsed.subtasks == ()
    assert parsed.subtask_images == ()


def test_parser_does_not_mutate_source():
    row = _plan_row()
    before = copy.deepcopy(row)
    parse_unicot_breakdown_record(row, manifest_id="manifest")
    assert row == before


def test_converter_config_is_versioned():
    config = breakdown_converter_config()
    assert config["unicot_breakdown_parser"] == "subtask_structure_v1"
    assert config["unicot_breakdown_max_subtask_slots"] == 3


def test_source_record_override_must_match_data_id():
    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record(
            _plan_row(data_id="source"),
            manifest_id="manifest",
            source_record_id="different",
        )
    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row: row.__setitem__("subtasks", None), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtasks"].pop(), RejectionReason.LENGTH_MISMATCH),
        (lambda row: row["subtasks"].append(None), RejectionReason.LENGTH_MISMATCH),
        (lambda row: row.__setitem__("subtask_images", None), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtask_images"].pop(), RejectionReason.LENGTH_MISMATCH),
        (lambda row: row.__setitem__("prompt", ""), RejectionReason.EMPTY_PROMPT),
        (lambda row: row.pop("prompt"), RejectionReason.MISSING_FIELD),
        (lambda row: row["subtasks"].__setitem__(0, None), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtasks"].__setitem__(0, ""), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row: row["subtasks"].__setitem__(2, "gap"), RejectionReason.CONTRADICTORY_TERMINAL),
        (lambda row: row["subtask_images"].__setitem__(2, "ghost.png"), RejectionReason.CONTRADICTORY_TERMINAL),
        (lambda row: row["subtask_images"].__setitem__(0, 42), RejectionReason.INVALID_FIELD_TYPE),
    ],
)
def test_malformed_plan_rows_fail_closed(mutation, reason):
    row = _plan_row(1)
    mutation(row)
    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record(row, manifest_id="manifest")
    assert error.value.reason is reason


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row["subtasks"].__setitem__(1, "unexpected plan"),
        lambda row: row["subtask_images"].__setitem__(0, "unexpected.png"),
    ],
)
def test_contradictory_no_breakdown_rows_fail_closed(mutation):
    row = _reflect_row()
    mutation(row)
    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_breakdown_record(row, manifest_id="manifest")
    assert error.value.reason is RejectionReason.CONTRADICTORY_TERMINAL


def test_non_mapping_and_empty_manifest_fail_closed():
    with pytest.raises(VisualReflectionDataError) as non_mapping:
        parse_unicot_breakdown_record("bad", manifest_id="manifest")  # type: ignore[arg-type]
    assert non_mapping.value.reason is RejectionReason.INVALID_FIELD_TYPE

    with pytest.raises(VisualReflectionDataError) as empty_manifest:
        parse_unicot_breakdown_record(_plan_row(), manifest_id="")
    assert empty_manifest.value.reason is RejectionReason.INVALID_FIELD_TYPE
