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
"""CPU tests for fail-closed UniCoT direct parsing."""

import copy
from pathlib import Path
from typing import Any

import pytest

from tests.utils.dataset._visual_reflection_fixtures import write_rgb_png
from verl_omni.utils.dataset.visual_reflection import (
    LocalImageResolver,
    RejectionReason,
    VisualReflectionDataError,
)
from verl_omni.utils.dataset.visual_reflection.unicot import UniCoTParserConfig, parse_unicot_record


def make_unicot_row(root: Path, *, edit_turns: int, data_id: str = "fixture") -> dict[str, Any]:
    """Create a valid row using the public UniCoT field shape."""
    state_count = edit_turns + 1
    input_paths = [(Path("images") / "inputs" / f"state_{index}.png").as_posix() for index in range(state_count)]
    for index in range(state_count):
        write_rgb_png(root / input_paths[index], (index * 70 % 255, index * 40 % 255, index * 20 % 255))
    output_paths: list[str | None] = []
    for index in range(edit_turns):
        output_path = Path("images") / "outputs" / f"state_{index + 1}.png"
        (root / output_path).parent.mkdir(parents=True, exist_ok=True)
        (root / output_path).write_bytes((root / input_paths[index + 1]).read_bytes())
        output_paths.append(output_path.as_posix())
    output_paths.append(None)

    evaluations = [f"Detailed evaluation for state {index}." for index in range(state_count)]
    summaries = [f"Summary for state {index}." for index in range(state_count)]
    if state_count > 1:
        summaries[1] = ""
        evaluations[1] = "  Detailed\n evaluation   fallback for state 1.  "
    edits = [f"Apply delta edit {index}." for index in range(edit_turns)]
    edits.append("Everything is good. No editing needed.")
    return {
        "data_id": data_id,
        "prompt": "A precise visual prompt.",
        "eval": evaluations,
        "eval_summary": summaries,
        "edit": edits,
        "input_image": input_paths,
        "output_image": output_paths,
    }


@pytest.mark.parametrize("edit_turns", [0, 1, 2])
def test_parse_valid_zero_one_and_two_edit_rows(tmp_path, edit_turns):
    row = make_unicot_row(tmp_path, edit_turns=edit_turns, data_id=f"row-{edit_turns}")

    trajectory = parse_unicot_record(
        row,
        manifest_id="manifest-test",
        image_resolver=LocalImageResolver(tmp_path),
    )

    assert len(trajectory["images"]) == edit_turns + 1
    assert [step["action"] for step in trajectory["steps"]] == ["continue"] * edit_turns + ["stop"]
    assert trajectory["steps"][-1]["edit"] == ""
    assert trajectory["source_record_id"] == f"row-{edit_turns}"
    assert trajectory["pipeline_variant"] == "direct_parse"
    if edit_turns:
        assert trajectory["steps"][1]["reflection"] == "Detailed evaluation fallback for state 1."


def test_parser_uses_summary_before_eval_and_never_infers_action_from_language(tmp_path):
    row = make_unicot_row(tmp_path, edit_turns=1)
    row["eval_summary"][0] = "The text says stop, but a transition exists."
    row["eval"][0] = "continue"
    row["eval_summary"][1] = "The text says continue, but output is null."

    trajectory = parse_unicot_record(
        row,
        manifest_id="manifest-test",
        image_resolver=LocalImageResolver(tmp_path),
    )

    assert trajectory["steps"][0] == {
        "reflection": "The text says stop, but a transition exists.",
        "action": "continue",
        "edit": "Apply delta edit 0.",
    }
    assert trajectory["steps"][1]["action"] == "stop"


def test_eval_fallback_is_whitespace_cleaned_and_character_limited(tmp_path):
    row = make_unicot_row(tmp_path, edit_turns=0)
    row["eval_summary"][0] = ""
    row["eval"][0] = "  alpha\n beta    gamma  "

    trajectory = parse_unicot_record(
        row,
        manifest_id="manifest-test",
        image_resolver=LocalImageResolver(tmp_path),
        config=UniCoTParserConfig(fallback_max_chars=10),
    )

    assert trajectory["steps"][0]["reflection"] == "alpha beta"


@pytest.mark.parametrize("value", [True, 1.5, 0, -1])
def test_unicot_parser_config_rejects_invalid_character_limits(value):
    with pytest.raises((TypeError, ValueError)):
        UniCoTParserConfig(fallback_max_chars=value)


def test_terminal_exact_sentinel_is_case_whitespace_normalized_then_cleared(tmp_path):
    row = make_unicot_row(tmp_path, edit_turns=0)
    row["edit"][0] = "  EVERYTHING   is good.\nNo editing needed.  "

    trajectory = parse_unicot_record(
        row,
        manifest_id="manifest-test",
        image_resolver=LocalImageResolver(tmp_path),
    )

    assert trajectory["steps"][0] == {
        "reflection": "Summary for state 0.",
        "action": "stop",
        "edit": "",
    }


def test_transition_hash_matches_even_when_output_and_next_input_uris_differ(tmp_path):
    row = make_unicot_row(tmp_path, edit_turns=1)
    assert row["output_image"][0] != row["input_image"][1]

    trajectory = parse_unicot_record(
        row,
        manifest_id="manifest-test",
        image_resolver=LocalImageResolver(tmp_path),
    )

    assert trajectory["images"][1]["uri"] == row["output_image"][0]


def test_parser_does_not_mutate_source_record(tmp_path):
    row = make_unicot_row(tmp_path, edit_turns=2)
    before = copy.deepcopy(row)

    parse_unicot_record(
        row,
        manifest_id="manifest-test",
        image_resolver=LocalImageResolver(tmp_path),
    )

    assert row == before


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda row, root: row.__setitem__("edit", None), RejectionReason.INVALID_FIELD_TYPE),
        (lambda row, root: row["eval"].pop(), RejectionReason.LENGTH_MISMATCH),
        (
            lambda row, root: (row["eval_summary"].__setitem__(0, ""), row["eval"].__setitem__(0, "")),
            RejectionReason.EMPTY_REFLECTION,
        ),
        (lambda row, root: row["edit"].__setitem__(0, "  "), RejectionReason.EMPTY_CONTINUE_EDIT),
        (
            lambda row, root: row["edit"].__setitem__(0, "Everything is good. No editing needed."),
            RejectionReason.CONTRADICTORY_CONTINUE_EDIT,
        ),
        (lambda row, root: row["output_image"].__setitem__(0, None), RejectionReason.CONTRADICTORY_TERMINAL),
        (
            lambda row, root: row["output_image"].__setitem__(-1, row["input_image"][-1]),
            RejectionReason.MISSING_TERMINAL_STOP,
        ),
        (
            lambda row, root: row["edit"].__setitem__(-1, "Keep changing the composition."),
            RejectionReason.CONTRADICTORY_TERMINAL_EDIT,
        ),
        (lambda row, root: row["edit"].__setitem__(-1, ""), RejectionReason.CONTRADICTORY_TERMINAL_EDIT),
        (
            lambda row, root: Path(root / row["input_image"][0]).unlink(),
            RejectionReason.MISSING_IMAGE,
        ),
        (
            lambda row, root: write_rgb_png(root / row["input_image"][1], (255, 255, 255)),
            RejectionReason.TRANSITION_HASH_MISMATCH,
        ),
    ],
)
def test_malformed_rows_fail_closed_without_partial_trajectory(tmp_path, mutation, reason):
    row = make_unicot_row(tmp_path, edit_turns=2)
    mutation(row, tmp_path)

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_record(
            row,
            manifest_id="manifest-test",
            image_resolver=LocalImageResolver(tmp_path),
        )

    assert error.value.reason is reason


@pytest.mark.parametrize("missing_value", [None, "missing"])
def test_eval_field_must_be_a_complete_list_even_when_summaries_are_valid(tmp_path, missing_value):
    row = make_unicot_row(tmp_path, edit_turns=0)
    if missing_value == "missing":
        del row["eval"]
    else:
        row["eval"] = missing_value

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_record(
            row,
            manifest_id="manifest-test",
            image_resolver=LocalImageResolver(tmp_path),
        )

    assert error.value.reason in {RejectionReason.MISSING_FIELD, RejectionReason.INVALID_FIELD_TYPE}


def test_source_record_override_must_match_public_data_id(tmp_path):
    row = make_unicot_row(tmp_path, edit_turns=0, data_id="source-id")

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_record(
            row,
            manifest_id="manifest-test",
            source_record_id="wrong-id",
            image_resolver=LocalImageResolver(tmp_path),
        )

    assert error.value.reason is RejectionReason.DUPLICATE_SOURCE_RECORD


@pytest.mark.parametrize(("field", "value"), [("eval", None), ("eval_summary", None)])
def test_reflection_array_elements_must_be_strings(tmp_path, field, value):
    row = make_unicot_row(tmp_path, edit_turns=0)
    row[field][0] = value

    with pytest.raises(VisualReflectionDataError) as error:
        parse_unicot_record(
            row,
            manifest_id="manifest-test",
            image_resolver=LocalImageResolver(tmp_path),
        )

    assert error.value.reason is RejectionReason.INVALID_FIELD_TYPE
