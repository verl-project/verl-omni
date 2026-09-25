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
"""CPU tests for the plan-protocol primitives.

Two contracts matter and both are exercised against the real UniCoT-Breakdown
wordings, not invented ones: the plan-line grammar the loop uses to tell "the model
wrote its plan" from "the model stopped talking", and the edit-lead-in strip used to
flatten source subtasks for a harness that has no image-editing tool.
"""

from __future__ import annotations

import pytest

from verl_omni.utils.agentic.plan_protocol import (
    MIN_PLAN_LINE_TOKENS,
    blank_tool_payloads,
    delta_subtasks,
    plan_lines_from_prose,
    strip_edit_lead_in,
)


def test_plan_lines_are_read_after_the_header():
    prose = (
        "I will not number these lines at the top.\n"
        "Plan:\n"
        "1. A librarian floats in an underwater cave library with fish nearby.\n"
        "2. Add a gold satin gown and filtered light beams from above.\n"
    )

    lines = plan_lines_from_prose(prose)

    assert len(lines) == 2
    assert lines[0].startswith("A librarian floats")
    # The header bounds the scan, so the preamble is not treated as a plan item.
    assert all("not number these lines" not in line for line in lines)


def test_plan_lines_accept_bullets_and_reject_short_items():
    assert plan_lines_from_prose("- An African librarian floats in a cave.\n") == [
        "An African librarian floats in a cave."
    ]
    # A heading is not a subtask prompt: the threshold is MIN_PLAN_LINE_TOKENS words.
    assert plan_lines_from_prose("1. red\n2. blue\n") == []
    assert MIN_PLAN_LINE_TOKENS == 4


def test_plan_lines_scans_the_whole_text_without_a_header():
    """A policy that skips the literal ``Plan:`` header still emitted a plan."""
    prose = "1. A librarian floats in an underwater cave library.\n"

    assert plan_lines_from_prose(prose) == ["A librarian floats in an underwater cave library."]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The dominant subtask-1 form, with and without the article.
        (
            "Keep the outline of the image unchanged and edit with the following details. The chair is high-tech.",
            "The chair is high-tech.",
        ),
        (
            "Keep the outline of the image unchanged and edit with following details. The fish is mid-swim.",
            "The fish is mid-swim.",
        ),
        (
            "Keep the compositional outline unchanged and make the following edits: the piano keys are white.",
            "the piano keys are white.",
        ),
        (
            "Keep the layout the same and add the following details: the location is Houston.",
            "the location is Houston.",
        ),
        # The dominant subtask-2 form: a preserve sentence, then the content.
        (
            "Keep all previously rendered elements unchanged. Apply cinematic lighting with warm rim light.",
            "Apply cinematic lighting with warm rim light.",
        ),
        # Preserve clause and content share one comma-joined sentence.
        (
            "Without changing any composed details, apply shimmering ornamental highlights to the scene.",
            "apply shimmering ornamental highlights to the scene.",
        ),
        # Preserve clause and content share one conjunction-joined sentence.
        (
            "Maintain all previously rendered elements and apply a cubist style to the whole composition.",
            "apply a cubist style to the whole composition.",
        ),
        (
            "Keep all previously rendered elements, and make sure the scene conveys a supernatural mood.",
            "make sure the scene conveys a supernatural mood.",
        ),
        # A base subtask carries no lead-in and must survive untouched.
        (
            "An African American librarian floats while reading a book in an underwater library.",
            "An African American librarian floats while reading a book in an underwater library.",
        ),
        # Nothing follows the preserve clause, so there is no content to keep.
        ("Keep all previously rendered elements unchanged.", "Keep all previously rendered elements unchanged."),
        ("", ""),
    ],
)
def test_strip_edit_lead_in(raw: str, expected: str):
    assert strip_edit_lead_in(raw) == expected


def test_strip_edit_lead_in_does_not_swallow_content_across_sentences():
    """Regression: an unbounded match ate the middle of a subtask.

    ``_DETAILS_CLAUSE_RE`` originally used ``.*?`` with ``DOTALL``, so a later
    ``these details`` in a *different* sentence anchored the match and everything
    between it and the opening ``Keep`` was discarded — real content, kept only
    because the 35% guard happened to let it through. The clause must stay inside the
    opening sentence.
    """
    raw = (
        "Keep the overall outline and composition. Make the silhouette tall and centered, "
        "surrounded by thick swirling mist. Now apply these details: add vibrant concentric spirals."
    )

    assert strip_edit_lead_in(raw) == raw


def test_strip_edit_lead_in_keeps_the_original_when_the_strip_is_implausible():
    """A strip that retains almost nothing is a mis-parse, not a lead-in.

    The conjunction form is bounded to one sentence but can still leave a tiny
    remainder; below ``_MIN_RETAINED_FRACTION`` the source text is returned intact so a
    fragment never reaches the reward as a reference.
    """
    raw = "Maintain all of the previously rendered elements and their exact spatial arrangement, and apply a tint."

    assert strip_edit_lead_in(raw) == raw


def test_delta_subtasks_drop_the_edit_framing_and_keep_each_part():
    """The canvas sample: each item is the part it contributes, not a re-render.

    Source subtasks 1 and 2 are written for an image-*edit* tool ("keep the outline
    unchanged and edit …"). Plan mode sends the whole list as one prompt, so an item is
    a part of that description; the edit instruction is something a stateless
    text-to-image tool cannot act on and must not reach the reward as content.
    """
    source = (
        "An African American librarian floats while reading a book in an underwater library in a cave.",
        "Keep the outline of the image unchanged and edit with the following details. "
        "The librarian wears a luxurious gold satin gown.",
        "Keep all previously rendered elements unchanged. "
        "Apply visual effects that reinforce the elegant and ethereal mood.",
    )

    parts = delta_subtasks(source)

    assert len(parts) == len(source)
    # Each part is the source subtask with its lead-in removed, never the accumulated chain.
    assert parts[0] == source[0]
    assert "gold satin gown" in parts[1]
    assert source[0] not in parts[1]
    assert "ethereal mood" in parts[2]
    # The edit framing reaches none of the parts.
    assert not any("Keep the outline" in part for part in parts)
    assert not any("Keep all previously rendered" in part for part in parts)
    assert not any("following details" in part for part in parts)


def test_delta_subtasks_handle_empty_input():
    assert delta_subtasks(()) == ()
    assert delta_subtasks(["", "   "]) == ()


def test_plan_lines_ignore_the_plan_copied_into_a_tool_call():
    """A call's prompt is the plan *again*, so counting it would double every revision.

    Plan mode now passes the numbered list as ``generate_image``'s ``prompt``. If the
    extractor read the raw transcript, the call-only turn that carries it would look like
    a second plan turn — and the labeler, which classifies raw decode, would tag a call
    turn as a plan turn.
    """
    plan = "1. A librarian floats in an underwater cave library with fish nearby.\n"
    call = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": ' + repr(plan) + "}}\n</tool_call>\n"

    assert plan_lines_from_prose(plan) == ["A librarian floats in an underwater cave library with fish nearby."]
    assert plan_lines_from_prose(call) == []
    assert plan_lines_from_prose(plan + call) == plan_lines_from_prose(plan)
    # Blanking preserves length, so a caller comparing positions still can: the number
    # of characters removed and the number of blanks inserted are equal.
    assert len(blank_tool_payloads(plan + call)) == len(plan + call)
