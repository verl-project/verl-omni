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
"""Plan protocol primitives shared by the builder and the loop.

Two unrelated problems live here because they are two halves of the same shape.

**Reading a plan.** The plan data asks the policy to write a numbered list of subtask
prompts before it starts generating. The loop needs a predicate to tell "the model wrote
its plan" from "the model stopped talking", so the line grammar has one definition here
instead of one per consumer.

**Writing the reference.** UniCoT-Breakdown subtasks were authored for an *image edit*
tool: from subtask 1 onward every row opens by asserting the previous render is preserved
("Keep the outline of the image unchanged and edit with the following details. …v"). This
harness has no edit tool — ``generate_image`` is stateless text-to-image — and the plan is
sent as the call's prompt, so an item is one part of a single description rather than a
step rendered on its own. :func:`delta_subtasks` therefore drops the edit-tool lead-in and
keeps the part, which is what the reference shown to the agent should be and what the
system prompt asks the policy to produce.

This module holds no reward and no scoring. Every data source is graded by the same
formula in ``agentic_multidim_reward``; reflect and plan rollouts differ only in their
system prompt.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

__all__ = [
    "MIN_PLAN_LINE_TOKENS",
    "blank_tool_payloads",
    "delta_subtasks",
    "plan_lines_from_prose",
    "strip_edit_lead_in",
]

#: A plan line needs enough distinct words to be a subtask prompt rather than a heading.
MIN_PLAN_LINE_TOKENS = 4

#: Word-ish tokens, matching ``agentic_multidim_reward._tokens`` so the ">= 4 tokens"
#: filter here and the coverage metric there agree on what counts as content.
_TOKEN_RE = re.compile(r"[a-z0-9_']+")
_PLAN_HEADER_RE = re.compile(r"\bPlan\s*:", re.IGNORECASE)
_PLAN_LINE_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*.*?\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


def blank_tool_payloads(text: str) -> str:
    """Replace tool-call blocks and think tags with same-length blanks.

    A plan is what the policy writes as *prose*. The plan is then copied into the
    ``generate_image`` call's prompt, so a raw transcript contains it twice: once as the
    plan turn and again inside the tool call. Reading the second copy would make a
    call-only turn look like a plan turn. Length is preserved so a caller that needs
    offsets can still compare them against the original text.
    """

    def blank(match: re.Match[str]) -> str:
        return "\n" * (match.end() - match.start())

    return _THINK_TAG_RE.sub(blank, _TOOL_CALL_BLOCK_RE.sub(blank, text or ""))


def plan_lines_from_prose(prose: str) -> list[str]:
    """Extract the numbered/bulleted subtask prompts from already-stripped prose.

    Tool-call and think markup is blanked first (see :func:`blank_tool_payloads`), so the
    plan copied into a call's ``prompt`` is not read back as a second plan turn. A
    ``Plan:`` header, when present, bounds where the list starts; otherwise the whole text
    is scanned, because a policy that omits the header is still emitting a plan.

    Args:
        prose: Assistant text with tool-call payloads already removed, line breaks intact.

    Returns:
        Plan item texts in order, each with at least :data:`MIN_PLAN_LINE_TOKENS`
        distinct word tokens. Empty when no plan is present.
    """
    text = blank_tool_payloads(prose)
    header = _PLAN_HEADER_RE.search(text)
    lines = []
    for match in _PLAN_LINE_RE.finditer(text, header.end() if header else 0):
        line = match.group(1).strip()
        if len(set(_TOKEN_RE.findall(line.lower()))) >= MIN_PLAN_LINE_TOKENS:
            lines.append(line)
    return lines


#: Words that open a "preserve the previous render, then edit it" clause.
_PRESERVE_START = r"(?:keep|maintain|preserve|retain|do not change|without changing)"
_EDIT_VERB = r"(?:edit|add|refine|make|apply|render|enhance|integrate|illuminate|ensure|adjust|update)"

#: ``… and edit with the following details. <content>`` — the dominant subtask-1 form,
#: plus the ``make the following edits:`` and ``add the following details:`` variants.
#: Bounded to one sentence (``[^.!?]``, no ``DOTALL``): an unbounded ``.*?`` would find a
#: later ``these details`` and swallow the real content between the two.
_DETAILS_CLAUSE_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?\b(?:following|these)\b[^.!?]*?\b(?:details?|edits?)\b\s*[:.;]?\s*",
    re.IGNORECASE,
)
#: ``Keep all previously rendered elements unchanged. <content>`` — the dominant
#: subtask-2 form. The preserve sentence ends before the content starts.
_PRESERVE_SENTENCE_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?\b(?:unchanged|the same|as (?:is|before)|intact)\b\s*[.!?]\s*",
    re.IGNORECASE,
)
#: ``Without changing any composed details, apply …`` — preserve clause and content share
#: one comma-joined sentence, so split at the comma before the edit verb.
_PRESERVE_COMMA_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?,\s*(?={_EDIT_VERB}\b)(?P<content>.+)$",
    re.IGNORECASE | re.DOTALL,
)
#: ``Maintain all previously rendered elements, and make sure …`` — preserve clause and
#: content share one sentence, so split at the conjunction before the edit verb.
_PRESERVE_CONJUNCTION_RE = re.compile(
    rf"^\s*{_PRESERVE_START}\b[^.!?]*?(?:,\s*|\s+)and\s+(?={_EDIT_VERB}\b)(?P<content>.+)$",
    re.IGNORECASE | re.DOTALL,
)
_LEAD_IN_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (_DETAILS_CLAUSE_RE, False),
    (_PRESERVE_SENTENCE_RE, False),
    (_PRESERVE_COMMA_RE, True),
    (_PRESERVE_CONJUNCTION_RE, True),
)
#: A strip that removes most of the text is a mis-parse, not a lead-in, so the original is
#: kept. Deliberately low: a legitimate subtask can have a long preserve clause and one
#: short content sentence, and the patterns are already bounded to the opening sentence.
_MIN_RETAINED_FRACTION = 0.15


def strip_edit_lead_in(subtask: str) -> str:
    """Drop an image-edit lead-in from a source subtask, keeping the content.

    Args:
        subtask: Raw UniCoT-Breakdown subtask text.

    Returns:
        The subtask content with the leading "preserve the previous render and edit it"
        clause removed and a dangling conjunction cleaned up. Returned unchanged when the
        text carries no lead-in, when stripping would empty it, or when the strip would
        retain less than :data:`_MIN_RETAINED_FRACTION` of the text (a mis-parse).
    """
    text = (subtask or "").strip()
    if not text:
        return text
    for pattern, uses_content_group in _LEAD_IN_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        retained = (match.group("content") if uses_content_group else text[match.end() :]).strip()
        retained = re.sub(r"^(?:,|and|then)\s+", "", retained, flags=re.IGNORECASE).strip()
        if not retained or len(retained) < _MIN_RETAINED_FRACTION * len(text):
            continue
        return retained
    return text


def delta_subtasks(subtasks: Sequence[str]) -> tuple[str, ...]:
    """Drop the edit-tool framing from each source subtask, keeping the part itself.

    The plan is sent as one prompt, so what an item has to be is the *part of the
    picture* it contributes, not a step that re-renders everything before it. The source
    UniCoT-Breakdown items assert that the previous render is preserved before saying what
    changes; a stateless text-to-image tool has no earlier render, so that clause asks for
    something the tool cannot do and is dropped.

    Args:
        subtasks: Source subtasks in order, without any sentinel.

    Returns:
        One part per source subtask, with its lead-in stripped. Empty input returns
        ``()``.
    """
    parts = []
    for subtask in subtasks:
        stripped = strip_edit_lead_in(str(subtask).strip())
        if stripped.strip():
            parts.append(stripped.strip())
    return tuple(parts)
