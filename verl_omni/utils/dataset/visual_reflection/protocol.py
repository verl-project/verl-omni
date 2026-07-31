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
"""Strict formatter and parser for the visible reflection protocol."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .contracts import ReflectionStep, RejectionReason, VisualReflectionDataError, validate_reflection_step

_HEADER_RE = re.compile(r"^(Reflection|Action|Edit):", re.MULTILINE)


def format_reflection_response(step: Mapping[str, Any]) -> str:
    """Serialize one validated step in the only supported field order."""
    normalized = validate_reflection_step(step)
    edit_line = f"Edit: {normalized['edit']}" if normalized["edit"] else "Edit:"
    return f"Reflection: {normalized['reflection']}\nAction: {normalized['action']}\n{edit_line}"


def parse_reflection_response(text: str) -> ReflectionStep:
    """Parse strict Reflection/Action/Edit text without guessing or rewriting action."""
    if not isinstance(text, str):
        raise _protocol_error("response must be a string")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(_HEADER_RE.finditer(normalized))
    names = [match.group(1) for match in matches]
    if names != ["Reflection", "Action", "Edit"] or not matches or matches[0].start() != 0:
        raise _protocol_error("response must contain exactly Reflection, Action, and Edit once in that order")

    reflection = normalized[matches[0].end() : matches[1].start()].strip()
    action = normalized[matches[1].end() : matches[2].start()].strip()
    edit = normalized[matches[2].end() :].strip()
    if "\n" in action:
        raise _protocol_error("Action must be a single line")

    try:
        return validate_reflection_step({"reflection": reflection, "action": action, "edit": edit})
    except VisualReflectionDataError as exc:
        raise _protocol_error(str(exc)) from exc


def _protocol_error(message: str) -> VisualReflectionDataError:
    return VisualReflectionDataError(
        RejectionReason.PROTOCOL_PARSE_ERROR,
        message,
        field="response",
    )
