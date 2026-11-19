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

"""UND decode parsing: every tool-call dialect the Bagel checkpoint emits, and the turn classifier."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

HERMES_SPECIAL_TOKENS: tuple[str, ...] = ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")

#: Stop strings for the UND lane.
UND_STOP_SEQUENCES: tuple[str, ...] = ("<output>", "</tool_call>", "</tools>")

#: Chat-template wrapper for an environment observation (Qwen/Hermes ``<tool_response>`` user turn).
TOOL_RESPONSE_TEMPLATE = (
    "\n<|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
)
IM_END = "<|im_end|>"

GENERATE_IMAGE_TOOL = "generate_image"
JUDGE_IMAGE_TOOL = "judge_image"
#: Tool names the checkpoint uses to *stop*; a stop payload is the terminal signal, not an unsupported tool.
STOP_TOOL_NAMES = frozenset({"done", "done.", "stop", "finish", "finished", "finalize", "end", "complete", "terminate"})

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
#: ``<tools>`` is the tag the prompt uses to *describe* the functions; the checkpoint re-emits it as a call tag.
_TAGGED_CALL_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
#: ``Done`` / ``Done.`` optionally followed by special tokens (the engine decodes with ``skip_special_tokens=False``).
_DONE_RE = re.compile(r"\bDone\.?\s*(?:<\|[^|]*?\|>\s*)*$", re.IGNORECASE)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")
_END_OF_CONVERSATION = "end of conversation"
_ROLE_ECHO_RE = re.compile(r"^\s*(?:assistant|system|user)\s*[:\-]?\s*", re.IGNORECASE)
_LEADING_MARKER_RE = re.compile(r"^\s*(?:<[^>{}\n]{0,48}>\s*)+")
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+.-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
#: Longest label a fenced call may carry in front of its fence; a quoted example inside prose exceeds it.
_FENCED_PREAMBLE_LIMIT = 120
_NATIVE_OPEN_TAGS = (
    "start of ToolCall",
    "begin of ToolCall",
    "begin_of ToolCall",
    "begin_of_ToolCall",
    "FunctionCallBegin",
)
_NATIVE_CLOSE_TAGS = (
    "end of ToolCall",
    "end_of ToolCall",
    "end of the ToolCall",
    "end_of the ToolCall",
    "end_of_ToolCall",
    "FunctionCallEnd",
)
_NATIVE_CALL_RE = re.compile(
    rf"<\|(?:{'|'.join(re.escape(t) for t in _NATIVE_OPEN_TAGS)})\|>\s*(.*?)\s*"
    rf"(?:<\|(?:{'|'.join(re.escape(t) for t in _NATIVE_CLOSE_TAGS)})\|>|\Z)",
    re.DOTALL,
)
_NAMED_OBJECT_START_RE = re.compile(r'\{\s*"(?:name|function)"\s*:')
_TRAILING_JUNK_RE = re.compile(r"(?:<\|[^|]*?\|>|[^\w])*")
#: An in-turn "Short reflection: ..." tail the checkpoint appends to the diffusion prompt itself.
_REFLECTION_TAIL_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*"
    r"(?:(?:short|brief|private|self|final|quick|inner)\s+)*"
    r"(?:(?:self-)?reflection|thinking|thoughts?|deliberation)\s*:",
    re.IGNORECASE,
)


def strip_reflection_tail(prompt: str) -> str:
    """Drop a trailing ``Short reflection: ...`` / ``Reflection: ...`` clause."""
    match = _REFLECTION_TAIL_RE.search(prompt)
    if match is None:
        return prompt
    return prompt[: match.start()].rstrip(" \t\n.,;:-") or prompt


def parse_hermes_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a Hermes ``<tool_call>{...}</tool_call>`` span, or ``None`` if absent."""
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    payload = json.loads(match.group(1))
    if "name" not in payload and "function" in payload:
        payload = payload["function"]
    return payload


def _leading_json_object(text: str) -> dict[str, Any] | None:
    """Parse the JSON object that starts ``text`` with a string-aware brace counter."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _named_payload(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    """Accept ``{"function": {...}}`` (OpenAI shape) and require an action ``name``."""
    if payload is None:
        return None
    if "name" not in payload and "function" in payload:
        function = payload["function"]
        payload = function if isinstance(function, dict) else payload
    return payload if "name" in payload else None


def _labelled_fenced_call(text: str) -> dict[str, Any] | None:
    """Parse the first *labelled* fenced block in ``text`` that names a tool."""
    cursor = 0
    for match in _FENCED_BLOCK_RE.finditer(text):
        label = text[cursor : match.start()]
        cursor = match.end()
        if len(label) > _FENCED_PREAMBLE_LIMIT or "{" in label or "}" in label:
            continue
        payload = _named_payload(_leading_json_object(match.group(1)))
        if payload is not None:
            return payload
    return None


def _tools_tagged_call(text: str) -> dict[str, Any] | None:
    """Parse the first ``<tools>{...}</tools>`` span that carries a *complete call*."""
    for match in _TAGGED_CALL_RE.finditer(text):
        payload = _named_payload(_leading_json_object(match.group(1)))
        if payload is not None and "arguments" in payload:
            return payload
    return None


def _native_candidates(raw: str):
    """Yield the substrings of a native tag body that could hold the call (array, object, raw)."""
    yield raw
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        yield raw[start : end + 1]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        yield raw[start : end + 1]


def _native_payload(raw: str) -> dict[str, Any] | None:
    """Read the first named call out of a native tag body (array *or* object; tolerates a cut tail)."""
    raw = raw.strip()
    for candidate in _native_candidates(raw):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, list):
            for item in loaded:
                payload = _named_payload(item if isinstance(item, dict) else None)
                if payload is not None:
                    return payload
            continue
        payload = _named_payload(loaded if isinstance(loaded, dict) else None)
        if payload is not None:
            return payload
    return _named_payload(_leading_json_object(raw))


def _native_tagged_call(text: str) -> dict[str, Any] | None:
    """Parse a ``<|start of ToolCall|>[...]<|end of ToolCall|>`` span."""
    for match in _NATIVE_CALL_RE.finditer(text):
        payload = _native_payload(match.group(1))
        if payload is not None:
            return payload
    return None


def _trailing_call(text: str) -> dict[str, Any] | None:
    """Parse a named call that *ends* the turn, after a prose plan."""
    decoder = json.JSONDecoder()
    run: list[dict[str, Any]] = []
    tail_start = len(text)
    for match in reversed(list(_NAMED_OBJECT_START_RE.finditer(text))):
        try:
            payload, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if end > tail_start or not isinstance(payload, dict):
            continue
        if _TRAILING_JUNK_RE.fullmatch(text[end:tail_start]) is None:
            break
        run.insert(0, payload)
        tail_start = match.start()
    for payload in run:
        payload = _named_payload(payload)
        if payload is not None and "arguments" in payload:
            return payload
    return None


def parse_und_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a tool call emitted by the UND lane -- tagged, native, ``<tools>``, bare, or fenced."""
    tagged = parse_hermes_tool_call(text)
    if tagged is not None:
        return tagged
    native = _native_tagged_call(text)
    if native is not None:
        return native
    head = _LEADING_MARKER_RE.sub("", _ROLE_ECHO_RE.sub("", text, count=1).lstrip())
    if head.startswith("{"):
        return _named_payload(_leading_json_object(head))
    tools_tagged = _tools_tagged_call(text)
    if tools_tagged is not None:
        return tools_tagged
    if "```" in text:
        fenced = _labelled_fenced_call(text)
        if fenced is not None:
            return fenced
    return _trailing_call(text)


def is_terminal_only(text: str) -> bool:
    """True when a decode carries only control markers / the end-of-conversation line."""
    residue = _SPECIAL_TOKEN_RE.sub(" ", text).strip().strip(".").strip()
    return not residue or residue.lower() == _END_OF_CONVERSATION


def und_turn_kind(text: str) -> str:
    """Classify an UND decode: ``generate_image``, ``judge_image``, ``done``, or ``continue``.

    A Hermes-tagged call naming any other tool is fail-closed (``ValueError``); a bare
    payload naming an unknown tool is inert and classified ``continue``.
    """
    tagged = parse_hermes_tool_call(text)
    call = tagged if tagged is not None else parse_und_tool_call(text)
    if call is not None:
        name = str(call.get("name", ""))
        normalized = name.strip().strip(".\"' ").lower()
        if normalized == GENERATE_IMAGE_TOOL:
            return "generate_image"
        if normalized in STOP_TOOL_NAMES:
            logger.debug("bagel_corl_und_stop_tool name=%s (terminal signal)", name)
            return "done"
        if normalized == JUDGE_IMAGE_TOOL:
            return "judge_image"
        if tagged is not None:
            raise ValueError(f"Bagel CoRL UND emitted unsupported tool {name!r}; Qwen/other tools are fail-closed")
        logger.debug("bagel_corl_und_inert_tool name=%s (no loop tool; RM judges)", name)
        return "continue"
    if _DONE_RE.search(text.strip()) or is_terminal_only(text):
        return "done"
    return "continue"
