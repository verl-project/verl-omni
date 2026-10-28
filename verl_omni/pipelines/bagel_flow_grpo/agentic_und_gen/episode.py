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

"""Hermes tool protocol, J×K IDs, serial episode, and flatten for Bagel Co-RL (Joint-Training)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Single trajectory-state layer (audit T1.6): the bagel lane shares the Mode-2a
# tool loop's registry/latch/context under tools.trajectory. The legacy
# agent_loop/image_gen_trajectory_context module remains only for its own
# deprecated consumers and must not be imported here.
from verl_omni.agent_loop.utils import derive_rollout_seed
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import (
    _force_info_logging,
    serving_rollout_provenance,
)
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.turn_protocol import (
    build_forced_reflection,
    derive_good_enough_from_scores,
    format_rm_scores_as_judge_text,
)
from verl_omni.tools.trajectory.episode import EpisodeState, current_episode, episode_scope
from verl_omni.tools.trajectory.episode_artifacts import build_generate_call_meta, register_tool_artifact
from verl_omni.tools.trajectory.episode_context import get_active_user_prompt
from verl_omni.tools.trajectory.episode_judge import (
    clear_good_enough_yes_reached,
    get_good_enough_yes_reached,
    set_good_enough_yes_reached,
)

logger = logging.getLogger(__name__)


# ``BAGEL_CORL_DEBUG=1`` re-enables this package's INFO diagnostics. vLLM calls ``logging.basicConfig(level=ERROR)``
# in every process that imports it, which raises the *root* level and silently swallows the ``bagel_corl_sync``
# lifecycle traces, the per-turn timing below and ``_wake_und_rollout_replicas``' confirmation -- the driver log goes
# quiet at the exact moment the engines start, so a stalled rollout looks like an idle one.
if os.getenv("BAGEL_CORL_DEBUG") == "1":
    _force_info_logging()
# Per-turn trace of the serial UND/GEN episode. Off by default: one line per decode is
# too noisy for a long run, but it is the only way to tell "the AR engine is slow" from
# "the client never got its response" when an episode stops making progress.
_BAGEL_CORL_TURN_DEBUG = os.getenv("BAGEL_CORL_DEBUG") == "1"

GENERATE_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image from a text prompt using the Bagel GEN pathway.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Text prompt for image generation."},
            },
            "required": ["prompt"],
        },
    },
}

JUDGE_IMAGE_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "judge_image",
        "description": (
            "Judge the last generated image. Returns correctness/aesthetics scores, "
            "findings, suggested prompt fixes, and a good_enough verdict. Call it after generate_image; "
            "on a planned task, call it once, after the final planned image."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user_request": {
                    "type": "string",
                    "description": "Prefer exactly ``same as user message``.",
                },
                "image_prompt": {
                    "type": "string",
                    "description": "Prefer exactly ``last``, or a short echo of the diffusion prompt.",
                },
            },
            "required": ["user_request", "image_prompt"],
        },
    },
}

# Agentic reflection protocol. UND turns are the reasoner: generate, then
# rewrite from the judge, up to three images. The chat template appends the tool schemas.
REFLECTION_SYSTEM_PROMPT = """You are a visual creation agent. You have two tools:
generate_image draws an image from a complete diffusion prompt; judge_image inspects the
last generated image and returns structured feedback.

The loop, and there is no other: generate an image, judge it, then rewrite the diffusion
prompt from the judge's findings and generate again. Keep going while the judge finds
fault. Once the judge is good enough — or another pass cannot improve the image — you are
done, and that is a reply of the single word Done with no tool call.

Rewriting rules — the rewrite is the work, not a formality:
- The user's request is the content spec; your diffusion prompt is a rendering
  recipe for the generate_image tool. The tool never sees the request, so anything
  the recipe does not say does not get drawn. A rewrite keeps the content and
  changes the recipe.
- Rewrite drastically. A prompt that grows by one quality assertion per pass is not
  a rewrite: the tool reads the same recipe and returns the same defect. Never
  append reassurances such as "the text is legible", "clearly rendered" or "high
  quality" — change what the tool actually reads.
- Change at least two things that reach the pixels: subject and framing, composition
  and layout grid, medium and style, palette and contrast, lighting, camera angle or
  aspect ratio, level of detail, and — when the image must show text — the typography
  strategy (how many words, which lines, weight, case, size on the canvas, placement,
  and the background behind the glyphs). Never ask for more words than the tool can
  draw legibly.
- Fail forward from the last judge: resolve every finding and apply the
  suggested_fixes. Never re-send a recipe this rollout already used — the earlier
  prompts are in the conversation, so mine them for what already failed and try a
  materially different one rather than a reworded repeat.
- Preserve every explicit requirement of the request: subjects, colours, counts, and
  any literal strings that must appear. Reordering or translating the wording is
  allowed; silently dropping requested content is not.

Always generate before judging, judge before deciding, and use no other tools.
The brevity note on the user turn bounds your private thinking and your reflection
only — never let it shorten or water down a diffusion prompt."""


def resolve_task_type(*, extra_info: Any = None, ground_truth: Any = None) -> Any:
    """Return the row's ``task_type`` label from the rollout kwargs."""
    for source in (extra_info, ground_truth):
        if isinstance(source, dict):
            label = source.get("task_type")
            if label:
                return label
    return None


def with_reflection_system_prompt(raw_prompt: Any, *, task_type: Any = None) -> Any:
    """Replace the dataset system turn with the reflection protocol."""
    if str(task_type or "").strip().lower() == "plan":
        return raw_prompt
    if not isinstance(raw_prompt, list | tuple) or not raw_prompt:
        return raw_prompt
    messages = [dict(m) if isinstance(m, dict) else m for m in raw_prompt]
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0]["content"] = REFLECTION_SYSTEM_PROMPT
        return messages
    return [{"role": "system", "content": REFLECTION_SYSTEM_PROMPT}, *messages]


HERMES_SPECIAL_TOKENS: tuple[str, ...] = (
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Tagged-call dialect (``<tools>{...}</tools>``). NOT a synonym for ``<tool_call>``: ``<tools>`` is
# the tag the recipe's prompt uses to *describe* the available functions, so the checkpoint
# re-emits it as its own call tag. See ``_tools_tagged_call``.
_TAGGED_CALL_RE = re.compile(r"<tools>\s*(.*?)\s*</tools>", re.DOTALL)
# The terminal ``Done.`` decode comes back from the engine with ``skip_special_tokens=False``, so
# the raw text is ``Done.<|im_end|>``. The trailing special tokens are therefore tolerated;
# without that the loop reads its own stop word as ``continue`` and appends a second ``Done.``.
# The period is optional: the checkpoint writes ``Done`` as often as ``Done.`` (16 of 96
# episodes in steps 78-82 of ``bagel_corl_plateaufix_2356``), and a missing period used to make
# the loop re-sample its own stop word until ``max_und_turns``.
_DONE_RE = re.compile(r"\bDone\.?\s*(?:<\|[^|]*?\|>\s*)*$", re.IGNORECASE)

# Control markers, which the engine returns inside ``text`` because it decodes with
# ``skip_special_tokens=False`` (``<|im_end|>``, ``<|end of conversation|>``, ...). A turn made
# of nothing else carries no policy content at all.
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]*?\|>")
#: The Bagel template's own end-of-conversation line, which this checkpoint emits as a *whole
#: turn*. Matched case-insensitively after the trailing period is stripped.
_END_OF_CONVERSATION = "end of conversation"

# Role labels the checkpoint sometimes re-emits as *text* before a payload
# ("ASSISTANT\n {json}"), and control/tag markers it echoes back from the prompt itself
# ("<tools>\n {json}"). Both are stripped before a bare-payload call is recognized; prose is
# not, which is what keeps a JSON example inside the model's plan from matching.
_ROLE_ECHO_RE = re.compile(r"^\s*(?:assistant|system|user)\s*[:\-]?\s*", re.IGNORECASE)
_LEADING_MARKER_RE = re.compile(r"^\s*(?:<[^>{}\n]{0,48}>\s*)+")

# Fenced dialect (see ``parse_und_tool_call``). The optional info string (```json) is dropped.
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+.-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)
# Longest label a fenced call may carry in front of its fence. The bound is what separates
# "the turn *is* a labelled call" from "a prose plan that happens to quote one": a quoted
# example sits inside paragraphs, a call does not. It is applied
# per fenced block -- the text between one fence and the next -- so a turn that lists one fenced
# object per step (a plan, then the call) still counts, while a lone quoted example does not.
_FENCED_PREAMBLE_LIMIT = 120

# Native call tags.
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
_NATIVE_OPEN_ALT = "|".join(re.escape(tag) for tag in _NATIVE_OPEN_TAGS)
_NATIVE_CLOSE_ALT = "|".join(re.escape(tag) for tag in _NATIVE_CLOSE_TAGS)
_NATIVE_CALL_RE = re.compile(
    rf"<\|(?:{_NATIVE_OPEN_ALT})\|>\s*(.*?)\s*(?:<\|(?:{_NATIVE_CLOSE_ALT})\|>|\Z)",
    re.DOTALL,
)

_GENERATE_IMAGE_TOOL = "generate_image"
_JUDGE_IMAGE_TOOL = "judge_image"
#: Tool names the checkpoint uses to *stop*. A stop payload is exactly the terminal signal the
#: loop wants; the fail-closed guard exists for *Qwen* tools, not for the model's own stop word
#: (treating ``{"name": "Done"}`` as unsupported killed the episode and padded the batch).
_STOP_TOOL_NAMES = frozenset(
    {"done", "done.", "stop", "finish", "finished", "finalize", "end", "complete", "terminate"}
)

# The recipe's system prompt asks the UND lane to ``judge_image`` after the last image, but the loop's tool registry
# has only ``generate_image`` (judging is the RM's job, see ``agentic_und_gen.reward``).
_INERT_BARE_TOOLS = frozenset({"judge_image"})

# The user prompt asks for a short private reflection in the same turn (``_BREVITY_SUFFIX`` in
# ``build_unicot_agentic_rl``), and the checkpoint appends that reflection to the ``generate_image`` prompt itself,
# e.g. "...
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


class GenerateImageCapError(RuntimeError):
    """Raised when a second GEN call is attempted under ``max_generate_passes=1``."""


def parse_hermes_tool_call(text: str) -> dict[str, Any] | None:
    """Parse a Hermes ``<tool_call>{...}</tool_call>`` span.

    Returns:
        Parsed JSON object, or ``None`` if no tool call is present.
    """
    match = _TOOL_CALL_RE.search(text)
    if match is None:
        return None
    payload = json.loads(match.group(1))
    if "name" not in payload and "function" in payload:
        payload = payload["function"]
    return payload


def _leading_json_object(text: str) -> dict[str, Any] | None:
    """Parse the JSON object that starts ``text`` (ignoring anything that follows it).

    A brace counter that respects string state, so a ``prompt`` containing ``{``/``}`` or an
    escaped quote does not end the scan early.
    """
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
    """Accept ``{"function": {...}}`` (OpenAI shape) and require an action ``name``.

    A payload with no ``name`` is *not* a call: that is the guard that keeps a JSON example
    inside the model's prose from being executed.
    """
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
        if payload is None or "arguments" not in payload:
            continue
        return payload
    return None


def _native_candidates(raw: str):
    """Yield the substrings of a native tag body that could hold the call.

    The body is normally a bare JSON array, but a turn cut at the token cap has no closing
    bracket, so the bracketed scan is offered as a fallback rather than assumed.
    """
    yield raw
    start, end = raw.find("["), raw.rfind("]")
    if start >= 0 and end > start:
        yield raw[start : end + 1]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        yield raw[start : end + 1]


def _native_payload(raw: str) -> dict[str, Any] | None:
    """Read the first named call out of a native tag body (array *or* object)."""
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
    # Truncated body: fall back to the leading-object scan, which tolerates a cut tail.
    return _named_payload(_leading_json_object(raw))


def _native_tagged_call(text: str) -> dict[str, Any] | None:
    """Parse a ``<|start of ToolCall|>[...]<|end of ToolCall|>`` span."""
    for match in _NATIVE_CALL_RE.finditer(text):
        payload = _native_payload(match.group(1))
        if payload is not None:
            return payload
    return None


_NAMED_OBJECT_START_RE = re.compile(r'\{\s*"(?:name|function)"\s*:')
_TRAILING_JUNK_RE = re.compile(r"(?:<\|[^|]*?\|>|[^\w])*")


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


# : Stop strings for the UND lane.
UND_STOP_SEQUENCES: tuple[str, ...] = ("<output>", "</tool_call>", "</tools>")

#: Chat-template wrapper for an environment observation: close the assistant turn, hand the
#: tool's answer back as a ``user`` ``<tool_response>`` turn (the Qwen/Hermes template's own
#: shape, and what the Mode-2a loop renders), and open the next assistant turn.
TOOL_RESPONSE_TEMPLATE = (
    "\n<|im_start|>user\n<tool_response>\n{body}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"
)
_IM_END = "<|im_end|>"

#: Fraction of the *remaining* context a single UND turn may consume.
UND_TURN_FRACTION: float = 0.5

#: Substring of the rejection ``AsyncOmni.generate`` raises while any sleep tag is still set.
#: The engine reports the exact tag set, e.g. ``Currently sleeping tags: ['weights', 'kv_cache']``.
ENGINE_ASLEEP_MARKER: str = "partially or fully asleep"

#: Backoff schedule for re-issuing a decode that was rejected by a still-waking engine.
#: A level-1 wake re-maps its weights through ``create_and_map`` and takes on the order of tens
#: of seconds, so the schedule has to outlast that. Total ~62 s across 5 retries; a genuine,
#: never-waking engine still fails loudly rather than spinning forever.
ENGINE_ASLEEP_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 10.0, 15.0, 30.0)


def _iter_exception_chain(exc: BaseException):
    """Yield ``exc`` followed by every exception it wraps.

    Both spellings are followed because the two wrappers in this path disagree: Python sets
    ``__cause__``/``__context__``, while ``ray.exceptions.RayTaskError`` also carries an
    explicit ``cause`` attribute.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        wrapped = getattr(current, "cause", None)
        if not isinstance(wrapped, BaseException):
            wrapped = current.__cause__ if isinstance(current.__cause__, BaseException) else current.__context__
        current = wrapped if isinstance(wrapped, BaseException) else None


def exception_search_text(exc: BaseException) -> str:
    """Flatten ``exc`` and everything it wraps into one searchable string."""
    parts: list[str] = []
    for current in _iter_exception_chain(exc):
        parts.append(f"{type(current).__name__}: {current}")
        parts.append(repr(current))
    try:
        parts.append("".join(traceback.format_exception(exc)))
    except Exception:  # noqa: BLE001 -- classification must never raise
        pass
    return "\n".join(parts)


def engine_asleep_retry_delay(attempt: int, message: str) -> float | None:
    """Seconds to wait before re-issuing a decode rejected by a still-waking engine."""
    if ENGINE_ASLEEP_MARKER not in message:
        return None
    if attempt >= len(ENGINE_ASLEEP_BACKOFF_S):
        return None
    return ENGINE_ASLEEP_BACKOFF_S[attempt]


async def generate_retrying_engine_wake(attempt_fn, *, on_retry=None, sleep=None):
    """Await ``attempt_fn()``, re-issuing while the engine reports a still-waking sleep tag."""
    import asyncio

    sleeper = sleep or asyncio.sleep
    attempt = 0
    while True:
        try:
            return await attempt_fn()
        except Exception as exc:  # noqa: BLE001 — only the wake-window rejection is swallowed
            delay = engine_asleep_retry_delay(attempt, exception_search_text(exc))
            if delay is None:
                raise
            if on_retry is not None:
                on_retry(attempt, delay, exc)
            await sleeper(delay)
            attempt += 1


def und_turn_max_tokens(
    *,
    context_used: int,
    max_context_tokens: int,
    fraction: float = UND_TURN_FRACTION,
) -> int:
    """Per-turn ``max_tokens`` for the UND lane.

    Args:
        context_used: ``len(prompt_ids) + len(response_ids)`` already in the episode.
        max_context_tokens: the episode budget (``prompt_length + response_length``).
        fraction: share of the remaining budget one turn may take; ``1.0`` disables the cap.

    Returns:
        The per-turn token cap, or 0 when the budget is already spent."""
    remaining = int(max_context_tokens) - int(context_used)
    if remaining <= 0:
        return 0
    if fraction >= 1.0:
        return remaining
    if fraction <= 0.0:
        raise ValueError(f"und turn fraction must be in (0, 1], got {fraction!r}")
    # ceil: never hand back 0 while there is room, or the turn would be a no-op that still
    # consumes an episode step.
    share = max(1, math.ceil(remaining * float(fraction)))
    return min(remaining, share)


#: Prompt for the AR-replica health canary. Deliberately trivial: a working language model
#: answers it in a handful of varied tokens, so anything else is a finding.
UND_HEALTH_CANARY: str = "Name three fruits."

#: A canary answer shorter than this cannot be judged (``""`` is degenerate on its own; a
#: 2-3 token answer is just a terse model).
UND_HEALTH_MIN_WORDS: int = 6

#: Share of the canary answer one repeated token may occupy before the replica is called
#: degenerate. A collapsed replica emits ``' . . . . . .'`` (share ~1.0); a healthy answer
#: like ``'Apples, bananas, and cherries.'`` sits near 0.14.
UND_HEALTH_REPEAT_SHARE: float = 0.4


def und_health_verdict(
    text: str,
    *,
    min_words: int = UND_HEALTH_MIN_WORDS,
    repeat_share: float = UND_HEALTH_REPEAT_SHARE,
) -> tuple[bool, float]:
    """Judge an AR-replica canary answer as coherent or degenerate.

    Args:
        text: the decoded canary answer.
        min_words: answers shorter than this are unjudgeable and called degenerate.
        repeat_share: a single repeated token taking more than this share is a loop.

    Returns:
        ``(degenerate, top_token_share)``. An empty answer is ``(True, 1.0)``."""
    words = text.split()
    if not words:
        return True, 1.0
    counts: dict[str, int] = {}
    for word in words:
        key = word.lower().strip(",.")
        counts[key] = counts.get(key, 0) + 1
    top_share = max(counts.values()) / len(words)
    degenerate = len(words) < min_words or top_share > repeat_share
    return degenerate, top_share


#: A UND turn must carry at least this many tokens before a repeat can be called a loop. Shorter
#: ``continue`` turns are legitimate (a terse plan, a clarifying sentence); a collapsed replica
#: emitted ``assistant\n`` hundreds of times.
UND_LOOP_MIN_TOKENS: int = 6


def und_turn_kind(text: str) -> str:
    """Classify an UND decode: ``generate_image``, ``judge_image``, ``done``, or ``continue``."""
    tagged = parse_hermes_tool_call(text)
    call = tagged if tagged is not None else parse_und_tool_call(text)
    if call is not None:
        name = str(call.get("name", ""))
        normalized = name.strip().strip(".\"' ").lower()
        if normalized == _GENERATE_IMAGE_TOOL:
            return "generate_image"
        if normalized in _STOP_TOOL_NAMES:
            logger.debug("bagel_corl_und_stop_tool name=%s (terminal signal)", name)
            return "done"
        # ``judge_image`` is a real loop tool: ``run_serial_episode`` answers it with the
        # in-loop RM verdict on the last generated image (Mode-2a judge observation).
        if normalized == _JUDGE_IMAGE_TOOL:
            return "judge_image"
        if tagged is not None:
            raise ValueError(f"Bagel CoRL UND emitted unsupported tool {name!r}; Qwen/other tools are fail-closed")
        logger.debug("bagel_corl_und_inert_tool name=%s (no loop tool; RM judges)", name)
        return "continue"
    if _DONE_RE.search(text.strip()):
        return "done"
    if _is_terminal_only(text):
        return "done"
    return "continue"


def _is_terminal_only(text: str) -> bool:
    """True when a decode carries only control markers / the end-of-conversation line."""
    residue = _SPECIAL_TOKEN_RE.sub(" ", text).strip().strip(".").strip()
    return not residue or residue.lower() == _END_OF_CONVERSATION


def bind_episode_ids(
    *,
    dataset_task_uid: str,
    episode_uid: str | None = None,
    policy_version: int = 0,
    gen_call_id: str | None = None,
) -> dict[str, str | int]:
    """RFC component-1 IDs. FlowGRPO groups by ``gen_group_uid``, never by semantic slot."""
    episode_uid = episode_uid or str(uuid.uuid4())
    gen_call_id = gen_call_id or str(uuid.uuid4())
    return {
        "dataset_task_uid": dataset_task_uid,
        "und_group_uid": dataset_task_uid,
        "episode_uid": episode_uid,
        "policy_version": int(policy_version),
        "gen_call_id": gen_call_id,
        "gen_group_uid": gen_call_id,
    }


def gen_sample_uid(gen_call_id: str, seed_index: int) -> str:
    return f"{gen_call_id}:{seed_index}"


def conditioning_uid(token_ids: list[int]) -> str:
    """RFC §4.4.2a ``cond_uid``: content hash of the conditioning token slice."""
    digest = hashlib.sha256()
    for token in token_ids:
        digest.update(int(token).to_bytes(8, "big", signed=True))
    return digest.hexdigest()


@dataclass
class GenSample:
    """One of S FlowGRPO seeds for a single ``generate_image`` turn."""

    gen_sample_uid: str
    gen_group_uid: str
    seed_index: int
    valid: bool
    prompt_token_ids: list[int]
    all_latents: Any | None = None
    timesteps: Any | None = None
    rollout_log_probs: Any | None = None
    rm_score: float | None = None
    rm_score_source: str | None = None
    image_path: str | None = None
    call_role: str = "initial"
    gen_call_index: int | None = None
    good_enough: bool | None = None
    # Per-axis RM breakdown when the backend reports one (UnifiedReward 2.0:
    # ``alignment`` / ``coherence`` / ``style`` on 1-5). Feeds the judge observation only;
    # ``rm_score`` stays the reward.
    rm_axes: dict[str, float] | None = None
    # Mode-2a rubric verdict (``vlm_judge`` backend): ``correctness`` / ``aesthetics`` on
    # [0, 1], ``findings``, ``suggested_fixes``, ``rubber_stamp`` and the facet grids.
    rm_judge: dict[str, Any] | None = None
    # RFC §4.4.2a conditioning identity. Derived by the GEN path from the token
    # slice it actually encodes; constant across the S seeds of one call (that is
    # the reuse group R2 exploits) and equal across calls whose conditioning
    # coincides. ``None`` when the serving path did not report it.
    cond_uid: str | None = None
    cond_len: int = 0
    rollout_policy_version: int | None = None
    rollout_replica_id: str | int | None = None
    serving_provenance_verified: bool = False


@dataclass
class EpisodeRollout:
    """One serial UND episode (zero or more GEN calls).

    Runtime counters (RFC): ``turns`` is in-episode ``J`` (UND policy turns);
    ``num_gen_calls`` is in-episode ``K`` (``generate_image`` invocations). Invariant ``J >= K``.
    """

    und_group_uid: str
    episode_uid: str
    policy_version: int | None
    prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]
    turns: int
    # π_rollout per response position (the AR replica's own sampling log-probs). Published to the
    # TransferQueue as ``rollout_log_probs`` so the trainer's π_rollout/π_θ correction has a
    # rollout distribution; empty on a replay that never sampled (tests, pattern-3 stubs).
    rollout_log_probs: list[float] = field(default_factory=list)
    requested_policy_version: int | None = None
    rollout_replica_id: str | int | None = None
    serving_provenance_verified: bool = False
    und_serving_provenance: list[dict[str, Any]] = field(default_factory=list)
    gen_samples: list[GenSample] = field(default_factory=list)
    used_image_credit: bool = False
    forced_reflection: bool = False
    autonomous_done: bool = False
    corl_reward_version: str = "v1"
    bagel_corl_reward: dict[str, Any] = field(default_factory=dict)
    und_reward: float = 0.0
    terminal_reward: float | None = None
    judge_text: str | None = None
    stop_required: bool = False
    num_gen_calls: int = 0
    # In-memory observations for inspecting the episode's tool protocol.
    turn_trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """π_rollout must stay positionally aligned with the response it describes."""
        if self.rollout_log_probs and len(self.rollout_log_probs) != len(self.response_ids):
            raise ValueError(
                "Bagel Co-RL rollout_log_probs must align with response_ids "
                f"(got {len(self.rollout_log_probs)} for {len(self.response_ids)} tokens)"
            )


class BagelGenerateImageTool:
    """One GEN turn per call (RFC ``K`` += 1)."""

    def __init__(
        self,
        *,
        gen_samples_per_call: int,
        max_generate_passes: int = 1,
        generate_fn: Callable[..., Any] | None = None,
        seed_base: int = 0,
    ):
        if gen_samples_per_call < 1:
            raise ValueError("gen_samples_per_call must be >= 1")
        if max_generate_passes < 1:
            raise ValueError("max_generate_passes must be >= 1")
        self.s = int(gen_samples_per_call)
        self.max_generate_passes = int(max_generate_passes)
        self._passes = 0
        self._generate_fn = generate_fn
        # Episode-scoped seed base + call counter for the default FlowGRPO seeds.
        self._seed_base = int(seed_base)
        self._call_index = 0

    def remaining_passes(self) -> int:
        return max(0, self.max_generate_passes - self._passes)

    def _default_seeds(self, call_index: int) -> list[int]:
        """Fresh FlowGRPO seeds for one ``generate_image`` turn."""
        return [derive_rollout_seed(self._seed_base, call_index * self.s + seed_index) for seed_index in range(self.s)]

    async def __call__(
        self,
        *,
        prompt: str,
        prompt_token_ids: list[int],
        gen_call_id: str,
        seeds: list[int] | None = None,
    ) -> list[GenSample]:
        if self._passes >= self.max_generate_passes:
            raise GenerateImageCapError(
                f"max_generate_passes={self.max_generate_passes} refuses a second generate_image call"
            )
        self._passes += 1
        call_index = self._call_index
        self._call_index += 1
        if seeds is None:
            seeds = self._default_seeds(call_index)
        if len(seeds) != self.s:
            raise ValueError(f"expected S={self.s} FlowGRPO seeds for one GEN turn, got {len(seeds)}")
        raw_rows: list[dict[str, Any]]
        if self._generate_fn is None:
            raw_rows = [{"valid": True} for _ in seeds]
        else:
            maybe = self._generate_fn(
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                seeds=seeds,
                gen_call_id=gen_call_id,
            )
            raw_rows = await maybe if asyncio.iscoroutine(maybe) else maybe
        samples: list[GenSample] = []
        for seed_index, (seed, row) in enumerate(zip(seeds, raw_rows, strict=True)):
            valid = bool(row.get("valid", True))
            cond_uid = row.get("cond_uid")
            samples.append(
                GenSample(
                    gen_sample_uid=gen_sample_uid(gen_call_id, seed_index),
                    gen_group_uid=gen_call_id,
                    seed_index=seed_index,
                    valid=valid,
                    prompt_token_ids=list(prompt_token_ids),
                    all_latents=row.get("all_latents"),
                    timesteps=row.get("timesteps"),
                    rollout_log_probs=row.get("rollout_log_probs"),
                    image_path=row.get("image_path"),
                    call_role="initial",
                    cond_uid=str(cond_uid) if cond_uid else None,
                    cond_len=int(row.get("cond_len") or 0),
                    **serving_rollout_provenance(row),
                )
            )
            _ = seed
        return samples


def compact_image_observation(path: str) -> str:
    """UND-facing observation: path only; S seed trajectories stay on the GEN batch."""
    return f"path={path}"


def judge_text_from_gen_samples(
    samples: list[GenSample],
    *,
    reduction: str = "any",
    threshold: float | None = None,
) -> str | None:
    """Reduce RM scores into Mode-2a judge text for ``build_forced_reflection``.

    Args:
        samples: GEN samples of one ``generate_image`` call (up to ``S`` seeds).
        reduction: episode-level stop bit over per-seed explicit flags —
            ``"any"`` (default, best-of-S), ``"all"``, or ``"mean"``
            (fraction of YES flags >= ``threshold``).
        threshold: ``good_enough_threshold`` from the single SoT
            (``agentic_image_gen.good_enough_threshold``); ``None`` keeps the
            turn-protocol default for callers that did not wire the knob."""
    scores = [float(s.rm_score) for s in samples if s.valid and s.rm_score is not None]
    if not scores:
        return None
    mean = float(sum(scores) / len(scores))
    explicit = [bool(s.good_enough) for s in samples if s.good_enough is not None]
    if explicit:
        if reduction == "all":
            good_enough = all(explicit)
        elif reduction == "mean":
            effective_threshold = float(threshold) if threshold is not None else 0.7
            good_enough = (sum(1.0 for flag in explicit if flag) / len(explicit)) >= effective_threshold
        else:
            good_enough = any(explicit)
    else:
        good_enough = derive_good_enough_from_scores(
            correctness=mean,
            aesthetics=mean,
            similarity=None,
            **({"threshold": float(threshold)} if threshold is not None else {}),
        )
    for sample in samples:
        if sample.good_enough is None:
            sample.good_enough = good_enough
    correctness, aesthetics = mean, mean
    findings, fixes = "", ""
    rubber_stamp = False
    judges = [s.rm_judge for s in samples if s.valid and s.rm_judge]
    axes = _mean_rm_axes(samples)
    if judges:
        correctness = float(np.mean([float(j.get("correctness", 0.0)) for j in judges]))
        aesthetics = float(np.mean([float(j.get("aesthetics", 0.0)) for j in judges]))
        # With S > 1 seeds, explain the weakest image: that is the one the rewrite must fix.
        worst = min(judges, key=lambda j: float(j.get("correctness", 0.0)) + float(j.get("aesthetics", 0.0)))
        findings = str(worst.get("findings") or "")
        fixes = str(worst.get("suggested_fixes") or "")
        rubber_stamp = any(bool(j.get("rubber_stamp")) for j in judges)
    elif axes is not None:
        correctness, aesthetics, findings, fixes = _judge_facets_from_axes(axes)
    return format_rm_scores_as_judge_text(
        correctness=correctness,
        aesthetics=aesthetics,
        good_enough=good_enough,
        findings=findings,
        suggested_fixes=fixes,
        rubber_stamp=rubber_stamp,
        similarity=None,
    )


_UNIFIED_AXES = ("alignment", "coherence", "style")
_AXIS_FINDINGS = {
    "alignment": "content vs the prompt",
    "coherence": "glitches / distorted objects",
    "style": "aesthetic appeal",
}
_AXIS_FIXES = {
    "alignment": "state every requested subject, attribute, count and relation explicitly in the diffusion prompt",
    "coherence": "simplify the composition and describe shapes, poses and spatial layout concretely",
    "style": "specify the medium, lighting, palette and camera framing",
}


def _mean_rm_axes(samples: list[GenSample]) -> dict[str, float] | None:
    """Average the UnifiedReward axes over the call's valid seeds, or ``None`` if absent."""
    rows = [s.rm_axes for s in samples if s.valid and s.rm_axes and set(_UNIFIED_AXES) <= set(s.rm_axes)]
    if not rows:
        return None
    return {axis: float(sum(float(r[axis]) for r in rows) / len(rows)) for axis in _UNIFIED_AXES}


def _judge_facets_from_axes(axes: dict[str, float]) -> tuple[float, float, str, str]:
    """Map 1-5 UnifiedReward axes onto the Mode-2a judge facets.

    ``correctness`` is the alignment axis and ``aesthetics`` the mean of coherence and style,
    both on [0, 1]; the findings name every axis and the fix targets the weakest one, so the
    reflection has something concrete to rewrite from.
    """

    def unit(value: float) -> float:
        return (min(max(float(value), 1.0), 5.0) - 1.0) / 4.0

    correctness = unit(axes["alignment"])
    aesthetics = (unit(axes["coherence"]) + unit(axes["style"])) / 2.0
    findings = "; ".join(f"{axis}={axes[axis]:.1f}/5 ({_AXIS_FINDINGS[axis]})" for axis in _UNIFIED_AXES)
    weakest = min(_UNIFIED_AXES, key=lambda axis: axes[axis])
    fixes = f"{weakest} is weakest: {_AXIS_FIXES[weakest]}" if axes[weakest] < 5.0 else ""
    return correctness, aesthetics, f"UnifiedReward axes {findings}", fixes


def _encode_text(tokenizer: Any | None, text: str, *, fallback_ids: list[int] | None = None) -> list[int]:
    if tokenizer is not None:
        return list(tokenizer.encode(text, add_special_tokens=False))
    return list(fallback_ids or [])


async def run_serial_episode(
    *,
    dataset_task_uid: str,
    policy_version: int,
    prompt_ids: list[int],
    und_decode: Callable[..., Any],
    generate_tool: BagelGenerateImageTool,
    score_fn: Callable[[list[GenSample]], Any] | None = None,
    build_judge_text_fn: Callable[[list[GenSample]], str | None] | None = None,
    tokenizer: Any | None = None,
    user_prompt: str = "",
    max_und_turns: int = 8,
    max_context_tokens: int | None = None,
    max_response_tokens: int | None = None,
    forced_reflection_text: str = "Done.",
    episode_uid: str | None = None,
    non_image_reward: float | None = None,
    good_enough_threshold: float | None = None,
    good_enough_reduction: str = "any",
    corl_reward_version: str = "v1",
    fixed_gen_prompt: str = "",
    task_type: str | None = None,
    expected_num_images: int = 1,
) -> EpisodeRollout:
    """Serial UND turn(s) → optional one GEN turn (S FlowGRPO seeds) → RM → reflection / Done."""
    ids = bind_episode_ids(
        dataset_task_uid=dataset_task_uid,
        episode_uid=episode_uid,
        policy_version=policy_version,
    )
    with episode_scope(current_episode() or EpisodeState(episode_id=ids["episode_uid"], user_prompt=user_prompt)):
        response_ids: list[int] = []
        response_mask: list[int] = []
        # π_rollout per response position, from the AR replica's own sampling. Kept in lockstep
        # with ``response_ids``/``response_mask`` by ``_append_response`` below.
        response_log_probs: list[float] = []
        gen_samples: list[GenSample] = []
        used_image_credit = False
        forced = False
        autonomous_done = False
        if corl_reward_version not in ("v1", "v2"):
            raise ValueError("corl_reward_version must be v1 or v2")
        turns = 0
        num_gen_calls = 0
        judge_text: str | None = None
        stop_required = False
        planned_images = max(1, int(expected_num_images)) if str(task_type or "").strip().lower() == "plan" else 1
        images_rendered = 0
        early_judge_declined = False
        # Per-turn trajectory trace (see ``EpisodeRollout.turn_trace``). Collected even when
        # ``BAGEL_CORL_DEBUG`` is off: the live per-turn ``logger.info`` below is the only
        # other place the raw UND text surfaces, and it is exactly the text that explains a
        # K=0 episode -- so the dump must not depend on the debug flag being set.
        trace: list[dict[str, Any]] = []
        clear_good_enough_yes_reached()
        active_user_prompt = user_prompt or get_active_user_prompt() or ""
        if _BAGEL_CORL_TURN_DEBUG:
            # Re-assert INFO logging per episode: the first decode of the step lazily imports
            # ``vllm``, whose logger setup reconfigures the *root* logger (``force=True``) and
            # would otherwise drop every INFO/WARNING line emitted from here on -- the exact
            # failure that made a healthy rollout look wedged (see ``_force_info_logging``).
            _force_info_logging()

        observed_und_version: int | None = None
        observed_und_replicas: set[str | int] = set()
        und_provenance: list[dict[str, Any]] = []
        all_und_provenance_verified = True

        def _observe_serving(step: dict[str, Any]) -> None:
            nonlocal observed_und_version, all_und_provenance_verified
            provenance = serving_rollout_provenance(step)
            version = provenance["rollout_policy_version"]
            if version is not None:
                if observed_und_version is not None and version != observed_und_version:
                    raise ValueError("Synchronous episode mixed observed UND serving policy versions")
                observed_und_version = version
            replica = provenance["rollout_replica_id"]
            if replica is not None:
                observed_und_replicas.add(replica)
            all_und_provenance_verified &= provenance["serving_provenance_verified"]
            und_provenance.append(
                {
                    **provenance,
                    "response_start": len(response_ids),
                    "response_end": len(response_ids) + len(step.get("token_ids") or []),
                }
            )

        def _check_gen_serving(samples: list[GenSample]) -> None:
            for sample in samples:
                version = sample.rollout_policy_version
                if version is not None and observed_und_version is not None and version != observed_und_version:
                    raise ValueError("Synchronous episode mixed GEN and UND serving policy versions")

        def _context_budget_spent() -> bool:
            """True once ``prompt_ids + response_ids`` has used up ``max_context_tokens``.

            One predicate for both decode sites below: the engine rejects a prompt with no
            room left for generation, so neither may be attempted past the budget.
            """
            if max_response_tokens is not None and len(response_ids) >= max_response_tokens:
                return True
            return max_context_tokens is not None and len(prompt_ids) + len(response_ids) >= max_context_tokens

        def _append_response(tokens: Any, mask_value: int, log_probs: Any | None = None) -> None:
            """Extend the episode with ``tokens``, keeping π_rollout aligned with ``response_ids``.

            Args:
                tokens: token ids (or tokens already extracted from a decode step).
                mask_value: 1 for policy tokens, 0 for forced/environment tokens.
                log_probs: the AR engine's per-token log-probs, if this decode sampled them."""
            token_list = [int(t) for t in tokens]
            response_ids.extend(token_list)
            response_mask.extend([int(mask_value)] * len(token_list))
            score = None if log_probs is None else [float(v) for v in log_probs]
            if score is not None and len(score) != len(token_list):
                raise ValueError(
                    "Bagel Co-RL UND pass: rollout_log_probs must align with response_ids "
                    f"(got {len(score)} log-probs for {len(token_list)} tokens)"
                )
            response_log_probs.extend(score if score is not None else [0.0] * len(token_list))

        # (judge text, judged image path, the GEN call's samples): the RM verdict on the last
        # generate_image call, held until the policy asks for it with ``judge_image``.
        pending_judge: tuple[str, str, list[GenSample]] | None = None

        def _append_env_turn(body: str, policy_text: str, fallback_ids: list[int]) -> None:
            """Answer the policy's tool call with a ``<tool_response>`` user turn (mask 0)."""
            close = "" if policy_text.rstrip().endswith(_IM_END) else _IM_END
            obs_ids = _encode_text(
                tokenizer,
                close + TOOL_RESPONSE_TEMPLATE.format(body=body),
                fallback_ids=fallback_ids,
            )
            if obs_ids:
                _append_response(obs_ids, 0)
            trace.append({"record": "env_obs", "turn": turns, "text": body})

        async def _reflect_on_judge(verdict: str, judged_samples: list[GenSample], judged_step: dict[str, Any]) -> bool:
            """Inject the Mode-2a reflection after a judge response; True ends the episode."""
            nonlocal forced, stop_required
            remaining = generate_tool.remaining_passes()
            reflection = build_forced_reflection(
                verdict,
                force_done=remaining == 0,
                generate_pass=generate_tool.max_generate_passes - remaining,
                max_passes=generate_tool.max_generate_passes,
            )
            if reflection is None:
                return False
            reflection_text, stop_required = reflection
            forced_ids = _encode_text(
                tokenizer,
                reflection_text + "\n",
                fallback_ids=list(judged_step.get("forced_token_ids") or []),
            )
            if forced_ids:
                forced = True
                _append_response(forced_ids, 0)
            trace.append({"record": "reflection", "turn": turns, "text": reflection_text})
            reached = (
                ("good_enough=YES" in verdict)
                if corl_reward_version == "v2"
                else (
                    any(s.good_enough for s in judged_samples if s.good_enough is not None)
                    or "good_enough=YES" in verdict
                )
            )
            state = current_episode()
            judged_call = judged_samples[-1].gen_group_uid if judged_samples else None
            if state is not None and any(record.call_id == judged_call for record in state.generations):
                state.record_judgment(judged_call, good_enough=reached)
            elif reached:
                set_good_enough_yes_reached(True)
            if not stop_required:
                return False
            if _context_budget_spent():
                # Same rule as the turn loop: with the context spent there is no room for the
                # closing ``Done.`` decode, so take the tokenizer fallback below instead of
                # asking the engine for a turn it cannot generate.
                done_step: dict[str, Any] = {}
            else:
                done_decode = und_decode(prompt_ids=prompt_ids, response_ids=response_ids)
                done_step = await done_decode if asyncio.iscoroutine(done_decode) else done_decode
                _observe_serving(done_step)
            done_ids = list(done_step.get("token_ids") or [])
            done_text = str(done_step.get("text") or "")
            try:
                done_kind = und_turn_kind(done_text)
            except ValueError as exc:
                # A stop decode is the one place the fail-closed guard has nothing to protect:
                # the episode is ending either way, and the fallback below appends the
                # tokenizer's literal ``Done.``. Raising here instead dropped the whole episode
                # (e.g. a ``{"name": "Done"}`` payload) and shrank the batch with a synthetic pad.
                logger.warning("bagel_corl_und_done_decode_unparsed err=%s; using Done. fallback", exc)
                done_kind = "continue"
            if done_step:
                trace.append(
                    {
                        "record": "und_turn",
                        "turn": turns,
                        "kind": done_kind,
                        "req_ctx": len(prompt_ids) + len(response_ids),
                        "out_tokens": len(done_ids),
                        "text": done_text,
                    }
                )
            if done_ids and done_kind == "done":
                _append_response(done_ids, 1, done_step.get("log_probs"))
            else:
                fallback = _encode_text(
                    tokenizer,
                    "Done.",
                    fallback_ids=list(done_step.get("done_token_ids") or judged_step.get("done_token_ids") or []),
                )
                if fallback:
                    _append_response(fallback, 0 if corl_reward_version == "v2" else 1)
            return True

        # Defined before the loop so the budget guard below can break on its very first
        # iteration (a prompt that already fills the context) and still leave the post-loop
        # fallbacks that read ``step`` well-defined. A valid launch cannot reach that: the
        # framework caps the prompt at ``rollout.prompt_length`` while the budget is
        # ``prompt_length + response_length``.
        step: dict[str, Any] = {}

        for _ in range(max_und_turns):
            if _context_budget_spent():
                # Episode context exhausted: the next decode would leave the AR engine no
                # room for a single token. Keep the turns that fit and fall through to the
                # forced reflection / Done handling rather than raising.
                break
            turns += 1
            if _BAGEL_CORL_TURN_DEBUG:
                # Logged *before* the await: when an episode stops making progress, the last
                # line here names the turn that never returned, which the after-the-fact log
                # below cannot do. ``ctx`` is what the engine is asked to prefill this turn.
                logger.info(
                    "bagel_corl_turn episode=%s turn=%d req_ctx=%d resp_tokens=%d",
                    ids["episode_uid"][:8],
                    turns,
                    len(prompt_ids) + len(response_ids),
                    len(response_ids),
                )
            _turn_t0 = time.perf_counter()
            _req_ctx = len(prompt_ids) + len(response_ids)
            decode = und_decode(prompt_ids=prompt_ids, response_ids=response_ids)
            step = await decode if asyncio.iscoroutine(decode) else decode
            _observe_serving(step)
            token_ids: list[int] = list(step["token_ids"])
            text: str = str(step["text"])
            kind = und_turn_kind(text)
            trace.append(
                {
                    "record": "und_turn",
                    "turn": turns,
                    "kind": kind,
                    "req_ctx": _req_ctx,
                    "out_tokens": len(token_ids),
                    "decode_s": round(time.perf_counter() - _turn_t0, 3),
                    "text": text,
                }
            )
            if _BAGEL_CORL_TURN_DEBUG:
                logger.info(
                    "bagel_corl_turn episode=%s turn=%d kind=%s out_tokens=%d decode_s=%.1f text=%r",
                    ids["episode_uid"][:8],
                    turns,
                    kind,
                    len(token_ids),
                    time.perf_counter() - _turn_t0,
                    # The raw UND text is the only way to tell a model that never emits the
                    # Hermes grammar from one whose ``<tool_call>`` block is being dropped or
                    # mangled on the way out of the engine (see the tools-schema note in
                    # ``agentic_und_gen.agent_loop.BagelMultiturnAgentLoop.run``). Truncated: one line per turn.
                    text[:240],
                )
            _append_response(token_ids, 1, step.get("log_probs"))
            if kind == "done":
                remaining = generate_tool.remaining_passes()
                if remaining > 0 and not get_good_enough_yes_reached() and not _is_terminal_only(text):
                    # The episode ends at min(max_generate_passes, first good_enough=YES); an
                    # earlier Done. is declined so the policy rewrites instead of quitting on NO.
                    # A silent turn (bare ``<|im_end|>``) is not a Done the policy chose to write:
                    # re-sampling it produced prompt fragments repeated to ``max_und_turns``
                    # (``sample_9003.01``, step 82), so it ends the episode as before.
                    next_step = (
                        "call judge_image on the new image"
                        if pending_judge is not None
                        else "rewrite the diffusion prompt from the judge's findings and call generate_image"
                    )
                    _append_env_turn(
                        "agentic_stop ok=0\n"
                        f"good_enough=YES has not been reached and {remaining} of "
                        f"{generate_tool.max_generate_passes} generate_image calls remain: {next_step}. "
                        "Done. is accepted after good_enough=YES or the final generate_image call.",
                        text,
                        list(step.get("obs_token_ids") or []),
                    )
                    continue
                autonomous_done = not forced
                break
            if kind == "judge_image":
                if pending_judge is None:
                    _append_env_turn(
                        "agentic_judge ok=0 parse_ok=0 stub=0 backend=reward_loop\n"
                        "No new image to judge: call generate_image first.",
                        text,
                        list(step.get("obs_token_ids") or []),
                    )
                    continue
                if (
                    not early_judge_declined
                    and images_rendered < planned_images
                    and generate_tool.remaining_passes() > 0
                ):
                    # Declined once only, so a policy that keeps judging still gets a verdict
                    # within ``max_und_turns``.
                    early_judge_declined = True
                    _append_env_turn(
                        "agentic_judge ok=0 parse_ok=1 stub=0 backend=reward_loop\n"
                        f"Plan step {images_rendered}/{planned_images}: call generate_image for subtask "
                        f"{images_rendered + 1} first. judge_image runs once, after the final subtask image.",
                        text,
                        list(step.get("obs_token_ids") or []),
                    )
                    continue
                judged_text, judged_path, judged_samples = pending_judge
                pending_judge = None
                _append_env_turn(
                    f"VL judge on the last generated image:\n  path={judged_path}\n{judged_text}",
                    text,
                    list(step.get("obs_token_ids") or []),
                )
                if await _reflect_on_judge(judged_text, judged_samples, step):
                    break
                continue
            if kind != "generate_image":
                continue

            if get_good_enough_yes_reached():
                # Env hard-stop: prior YES latch blocks further generate_image.
                break
            if generate_tool.remaining_passes() <= 0:
                _append_env_turn(
                    f"agentic_tool ok=0 images=0 backend=vllm_omni\n"
                    f"generate_image pass cap ({generate_tool.max_generate_passes}) reached; reply Done.",
                    text,
                    list(step.get("obs_token_ids") or []),
                )
                continue

            call = parse_und_tool_call(text) or {}
            arguments = call.get("arguments", call.get("parameters", {}))
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            # Strip any in-turn "Short reflection: ..." tail before the prompt conditions the
            # diffusion model -- see ``strip_reflection_tail``. Applies at extraction so the
            # recorded trajectory ``prompt``, ``build_generate_call_meta`` and the GEN request
            # all see the same sanitized text.
            prompt = strip_reflection_tail(str(arguments.get("prompt", "")))
            if fixed_gen_prompt:
                prompt = active_user_prompt if fixed_gen_prompt == "__user_request__" else fixed_gen_prompt
                if not prompt:
                    raise ValueError("fixed generator prompt requires a nonempty original request")
            meta = build_generate_call_meta(prompt=prompt, user_prompt=active_user_prompt)
            # RFC component-1 IDs: GEN TQ keys must be deterministic per (episode, call)
            # so a replay reproduces the same keys. Call 0 keeps the episode's base
            # ``gen_call_id``; later calls derive from the episode id and their local
            # index instead of a fresh uuid4, which left every K > 1 key unreproducible
            # (``num_gen_calls`` is the 0-based local call index here; it increments below).
            gen_call_id = (
                str(ids["gen_call_id"]) if generate_tool._passes == 0 else f"{ids['episode_uid']}:call{num_gen_calls}"
            )
            num_gen_calls += 1
            call_samples = await generate_tool(
                prompt=prompt,
                prompt_token_ids=list(prompt_ids) + list(response_ids),
                gen_call_id=gen_call_id,
            )
            call_samples = sorted(call_samples, key=lambda sample: sample.seed_index)
            _check_gen_serving(call_samples)
            call_role = str(meta.get("call_role") or "initial")
            for sample in call_samples:
                sample.call_role = call_role
                sample.gen_call_index = num_gen_calls - 1

            valid_paths = [s.image_path for s in call_samples if s.valid and s.image_path]
            if valid_paths:
                register_tool_artifact(prompt=prompt, paths=[str(p) for p in valid_paths], call_id=gen_call_id)
            trace.append(
                {
                    "record": "gen_call",
                    "turn": turns,
                    "gen_call_id": gen_call_id,
                    "call_role": call_role,
                    "prompt": prompt,
                    "num_samples": len(call_samples),
                    "num_valid": sum(1 for s in call_samples if s.valid),
                    "image_paths": [str(p) for p in valid_paths],
                }
            )

            if score_fn is not None:
                scored = score_fn(call_samples)
                if asyncio.iscoroutine(scored):
                    scored = await scored
                call_samples = scored
                _check_gen_serving(call_samples)

            # Accumulate — every generate_image call contributes its S seeds to the
            # episode (RFC: all K calls survive to gen_batch). Replacing the list here
            # silently dropped all but the last call's trajectories for K > 1.
            gen_samples.extend(call_samples)

            if valid_paths:
                used_image_credit = True
                images_rendered += 1
                observation = (
                    f"vLLM-Omni generated the requested image. {compact_image_observation(valid_paths[0])} "
                    f"agentic_tool ok=1 images={len(valid_paths)} backend=vllm_omni"
                )
                if images_rendered < planned_images and generate_tool.remaining_passes() > 0:
                    observation += (
                        f"\nPlan step {images_rendered}/{planned_images} rendered. Next: call generate_image "
                        f"with the complete prompt for subtask {images_rendered + 1} of your plan."
                    )
            else:
                observation = (
                    "agentic_tool ok=0 images=0 backend=vllm_omni\n"
                    "Image generation failed; rewrite the prompt and call generate_image again."
                )

            call_judge_text: str | None = None
            if build_judge_text_fn is not None:
                maybe_text = build_judge_text_fn(call_samples)
                if asyncio.iscoroutine(maybe_text):
                    maybe_text = await maybe_text
                call_judge_text = maybe_text
            if not call_judge_text:
                call_judge_text = judge_text_from_gen_samples(
                    call_samples,
                    reduction="all" if corl_reward_version == "v2" else good_enough_reduction,
                    threshold=good_enough_threshold,
                )

            if valid_paths or call_judge_text:
                _append_env_turn(observation, text, list(step.get("obs_token_ids") or [0]))
            if call_judge_text:
                judge_text = call_judge_text
                # Answered on the policy's ``judge_image`` call, as in the Mode-2a loop.
                pending_judge = (call_judge_text, valid_paths[0] if valid_paths else "", call_samples)
                continue

            # No judge text (no in-loop RM): keep legacy forced/done token fallbacks.
            forced_ids = list(step.get("forced_token_ids") or [])
            if forced_ids:
                forced = True
                _append_response(forced_ids, 0)
            else:
                done_ids = list(step.get("done_token_ids") or [])
                if done_ids:
                    _append_response(done_ids, 1)
            break
        else:
            forced = True
            _ = forced_reflection_text

        trusted_fields = (
            trusted_image_reward_fields(
                gen_samples, num_gen_calls=num_gen_calls, autonomous_done=autonomous_done, forced_reflection=forced
            )
            if corl_reward_version == "v2"
            else {}
        )
        image_credit = (
            trusted_fields.get("image_reward_objective") if trusted_fields else image_credit_reward(gen_samples)
        )
        if image_credit is not None:
            und_reward = image_credit
        elif non_image_reward is not None:
            # Pattern 3 (K=0): use the non-image UND scalar (RFC und/no_image_credit),
            # never a hard-coded 0 that would zero out token GRPO signal.
            und_reward = float(non_image_reward)
        else:
            und_reward = 0.0

        if _BAGEL_CORL_TURN_DEBUG:
            logger.info(
                "bagel_corl_episode episode=%s turns=%d K=%d resp_tokens=%d forced=%s stop_required=%s reward=%.3f",
                str(ids["episode_uid"])[:8],
                turns,
                num_gen_calls,
                len(response_ids),
                forced,
                stop_required,
                und_reward,
            )

        if max_response_tokens is not None and len(response_ids) > max_response_tokens:
            # An environment turn (judge response + reflection) can land past the budget after
            # the last decode that fit; those tail tokens are mask 0 or the closing ``Done.``.
            logger.warning(
                "bagel_corl_episode episode=%s response %d > response_length %d; truncating",
                str(ids["episode_uid"])[:8],
                len(response_ids),
                max_response_tokens,
            )
            del response_ids[max_response_tokens:]
            del response_mask[max_response_tokens:]
            del response_log_probs[max_response_tokens:]

        return EpisodeRollout(
            und_group_uid=str(ids["und_group_uid"]),
            episode_uid=str(ids["episode_uid"]),
            policy_version=observed_und_version,
            requested_policy_version=int(ids["policy_version"]),
            rollout_replica_id=next(iter(observed_und_replicas)) if len(observed_und_replicas) == 1 else None,
            serving_provenance_verified=bool(und_provenance)
            and all_und_provenance_verified
            and len(observed_und_replicas) == 1,
            und_serving_provenance=und_provenance,
            prompt_ids=list(prompt_ids),
            response_ids=response_ids,
            response_mask=response_mask,
            rollout_log_probs=response_log_probs,
            turns=turns,
            gen_samples=gen_samples,
            used_image_credit=used_image_credit,
            forced_reflection=forced,
            autonomous_done=autonomous_done,
            corl_reward_version=corl_reward_version,
            bagel_corl_reward=trusted_fields,
            und_reward=und_reward,
            judge_text=judge_text,
            stop_required=stop_required,
            num_gen_calls=num_gen_calls,
            turn_trace=trace,
        )


def trusted_image_reward_fields(samples, *, num_gen_calls, autonomous_done=False, forced_reflection=False):
    """Build the harness-only V2 envelope from RM-marked samples, never transcript text.

    Group first appearance is call chronology; seed order is explicit and stable.
    A failed/missing image score rejects the episode instead of fabricating signal.
    """
    calls = {}
    call_indices = {}
    for sample in samples:
        if (
            not sample.valid
            or sample.rm_score_source != "bagel_corl_tool_rm"
            or sample.rm_score is None
            or not math.isfinite(float(sample.rm_score))
        ):
            raise ValueError("V2 requires finite trusted tool RM scores for every seed")
        uid = str(sample.gen_group_uid)
        call_index = getattr(sample, "gen_call_index", None)
        if uid in call_indices and call_indices[uid] != call_index:
            raise ValueError("inconsistent call index in V2 group")
        call_indices[uid] = call_index
        group = calls.setdefault(uid, {})
        if sample.seed_index in group:
            raise ValueError("duplicate seed index in V2 call")
        group[sample.seed_index] = float(sample.rm_score)
    if len(calls) != num_gen_calls:
        raise ValueError("V2 trusted call count does not match generated calls")
    if calls and all(index is not None for index in call_indices.values()):
        if sorted(call_indices.values()) != list(range(num_gen_calls)):
            raise ValueError("V2 call indices must be unique and contiguous")
        calls = dict(sorted(calls.items(), key=lambda pair: call_indices[pair[0]]))
    seed_scores = [[row[i] for i in sorted(row)] for row in calls.values()]
    means = [sum(row) / len(row) for row in seed_scores]
    first, final, best = (means[0], means[-1], max(means)) if means else (None, None, None)
    return {
        "origin": "bagel_corl_tool_rm",
        "version": "v2",
        "call_means": means,
        "call_seed_scores": seed_scores,
        "image_reward_first": first,
        "image_reward_final": final,
        "image_reward_best": best,
        "image_reward_gain": final - first if means else None,
        "image_reward_source": "bagel_corl_tool_rm",
        "num_gen_calls": num_gen_calls,
        "autonomous_done": bool(autonomous_done),
        "forced_reflection": bool(forced_reflection),
        "image_reward_objective": final - 0.01 * max(0, num_gen_calls - 1) if means else None,
    }


def image_credit_reward(samples: list[GenSample]) -> float | None:
    """RFC ``und/no_image_credit`` image credit: the **final** pass plus the rewrite gain."""
    ordered: dict[str, list[float]] = {}
    for sample in samples or []:
        score = sample.rm_score
        if not sample.valid or score is None:
            continue
        ordered.setdefault(str(sample.gen_group_uid), []).append(float(score))
    if not ordered:
        return None
    calls = [float(np.mean(scores)) for scores in ordered.values()]
    final = calls[-1]
    improvement = max(0.0, final - calls[0]) if len(calls) > 1 else 0.0
    return float(final + min(1.0, improvement))


def episode_und_reward(episode: EpisodeRollout) -> tuple[float, bool]:
    """RFC ``und/no_image_credit``: the episode reward carried on the UND row."""
    if episode.corl_reward_version == "v2" and episode.terminal_reward is not None:
        return float(episode.terminal_reward), not episode.used_image_credit
    if episode.corl_reward_version == "v2":
        value = episode.bagel_corl_reward.get("image_reward_objective")
        return (float(value), False) if value is not None else (float(episode.und_reward), True)
    image_credit = image_credit_reward(episode.gen_samples)
    if image_credit is not None:
        return image_credit, False
    return float(episode.und_reward), True


def und_episode_rm_scores(response_mask: Any, reward: float) -> list[float]:
    """Per-token UND episode reward, placed on the last *trainable* response position."""
    if hasattr(response_mask, "ndim") and response_mask.ndim != 1:
        raise ValueError(
            "Bagel Co-RL UND rm_scores expects a 1-D response mask for one episode, got shape "
            f"{tuple(response_mask.shape)}"
        )
    mask = list(response_mask)
    valid = [index for index, flag in enumerate(mask) if bool(flag)]
    if not valid:
        raise ValueError(
            "Bagel Co-RL UND rm_scores: response_mask has no trainable position, so the episode "
            f"reward {float(reward)} could never reach the advantage computation"
        )
    scores = [0.0] * len(mask)
    scores[valid[-1]] = float(reward)
    return scores


def aggregate_episode_metrics(records: list[Any]) -> dict[str, float]:
    """Batch-aggregate per-episode metrics into step-level logging values."""
    j_values: list[float] = []
    k_values: list[float] = []
    dropped = 0.0
    no_credit_flags: list[float] = []
    skipped_flags: list[float] = []
    k_zero_flags: list[float] = []
    # RFC §4.4.4 R2: counts are additive over the batch, ratios are batch means
    # (each episode's ratio was already measured against the engine counters).
    cond_sums: dict[str, float] = {
        "gen/cond_cache_hits": 0.0,
        "gen/cond_cache_misses": 0.0,
        "gen/cond_cache_bypassed": 0.0,
    }
    cond_means: dict[str, list[float]] = {
        "gen/cond_recompute_ratio": [],
        "gen/prompt_embed_cache_hit_rate": [],
        "gen/cond_amortization": [],
    }
    for rec in records:
        fields = rec.get("fields") if isinstance(rec, dict) else rec
        if not isinstance(fields, dict):
            fields = {}
        metrics = fields.get("bagel_corl_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        j = fields.get("episode_J", metrics.get("episode/J"))
        k = fields.get("episode_K", metrics.get("episode/K"))
        if j is not None:
            j_values.append(float(j))
        if k is not None:
            k_values.append(float(k))
        dropped += float(metrics.get("gen/dropped_incomplete_groups", 0.0) or 0.0)
        if "und/no_image_credit" in metrics:
            no_credit_flags.append(float(metrics["und/no_image_credit"] or 0.0))
        if "gen/skipped_no_groups" in metrics:
            skipped_flags.append(float(metrics["gen/skipped_no_groups"] or 0.0))
        if "episode/K_zero" in metrics:
            k_zero_flags.append(float(metrics["episode/K_zero"] or 0.0))
        for key in cond_sums:
            if key in metrics:
                cond_sums[key] += float(metrics[key] or 0.0)
        for key, values in cond_means.items():
            if key in metrics:
                values.append(float(metrics[key] or 0.0))
    aggregated: dict[str, float] = {}
    if j_values:
        aggregated["episode/J"] = float(sum(j_values) / len(j_values))
    if k_values:
        aggregated["episode/K"] = float(sum(k_values) / len(k_values))
    aggregated["gen/dropped_incomplete_groups"] = float(dropped)
    if no_credit_flags:
        aggregated["und/no_image_credit"] = float(sum(no_credit_flags) / len(no_credit_flags))
    if skipped_flags:
        aggregated["gen/skipped_no_groups"] = float(sum(skipped_flags) / len(skipped_flags))
    if k_zero_flags:
        aggregated["episode/K_zero"] = float(sum(k_zero_flags) / len(k_zero_flags))
    if any(value > 0.0 for value in cond_sums.values()):
        aggregated.update(cond_sums)
    for key, values in cond_means.items():
        if values:
            aggregated[key] = float(sum(values) / len(values))
    return aggregated


# : Reward dims republished on every *training* update. : : The validation pass already logs all of these as ``val-
# aux/<lane>/<dim>/mean@1``, but : training published only the scalar ``critic/score/mean``.
TRAIN_REWARD_DIM_KEYS: tuple[str, ...] = (
    "reward_reflect",
    "reward_plan",
    "reward_format",
    "reward_protocol",
    "reward_tool",
    "reward_result",
    "reward_image",
    "reward_done",
    "reward_tool_call",
)


def _record_fields(record: Any) -> dict[str, Any]:
    """Fields of one TQ row, accepting the ``{"fields": ...}`` wrapper or a bare dict."""
    if not isinstance(record, dict):
        return {}
    fields = record.get("fields")
    if isinstance(fields, dict):
        return fields
    return {} if "fields" in record else record


def und_record_field_keys(records: list[Any], *, limit: int | None = None) -> list[str]:
    """Field names the first UND row carries, for diagnosing an empty reward breakdown."""
    if not records:
        return []
    keys = sorted(_record_fields(records[0]))
    return keys[:limit] if limit is not None else keys


def _reward_breakdown(fields: dict[str, Any]) -> dict[str, Any] | None:
    """The episode's per-dim reward blob, wherever the row carries it.

    ``_agent_loop_postprocess`` nests it as ``extra_fields.reward_extra_info``; the
    TensorDict branch of ``_und_records_from_batch`` flattens ``extra_fields`` keys onto
    the row, so both shapes reach here.
    """
    nested = fields.get("extra_fields")
    nested_blob = nested.get("reward_extra_info") if isinstance(nested, dict) else None
    for candidate in (fields.get("reward_extra_info"), nested_blob):
        if isinstance(candidate, dict):
            return candidate
    return None


def _as_float(value: Any) -> float | None:
    """``float(value)`` or ``None`` — a value that cannot be read is not a zero."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _episode_score(value: Any) -> float | None:
    """The episode scalar from a per-token score vector, mirroring the trainer's ``sum(-1)``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return _as_float(value)
    if isinstance(value, str | bytes) or value is None:
        return None
    try:
        import numpy as np
    except ImportError:  # numpy is a hard dependency of this tree; degrade rather than crash
        return _as_float(value)
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    return float(array.sum())


def _episode_reward(fields: dict[str, Any]) -> float | None:
    """The episode's trained reward, read the way the trainer reads it."""
    for key in ("rm_scores", "token_level_scores"):
        if key not in fields:
            continue
        value = _episode_score(fields.get(key))
        if value is not None:
            return value
    return _as_float(fields.get("und_reward"))


def aggregate_reward_breakdown_metrics(records: list[Any]) -> dict[str, float]:
    """Batch-aggregate the per-episode reward breakdown into step-level logging values."""
    dim_values: dict[str, list[float]] = {key: [] for key in TRAIN_REWARD_DIM_KEYS}
    with_image: list[float] = []
    without_image: list[float] = []
    for record in records:
        fields = _record_fields(record)
        if not fields:
            continue
        breakdown = _reward_breakdown(fields)
        if breakdown is not None:
            for key in TRAIN_REWARD_DIM_KEYS:
                value = _as_float(breakdown.get(key))
                if value is not None:
                    dim_values[key].append(value)
        score = _episode_reward(fields)
        credit = fields.get("used_image_credit")
        if score is None or credit is None:
            continue
        (with_image if bool(credit) else without_image).append(score)

    aggregated: dict[str, float] = {}
    for key in TRAIN_REWARD_DIM_KEYS:
        values = dim_values[key]
        if not values:
            continue
        metric = key.removeprefix("reward_")
        aggregated[f"bagel_corl/reward_dim/{metric}/mean"] = float(sum(values) / len(values))
        aggregated[f"bagel_corl/reward_dim/{metric}/min"] = float(min(values))
        aggregated[f"bagel_corl/reward_dim/{metric}/max"] = float(max(values))
    for name, values in (("valid", with_image), ("no_image", without_image)):
        if not values:
            continue
        aggregated[f"bagel_corl/reward_{name}/mean"] = float(sum(values) / len(values))
        aggregated[f"bagel_corl/reward_{name}/count"] = float(len(values))
    return aggregated


def _as_gen_sample(item: Any) -> GenSample | None:
    if isinstance(item, GenSample):
        return item
    if not isinstance(item, dict):
        return None
    return GenSample(
        gen_sample_uid=str(item.get("gen_sample_uid", "")),
        gen_group_uid=str(item.get("gen_group_uid", "")),
        seed_index=int(item.get("seed_index", 0)),
        valid=bool(item.get("valid", False)),
        prompt_token_ids=list(item.get("prompt_token_ids") or []),
        all_latents=item.get("all_latents"),
        timesteps=item.get("timesteps"),
        rollout_log_probs=item.get("rollout_log_probs"),
        rm_score=item.get("rm_score"),
        rm_score_source=item.get("rm_score_source"),
        image_path=item.get("image_path"),
        call_role=str(item.get("call_role", "initial")),
        gen_call_index=item.get("gen_call_index"),
        good_enough=item.get("good_enough"),
        cond_uid=str(item["cond_uid"]) if item.get("cond_uid") else None,
        cond_len=int(item.get("cond_len") or 0),
        **serving_rollout_provenance(item),
    )


def strip_pixels_for_actor(row: dict[str, Any]) -> dict[str, Any]:
    """Bagel GEN trains on ``prompt_token_ids``, not pixels or ``prompt_embeds``."""
    cleaned = dict(row)
    for key in ("images", "pixel_values", "prompt_embeds", "negative_prompt_embeds"):
        cleaned.pop(key, None)
    return cleaned
