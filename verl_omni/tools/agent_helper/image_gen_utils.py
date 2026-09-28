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

"""Readers used by ``ImageGenToolAgentLoop``, not by frozen tool bodies."""

from __future__ import annotations

import json
import re
from typing import Any

from verl_omni.tools.trajectory.hydra_env import agentic_get, agentic_get_bool
from verl_omni.utils.agentic.max_passes import max_generate_passes

__all__ = [
    "build_forced_reflection",
    "count_successful_generates",
    "count_successful_judges",
    "fits_response_budget",
    "force_first_generate_probability",
    "hermes_tool_call",
    "last_live_generate_prompt",
    "last_user_text",
    "max_generate_passes",
    "messages_after_last_user",
    "tool_calls_are_premature_judge",
    "tool_message_text",
]


def fits_response_budget(mask_len: int, n_new_ids: int, response_length: int) -> bool:
    """Return whether appending new token ids stays under the response budget.

    Args:
        mask_len: Current response mask length.
        n_new_ids: Number of token ids to append.
        response_length: Max response length.

    Returns:
        True if ``n_new_ids > 0`` and ``mask_len + n_new_ids < response_length``.
    """
    return n_new_ids > 0 and mask_len + n_new_ids < response_length


def force_first_generate_probability(step: Any, *, validate: bool = False) -> float:
    """Return the linearly annealed probability of forcing the first tool call.

    Args:
        step: Global training step. ``None`` is treated as step 0 (full force).
        validate: If True, always return 0 (no teacher-force at val).

    Returns:
        Probability in ``[0, 1]``.

    Raises:
        ValueError: If ``step`` or Hydra warmup/end knobs are not integers.
    """
    if validate or not agentic_get_bool("force_first_generate", False):
        return 0.0
    if step is None:
        step_i = 0
    else:
        try:
            step_i = max(0, int(step))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"force_first_generate step must be an int or None, got {step!r}") from exc
    warmup_raw = agentic_get("force_first_warmup_steps")
    try:
        warmup = max(0, int(str(warmup_raw).strip()))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"agentic_image_gen.force_first_warmup_steps must be an integer >= 0, got {warmup_raw!r}"
        ) from exc
    end_raw = agentic_get("force_first_end_step")
    try:
        end = int(str(end_raw).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"agentic_image_gen.force_first_end_step must be an integer, got {end_raw!r}") from exc
    end = max(warmup + 1, end)
    if step_i <= warmup:
        return 1.0
    if step_i >= end:
        return 0.0
    return float(end - step_i) / float(end - warmup)


def tool_calls_are_premature_judge(tool_calls: list[Any] | None) -> bool:
    """Return whether every pending tool call is a premature ``judge_image``.

    Args:
        tool_calls: Parsed tool calls from the latest assistant turn, or ``None``.

    Returns:
        True when the list is non-empty and every call is named ``judge_image``.
    """
    if not tool_calls:
        return False
    names = [getattr(tool_call, "name", None) for tool_call in tool_calls]
    return bool(names) and all(name == "judge_image" for name in names)


def last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the text of the last user message.

    Args:
        messages: Chat messages (dicts with ``role`` / ``content``).

    Returns:
        Stripped user text, or ``""`` if none.
    """
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            return " ".join(
                str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        return str(content or "").strip()
    return ""


def hermes_tool_call(name: str, **arguments: str) -> str:
    """Format a Hermes ``<tool_call>`` XML block.

    Args:
        name: Tool name.
        **arguments: String tool arguments.

    Returns:
        Hermes tool-call markup string.
    """
    payload = {"name": name, "arguments": dict(arguments)}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def messages_after_last_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the live suffix after the last user turn.

    Args:
        messages: Full chat message list.

    Returns:
        Messages after the last ``role=user`` turn (may be empty).
    """
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    return list(messages[last_user + 1 :]) if last_user >= 0 else list(messages)


def tool_message_text(message: dict[str, Any]) -> str:
    """Extract plain text from a tool message.

    Args:
        message: Chat message dict.

    Returns:
        Concatenated text content.
    """
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return str(content or "")


def _is_successful_judge(text: str) -> bool:
    return re.search(r"\bagentic_judge\s+ok=1\b", text, re.IGNORECASE) is not None


def _is_fewshot_observation(text: str) -> bool:
    return re.search(r"\bbackend\s*=\s*fewshot\b", text, re.IGNORECASE) is not None


def _is_live_generate_observation(text: str) -> bool:
    return (
        re.search(r"\bagentic_tool\s+ok=1\b", text, re.IGNORECASE) is not None
        and re.search(r"\bbackend\s*=\s*(?!fewshot\b)[A-Za-z0-9_]+\b", text, re.IGNORECASE) is not None
    )


def count_successful_judges(messages: list[dict[str, Any]]) -> int:
    """Count successful live judge observations after the live user turn.

    Args:
        messages: Full chat message list.

    Returns:
        Number of live ``agentic_judge ok=1`` tool observations.
    """
    return sum(
        1
        for message in messages_after_last_user(messages)
        if message.get("role") == "tool"
        and _is_successful_judge(tool_message_text(message))
        and not _is_fewshot_observation(tool_message_text(message))
    )


def last_live_generate_prompt(messages: list[dict[str, Any]]) -> str:
    """Return the diffusion prompt from the last successful live generation.

    Args:
        messages: Full chat message list.

    Returns:
        Prompt string, or ``""`` if none.
    """
    for message in reversed(messages_after_last_user(messages)):
        if message.get("role") != "tool":
            continue
        text = tool_message_text(message)
        if not _is_live_generate_observation(text) or _is_fewshot_observation(text):
            continue
        match = re.search(r"prompt='([^']*)'", text)
        if match:
            return match.group(1).strip()
        match = re.search(r'prompt="([^"]*)"', text)
        if match:
            return match.group(1).strip()
    return ""


def count_successful_generates(messages: list[dict[str, Any]]) -> int:
    """Count successful live generation observations after the live user turn.

    Args:
        messages: Full chat message list.

    Returns:
        Number of live successful ``generate_image`` tool observations.
    """
    return sum(
        1
        for message in messages_after_last_user(messages)
        if message.get("role") == "tool"
        and _is_live_generate_observation(tool_message_text(message))
        and not _is_fewshot_observation(tool_message_text(message))
    )


def _field(text: str, pattern: str, *, flags: int = re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1) if match else None


def build_forced_reflection(
    tool_text: str,
    *,
    force_done: bool = False,
    generate_pass: int = 0,
    max_passes: int = 3,
) -> tuple[str, bool] | None:
    """Build forced-Reflection assistant text from a successful judge observation.

    Args:
        tool_text: Last ``judge_image`` tool observation.
        force_done: If True, require ``Done.`` after max generate passes.
        generate_pass: Current generate count (for the max-pass message).
        max_passes: Max generate_image calls allowed.

    Returns:
        ``(assistant_text, stop_required)``, or ``None`` if the obs is not a live judge.
    """
    if not _is_successful_judge(tool_text or ""):
        return None
    correctness = _field(tool_text, r"\bcorrectness\s*=\s*([0-9.]+)") or "?"
    aesthetics = _field(tool_text, r"\baesthetics\s*=\s*([0-9.]+)") or "?"
    good_enough_value = _field(tool_text, r"\bgood_enough\s*=\s*(YES|NO)")
    findings_value = _field(
        tool_text,
        r"\bfindings:\s*(.+?)(?:\n\s*suggested_fixes:|\n\s*agentic_judge\b)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    fixes_value = _field(
        tool_text,
        r"\bsuggested_fixes:\s*(.+?)(?:\n\s*agentic_judge\b|\n\n|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    good_enough = (good_enough_value or "").upper() == "YES"
    findings = re.sub(r"\s+", " ", (findings_value or "").strip())[:220]
    fixes = re.sub(r"\s+", " ", (fixes_value or "").strip())[:160]
    if not findings:
        findings = "see VL facet scores above"

    if good_enough:
        text = (
            f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
            f"good_enough=YES. {findings} Stop now; do not call another tool. "
            "Your next and only action must be exactly Done. agentic_stop_decision_required=1"
        )
        return text, True
    if force_done:
        text = (
            f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
            f"good_enough=NO after generate_image pass {generate_pass}/{max_passes}. "
            f"{findings} {max_passes}-pass max reached; stop now and do not call another tool. "
            "Your next and only action must be exactly Done. "
            "agentic_force_stop_max_passes=1 agentic_stop_decision_required=1"
        )
        return text, True
    fix_note = f" Suggested fixes: {fixes}." if fixes and fixes.lower() != "none" else ""
    text = (
        f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
        f"good_enough=NO. {findings}.{fix_note} Rewriting the diffusion prompt next."
    )
    return text, False
