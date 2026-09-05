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

"""Shared helpers for agent-loop rollout seeding and image-generation control."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

__all__ = [
    "build_forced_reflection",
    "count_successful_generates",
    "count_successful_judges",
    "derive_rollout_seed",
    "fits_response_budget",
    "force_enabled",
    "force_first_generate_probability",
    "hermes_tool_call",
    "last_live_generate_prompt",
    "last_user_text",
    "max_generate_passes",
    "maybe_per_rollout_seeds",
    "messages_after_last_user",
    "rewrite_judge_before_generate",
    "tool_calls_are_premature_judge",
    "tool_message_text",
]


def messages_to_text(messages: Any) -> str:
    """Extract message text without applying a chat template."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, dict):
        messages = [messages]

    parts = []
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
            continue
        for item in content or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
    return "\n".join(part for part in parts if part).strip()


def derive_rollout_seed(base_seed: int, rollout_index: int) -> int:
    """Map per-step rollout base and expanded row index to a vLLM seed.
    Row index is 0 .. num_prompts * rollout.n - 1 after interleaved repeat."""
    max_seed = (1 << 63) - 1
    return (int(base_seed) * 1_000_003 + int(rollout_index)) % max_seed


def maybe_per_rollout_seeds(meta_info: dict, batch_size: int, global_indices=None) -> Optional[list[int]]:
    """Build one seed per post-repeat rollout row.

    When ``global_indices`` is provided (trainer sets ``_rollout_seed_global_idx``),
    seeds are derived from those stable indices rather than the local chunk
    position, so multi-worker splits and request packing order cannot remap RNG
    state across rows.
    """
    base = meta_info.get("rollout_seed")
    if base is None:
        return None
    base = int(base)

    if global_indices is None:
        return [derive_rollout_seed(base, i) for i in range(batch_size)]

    indices = [int(idx) for idx in list(global_indices)]
    if len(indices) != batch_size:
        raise ValueError(f"Expected {batch_size} global rollout indices, got {len(indices)}")

    return [derive_rollout_seed(base, idx) for idx in indices]


def force_enabled() -> bool:
    return os.getenv("AGENTIC_FORCE_REFLECTION_AFTER_JUDGE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
        "",
    }


def max_generate_passes() -> int:
    try:
        return max(1, int(os.getenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "3")))
    except ValueError:
        return 3


def fits_response_budget(mask_len: int, n_new_ids: int, response_length: int) -> bool:
    """Return whether appending ``n_new_ids`` stays below ``response_length``."""
    return n_new_ids > 0 and mask_len + n_new_ids < response_length


def force_first_generate_probability(step: Any, *, validate: bool = False) -> float:
    """Return the linearly annealed probability of forcing the first tool call."""
    enabled = os.getenv("AGENTIC_FORCE_FIRST_GENERATE", "0").strip().lower()
    if validate or enabled not in {"1", "true", "yes", "on"}:
        return 0.0
    try:
        step_i = max(0, int(step))
    except (TypeError, ValueError):
        step_i = 0
    try:
        warmup = max(0, int(os.getenv("AGENTIC_FORCE_FIRST_WARMUP_STEPS", "10")))
    except ValueError:
        warmup = 10
    try:
        end = max(warmup + 1, int(os.getenv("AGENTIC_FORCE_FIRST_END_STEP", "20")))
    except ValueError:
        end = 20
    if step_i <= warmup:
        return 1.0
    if step_i >= end:
        return 0.0
    return float(end - step_i) / float(end - warmup)


def rewrite_judge_before_generate() -> bool:
    """Return whether a first-turn judge call is rewritten to generate."""
    return os.getenv("AGENTIC_REWRITE_JUDGE_BEFORE_GENERATE", "1").strip().lower() not in {
        "0",
        "false",
        "off",
        "no",
    }


def tool_calls_are_premature_judge(tool_calls: list[Any] | None) -> bool:
    if not tool_calls:
        return False
    names = [getattr(tool_call, "name", None) for tool_call in tool_calls]
    return bool(names) and all(name == "judge_image" for name in names)


def last_user_text(messages: list[dict[str, Any]]) -> str:
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
    payload = {"name": name, "arguments": dict(arguments)}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def messages_after_last_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the live suffix after the last user turn, excluding few-shot tools."""
    last_user = -1
    for index, message in enumerate(messages):
        if message.get("role") == "user":
            last_user = index
    return list(messages[last_user + 1 :]) if last_user >= 0 else list(messages)


def tool_message_text(message: dict[str, Any]) -> str:
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
    """Count successful live judge observations after the live user turn."""
    return sum(
        1
        for message in messages_after_last_user(messages)
        if message.get("role") == "tool"
        and _is_successful_judge(tool_message_text(message))
        and not _is_fewshot_observation(tool_message_text(message))
    )


def last_live_generate_prompt(messages: list[dict[str, Any]]) -> str:
    """Return the diffusion prompt from the last successful live generation."""
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
    """Count successful live generation observations after the live user turn."""
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
    """Build ``(assistant_text, stop_required)`` from a successful judge observation."""
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
