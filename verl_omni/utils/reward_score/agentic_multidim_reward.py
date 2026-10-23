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
"""RPCO multi-dimensional reward for agentic image-generation trajectories.

The scorer is intentionally self-contained so the reward PR can be reviewed
and merged independently of RFC #302's rollout and data PRs. It accepts the
trajectory text emitted by those PRs once they are present, but does not import
their agent-loop or dataset modules.

Wire with ``reward.reward_manager.name=naive`` so verl passes ``solution_str``.
The runtime must also pass ``trajectory_messages`` as a keyword argument or in
``extra_info``. This is a list of runtime-owned chat messages: assistant entries
contain ``content`` and OpenAI-style ``tool_calls`` (``id`` and ``function`` with
``name`` / ``arguments``); tool entries contain ``role="tool"``, ``tool_call_id``
and text ``content``. Optional tool ``name`` must match the call. These messages
must come from the runtime, never from parsing policy-authored text or JSON.
Structured messages are authoritative when present. Tool results are matched
once per call ID, and judge image paths must match the latest successful image
execution at the time of both the call and response. Plain ``solution_str`` or
decoded tokens alone cannot authenticate tool outcomes and receive zero reward.
``VisualRewardManager``'s ``solution_image`` is the wrong modality and raises.

The active reward set is ``{reflect, plan, format, tool, result}``. ``plan`` is
active only for plan rows. ``done`` and ``tool_call`` reproduce the PR1
closed-loop indicators for metrics, but are not additional score dimensions.
Invalid rollouts (no parsed ``generate_image`` call or no successful PNG)
receive score zero and ``rollout_valid=0``. ``task_type`` is required.

Judge C/A is trusted only in a matched runtime ``judge_image`` observation
with the tool observation header and finite scores in [0, 1]. Coverage is token F1 (not recall-only), so dumping
reference words into a long blob does not max ``R_reflect`` / ``R_plan``.
Rewrite-after-YES zeros ``R_result`` as well as the Done indicator. Reflect
``R_result`` requires a terminal trusted YES. ``R_tool`` needs a successful
PNG generate plus a trusted judge (not merely a parsed tool call).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

DIMS = ("reflect", "plan", "format", "tool", "result")
# Names consumed by AgenticMetricsAgentLoopManager when PR1 and PR3 are
# composed. Keeping them here lets this independent PR specify that contract.
REWARD_COMPONENTS = tuple(f"reward_{name}" for name in (*DIMS, "done", "tool_call"))

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_TOOL_OBS_LINE_RE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
_REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)


@dataclass(frozen=True)
class _ToolObservation:
    """A runtime tool response bound to one call and its trajectory position."""

    name: str
    content: str
    end: int


def _resolve_trajectory(text: str, messages: Any) -> tuple[str, list[_ToolObservation]]:
    """Separate policy text from runtime observations, binding call and image IDs."""
    if not isinstance(messages, list):
        return text, []
    parts: list[str] = []
    observations: list[_ToolObservation] = []
    pending: dict[str, tuple[str, tuple[str, str] | None]] = {}
    seen_ids: set[str] = set()
    latest_image: tuple[str, str] | None = None
    position = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue
            if tool_calls:
                content = _TOOL_CALL_RE.sub("", content)
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
                    continue
                name = _tool_name(function)
                content += f"\n<tool_call>{json.dumps(function)}</tool_call>"
                if call_id in seen_ids:
                    pending.pop(call_id, None)
                    continue
                seen_ids.add(call_id)
                pending[call_id] = (name, latest_image)
            parts.append(content + "\n")
            position += len(content) + 1
        elif role == "tool":
            call_id = message.get("tool_call_id")
            call = pending.pop(call_id, None) if isinstance(call_id, str) else None
            if call is None:
                continue
            name, image_at_call = call
            if message.get("name", name) != name:
                continue
            paths = re.findall(r"\bpath=([^\s'\"]+)", content, re.IGNORECASE)
            if name == "generate_image":
                if not re.search(r"\bagentic_tool\s+ok=1\b", content, re.IGNORECASE):
                    continue
                if len(paths) != 1 or not paths[0].lower().endswith(".png"):
                    continue
                latest_image = (call_id, paths[0])
            elif name == "judge_image":
                if latest_image is None or image_at_call != latest_image or paths != [latest_image[1]]:
                    continue
            else:
                continue
            observations.append(_ToolObservation(name, content, position + len(content)))
            parts.append(" " * len(content) + "\n")
            position += len(content) + 1
    return "".join(parts), observations


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    """Return the complete metric schema with all rewards disabled."""
    return {
        "score": 0.0,
        **{f"reward_{dim}": 0.0 for dim in DIMS},
        "reward_done": 0.0,
        "reward_tool_call": 0.0,
        "num_hermes_tool_calls": 0,
        "num_generate_image_prompts": 0,
        "num_judge_image_calls": 0,
        "judge_parse_ok": 0,
        "judge_parse_fail": 0,
        "judge_parse_ok_rate": 0.0,
        "protocol_ok": 0,
        "rewrite_after_yes": 0,
        "rollout_valid": 0,
        "terminal_done": 0,
        "terminal_policy_reflection": 0,
        "forced_reflection_context": 0,
        "n_successful_generates": 0,
        "expected_num_images": 0,
        "task_type": "",
        "method": method,
    }


def _as_dict(value: Any) -> dict[str, Any]:
    """Normalize mapping or JSON ground truth without mutating caller data."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
        return {"user_request": raw}
    return {}


def _parse_tool_call_body(body: str) -> dict[str, Any] | None:
    """Parse JSON or Qwen XML tool-call syntax without executing it."""
    if not body:
        return None
    if body.lstrip().startswith("{"):
        try:
            call = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return None
        return call if isinstance(call, dict) and call.get("name") else None

    function = re.search(r"<function=([^>\s]+)\s*>(.*?)</function>", body, re.IGNORECASE | re.DOTALL)
    if function is None:
        return None
    name = (function.group(1) or "").strip()
    if not name:
        return None
    arguments = {
        match.group(1).strip(): (match.group(2) or "").strip()
        for match in re.finditer(
            r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>",
            function.group(2) or "",
            re.IGNORECASE | re.DOTALL,
        )
        if match.group(1).strip()
    }
    return {"name": name, "arguments": arguments}


def _extract_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    """Collect recognized tool calls and their offsets in policy text."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        call = _parse_tool_call_body((match.group(1) or "").strip())
        if call is not None:
            calls.append((match.start(), match.end(), call))
    return calls


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Decode mapping or JSON-string tool arguments."""
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def _tool_name(call: dict[str, Any]) -> str:
    """Normalize the name of a parsed tool call."""
    return str(call.get("name") or "").strip().lower()


def _generate_prompts(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    """Collect nonempty prompts from parsed generate-image calls."""
    prompts = []
    for _, _, call in calls:
        if _tool_name(call) != "generate_image":
            continue
        prompt = str(_call_arguments(call).get("prompt") or "").strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def _assistant_prose(text: str) -> str:
    """Remove tool syntax and injected reflection from policy prose."""
    prose = _TOOL_CALL_RE.sub(" ", text or "")
    prose = _TOOL_OBS_LINE_RE.sub(" ", prose)
    prose = re.sub(r"</?think>", " ", prose, flags=re.IGNORECASE)
    # Masked, environment-injected reflection text cannot earn policy credit.
    prose = re.sub(
        r"(?is)\bReflection\s*:.*?(?:agentic_forced_reflection=1|agentic_force_stop_max_passes=1)\S*",
        " ",
        prose,
    )
    return re.sub(r"\s+", " ", prose).strip()


def _assistant_prose_lines(text: str) -> str:
    """Strip protocol payloads while preserving plan-item line boundaries."""
    prose = _TOOL_CALL_RE.sub("\n", text or "")
    prose = _TOOL_OBS_LINE_RE.sub("", prose)
    prose = re.sub(r"</?think>", "", prose, flags=re.IGNORECASE)
    return prose


def _tokens(text: str) -> set[str]:
    """Extract normalized unique word tokens for coverage scoring."""
    return set(re.findall(r"[a-z0-9_']+", (text or "").lower()))


def _coverage(candidate: str, reference: str) -> float:
    """Token F1 of candidate vs reference (recall-only bag-of-words is not enough).

    Precision penalizes dumping the reference tokens into a long unrelated blob;
    recall still rewards covering the reference. Exact copy scores 1.0.
    """
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = len(reference_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    recall = overlap / len(reference_tokens)
    precision = overlap / len(candidate_tokens)
    return 2.0 * precision * recall / (precision + recall)


def _good_enough(window: str) -> bool | None:
    """Read an explicit judge acceptance decision from one observation."""
    matches = list(re.finditer(r"\bgood_enough\s*=\s*(YES|NO|1|0|true|false)\b", window or "", re.IGNORECASE))
    if not matches:
        return None
    value = matches[-1].group(1).lower()
    return value in {"yes", "1", "true"}


def _successful_judges(observations: list[_ToolObservation]) -> list[tuple[float, float, bool | None, int]]:
    """Parse judge values only inside validated runtime judge observations."""
    hits = []
    for observation in observations:
        if observation.name != "judge_image":
            continue
        window = observation.content
        if len(_JUDGE_OK_RE.findall(window)) != 1:
            continue
        if "VL judge on the last generated image" not in window:
            continue
        if re.search(r"\b(?:parse_ok|ok)\s*=\s*0\b|\bstub\s*=\s*1\b", window, re.IGNORECASE):
            continue
        values = []
        for field in ("correctness", "aesthetics"):
            matches = re.findall(r"(?im)^\s*" + field + r"\s*=\s*(\S+)\s*$", window)
            if len(matches) != 1:
                break
            try:
                value = float(matches[0])
            except ValueError:
                break
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                break
            values.append(value)
        if len(values) == 2 and _good_enough(window) is not None:
            hits.append((values[0], values[1], _good_enough(window), observation.end))
    return hits


def _terminal_decision(text: str, judges: list[tuple[float, float, bool | None, int]]) -> tuple[bool, bool, bool]:
    """Recognize policy completion following the final trusted judge."""
    if not judges:
        return False, False, False

    judge_end = judges[-1][3]
    line_end = text.find("\n", judge_end)
    anchor = len(text) if line_end < 0 else line_end + 1
    forced_context = False
    for marker in re.finditer(r"\bagentic_forced_reflection=1\b", text, re.IGNORECASE):
        if marker.start() < anchor:
            continue
        anchor = marker.end()
        forced_context = True

    suffix = _assistant_prose(text[anchor:])
    suffix = re.sub(r"<\|[^>]+\|>|</?tool_response>|</?assistant>", " ", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"^\s*(?:assistant|user)\s+", "", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"\s+", " ", suffix).strip()
    policy_reflection = bool(_REFLECTION_RE.search(suffix))
    if policy_reflection:
        terminal_done = bool(re.search(r"\bDone\.\s*$", suffix, re.IGNORECASE))
    else:
        terminal_done = bool(re.fullmatch(r"Done\.", suffix, re.IGNORECASE))
    return terminal_done, policy_reflection, forced_context


def _generates_after_first_yes(
    calls: list[tuple[int, int, dict[str, Any]]], judges: list[tuple[float, float, bool | None, int]]
) -> int:
    """Count generation calls made after the first trusted acceptance."""
    yes_position = next((end for _, _, accepted, end in judges if accepted is True), None)
    if yes_position is None:
        return 0
    return sum(1 for start, _, call in calls if start > yes_position and _tool_name(call) == "generate_image")


def _extract_plan_lines(text: str) -> list[str]:
    """Extract substantive numbered or bulleted policy plan items."""
    prose = _assistant_prose_lines(text)
    header = re.search(r"\bPlan\s*:", prose, re.IGNORECASE)
    body = prose[header.end() :] if header else prose
    return [
        line
        for match in re.finditer(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$", body)
        if len(_tokens(line := match.group(1).strip())) >= 4
    ]


def _reflection_text(text: str) -> str:
    """Extract policy reflection prose without its terminal Done marker."""
    prose = _assistant_prose(text)
    match = re.search(r"\bReflection\s*:(.*?)(?:\bDone\.\s*$|$)", prose, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _reflection_reward(
    text: str,
    ground_truth: dict[str, Any],
    judges: list[tuple[float, float, bool | None, int]],
    observations: list[_ToolObservation],
) -> float:
    """Blend trusted judge quality with reference or trusted-feedback coverage."""
    if not judges:
        return 0.0
    quality = 0.0
    if judges:
        preferred = next((hit for hit in judges if hit[2] is True), judges[-1])
        quality = 0.5 * (preferred[0] + preferred[1])

    steps = ground_truth.get("reference_steps") or []
    reference = " ".join(str(step.get("reflection") or "") for step in steps if isinstance(step, dict)).strip()
    if not reference and judges:
        feedback = []
        trusted_ends = {end for _, _, _, end in judges}
        for observation in observations:
            if observation.end not in trusted_ends:
                continue
            window = observation.content
            feedback.extend(
                match.group(1).strip()
                for match in re.finditer(r"(?im)^\s*(?:findings|suggested_fixes)\s*:\s*(.*)$", window)
            )
        reference = " ".join(item for item in feedback if item.lower() not in {"", "none", "n/a"}).strip()
    if not reference:
        return quality
    return 0.5 * quality + 0.5 * _coverage(_reflection_text(text), reference)


def _plan_reward(text: str, ground_truth: dict[str, Any]) -> float:
    """Average the best policy-plan coverage of each reference subtask."""
    references = [str(item).strip() for item in ground_truth.get("reference_subtasks") or [] if str(item).strip()]
    candidates = _extract_plan_lines(text)
    if not references or not candidates:
        return 0.0
    return sum(max(_coverage(candidate, reference) for candidate in candidates) for reference in references) / len(
        references
    )


def _format_reward(
    text: str,
    *,
    task_type: str,
    successful_generates: int,
    forced_context: bool = False,
    judges: list[tuple[float, float, bool | None, int]],
) -> float:
    """Score structural protocol checks using trusted execution evidence."""
    raw_blocks = len(_TOOL_CALL_RE.findall(text))
    calls = _extract_tool_calls(text)
    names = [_tool_name(call) for _, _, call in calls]
    generates = [index for index, name in enumerate(names) if name == "generate_image"]
    judge_calls = [index for index, name in enumerate(names) if name == "judge_image"]
    terminal_done, policy_reflection, _ = _terminal_decision(text, judges)
    checks = [
        raw_blocks > 0 and len(calls) == raw_blocks,
        successful_generates >= 1,
        bool(judge_calls) and (not generates or max(judge_calls) > max(generates)),
        terminal_done,
    ]
    if task_type == "plan":
        checks.extend((bool(_extract_plan_lines(text)), policy_reflection or forced_context))
    else:
        # #409 force-injects Reflection (stripped from prose) then policy Done.
        # Count forced_context so the default curriculum can still saturate format.
        checks.append(bool(_REFLECTION_RE.search(_assistant_prose(text))) or forced_context)
    return sum(checks) / len(checks)


def _result_reward(
    *,
    task_type: str,
    expected: int,
    successful_generates: int,
    terminal_done: bool,
    blocked: bool,
    rewrite_after_yes: int,
    judges: list[tuple[float, float, bool | None, int]],
) -> float:
    """Require terminal trusted judging and task-specific generation counts."""
    if blocked or not terminal_done or successful_generates < 1 or rewrite_after_yes > 0:
        return 0.0
    if task_type == "plan":
        return 1.0 if successful_generates == expected else 0.0
    final_yes = bool(judges) and judges[-1][2] is True
    # Fail closed on a terminal NO: early-stop alone is not a free result point.
    return 1.0 if final_yes and successful_generates <= expected else 0.0


def _resolve_solution_text(
    solution_str: str,
    *,
    kwargs: dict[str, Any],
    extra_info: dict[str, Any],
) -> str:
    """Resolve trajectory text for NaiveRewardManager (and optional decode).

    ``solution_image`` from VisualRewardManager is the wrong modality — raise
    instead of scoring an empty blob as zeros.
    """
    blob = (solution_str or "").strip()
    if not blob:
        alt = kwargs.get("solution_str")
        if isinstance(alt, str):
            blob = alt.strip()
    if blob:
        return blob

    responses = kwargs.get("responses")
    tokenizer = kwargs.get("tokenizer") or extra_info.get("tokenizer")
    if responses is not None and tokenizer is not None:
        try:
            if hasattr(responses, "tolist"):
                ids = responses.tolist()
            else:
                ids = list(responses)
            if ids and isinstance(ids[0], list | tuple):
                ids = list(ids[0])
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "agentic_multidim_reward.compute_score failed to decode responses into solution_str"
            ) from exc

    if "solution_image" in kwargs:
        raise ValueError(
            "agentic_multidim_reward.compute_score requires solution_str (text trajectory). "
            "Got solution_image from VisualRewardManager — set "
            "reward.reward_manager.name=naive for Mode (2a)."
        )
    return ""


def _require_task_type(ground_truth: dict[str, Any], extra_info: dict[str, Any]) -> str | None:
    """Require an explicit supported task type from ground truth or metadata."""
    raw = ground_truth.get("task_type")
    if raw is None:
        raw = extra_info.get("task_type")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    task_type = str(raw).strip()
    if task_type not in {"reflect", "plan"}:
        return None
    return task_type


def _active_weights(
    ground_truth: dict[str, Any], extra_info: dict[str, Any], *, task_type: str
) -> dict[str, float] | None:
    """Return positive active-set weights, or None if a ``w_*`` value is garbage."""
    weights = {}
    for dim in DIMS:
        if dim == "plan" and task_type != "plan":
            continue
        raw = ground_truth.get(f"w_{dim}")
        if raw is None:
            raw = extra_info.get(f"w_{dim}")
        if raw is None:
            value = 1.0
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError, OverflowError):
                return None
            if not math.isfinite(value) or value < 0:
                return None
        if value > 0:
            weights[dim] = value
    return weights


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Compute the RFC #302 stage-3 reward and its complete metric schema.

    Args:
        data_source: Unused; kept for the verl ``compute_score`` signature.
        solution_str: Decoded trajectory text (NaiveRewardManager).
        ground_truth: Must include ``task_type`` (``reflect`` / ``plan``) plus
            optional references and ``w_*`` weights.
        extra_info: Fallback for task type / weights, tokenizer, and runtime
            ``trajectory_messages`` with role and tool-call identity preserved.
        **kwargs: May include runtime ``trajectory_messages``, decoded
            ``responses`` + tokenizer, or ``solution_image`` (rejected).

    Returns:
        Dict with ``score``, per-dim ``reward_*``, and metric schema fields.
    """
    del data_source
    gt = _as_dict(ground_truth)
    metadata = dict(extra_info or {})
    task_type = _require_task_type(gt, metadata)
    if task_type is None:
        return _zero_result(method="agentic_multidim_missing_task_type")
    weights = _active_weights(gt, metadata, task_type=task_type)
    if weights is None:
        return _zero_result(method="agentic_multidim_bad_weights")

    try:
        expected = max(1, int(gt.get("expected_num_images", metadata.get("expected_num_images", 1))))
    except (TypeError, ValueError):
        expected = 1

    text = _resolve_solution_text(solution_str, kwargs=kwargs, extra_info=metadata)
    kwargs.pop("solution_image", None)
    if not text.strip() and "trajectory_messages" not in kwargs and "trajectory_messages" not in metadata:
        result = _zero_result(method="agentic_multidim_empty")
        result.update(task_type=task_type, expected_num_images=expected)
        return result

    messages = kwargs.get("trajectory_messages", metadata.get("trajectory_messages"))
    text, observations = _resolve_trajectory(text, messages)
    calls = _extract_tool_calls(text)
    prompts = _generate_prompts(calls)
    names = [_tool_name(call) for _, _, call in calls]
    judges = _successful_judges(observations)
    judge_ok = len(judges)
    judge_failed = sum(observation.name == "judge_image" for observation in observations) - judge_ok
    judge_rate = judge_ok / (judge_ok + judge_failed) if judge_ok + judge_failed else 0.0
    successful_generates = sum(observation.name == "generate_image" for observation in observations)
    last_generate_end = max((obs.end for obs in observations if obs.name == "generate_image"), default=0)
    last_judge_end = max((obs.end for obs in observations if obs.name == "judge_image"), default=0)
    terminal_judges = judges if judges and judges[-1][3] == last_judge_end > last_generate_end else []
    terminal_done, policy_reflection, forced_context = _terminal_decision(text, terminal_judges)
    blocked = bool(
        re.search(
            r"\b(?:blocked_after_yes|blocked_after_max_passes)=1\b|generate_image blocked:",
            text,
            re.IGNORECASE,
        )
    )
    rewrites_after_yes = _generates_after_first_yes(calls, judges)
    tool_reward = float(successful_generates >= 1 and bool(judges))

    result = _zero_result(method="agentic_multidim")
    result.update(
        num_hermes_tool_calls=len(calls),
        num_generate_image_prompts=len(prompts),
        num_judge_image_calls=sum(name == "judge_image" for name in names),
        judge_parse_ok=judge_ok,
        judge_parse_fail=judge_failed,
        judge_parse_ok_rate=float(judge_rate),
        terminal_done=int(terminal_done),
        terminal_policy_reflection=int(policy_reflection),
        forced_reflection_context=int(forced_context),
        n_successful_generates=successful_generates,
        expected_num_images=expected,
        task_type=task_type,
        rewrite_after_yes=rewrites_after_yes,
        reward_tool_call=float(bool(calls)),
        reward_tool=tool_reward,
    )
    if not prompts or successful_generates == 0:
        return result

    valid_terminal_context = judge_ok > 0 and not blocked and rewrites_after_yes == 0
    closed = valid_terminal_context and terminal_done and (policy_reflection or forced_context)
    rewards = {
        "reflect": _reflection_reward(text, gt, judges, observations),
        "plan": _plan_reward(text, gt),
        "format": _format_reward(
            text,
            task_type=task_type,
            successful_generates=successful_generates,
            forced_context=forced_context,
            judges=terminal_judges,
        ),
        "tool": tool_reward,
        "result": _result_reward(
            task_type=task_type,
            expected=expected,
            successful_generates=successful_generates,
            terminal_done=terminal_done,
            blocked=blocked,
            rewrite_after_yes=rewrites_after_yes,
            judges=judges,
        ),
    }
    scale = max(weights.values(), default=1.0)
    normalized_weights = {dim: weight / scale for dim, weight in weights.items()}
    weight_sum = sum(normalized_weights.values())
    score = sum(normalized_weights[dim] * rewards[dim] for dim in weights) / weight_sum if weight_sum else 0.0
    result.update(
        score=float(min(1.0, score)),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_done=float(closed),
        reward_tool_call=float(bool(calls)),
        protocol_ok=int(rewards["format"] == 1.0),
        rollout_valid=1,
    )
    return result
