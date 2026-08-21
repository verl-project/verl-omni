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

The active reward set is ``{reflect, plan, format, tool, result}``. ``plan`` is
active only for plan rows. ``done`` and ``tool_call`` reproduce the PR1
closed-loop indicators for metrics, but are not additional score dimensions.
Invalid rollouts (no parsed ``generate_image`` call or no successful PNG)
receive score zero and ``rollout_valid=0``.

Judge C/A is trusted only after a parsed ``judge_image`` ``<tool_call>`` and the
tool observation header. Coverage is token F1 (not recall-only), so dumping
reference words into a long blob does not max ``R_reflect`` / ``R_plan``.
Rewrite-after-YES zeros ``R_result`` as well as the Done indicator.
"""

from __future__ import annotations

import json
import re
from typing import Any

DIMS = ("reflect", "plan", "format", "tool", "result")
# Names consumed by AgenticMetricsAgentLoopManager when PR1 and PR3 are
# composed. Keeping them here lets this independent PR specify that contract.
REWARD_COMPONENTS = tuple(f"reward_{name}" for name in (*DIMS, "done", "tool_call"))

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL)
_TOOL_OK_RE = re.compile(r"\bagentic_tool\s+ok=1\b", re.IGNORECASE)
_PATH_RE = re.compile(r"\bpath=([^\s'\"]+)", re.IGNORECASE)
_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_JUDGE_FAIL_RE = re.compile(r"\bagentic_judge\s+ok=0\b|\bparse_ok\s*=\s*0\b", re.IGNORECASE)
_CORRECTNESS_RE = re.compile(r"\bcorrectness\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_AESTHETICS_RE = re.compile(r"\baesthetics\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_GOOD_ENOUGH_RE = re.compile(r"\bgood_enough\s*=\s*(YES|NO|1|0|true|false)\b", re.IGNORECASE)
_FORCED_REFLECTION_RE = re.compile(r"\bagentic_forced_reflection=1\b", re.IGNORECASE)
_BLOCKED_GENERATE_RE = re.compile(
    r"\b(?:blocked_after_yes|blocked_after_max_passes)=1\b|generate_image blocked:",
    re.IGNORECASE,
)
_TOOL_OBS_LINE_RE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
_REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)
_JUDGE_OBS_HEADER = "VL judge on the last generated image"
_PLAN_HEADER_RE = re.compile(r"\bPlan\s*:", re.IGNORECASE)
_PLAN_ITEM_RE = re.compile(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
_FINDINGS_RE = re.compile(r"(?im)^\s*(?:findings|suggested_fixes)\s*:\s*(.*)$")
_TOKEN_RE = re.compile(r"[a-z0-9_']+")

_BASE_SCHEMA: dict[str, float | str | int | None] = {
    "score": 0.0,
    **{f"reward_{dim}": 0.0 for dim in DIMS},
    "reward_done": 0.0,
    "reward_tool_call": 0.0,
    "num_hermes_tool_calls": 0,
    "num_generate_image_prompts": 0,
    "num_judge_image_calls": 0,
    "judge_parse_ok": 0,
    "judge_parse_fail": 0,
    "judge_parse_ok_rate": 1.0,
    "protocol_ok": 0,
    "rewrite_after_yes": 0,
    "rollout_valid": 0,
    "terminal_done": 0,
    "terminal_policy_reflection": 0,
    "forced_reflection_context": 0,
    "n_successful_generates": 0,
    "expected_num_images": 0,
    "task_type": "",
    "method": "",
}


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    result = dict(_BASE_SCHEMA)
    result["method"] = method
    return result


def _as_dict(value: Any) -> dict[str, Any]:
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
    if not body:
        return None
    if body.lstrip().startswith("{"):
        try:
            call = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return None
        return call if isinstance(call, dict) and call.get("name") else None

    function = _FUNCTION_RE.search(body)
    if function is None:
        return None
    name = (function.group(1) or "").strip()
    if not name:
        return None
    arguments = {
        match.group(1).strip(): (match.group(2) or "").strip()
        for match in _PARAM_RE.finditer(function.group(2) or "")
        if match.group(1).strip()
    }
    return {"name": name, "arguments": arguments}


def _extract_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    calls = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        call = _parse_tool_call_body((match.group(1) or "").strip())
        if call is not None:
            calls.append((match.start(), match.end(), call))
    return calls


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or "").strip().lower()


def _follows_judge_image_call(pos: int, calls: list[tuple[int, int, dict[str, Any]]]) -> bool:
    """True when ``pos`` is after at least one parsed ``judge_image`` ``<tool_call>``."""
    return any(end <= pos for _, end, call in calls if _tool_name(call) == "judge_image")


def _generate_prompts(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    prompts = []
    for _, _, call in calls:
        if _tool_name(call) != "generate_image":
            continue
        prompt = str(_call_arguments(call).get("prompt") or "").strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def _assistant_prose(text: str) -> str:
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
    return set(_TOKEN_RE.findall((text or "").lower()))


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


def _count_successful_generates(text: str) -> int:
    return sum(
        1
        for line in (text or "").splitlines()
        if _TOOL_OK_RE.search(line) and any(path.lower().endswith(".png") for path in _PATH_RE.findall(line))
    )


def _judge_parse_stats(text: str, calls: list[tuple[int, int, dict[str, Any]]] | None = None) -> tuple[int, int, float]:
    blob = text or ""
    parsed_calls = calls if calls is not None else _extract_tool_calls(blob)
    ok = 0
    for marker in _JUDGE_OK_RE.finditer(blob):
        if _follows_judge_image_call(marker.start(), parsed_calls):
            ok += 1
    failed = 0
    for marker in re.finditer(r"\bagentic_judge\s+ok=0\b", blob, flags=re.IGNORECASE):
        if _follows_judge_image_call(marker.start(), parsed_calls):
            failed += 1
    if failed == 0:
        for marker in _JUDGE_FAIL_RE.finditer(blob):
            if _follows_judge_image_call(marker.start(), parsed_calls):
                failed += 1
    total = ok + failed
    return ok, failed, ok / total if total else 1.0


def _good_enough(window: str) -> bool | None:
    matches = list(_GOOD_ENOUGH_RE.finditer(window or ""))
    if not matches:
        return None
    value = matches[-1].group(1).lower()
    return value in {"yes", "1", "true"}


def _successful_judges(text: str) -> list[tuple[float, float, bool | None, int]]:
    """Return trusted ``(correctness, aesthetics, good_enough, end)`` values.

    Hits must follow a parsed ``judge_image`` ``<tool_call>`` and the tool's
    ``VL judge on the last generated image`` header.
    """
    blob = text or ""
    calls = _extract_tool_calls(blob)
    hits = []
    for marker in _JUDGE_OK_RE.finditer(blob):
        if not _follows_judge_image_call(marker.start(), calls):
            continue
        window = blob[max(0, marker.start() - 1400) : marker.end()]
        if _JUDGE_OBS_HEADER not in window:
            continue
        if re.search(r"\bparse_ok\s*=\s*0\b", window, re.IGNORECASE):
            continue
        correctness = list(_CORRECTNESS_RE.finditer(window))
        aesthetics = list(_AESTHETICS_RE.finditer(window))
        if not correctness or not aesthetics:
            continue
        try:
            c = min(1.0, max(0.0, float(correctness[-1].group(1))))
            a = min(1.0, max(0.0, float(aesthetics[-1].group(1))))
        except ValueError:
            continue
        hits.append((c, a, _good_enough(window), marker.end()))
    return hits


def _terminal_decision(text: str) -> tuple[bool, bool, bool]:
    judges = _successful_judges(text)
    if not judges:
        return False, False, False

    judge_end = judges[-1][3]
    line_end = text.find("\n", judge_end)
    anchor = len(text) if line_end < 0 else line_end + 1
    forced_context = False
    for marker in _FORCED_REFLECTION_RE.finditer(text, anchor):
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


def _generates_after_first_yes(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> int:
    yes_position = next((end for _, _, accepted, end in _successful_judges(text) if accepted is True), None)
    if yes_position is None:
        return 0
    return sum(1 for start, _, call in calls if start > yes_position and _tool_name(call) == "generate_image")


def _extract_plan_lines(text: str) -> list[str]:
    prose = _assistant_prose_lines(text)
    header = _PLAN_HEADER_RE.search(prose)
    body = prose[header.end() :] if header else prose
    return [line for match in _PLAN_ITEM_RE.finditer(body) if len(_tokens(line := match.group(1).strip())) >= 4]


def _reflection_text(text: str) -> str:
    prose = _assistant_prose(text)
    match = re.search(r"\bReflection\s*:(.*?)(?:\bDone\.\s*$|$)", prose, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _reflection_reward(text: str, ground_truth: dict[str, Any]) -> float:
    judges = _successful_judges(text)
    quality = 0.0
    if judges:
        preferred = next((hit for hit in judges if hit[2] is True), judges[-1])
        quality = 0.5 * (preferred[0] + preferred[1])

    steps = ground_truth.get("reference_steps") or []
    reference = " ".join(str(step.get("reflection") or "") for step in steps if isinstance(step, dict)).strip()
    if not reference and judges:
        feedback = []
        for _, _, _, end in judges:
            window = text[max(0, end - 1400) : end]
            feedback.extend(match.group(1).strip() for match in _FINDINGS_RE.finditer(window))
        reference = " ".join(item for item in feedback if item.lower() not in {"", "none", "n/a"}).strip()
    if not reference:
        return quality
    return 0.5 * quality + 0.5 * _coverage(_reflection_text(text), reference)


def _plan_reward(text: str, ground_truth: dict[str, Any]) -> float:
    references = [str(item).strip() for item in ground_truth.get("reference_subtasks") or [] if str(item).strip()]
    candidates = _extract_plan_lines(text)
    if not references or not candidates:
        return 0.0
    return sum(max(_coverage(candidate, reference) for candidate in candidates) for reference in references) / len(
        references
    )


def _format_reward(text: str, *, task_type: str, successful_generates: int) -> float:
    raw_blocks = len(_TOOL_CALL_RE.findall(text))
    calls = _extract_tool_calls(text)
    names = [_tool_name(call) for _, _, call in calls]
    generates = [index for index, name in enumerate(names) if name == "generate_image"]
    judges = [index for index, name in enumerate(names) if name == "judge_image"]
    terminal_done, policy_reflection, _ = _terminal_decision(text)
    checks = [
        raw_blocks > 0 and len(calls) == raw_blocks,
        successful_generates >= 1,
        bool(judges) and (not generates or max(judges) > max(generates)),
        terminal_done,
    ]
    if task_type == "plan":
        checks.extend((bool(_extract_plan_lines(text)), policy_reflection))
    else:
        checks.append(bool(_REFLECTION_RE.search(_assistant_prose(text))))
    return sum(checks) / len(checks)


def _result_reward(
    text: str,
    *,
    task_type: str,
    expected: int,
    successful_generates: int,
    terminal_done: bool,
    blocked: bool,
    rewrite_after_yes: int,
) -> float:
    if blocked or not terminal_done or successful_generates < 1 or rewrite_after_yes > 0:
        return 0.0
    if task_type == "plan":
        return 1.0 if successful_generates == expected else 0.0
    judges = _successful_judges(text)
    final_yes = bool(judges) and judges[-1][2] is True
    return 1.0 if successful_generates <= expected or final_yes else 0.0


def _active_weights(ground_truth: dict[str, Any], extra_info: dict[str, Any], *, task_type: str) -> dict[str, float]:
    weights = {}
    for dim in DIMS:
        if dim == "plan" and task_type != "plan":
            continue
        raw = ground_truth.get(f"w_{dim}")
        if raw is None:
            raw = extra_info.get(f"w_{dim}")
        try:
            value = float(raw if raw is not None else 1.0)
        except (TypeError, ValueError):
            value = 1.0
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
    """Compute the RFC #302 stage-3 reward and its complete metric schema."""
    del data_source, kwargs
    text = solution_str or ""
    gt = _as_dict(ground_truth)
    metadata = dict(extra_info or {})
    task_type = str(gt.get("task_type") or metadata.get("task_type") or "reflect")
    if task_type not in {"reflect", "plan"}:
        task_type = "reflect"
    try:
        expected = max(1, int(gt.get("expected_num_images", metadata.get("expected_num_images", 1))))
    except (TypeError, ValueError):
        expected = 1

    if not text.strip():
        result = _zero_result(method="agentic_multidim_empty")
        result.update(task_type=task_type, expected_num_images=expected)
        return result

    calls = _extract_tool_calls(text)
    prompts = _generate_prompts(calls)
    names = [_tool_name(call) for _, _, call in calls]
    judge_ok, judge_failed, judge_rate = _judge_parse_stats(text, calls)
    successful_generates = _count_successful_generates(text)
    terminal_done, policy_reflection, forced_context = _terminal_decision(text)
    blocked = bool(_BLOCKED_GENERATE_RE.search(text))
    rewrites_after_yes = _generates_after_first_yes(text, calls)

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
    )
    if not prompts or successful_generates == 0:
        return result

    valid_terminal_context = judge_ok > 0 and not blocked and rewrites_after_yes == 0
    closed = valid_terminal_context and terminal_done and (policy_reflection or forced_context)
    rewards = {
        "reflect": _reflection_reward(text, gt),
        "plan": _plan_reward(text, gt),
        "format": _format_reward(text, task_type=task_type, successful_generates=successful_generates),
        "tool": float(bool(calls)),
        "result": _result_reward(
            text,
            task_type=task_type,
            expected=expected,
            successful_generates=successful_generates,
            terminal_done=terminal_done,
            blocked=blocked,
            rewrite_after_yes=rewrites_after_yes,
        ),
    }
    weights = _active_weights(gt, metadata, task_type=task_type)
    weight_sum = sum(weights.values())
    score = sum(weights[dim] * rewards[dim] for dim in weights) / weight_sum if weight_sum else 0.0
    result.update(
        score=float(min(1.0, score)),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_done=float(closed),
        reward_tool_call=float(bool(calls)),
        protocol_ok=int(rewards["format"] == 1.0),
        rollout_valid=1,
    )
    return result
