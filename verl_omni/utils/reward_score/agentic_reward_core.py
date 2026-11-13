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
"""Shared RPCO trajectory parsing and reward primitives.

Tool outcomes require runtime-owned messages. ``TrajectoryAssessment`` evaluates
five Hermes reward dimensions; other scorers can compose the primitives with
their own dialect, active dimensions, weights, and metric schema.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

TASK_TYPES = ("reflect", "plan")

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
JUDGE_OBS_HEADER = "VL judge on the last generated image"
TOOL_OBS_LINE_RE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)
#: Harness-injected stop cues. Text after one is context (``response_mask=0``), not policy prose.
FORCED_CONTEXT_CUE_RE = re.compile(
    r"\bagentic_(?:forced_reflection|force_stop_max_passes|stop_decision_required)=1(?!\d)",
    re.IGNORECASE,
)
BLOCKED_RE = re.compile(
    r"\b(?:blocked_after_yes|blocked_after_max_passes)=1\b|generate_image blocked:",
    re.IGNORECASE,
)
_GENERATE_OK_RE = re.compile(r"\bagentic_tool\s+ok=1\b", re.IGNORECASE)
_PATH_RE = re.compile(r"\bpath=([^\s'\"]+)", re.IGNORECASE)
# ``image_gen`` echoes the policy's prompt as a Python repr; its contents are policy-authored.
_PROMPT_ECHO_RE = re.compile(r"\bprompt=(?:'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")")


@dataclass(frozen=True)
class ToolObservation:
    """A runtime tool response accepted by the trajectory resolver.

    Attributes:
        name: Matched tool name, either ``generate_image`` or ``judge_image``.
        content: Original response text, retained for judge parsing.
        end: Exclusive end offset of the blanked response in ``Trajectory.text``.
    """

    name: str
    content: str
    end: int


class JudgeHit(NamedTuple):
    """One trusted judge verdict; ``end`` is its offset in the resolved text."""

    correctness: float
    aesthetics: float
    good_enough: bool | None
    end: int


@dataclass(frozen=True)
class Trajectory:
    """Policy text and accepted tool observations in a shared offset space.

    Attributes:
        text: Policy prose and structured calls; accepted tool response bodies are
            replaced by spaces to preserve their offsets.
        observations: Matched runtime tool responses that passed provenance checks.
        blocked: Whether a block marker appeared in a runtime tool response, or in
            the input text when runtime messages were not supplied.
        authenticated: Whether a runtime message list was supplied. Individual
            observations still require call and image matching before being trusted.
    """

    text: str
    observations: tuple[ToolObservation, ...]
    blocked: bool
    authenticated: bool


class TerminalDecision(NamedTuple):
    """How the policy closed after the final trusted verdict."""

    terminal_done: bool
    policy_reflection: bool
    forced_context: bool
    anchor: int


@dataclass(frozen=True)
class TrajectoryAssessment:
    """Cache a resolved Hermes trajectory for diagnostics and five RPCO dimensions.

    Create instances with ``from_messages``. The private fields hold the resolved
    trajectory, parsed calls, trusted judge verdicts, terminal decision, and counts
    used by ``metrics`` and ``dimension_scores``. Scorers choose active dimensions
    and weights separately.
    """

    _trajectory: Trajectory
    _calls: tuple[tuple[int, int, dict[str, Any]], ...]
    _judges: tuple[JudgeHit, ...]
    _decision: TerminalDecision
    _successful_generates: int
    _judge_failed: int
    _blocked: bool
    _rewrites_after_yes: int

    @classmethod
    def from_messages(cls, text: str, messages: Any) -> TrajectoryAssessment:
        """Resolve provenance, calls, verdicts, and the final policy decision."""
        trajectory = resolve_trajectory(text, messages)
        calls = extract_tool_calls(trajectory.text)
        judges = trusted_judges(trajectory.observations)
        last_generate = max((obs.end for obs in trajectory.observations if obs.name == "generate_image"), default=0)
        last_judge = max((obs.end for obs in trajectory.observations if obs.name == "judge_image"), default=0)
        terminal_judges = judges if judges and judges[-1].end == last_judge > last_generate else []
        return cls(
            _trajectory=trajectory,
            _calls=tuple(calls),
            _judges=tuple(judges),
            _decision=terminal_decision(trajectory.text, terminal_judges),
            _successful_generates=sum(obs.name == "generate_image" for obs in trajectory.observations),
            _judge_failed=sum(obs.name == "judge_image" for obs in trajectory.observations) - len(judges),
            _blocked=trajectory.blocked or bool(BLOCKED_RE.search(trajectory.text)),
            _rewrites_after_yes=generates_after_first_yes(calls, judges),
        )

    @property
    def valid(self) -> bool:
        """Whether a parsed generation call has a successful runtime image result."""
        return bool(generate_prompts(self._calls)) and self._successful_generates > 0

    @property
    def closed(self) -> bool:
        """Whether the final policy decision closes a valid, unblocked judge context."""
        return bool(
            self._judges
            and not self._blocked
            and self._rewrites_after_yes == 0
            and self._decision.terminal_done
            and (self._decision.policy_reflection or self._decision.forced_context)
        )

    @property
    def _tool_reward(self) -> float:
        return float(self._successful_generates >= 1 and bool(self._judges))

    def metrics(self) -> dict[str, float | int]:
        """Return trajectory diagnostics without choosing the scorer's output schema."""
        judge_ok = len(self._judges)
        judge_total = judge_ok + self._judge_failed
        return {
            "num_hermes_tool_calls": len(self._calls),
            "num_generate_image_prompts": len(generate_prompts(self._calls)),
            "num_judge_image_calls": sum(tool_name(call) == "judge_image" for _, _, call in self._calls),
            "judge_parse_ok": judge_ok,
            "judge_parse_fail": self._judge_failed,
            "judge_parse_ok_rate": judge_ok / judge_total if judge_total else 0.0,
            "terminal_done": int(self._decision.terminal_done),
            "terminal_policy_reflection": int(self._decision.policy_reflection),
            "forced_reflection_context": int(self._decision.forced_context),
            "n_successful_generates": self._successful_generates,
            "rewrite_after_yes": self._rewrites_after_yes,
            "reward_tool_call": float(bool(self._calls)),
            "reward_tool": self._tool_reward,
        }

    def dimension_scores(self, *, task_type: str, expected: int, ground_truth: Mapping[str, Any]) -> dict[str, float]:
        """Evaluate the five RPCO dimensions, preserving task-specific result counts."""
        text = self._trajectory.text
        trusted_ends = {hit.end for hit in self._judges}
        reference = reference_reflection(ground_truth) or judge_feedback(
            obs.content for obs in self._trajectory.observations if obs.end in trusted_ends
        )
        names = [tool_name(call) for _, _, call in self._calls]
        generate_indices = [index for index, name in enumerate(names) if name == "generate_image"]
        judge_indices = [index for index, name in enumerate(names) if name == "judge_image"]
        plan_lines = extract_plan_lines(text)
        submission_images = (
            sum(obs.name == "generate_image" and obs.end < self._judges[0].end for obs in self._trajectory.observations)
            if task_type == "reflect" and self._judges
            else None
        )
        return {
            "reflect": reflection_score(judge_quality(self._judges), reflection_text(text), reference)
            if self._judges
            else 0.0,
            "plan": plan_coverage(plan_lines, reference_subtasks(ground_truth)),
            "format": format_score(
                text,
                task_type=task_type,
                successful_generates=self._successful_generates,
                judged_after_last_generate=bool(judge_indices)
                and (not generate_indices or max(judge_indices) > max(generate_indices)),
                terminal_done=self._decision.terminal_done,
                policy_reflection=self._decision.policy_reflection,
                forced_context=self._decision.forced_context,
                plan_lines=plan_lines,
            ),
            "tool": self._tool_reward,
            "result": result_score(
                task_type=task_type,
                expected=expected,
                successful_generates=self._successful_generates,
                terminal_done=self._decision.terminal_done,
                blocked=self._blocked,
                rewrite_after_yes=self._rewrites_after_yes,
                judges=self._judges,
                submission_images=submission_images,
            ),
        }


def message_text(content: Any) -> str | None:
    """Text of a chat ``content`` field (string or OpenAI-style part list)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    return None


def resolve_trajectory(text: str, messages: Any) -> Trajectory:
    """Separate policy text from runtime observations, binding call and image IDs.

    Each tool response is matched once to the assistant call with its
    ``tool_call_id``. A ``judge_image`` response is trusted only when it judges the
    latest successful ``generate_image`` image, both when called and when answered.
    Tool-call markup inside assistant ``content`` is dropped: only structured
    ``tool_calls`` are re-rendered, so policy-authored markup cannot pose as a call.
    A runtime block marker in any tool response sets ``blocked``, including the
    responses that are not trusted as observations.
    """
    if not isinstance(messages, list):
        return Trajectory(text, (), bool(BLOCKED_RE.search(text or "")), False)
    parts: list[str] = []
    observations: list[ToolObservation] = []
    pending: dict[str, tuple[str, tuple[str, str] | None]] = {}
    seen_ids: set[str] = set()
    latest_image: tuple[str, str] | None = None
    blocked = False
    position = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message_text(message.get("content"))
        if content is None:
            continue
        role = message.get("role")
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                continue
            content = TOOL_CALL_RE.sub("", content)
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                call_id = call.get("id")
                function = call.get("function")
                if not isinstance(call_id, str) or not call_id or not isinstance(function, dict):
                    continue
                name = tool_name(function)
                content += f"\n<tool_call>{json.dumps(function)}</tool_call>"
                if call_id in seen_ids:
                    pending.pop(call_id, None)
                    continue
                seen_ids.add(call_id)
                pending[call_id] = (name, latest_image)
            parts.append(content + "\n")
            position += len(content) + 1
        elif role == "tool":
            blocked = blocked or bool(BLOCKED_RE.search(content))
            call_id = message.get("tool_call_id")
            call = pending.pop(call_id, None) if isinstance(call_id, str) else None
            if call is None:
                continue
            name, image_at_call = call
            if message.get("name", name) != name:
                continue
            status = _PROMPT_ECHO_RE.sub("", content)
            paths = _PATH_RE.findall(status)
            if name == "generate_image":
                if not _GENERATE_OK_RE.search(status):
                    continue
                if len(paths) != 1 or not paths[0].lower().endswith(".png"):
                    continue
                latest_image = (call_id, paths[0])
            elif name == "judge_image":
                if latest_image is None or image_at_call != latest_image or paths != [latest_image[1]]:
                    continue
            else:
                continue
            observations.append(ToolObservation(name, content, position + len(content)))
            parts.append(" " * len(content) + "\n")
            position += len(content) + 1
    return Trajectory("".join(parts), tuple(observations), blocked, True)


def zero_result(*, method: str, components: Sequence[str]) -> dict[str, float | str | int | None]:
    """Return the complete metric schema with every reward component disabled."""
    return {
        "score": 0.0,
        **{component: 0.0 for component in components},
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


def as_dict(value: Any) -> dict[str, Any]:
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


def as_sequence(value: Any) -> list[Any]:
    """Coerce a dataset reference field into a list of items."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def parse_tool_call_body(body: str) -> dict[str, Any] | None:
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


def extract_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    """Collect recognized ``<tool_call>`` blocks and their offsets."""
    calls = []
    for match in TOOL_CALL_RE.finditer(text or ""):
        call = parse_tool_call_body((match.group(1) or "").strip())
        if call is not None:
            calls.append((match.start(), match.end(), call))
    return calls


def call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    """Decode mapping or JSON-string tool arguments."""
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    return arguments if isinstance(arguments, dict) else {}


def tool_name(call: dict[str, Any]) -> str:
    """Normalize the name of a parsed tool call."""
    return str(call.get("name") or "").strip().lower()


def generate_prompts(calls: Sequence[tuple[int, int, dict[str, Any]]]) -> list[str]:
    """Collect nonempty prompts from parsed ``generate_image`` calls."""
    prompts = []
    for _, _, call in calls:
        if tool_name(call) != "generate_image":
            continue
        prompt = str(call_arguments(call).get("prompt") or "").strip()
        if prompt:
            prompts.append(prompt)
    return prompts


def assistant_prose(text: str) -> str:
    """Remove tool syntax and injected reflection from policy prose."""
    prose = TOOL_CALL_RE.sub(" ", text or "")
    prose = TOOL_OBS_LINE_RE.sub(" ", prose)
    prose = re.sub(r"</?think>", " ", prose, flags=re.IGNORECASE)
    # Masked, environment-injected reflection text cannot earn policy credit.
    prose = re.sub(
        r"(?is)\bReflection\s*:.*?(?:agentic_forced_reflection=1|agentic_force_stop_max_passes=1)\S*",
        " ",
        prose,
    )
    return re.sub(r"\s+", " ", prose).strip()


def assistant_prose_lines(text: str) -> str:
    """Strip protocol payloads while preserving plan-item line boundaries."""
    prose = TOOL_CALL_RE.sub("\n", text or "")
    prose = TOOL_OBS_LINE_RE.sub("", prose)
    prose = re.sub(r"</?think>", "", prose, flags=re.IGNORECASE)
    return prose


def tokens(text: str) -> set[str]:
    """Extract normalized unique word tokens for coverage scoring."""
    return set(re.findall(r"[a-z0-9_']+", (text or "").lower()))


def coverage(candidate: str, reference: str) -> float:
    """Token F1 of candidate vs reference (recall-only bag-of-words is not enough).

    Precision penalizes dumping the reference tokens into a long unrelated blob;
    recall still rewards covering the reference. Exact copy scores 1.0.
    """
    reference_tokens = tokens(reference)
    candidate_tokens = tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return 0.0
    overlap = len(reference_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    recall = overlap / len(reference_tokens)
    precision = overlap / len(candidate_tokens)
    return 2.0 * precision * recall / (precision + recall)


def good_enough(window: str) -> bool | None:
    """Read an explicit judge acceptance decision from one observation."""
    matches = list(re.finditer(r"\bgood_enough\s*=\s*(YES|NO|1|0|true|false)\b", window or "", re.IGNORECASE))
    if not matches:
        return None
    value = matches[-1].group(1).lower()
    return value in {"yes", "1", "true"}


def trusted_judges(observations: Iterable[ToolObservation]) -> list[JudgeHit]:
    """Parse judge values only inside validated runtime ``judge_image`` observations."""
    hits = []
    for observation in observations:
        if observation.name != "judge_image":
            continue
        window = observation.content
        if len(JUDGE_OK_RE.findall(window)) != 1:
            continue
        if JUDGE_OBS_HEADER not in window:
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
        accepted = good_enough(window)
        if len(values) == 2 and accepted is not None:
            hits.append(JudgeHit(values[0], values[1], accepted, observation.end))
    return hits


def terminal_decision(text: str, judges: Sequence[JudgeHit], *, done: str = r"Done\.") -> TerminalDecision:
    """Recognize policy completion following the final trusted verdict.

    The prose anchor is the end of the verdict's line (or the verdict itself when
    nothing follows it on a new line), advanced past every harness stop cue after
    the verdict. ``done`` is the scorer's terminal-marker regex.
    """
    if not judges:
        return TerminalDecision(False, False, False, len(text))

    judge_end = judges[-1].end
    line_end = text.find("\n", judge_end)
    anchor = judge_end if line_end < 0 else line_end + 1
    forced_context = False
    for marker in FORCED_CONTEXT_CUE_RE.finditer(text):
        if marker.start() < judge_end:
            continue
        # Only advance: moving back into the verdict line would expose harness prose.
        anchor = max(anchor, marker.end())
        forced_context = True

    suffix = assistant_prose(text[anchor:])
    suffix = re.sub(r"<\|[^>]+\|>|</?tool_response>|</?assistant>", " ", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"^\s*(?:assistant|user)\s+", "", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"\s+", " ", suffix).strip()
    policy_reflection = bool(REFLECTION_RE.search(suffix))
    if policy_reflection:
        terminal_done = bool(re.search(rf"\b{done}\s*$", suffix, re.IGNORECASE))
    else:
        terminal_done = bool(re.fullmatch(done, suffix, re.IGNORECASE))
    return TerminalDecision(terminal_done, policy_reflection, forced_context, anchor)


def generates_after_first_yes(calls: list[tuple[int, int, dict[str, Any]]], judges: Sequence[JudgeHit]) -> int:
    """Count ``generate_image`` calls made after the first trusted acceptance."""
    yes_position = next((hit.end for hit in judges if hit.good_enough is True), None)
    if yes_position is None:
        return 0
    return sum(1 for start, _, call in calls if start > yes_position and tool_name(call) == "generate_image")


def extract_plan_lines(text: str) -> list[str]:
    """Extract substantive numbered or bulleted policy plan items."""
    prose = assistant_prose_lines(text)
    header = re.search(r"\bPlan\s*:", prose, re.IGNORECASE)
    body = prose[header.end() :] if header else prose
    return [
        line
        for match in re.finditer(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$", body)
        if len(tokens(line := match.group(1).strip())) >= 4
    ]


def reflection_text(text: str) -> str:
    """Extract policy reflection prose without its terminal ``Done.`` marker."""
    prose = assistant_prose(text)
    match = re.search(r"\bReflection\s*:(.*?)(?:\bDone\.\s*$|$)", prose, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def judge_quality(judges: Sequence[JudgeHit]) -> float:
    """Mean C/A of the first accepted verdict, else of the last verdict."""
    if not judges:
        return 0.0
    preferred = next((hit for hit in judges if hit.good_enough is True), judges[-1])
    return 0.5 * (preferred.correctness + preferred.aesthetics)


def reference_reflection(ground_truth: Mapping[str, Any]) -> str:
    """Join the dataset ``reference_steps[*].reflection`` texts."""
    steps = as_sequence(ground_truth.get("reference_steps"))
    return " ".join(str(step.get("reflection") or "") for step in steps if isinstance(step, dict)).strip()


def judge_feedback(windows: Iterable[str]) -> str:
    """Join the ``findings`` / ``suggested_fixes`` lines of trusted judge windows."""
    feedback = [
        match.group(1).strip()
        for window in windows
        for match in re.finditer(r"(?im)^\s*(?:findings|suggested_fixes)\s*:\s*(.*)$", window)
    ]
    return " ".join(item for item in feedback if item.lower() not in {"", "none", "n/a"}).strip()


def reflection_score(quality: float, reflection: str, reference: str) -> float:
    """Blend judge quality with the coverage of the policy reflection over ``reference``."""
    if not reference:
        return quality
    return 0.5 * quality + 0.5 * coverage(reflection, reference)


def reference_subtasks(ground_truth: Mapping[str, Any]) -> list[str]:
    """Nonempty dataset ``reference_subtasks`` as strings."""
    return [str(item).strip() for item in as_sequence(ground_truth.get("reference_subtasks")) if str(item).strip()]


def plan_coverage(candidates: Sequence[str], references: Sequence[str]) -> float:
    """Average, over reference subtasks, the best coverage by any policy plan line."""
    if not references or not candidates:
        return 0.0
    return sum(max(coverage(candidate, reference) for candidate in candidates) for reference in references) / len(
        references
    )


def format_score(
    text: str,
    *,
    task_type: str,
    successful_generates: int,
    judged_after_last_generate: bool,
    terminal_done: bool,
    policy_reflection: bool,
    forced_context: bool,
    plan_lines: Sequence[str],
) -> float:
    """Fraction of structural protocol checks the trajectory passes.

    ``judged_after_last_generate`` is scorer-defined: a ``judge_image`` call for a
    Hermes trajectory, a trusted verdict for a dialect without judge calls.
    """
    raw_blocks = len(TOOL_CALL_RE.findall(text))
    calls = extract_tool_calls(text)
    checks = [
        raw_blocks > 0 and len(calls) == raw_blocks,
        successful_generates >= 1,
        judged_after_last_generate,
        terminal_done,
    ]
    if task_type == "plan":
        checks.extend((bool(plan_lines), policy_reflection or forced_context))
    else:
        # The loop force-injects Reflection (stripped from prose) then the policy writes Done.
        # Count forced_context so the default curriculum can still saturate format.
        checks.append(bool(REFLECTION_RE.search(assistant_prose(text))) or forced_context)
    return sum(checks) / len(checks)


def result_score(
    *,
    task_type: str,
    expected: int,
    successful_generates: int,
    terminal_done: bool,
    blocked: bool,
    rewrite_after_yes: int,
    judges: Sequence[JudgeHit],
    submission_images: int | None = None,
) -> float:
    """Task result: exact image count (plan) or a terminal trusted YES (reflect).

    ``submission_images`` counts only the images rendered before the first
    verdict -- the actual submission. A rewrite after a ``NO`` verdict is the
    reflection the harness demands, so charging it against ``expected`` is what
    zeroed reflect results while the policy obeyed the loop. The Hermes assessment
    passes this override only for reflect rows, preserving exact counts for plan
    rows. Other consumers that supply it for plan rows get graded count coverage.
    """
    if blocked or not terminal_done or successful_generates < 1 or rewrite_after_yes > 0:
        return 0.0
    counted = successful_generates if submission_images is None else submission_images
    if task_type == "plan":
        if submission_images is not None:
            return submission_images / expected if submission_images <= expected else 0.0
        return 1.0 if successful_generates == expected else 0.0
    final_yes = bool(judges) and judges[-1].good_enough is True
    # Fail closed on a terminal NO: early-stop alone is not a free result point.
    return 1.0 if final_yes and counted <= expected else 0.0


def resolve_solution_text(
    solution_str: str,
    *,
    kwargs: dict[str, Any],
    extra_info: dict[str, Any],
    scorer: str,
) -> str:
    """Resolve trajectory text for ``NaiveRewardManager`` (and optional decode).

    ``solution_image`` from ``VisualRewardManager`` is the wrong modality: raise
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
            ids = responses.tolist() if hasattr(responses, "tolist") else list(responses)
            if ids and isinstance(ids[0], list | tuple):
                ids = list(ids[0])
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
            if isinstance(decoded, str) and decoded.strip():
                return decoded.strip()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"{scorer}.compute_score failed to decode responses into solution_str") from exc

    if "solution_image" in kwargs:
        raise ValueError(
            f"{scorer}.compute_score requires solution_str (text trajectory). "
            "Got solution_image from VisualRewardManager — set "
            "reward.reward_manager.name=naive for Mode (2a)."
        )
    return ""


def require_task_type(ground_truth: Mapping[str, Any], extra_info: Mapping[str, Any]) -> str | None:
    """Require an explicit supported task type from ground truth or metadata."""
    raw = ground_truth.get("task_type")
    if raw is None:
        raw = extra_info.get("task_type")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    task_type = str(raw).strip()
    return task_type if task_type in TASK_TYPES else None


def expected_num_images(ground_truth: Mapping[str, Any], extra_info: Mapping[str, Any]) -> int:
    """``expected_num_images`` from ground truth or metadata, at least 1."""
    try:
        return max(1, int(ground_truth.get("expected_num_images", extra_info.get("expected_num_images", 1))))
    except (TypeError, ValueError):
        return 1


def active_weights(
    ground_truth: Mapping[str, Any],
    extra_info: Mapping[str, Any],
    *,
    task_type: str,
    dims: Sequence[str],
    defaults: Mapping[str, float] | None = None,
) -> dict[str, float] | None:
    """Positive weights of the active dimension set, or ``None`` if a ``w_*`` is garbage.

    ``plan`` is active only for plan rows. A row's ``w_<dim>`` (ground truth first,
    then metadata) overrides ``defaults`` (1.0 when absent); a non-numeric,
    non-finite or negative value fails closed. A zero weight drops the dimension.
    """
    weights = {}
    for dim in dims:
        if dim == "plan" and task_type != "plan":
            continue
        raw = ground_truth.get(f"w_{dim}")
        if raw is None:
            raw = extra_info.get(f"w_{dim}")
        if raw is None:
            value = float((defaults or {}).get(dim, 1.0))
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


def weighted_score(rewards: Mapping[str, float], weights: Mapping[str, float]) -> float:
    """Weighted mean of the active dimensions, capped at 1.

    Weights are rescaled by their maximum first, so large finite ``w_*`` values
    cannot overflow the sum.
    """
    scale = max(weights.values(), default=0.0)
    if not scale:
        return 0.0
    normalized = {dim: weight / scale for dim, weight in weights.items()}
    weight_sum = sum(normalized.values())
    return float(min(1.0, sum(normalized[dim] * rewards[dim] for dim in normalized) / weight_sum))
