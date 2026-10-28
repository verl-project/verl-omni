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

Like the Hermes scorer (``agentic_multidim_reward``), this scorer is composed
from the shared ``agentic_reward_core`` primitives. It owns the BAGEL
observation dialect, verdict attribution, its extra dimensions and the
convergence objective, and imports no agent-loop or dataset modules.

Wire with ``reward.reward_manager.name=naive`` so verl passes ``solution_str``.
``VisualRewardManager``'s ``solution_image`` is the wrong modality and raises.

The active reward set is ``{reflect, plan, format, protocol, tool, result,
image, done}``. ``plan`` is active only for plan rows. ``tool_call`` reproduces
the PR1 closed-loop indicator for metrics and is not a score dimension. Invalid
rollouts (no parsed ``generate_image`` call or no successful PNG) receive score
zero and ``rollout_valid=0``, which is what keeps a text-only trajectory out of
the image dim: the dims below are only reached once an image exists. ``task_type``
is required.

``R_image`` / ``R_done`` / ``R_protocol`` are what make the scalar *composite*,
and each replaces a term that used to be metric-only or second-hand:

* ``R_image`` is the first-class image-quality dim: it scores the **final**
  trusted verdict, plus a bonus for the **improvement** a rewrite bought
  (``max(0, final - first)``, capped at 1). Before this dim the judge's C/A
  magnitude entered only inside ``R_reflect`` (halved, then averaged over five
  dims), so a rewrite that fixed the image barely moved the scalar.
* ``R_done`` is the Done credit: the ``closed`` indicator (trusted judge, no
  hard block, no rewrite after YES, a terminal ``Done.``) is now scored rather
  than reported, so properly terminating is worth reward.
* ``R_protocol`` is parse-level validity: every ``<tool_call>`` block parsed,
  no ``agentic_judge ok=0``, and no hard block. ``protocol_ok`` reports it.

Judge C/A is trusted only after a parsed ``judge_image`` ``<tool_call>`` and the
tool observation header. Coverage is token F1 (not recall-only), so dumping
reference words into a long blob does not max ``R_reflect`` / ``R_plan``.
Rewrite-after-YES zeros ``R_result``, ``R_image`` and the Done indicator.
Reflect ``R_result`` requires a terminal trusted YES. ``R_tool`` needs a
successful PNG generate plus a trusted judge (not merely a parsed tool call).

The Bagel Co-RL (Joint-Training) loop is the second producer of this reward and
speaks a different observation dialect: calls ride ``<tools>{...}</tools>``, a
generated image is announced as a bare ``path=<abs>.png``, and the in-loop RM
verdict arrives paraphrased in the forced ``Reflection: VL judge reports ...``
line instead of an ``agentic_judge ok=1`` observation. ``_normalize_trajectory_dialect``
maps the lexical differences and ``_bagel_judge_hits`` reads the verdict, gated on
the ``generate_image`` call it is attributable to. Both are inert on a Hermes
transcript. The loop publishes no runtime ``trajectory_messages``, so these
dims are read from the decoded transcript; under ``corl_reward_version=v2``
the image and done dims come from the trusted ``bagel_corl_reward`` envelope.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from verl_omni.utils.reward_score.agentic_reward_core import (
    BLOCKED_RE,
    JUDGE_OK_RE,
    TOOL_CALL_RE,
    JudgeHit,
    active_weights,
    as_dict,
    expected_num_images,
    extract_plan_lines,
    format_score,
    generates_after_first_yes,
    good_enough,
    judge_feedback,
    judge_quality,
    plan_coverage,
    reference_reflection,
    reference_subtasks,
    reflection_score,
    reflection_text,
    require_task_type,
    resolve_solution_text,
    result_score,
    terminal_decision,
    tool_name,
    weighted_score,
    zero_result,
)

# Bound as module globals so the scalar assembly can be exercised with parser stubs.
from verl_omni.utils.reward_score.agentic_reward_core import extract_tool_calls as _extract_tool_calls
from verl_omni.utils.reward_score.agentic_reward_core import generate_prompts as _generate_prompts

DIMS = ("reflect", "plan", "format", "protocol", "tool", "result", "image", "done")
#: Default weight per dim when the row carries no ``w_<dim>``. The five original
#: dims keep 1.0 so existing recipes are not rescaled relative to each other;
#: ``image`` is deliberately heavier because it is the only dim with a continuous,
#: task-specific magnitude -- the rest are mostly protocol gates that saturate.
#: ``done``/``protocol`` are 1.0 so termination and parse validity are trained at
#: the same order of magnitude as the behaviour dims they replace.
DEFAULT_DIM_WEIGHTS: dict[str, float] = {
    "reflect": 1.0,
    "plan": 1.0,
    "format": 1.0,
    "protocol": 1.0,
    "tool": 1.0,
    "result": 1.0,
    "image": 3.0,
    "done": 1.0,
}
# Names consumed by AgenticMetricsAgentLoopManager when PR1 and PR3 are
# composed. Keeping them here lets this independent PR specify that contract.
REWARD_COMPONENTS = tuple(f"reward_{name}" for name in (*DIMS, "tool_call"))

# --- Bagel Co-RL (Joint-Training) observation dialect -------------------------------
#
# ``agentic_und_gen.episode.parse_und_tool_call`` accepts four dialects because the Bagel
# checkpoint is handed a Hermes system prompt but does NOT answer in it: it emits its
# calls inside the *schema* tag (``<tools>{...}</tools>``) and never in the documented
# ``<tool_call>`` one. ``_TAGGED_CALL_RE`` deliberately accepts a JSON *object* only:
# the schema block the prompt used to show is a JSON *array* of function definitions,
# so it can never be mistaken for a call.
_TAGGED_CALL_RE = re.compile(r"<tools>\s*(\{.*?\})\s*</tools>", re.DOTALL)
# Bare payload at the **start of a line**: the checkpoint's other call dialect --
# ``{"name": "generate_image", "arguments": {...}}`` with no tag at all. The line-start
# anchor is what keeps a JSON example quoted *inside* prose ("emit {"name": ...} for each
# subtask") from being read as a call.
_LINE_START_OBJECT_RE = re.compile(r"(?m)^[ \t]*\{")
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
# The checkpoint's *native* call fence, and the third dialect the rollout accepts. Kept in
# sync with ``agentic_und_gen.episode._NATIVE_OPEN_TAGS`` / ``_NATIVE_CLOSE_TAGS`` by hand -- this
# scorer stays self-contained by design (see the module docstring), so the spellings are
# duplicated rather than imported. Unlike every other dialect the payload is a JSON *array* of
# call objects, so the rewrite has to pick the first element that carries an actionable
# ``name`` instead of re-emitting the body.
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
_NATIVE_CALL_BLOCK_RE = re.compile(
    rf"<\|(?:{_NATIVE_OPEN_ALT})\|>\s*(.*?)\s*(?:<\|(?:{_NATIVE_CLOSE_ALT})\|>|\Z)",
    re.DOTALL,
)
# Bagel's UND-facing image observation: ``agentic_und_gen.episode.compact_image_observation``
# appends the bare path of the one image the loop accepted. It is *absolute*, which is what
# keeps this from matching a ``path=`` that a prompt happened to mention, and it is glued to
# whatever the model wrote last (see ``_count_successful_generates``), so no left anchor.
_BAGEL_IMAGE_OBS_RE = re.compile(r"path=(/[^\s'\"]*?\.png)")
# Bagel has no ``agentic_judge ok=1`` observation either: the in-loop RM verdict is
# paraphrased by ``agentic_und_gen.turn_protocol.build_forced_reflection`` into the forced
# *Reflection* line, and only that paraphrase reaches the transcript. Same numbers as
# ``format_rm_scores_as_judge_text`` would emit, different vocabulary.
_BAGEL_VERDICT_RE = re.compile(
    r"Reflection:\s*VL judge reports\s+correctness\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*"
    r"aesthetics\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*good_enough\s*=\s*(YES|NO)\b",
    re.IGNORECASE,
)
# The terminal marker in the checkpoint's own dialect. ``build_forced_reflection`` tells the
# policy its next action "must be exactly Done.", but the Bagel checkpoint answers in the
# tool-call grammar it was trained on and emits ``{"name": "Done"}`` -- glued straight onto
# the stop cue (``...agentic_stop_decision_required=1{"name": "Done"}``). Because that object
# is not line-anchored, ``_normalize_trajectory_dialect`` does not wrap it in ``<tool_call>``
# and ``_extract_tool_calls`` never sees it, so ``_terminal_decision`` is the only reader that
# can recover the signal. Measured 2026-10-13 on ``_024557``: the policy terminated this way
# in 48/260 episodes and was scored as never having terminated at all.
_DONE_CALL_RE = re.compile(r'"name"\s*:\s*"Done"\s*\}', re.IGNORECASE)


def _balanced_object_end(text: str, start: int) -> int | None:
    """Index just past the JSON object starting at ``start`` (string-aware brace counter).

    ``text[start]`` must be ``{``. Returns ``None`` when the object never closes -- a turn cut
    by the token budget leaves a half-written payload, and wrapping that would inject a
    ``<tool_call>`` the model never emitted.
    """
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
                return index + 1
    return None


def _wrap_bare_line_payloads(text: str) -> str:
    """Wrap every line-anchored, named JSON object in ``<tool_call>`` tags."""
    protected = [match.span() for match in _TOOL_CALL_BLOCK_RE.finditer(text)]
    pieces: list[str] = []
    cursor = 0
    for match in _LINE_START_OBJECT_RE.finditer(text):
        start = match.end() - 1
        if start < cursor or any(low <= start < high for low, high in protected):
            continue
        end = _balanced_object_end(text, start)
        if end is None:
            continue
        body = text[start:end]
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict) or not payload.get("name"):
            continue
        pieces.append(text[cursor:start])
        pieces.append(f"<tool_call>\n{body}\n</tool_call>")
        cursor = end
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


def _first_named_object(raw: str) -> dict[str, Any] | None:
    """First ``{"name": ...}`` object inside a native call body (an array, or a bare object)."""
    if not raw:
        return None
    cursor = 0
    while True:
        start = raw.find("{", cursor)
        if start < 0:
            return None
        end = _balanced_object_end(raw, start)
        if end is None:
            return None
        try:
            payload = json.loads(raw[start:end])
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("name"):
            return payload
        cursor = end


def _wrap_native_call_payloads(text: str) -> str:
    """Rewrite every native ``<|... ToolCall|>[...]<|... ToolCall|>`` span into ``<tool_call>``."""

    def replace(match: re.Match[str]) -> str:
        payload = _first_named_object(match.group(1) or "")
        if payload is None:
            return match.group(0)
        return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"

    return _NATIVE_CALL_BLOCK_RE.sub(replace, text)


def _normalize_trajectory_dialect(text: str) -> str:
    """Rewrite the Bagel Co-RL call dialect into the Hermes vocabulary."""
    if not text:
        return text
    text = _wrap_native_call_payloads(text)
    text = _TAGGED_CALL_RE.sub(lambda match: f"<tool_call>\n{match.group(1)}\n</tool_call>", text)
    return _wrap_bare_line_payloads(text)


def _follows_generate_image_call(pos: int, calls: list[tuple[int, int, dict[str, Any]]]) -> bool:
    """True when ``pos`` is after at least one parsed ``generate_image`` call.

    The Bagel verdict is produced by the *reward loop* judging the images the policy asked
    for, so ``generate_image`` -- not ``judge_image`` -- is the action it is attributable
    to. A turn that never asked for an image can never earn the verdict.
    """
    return any(end <= pos for _, end, call in calls if tool_name(call) == "generate_image")


def _follows_judge_image_call(pos: int, calls: list[tuple[int, int, dict[str, Any]]]) -> bool:
    """True when ``pos`` is after at least one parsed ``judge_image`` ``<tool_call>``."""
    return any(end <= pos for _, end, call in calls if tool_name(call) == "judge_image")


def _count_successful_generates(text: str) -> int:
    """Images the environment confirmed, in either lane's vocabulary."""
    blob = text or ""
    strict = sum(
        1
        for line in blob.splitlines()
        if re.search(r"\bagentic_tool\s+ok=1\b", line, re.IGNORECASE)
        and any(path.lower().endswith(".png") for path in re.findall(r"\bpath=([^\s'\"]+)", line, re.IGNORECASE))
    )
    if strict:
        return strict
    if re.search(r"\bagentic_tool\s+ok=[01]\b", blob, re.IGNORECASE):
        return 0
    return len(_BAGEL_IMAGE_OBS_RE.findall(blob))


def _judge_parse_stats(text: str, calls: list[tuple[int, int, dict[str, Any]]] | None = None) -> tuple[int, int, float]:
    blob = text or ""
    parsed_calls = calls if calls is not None else _extract_tool_calls(blob)
    ok = 0
    for marker in JUDGE_OK_RE.finditer(blob):
        if _follows_judge_image_call(marker.start(), parsed_calls):
            ok += 1
    # Bagel's verdict is an observation over ``generate_image``; a malformed one cannot be
    # rendered at all (``build_forced_reflection`` bails when the numbers are absent), so it
    # only ever counts as a parse *success* and never inflates the failure rate.
    ok += sum(1 for _, _, _, _ in _bagel_judge_hits(blob, parsed_calls))
    failed = 0
    for marker in re.finditer(r"\bagentic_judge\s+ok=0\b", blob, flags=re.IGNORECASE):
        if _follows_judge_image_call(marker.start(), parsed_calls):
            failed += 1
    if failed == 0:
        for marker in re.finditer(r"\bagentic_judge\s+ok=0\b|\bparse_ok\s*=\s*0\b", blob, re.IGNORECASE):
            if _follows_judge_image_call(marker.start(), parsed_calls):
                failed += 1
    total = ok + failed
    return ok, failed, (ok / total) if total else 0.0


def _bagel_judge_hits(text: str, calls: list[tuple[int, int, dict[str, Any]]] | None = None) -> list[JudgeHit]:
    """Trusted verdicts from a Bagel Reflection paraphrase."""
    parsed_calls = calls if calls is not None else _extract_tool_calls(text or "")
    hits = []
    for match in _BAGEL_VERDICT_RE.finditer(text or ""):
        if not _follows_generate_image_call(match.start(), parsed_calls):
            continue
        try:
            c = min(1.0, max(0.0, float(match.group(1))))
            a = min(1.0, max(0.0, float(match.group(2))))
        except ValueError:
            continue
        hits.append(JudgeHit(c, a, match.group(3).upper() == "YES", match.end()))
    return hits


def _successful_judges(text: str) -> list[JudgeHit]:
    """Trusted verdicts in either dialect, in transcript order."""
    blob = text or ""
    calls = _extract_tool_calls(blob)
    hits = []
    for marker in JUDGE_OK_RE.finditer(blob):
        if not _follows_judge_image_call(marker.start(), calls):
            continue
        window = blob[max(0, marker.start() - 1400) : marker.end()]
        if "VL judge on the last generated image" not in window:
            continue
        if re.search(r"\bparse_ok\s*=\s*0\b", window, re.IGNORECASE):
            continue
        correctness = list(re.finditer(r"\bcorrectness\s*=\s*([0-9]*\.?[0-9]+)", window, re.IGNORECASE))
        aesthetics = list(re.finditer(r"\baesthetics\s*=\s*([0-9]*\.?[0-9]+)", window, re.IGNORECASE))
        if not correctness or not aesthetics:
            continue
        try:
            c = min(1.0, max(0.0, float(correctness[-1].group(1))))
            a = min(1.0, max(0.0, float(aesthetics[-1].group(1))))
        except ValueError:
            continue
        hits.append(JudgeHit(c, a, good_enough(window), marker.end()))
    hits.extend(_bagel_judge_hits(blob, calls))
    hits.sort(key=lambda hit: hit.end)
    return hits


def _terminal_decision(text: str) -> tuple[bool, bool, bool]:
    """``(terminal_done, policy_reflection, forced_context)`` after the final verdict.

    The Bagel verdict sits *inside* the harness's forced ``Reflection`` line, with the
    stop cue after it on the same line, so the core anchors at the verdict and walks
    every cue forward. The period of ``Done.`` is optional in this dialect.
    """
    decision = terminal_decision(text, _successful_judges(text), done=r"Done\.?")
    # Accept the checkpoint's own terminal form as well: ``{"name": "Done"}`` is the same
    # statement as prose ``Done.`` in the grammar this checkpoint actually speaks, and the
    # object's ``{`` is the last character before the anchor, so the suffix holds its tail
    # (``"name": "Done" }``). Read it from the whole post-verdict region rather than from
    # the prose suffix: when the policy puts the object on its own line,
    # ``_normalize_trajectory_dialect`` wraps it in a ``<tool_call>`` block and
    # ``assistant_prose`` drops that block, so the marker would be invisible in the suffix
    # (measured: the glued form in 48/260 ``_024557`` episodes, plus the wrapped form).
    # Treating it as *not* a termination is what zeroed ``reward_done`` and ``reward_result``
    # and docked ``reward_format`` (0.4 vs 0.8) on those episodes.
    terminal_done = decision.terminal_done or bool(_DONE_CALL_RE.search(text[decision.anchor :]))
    return terminal_done, decision.policy_reflection, decision.forced_context


def _generates_after_first_yes(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> int:
    return generates_after_first_yes(calls, _successful_judges(text))


def _extract_initial_plan_lines(text: str) -> list[str]:
    """V2 plans precede the first tool call, including within one sampled turn.

    Normalize every supported call dialect before finding the boundary. Removing
    tool JSON first would erase that boundary and let post-call feedback bullets
    masquerade as an initial subtask plan. Legacy extraction remains unchanged.
    """
    normalized = _normalize_trajectory_dialect(text)
    first_call = re.search(r"<tool_call\b", normalized, re.IGNORECASE)
    return extract_plan_lines(normalized[: first_call.start()] if first_call else normalized)


def _reflection_reward(text: str, ground_truth: dict[str, Any]) -> float:
    judges = _successful_judges(text)
    reference = reference_reflection(ground_truth) or judge_feedback(
        text[max(0, hit.end - 1400) : hit.end] for hit in judges
    )
    return reflection_score(judge_quality(judges), reflection_text(text), reference)


def _plan_reward(text: str, ground_truth: dict[str, Any], *, initial_only: bool = False) -> float:
    candidates = _extract_initial_plan_lines(text) if initial_only else extract_plan_lines(text)
    return plan_coverage(candidates, reference_subtasks(ground_truth))


def _format_reward(
    text: str,
    *,
    task_type: str,
    successful_generates: int,
    forced_context: bool = False,
    initial_plan_only: bool = False,
) -> float:
    calls = _extract_tool_calls(text)
    terminal_done, policy_reflection, _ = _terminal_decision(text)
    # The judge check gates on the *verdict*, not on a ``judge_image`` call index. Hermes
    # renders the verdict as a ``judge_image`` ``<tool_call>`` plus an ``agentic_judge ok=1``
    # observation; Bagel never emits a ``judge_image`` call at all (its judge is the colocated
    # reward loop) and carries the verdict inside the forced ``Reflection: VL judge reports
    # ...`` line. Gating on ``judge_image`` call indices therefore capped the Bagel lane's
    # ``R_format`` at 4/5 and pinned ``protocol_ok`` to 0 forever, on every episode, no matter
    # how clean the trajectory was. ``_successful_judges`` is the one list that already knows
    # both dialects, and requiring it to land after the images it judged is the property the
    # check was always trying to state.
    verdicts = _successful_judges(text)
    last_generate_end = max((end for _, end, call in calls if tool_name(call) == "generate_image"), default=-1)
    return format_score(
        text,
        task_type=task_type,
        successful_generates=successful_generates,
        judged_after_last_generate=bool(verdicts) and verdicts[-1].end > last_generate_end,
        terminal_done=terminal_done,
        policy_reflection=policy_reflection,
        forced_context=forced_context,
        plan_lines=_extract_initial_plan_lines(text) if initial_plan_only else extract_plan_lines(text),
    )


def _result_reward(
    text: str,
    *,
    task_type: str,
    expected: int,
    successful_generates: int,
    terminal_done: bool,
    blocked: bool,
    rewrite_after_yes: int,
    graded_plan: bool = False,
) -> float:
    judges = _successful_judges(text)
    submission_images = None
    if graded_plan:
        # Only images rendered before the first verdict are the *submission*: a rewrite after
        # a NO verdict is the reflection the harness demands, not an extra attempt. Charging
        # it against the budget zeroed every reflect result (val 0.04-0.08) while the policy
        # was doing exactly what the env asked.
        first_verdict_end = judges[0].end if judges else len(text)
        submission_images = min(
            successful_generates,
            sum(
                1
                for _, end, call in _extract_tool_calls(text)
                if tool_name(call) == "generate_image" and end <= first_verdict_end
            ),
        )
    return result_score(
        task_type=task_type,
        expected=expected,
        successful_generates=successful_generates,
        terminal_done=terminal_done,
        blocked=blocked,
        rewrite_after_yes=rewrite_after_yes,
        judges=judges,
        submission_images=submission_images,
    )


def _protocol_reward(
    *,
    raw_blocks: int,
    parsed_calls: int,
    judge_ok: int,
    judge_failed: int,
    blocked: bool,
) -> float:
    """Parse-level protocol validity: every call block and every verdict parsed, no hard block."""
    if parsed_calls < 1:
        return 0.0
    checks = [
        raw_blocks > 0 and parsed_calls == raw_blocks,
        judge_failed == 0,
        judge_ok >= 1,
        not blocked,
    ]
    return sum(checks) / len(checks)


def _image_reward(
    text: str,
    *,
    successful_generates: int,
    blocked: bool,
    rewrite_after_yes: int,
) -> float:
    """Image quality of the **final** pass plus the gain a rewrite bought."""
    if successful_generates < 1 or blocked or rewrite_after_yes > 0:
        return 0.0
    qualities = [0.5 * (correctness + aesthetics) for correctness, aesthetics, _, _ in _successful_judges(text)]
    if not qualities:
        return 0.0
    final = qualities[-1]
    improvement = max(0.0, final - qualities[0]) if len(qualities) > 1 else 0.0
    return min(1.0, final + improvement)


def _trusted_v2_reward(metadata):
    """Validate structured harness metadata; policy-written numbers are never parsed."""
    row = metadata.get("bagel_corl_reward")
    if not isinstance(row, dict) or row.get("origin") != "bagel_corl_tool_rm" or row.get("version") != "v2":
        raise ValueError("V2 requires trusted bagel_corl_reward envelope")
    calls = row.get("call_seed_scores")
    count = row.get("num_gen_calls")
    if not isinstance(calls, list) or not isinstance(count, int) or count < 0 or len(calls) != count:
        raise ValueError("invalid trusted V2 call count")
    means = []
    for scores in calls:
        if not scores or not all(isinstance(v, int | float) and math.isfinite(v) for v in scores):
            raise ValueError("invalid trusted V2 seed scores")
        means.append(sum(scores) / len(scores))
    for key in ("autonomous_done", "forced_reflection"):
        if not isinstance(row.get(key), bool):
            raise ValueError("missing trusted V2 closure flags")
    if row["autonomous_done"] and row["forced_reflection"]:
        raise ValueError("forced closure cannot be autonomous")
    if means:
        expected = {
            "image_reward_first": means[0],
            "image_reward_final": means[-1],
            "image_reward_best": max(means),
            "image_reward_gain": means[-1] - means[0],
            "image_reward_objective": means[-1] - 0.01 * max(0, count - 1),
        }
        for key, value in expected.items():
            if not isinstance(row.get(key), int | float) or not math.isclose(row[key], value, abs_tol=1e-9):
                raise ValueError("inconsistent trusted V2 " + key)
    return row


def _scalar_reward_fields(trusted):
    """The reward loop stacks every extra field with ``np.array`` across the batch, so the
    per-call score lists, whose length is the episode's K, would make it ragged and raise.
    Validation then averages every numeric field, so a K=0 episode's ``None`` image
    scores are reported as 0.0, the image reward that episode is scored with."""
    return {
        key: (0.0 if value is None and key.startswith("image_reward_") else value)
        for key, value in trusted.items()
        if not isinstance(value, list | tuple | dict)
    }


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
        extra_info: Fallback for ``task_type`` / weights; optional tokenizer.
        **kwargs: May include ``responses`` + tokenizer, or ``solution_image``
            (rejected).

    Returns:
        Dict with ``score``, per-dim ``reward_*``, and metric schema fields.
    """
    del data_source
    gt = as_dict(ground_truth)
    metadata = dict(extra_info or {})
    version = str(metadata.get("corl_reward_version", "v1"))
    if version not in ("v1", "v2"):
        raise ValueError("corl_reward_version must be v1 or v2")
    trusted = _trusted_v2_reward(metadata) if version == "v2" else None
    task_type = require_task_type(gt, metadata)
    if task_type is None:
        return zero_result(method="agentic_multidim_missing_task_type", components=REWARD_COMPONENTS)

    expected = expected_num_images(gt, metadata)

    text = resolve_solution_text(solution_str, kwargs=kwargs, extra_info=metadata, scorer="bagel_corl_multidim_reward")
    kwargs.pop("solution_image", None)
    if not text.strip():
        result = zero_result(method="agentic_multidim_empty", components=REWARD_COMPONENTS)
        result.update(task_type=task_type, expected_num_images=expected)
        return result
    # One canonical text for every helper below. The Bagel Co-RL loop renders the same
    # facts in a different vocabulary; rewriting it here (rather than teaching each dim
    # about two dialects) is what keeps ``_extract_tool_calls`` spans aligned with the
    # ``JUDGE_OK_RE`` positions they are gated against.
    text = _normalize_trajectory_dialect(text)

    calls = _extract_tool_calls(text)
    prompts = _generate_prompts(calls)
    names = [tool_name(call) for _, _, call in calls]
    judge_ok, judge_failed, judge_rate = _judge_parse_stats(text, calls)
    successful_generates = _count_successful_generates(text)
    terminal_done, policy_reflection, forced_context = _terminal_decision(text)
    blocked = bool(BLOCKED_RE.search(text))
    rewrites_after_yes = _generates_after_first_yes(text, calls)
    # Resolved *after* the trajectory is in hand: a row with no generate, or one whose
    # calls all failed to parse, returns its structural zero above, so a bad ``w_*`` must
    # not turn that zero into a different zero with a misleading ``method``.
    weights = active_weights(gt, metadata, task_type=task_type, dims=DIMS, defaults=DEFAULT_DIM_WEIGHTS)
    if weights is None:
        return zero_result(method="agentic_multidim_bad_weights", components=REWARD_COMPONENTS)
    tool_reward = float(successful_generates >= 1 and judge_ok >= 1)

    result = zero_result(method="agentic_multidim", components=REWARD_COMPONENTS)
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
        if trusted is not None:
            result.update(_scalar_reward_fields(trusted))
        return result

    valid_terminal_context = judge_ok > 0 and not blocked and rewrites_after_yes == 0
    closed = valid_terminal_context and terminal_done and (policy_reflection or forced_context)
    protocol_reward = _protocol_reward(
        raw_blocks=len(TOOL_CALL_RE.findall(text)),
        parsed_calls=len(calls),
        judge_ok=judge_ok,
        judge_failed=judge_failed,
        blocked=blocked,
    )
    rewards = {
        "reflect": _reflection_reward(text, gt),
        "plan": _plan_reward(text, gt, initial_only=version == "v2"),
        "format": _format_reward(
            text,
            task_type=task_type,
            successful_generates=successful_generates,
            forced_context=forced_context,
            initial_plan_only=version == "v2",
        ),
        "protocol": protocol_reward,
        "tool": tool_reward,
        "result": _result_reward(
            text,
            task_type=task_type,
            expected=expected,
            successful_generates=successful_generates,
            terminal_done=terminal_done,
            blocked=blocked,
            rewrite_after_yes=rewrites_after_yes,
            graded_plan=version == "v2",
        ),
        "image": _image_reward(
            text,
            successful_generates=successful_generates,
            blocked=blocked,
            rewrite_after_yes=rewrites_after_yes,
        ),
        "done": float(closed),
    }
    if trusted is not None:
        # Keep protocol/planning dimensions and weights explicit and unchanged.
        rewards["image"] = float(trusted["image_reward_objective"])
        if task_type == "plan":
            # The harness charges every call after the first as a rewrite; the planned
            # subtask images are the task, so only calls past ``expected`` are charged.
            count = trusted["num_gen_calls"]
            rewards["image"] = float(trusted["image_reward_final"]) - 0.01 * max(0, count - expected)
        rewards["done"] = float(trusted["autonomous_done"])
        result.update(_scalar_reward_fields(trusted))
        result["method"] = "agentic_multidim_corl_v2"
        if forced_context and not policy_reflection:
            # The harness wrote the reflection, so ``reflect`` is half the rubric verdict and
            # half the coverage of empty policy prose: a coarse copy of ``image``, not a
            # policy-controllable dimension.
            weights.pop("reflect", None)
        # ``done`` is ``autonomous_done``, and the harness sets that to False on every episode
        # it forced a reflection into -- which is nearly all of them. A dim that is 0 on
        # almost every row cannot teach termination; it only rescales the rest of the
        # composite down by its weight share. Keep reporting ``reward_done`` as a diagnostic,
        # but leave it out of the score.
        weights.pop("done", None)
    result.update(
        score=weighted_score(rewards, weights),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_tool_call=float(bool(calls)),
        protocol_ok=int(protocol_reward == 1.0),
        rollout_valid=1,
    )
    return result
