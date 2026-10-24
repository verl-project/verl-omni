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
transcript.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

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

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
# --- Bagel Co-RL (Joint-Training) observation dialect -------------------------------
#
# ``bagel_corl_lib.parse_und_tool_call`` accepts four dialects because the Bagel
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
# sync with ``bagel_corl_lib._NATIVE_OPEN_TAGS`` / ``_NATIVE_CLOSE_TAGS`` by hand -- this
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
# Bagel's UND-facing image observation: ``bagel_corl_lib.compact_image_observation``
# appends the bare path of the one image the loop accepted. It is *absolute*, which is what
# keeps this from matching a ``path=`` that a prompt happened to mention, and it is glued to
# whatever the model wrote last (see ``_count_successful_generates``), so no left anchor.
_BAGEL_IMAGE_OBS_RE = re.compile(r"path=(/[^\s'\"]*?\.png)")
# Bagel has no ``agentic_judge ok=1`` observation either: the in-loop RM verdict is
# paraphrased by ``rpco_turn_protocol.build_forced_reflection`` into the forced
# *Reflection* line, and only that paraphrase reaches the transcript. Same numbers as
# ``format_rm_scores_as_judge_text`` would emit, different vocabulary.
_BAGEL_VERDICT_RE = re.compile(
    r"Reflection:\s*VL judge reports\s+correctness\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*"
    r"aesthetics\s*=\s*([0-9]*\.?[0-9]+)\s*,\s*good_enough\s*=\s*(YES|NO)\b",
    re.IGNORECASE,
)
# Harness-injected stop cues. Text carrying one is context-only (``response_mask=0``) and
# must not be read as policy prose. ``agentic_forced_reflection`` is what the Mode-2a loops
# append (``tool_agent_loop`` / ``image_gen_tool_agent_loop``); the other two are what
# ``rpco_turn_protocol.build_forced_reflection`` writes, and are the *only* cue the Bagel
# Co-RL loop emits -- it never appends ``agentic_forced_reflection``.
_FORCED_CONTEXT_CUE_RE = re.compile(
    r"\bagentic_(?:forced_reflection|force_stop_max_passes|stop_decision_required)=1(?!\d)",
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
_TOOL_OBS_LINE_RE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
_REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    return {
        "score": 0.0,
        **{f"reward_{dim}": 0.0 for dim in DIMS},
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


def _as_sequence(value: Any) -> list[Any]:
    """Coerce a dataset reference field into a list of items.

    ``reference_steps`` / ``reference_subtasks`` are written as Python lists but do not
    survive the parquet round-trip as ones: they come back as ``ndarray``, 1-3 elements
    wide. ``value or []`` is therefore a latent crash -- on a >1-element array it raises
    ``ValueError: The truth value of an array with more than one element is ambiguous``.

    Coercion is deliberately permissive: ``None`` (the unused sibling key) and an empty
    string degrade to no references, a plain string stays *one* reference rather than being
    iterated into characters, and an unlistable scalar degrades to itself.
    """
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
    """Wrap every line-anchored, named JSON object in ``<tool_call>`` tags.

    Runs *after* the ``<tools>`` rewrite, and skips spans that rewrite already produced
    (tracked by ``_TOOL_CALL_BLOCK_RE``) so a ``<tools>`` call is never wrapped twice -- a
    nested ``<tool_call>`` would make ``raw_blocks`` and ``_extract_tool_calls`` disagree
    again, which is the very check ``_format_reward`` opens with.
    """
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
    """First ``{"name": ...}`` object inside a native call body (an array, or a bare object).

    The native fence wraps a JSON **array** of calls, so :func:`_parse_tool_call_body` cannot
    read it directly. Scans the body's own balanced objects in order and takes the first one
    that names a tool, which mirrors ``bagel_corl_lib._native_payload``: a multi-call array
    still drives the loop on its first actionable element. Returns ``None`` for a body with no
    complete named object -- a turn cut by the token budget must not become a fabricated call.
    """
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
    """Rewrite every native ``<|... ToolCall|>[...]<|... ToolCall|>`` span into ``<tool_call>``.

    Runs *before* ``_wrap_bare_line_payloads`` (which would otherwise wrap a pretty-printed
    element of the same array a second time) and before the ``<tools>`` rewrite is irrelevant
    -- the spans cannot overlap. A fence whose body carries no named object is left untouched,
    matching the loop, which also declines to execute it.
    """

    def replace(match: re.Match[str]) -> str:
        payload = _first_named_object(match.group(1) or "")
        if payload is None:
            return match.group(0)
        return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"

    return _NATIVE_CALL_BLOCK_RE.sub(replace, text)


def _normalize_trajectory_dialect(text: str) -> str:
    """Rewrite the Bagel Co-RL call dialect into the Hermes vocabulary.

    The scorer's dims are protocol-shaped (``format`` counts ``<tool_call>`` blocks
    against parsed calls, ``result`` needs a judge verdict), and they were written against
    the Mode-2a Hermes transcript. The Bagel Joint-Training loop renders the *same* facts in
    its own vocabulary, so every dim read zero:

    * calls ride the schema tag -- ``<tools>{json}</tools>`` -- not ``<tool_call>``;
    * or they ride the checkpoint's native ``<|start of ToolCall|>[{json}]<|end of ToolCall|>``
      fence (every spelling in ``_NATIVE_OPEN_TAGS``/``_NATIVE_CLOSE_TAGS``), whose payload is an
      array rather than an object;
    * or they ride *no* tag at all, as a bare line-anchored JSON object (see
      ``_LINE_START_OBJECT_RE``);
    * a generated image is announced as a bare ``path=<abs>.png`` with no
      ``agentic_tool ok=1`` prefix (handled in ``_count_successful_generates``);
    * the RM verdict arrives as the forced ``Reflection: VL judge reports ...`` line
      (handled in ``_bagel_judge_hits``).

    Only the calls are a safe rewrite. The observation is *not* rewritten: a turn that runs
    out of context is truncated mid-JSON, so the loop's ``compact_image_observation`` is
    glued to the model's last tokens (``...ideal for`` + ``path=...``), and any rewrite that
    inserts a line break to satisfy the line-oriented matcher would have to guess whether it
    is landing inside a JSON string. The verdict is *parsed* rather than rewritten because
    it is attributable to ``generate_image``, not to a policy ``judge_image`` action, and
    laundering it into the Hermes judge gate would credit the wrong action.

    The rewrite is inert on a Hermes transcript: ``_TAGGED_CALL_RE`` only matches a JSON
    object, so the prompt's ``<tools>`` schema block -- a JSON *array* of function
    definitions -- can never be mistaken for a call, and a Hermes transcript keeps its calls
    inside ``<tool_call>`` blocks, which ``_wrap_bare_line_payloads`` skips.
    """
    if not text:
        return text
    text = _wrap_native_call_payloads(text)
    text = _TAGGED_CALL_RE.sub(lambda match: f"<tool_call>\n{match.group(1)}\n</tool_call>", text)
    return _wrap_bare_line_payloads(text)


def _extract_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    calls = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        call = _parse_tool_call_body((match.group(1) or "").strip())
        if call is not None:
            calls.append((match.start(), match.end(), call))
    return calls


def _follows_generate_image_call(pos: int, calls: list[tuple[int, int, dict[str, Any]]]) -> bool:
    """True when ``pos`` is after at least one parsed ``generate_image`` call.

    The Bagel verdict is produced by the *reward loop* judging the images the policy asked
    for, so ``generate_image`` -- not ``judge_image`` -- is the action it is attributable
    to. A turn that never asked for an image can never earn the verdict.
    """
    return any(end <= pos for _, end, call in calls if _tool_name(call) == "generate_image")


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


def _count_successful_generates(text: str) -> int:
    """Images the environment confirmed, in either lane's vocabulary.

    Hermes writes ``agentic_tool ok=1 path=<p>.png`` on its own line. Bagel appends the bare
    path of ``bagel_corl_lib.compact_image_observation``, and that observation is *glued* to
    the model's preceding tokens whenever the turn was truncated mid-JSON (``...ideal for`` +
    ``path=/abs/gen_x.png``). The line-oriented matcher cannot see it there: ``\\bagentic_tool``
    needs a word boundary, and ``</tools`` + ``agentic_tool`` sits between two word characters.
    Counting the bare form by occurrence sidesteps the glue entirely.

    The bare form is only consulted when the transcript carries *no* Hermes generate
    observation at all. A Hermes lane that reported ``ok=0`` writes a ``path=`` too, and a
    failed generate is a hard zero -- counting it would resurrect exactly the zero-credit
    rollout ``test_empty_and_failed_generate_rollouts_are_hard_zero`` pins.
    """
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
    for marker in _JUDGE_OK_RE.finditer(blob):
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


def _good_enough(window: str) -> bool | None:
    matches = list(re.finditer(r"\bgood_enough\s*=\s*(YES|NO|1|0|true|false)\b", window or "", re.IGNORECASE))
    if not matches:
        return None
    value = matches[-1].group(1).lower()
    return value in {"yes", "1", "true"}


def _bagel_judge_hits(
    text: str, calls: list[tuple[int, int, dict[str, Any]]] | None = None
) -> list[tuple[float, float, bool, int]]:
    """Trusted ``(correctness, aesthetics, good_enough, end)`` from a Bagel Reflection verdict.

    The Bagel loop never writes an ``agentic_judge ok=1`` observation: the reward loop
    judges the GEN samples of a ``generate_image`` call and
    ``rpco_turn_protocol.build_forced_reflection`` paraphrases the verdict into the forced
    ``Reflection: VL judge reports correctness=..., aesthetics=..., good_enough=...`` line,
    which IS in the transcript. Reading it here is not a loosening of the Hermes gate -- the
    Bagel verdict is attributable to a different action (the image request, not a
    ``judge_image`` call) and is gated on that instead.
    """
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
        hits.append((c, a, match.group(3).upper() == "YES", match.end()))
    return hits


def _successful_judges(text: str) -> list[tuple[float, float, bool | None, int]]:
    """Return trusted ``(correctness, aesthetics, good_enough, end)`` values.

    Hermes hits must follow a parsed ``judge_image`` ``<tool_call>`` and the tool's
    ``VL judge on the last generated image`` header. Bagel hits follow a ``generate_image``
    call and carry the verdict inline (``_bagel_judge_hits``). Both feed one
    chronologically ordered list, because every consumer -- ``_terminal_decision``'s anchor,
    ``_result_reward``'s final verdict, ``_generates_after_first_yes`` -- is asking about
    "the last verdict", regardless of which lane wrote it.
    """
    blob = text or ""
    calls = _extract_tool_calls(blob)
    hits = []
    for marker in _JUDGE_OK_RE.finditer(blob):
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
        hits.append((c, a, _good_enough(window), marker.end()))
    hits.extend(_bagel_judge_hits(blob, calls))
    hits.sort(key=lambda hit: hit[3])
    return hits


def _terminal_decision(text: str) -> tuple[bool, bool, bool]:
    judges = _successful_judges(text)
    if not judges:
        return False, False, False

    judge_end = judges[-1][3]
    line_end = text.find("\n", judge_end)
    # Prose anchor. No newline follows the verdict in the Bagel Co-RL transcript: the verdict is
    # embedded *inside* the harness's forced ``Reflection`` line and the harness writes its stop
    # cue after it, so everything past the verdict is harness context (``response_mask=0``), not
    # policy prose. Anchoring at ``len(text)`` -- what the line-oriented rule does, and is right
    # when the verdict ends a message -- hides the trailing policy ``Done.`` completely:
    # ``terminal_done`` stayed False, so ``R_format`` lost its terminal check and ``R_result`` was
    # pinned to 0 on every Bagel episode.
    anchor = judge_end if line_end < 0 else line_end + 1
    # Walk every cue forward, scanning from the verdict rather than from the prose anchor.
    # ``build_forced_reflection`` writes ``...agentic_force_stop_max_passes=1
    # agentic_stop_decision_required=1`` on the *same line* as the verdict it paraphrases, so
    # starting the scan at ``line_end + 1`` skipped every cue and ``forced_context`` was a
    # constant 0. That is load-bearing twice over: ``closed`` requires ``policy_reflection or
    # forced_context``, so ``reward_done`` was 0 on every episode, and ``R_format``'s
    # forced-context check could then only be satisfied by reading the harness's own
    # ``Reflection:`` line as policy prose. Measured 2026-10-12
    # (``bagel_corl_rm1_0123_20261012_084724``): ``forced_reflection_context`` and
    # ``reward_done`` were 0 at all 150 validation steps on both val splits. The Mode-2a loops
    # append ``agentic_forced_reflection=1`` on a later line, so the anchor still advances to the
    # end of the harness text either way.
    forced_context = False
    for marker in _FORCED_CONTEXT_CUE_RE.finditer(text):
        if marker.start() < judge_end:
            continue
        # Only ever *advance* the anchor. A cue inside the verdict line has
        # ``marker.end() < line_end + 1``, and moving the anchor back into that line would
        # expose the harness's own ``rubber_stamp=`` / ``findings:`` prose as if it were
        # policy text, which is what the trailing-``Done.`` check is trying not to read.
        anchor = max(anchor, marker.end())
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
    # Accept the checkpoint's own terminal form as well: ``{"name": "Done"}`` is the same
    # statement as prose ``Done.`` in the grammar this checkpoint actually speaks, and the
    # object's ``{`` is the last character before the anchor, so the suffix holds its tail
    # (``"name": "Done" }``). Read it from the whole post-verdict region rather than from
    # ``suffix``: when the policy puts the object on its own line,
    # ``_normalize_trajectory_dialect`` wraps it in a ``<tool_call>`` block and
    # ``_assistant_prose`` drops that block, so the marker would be invisible in ``suffix``
    # (measured: the glued form in 48/260 ``_024557`` episodes, plus the wrapped form).
    # Treating it as *not* a termination is what zeroed ``reward_done`` and ``reward_result``
    # and docked ``reward_format`` (0.4 vs 0.8) on those episodes.
    terminal_done = terminal_done or bool(_DONE_CALL_RE.search(text[anchor:]))
    return terminal_done, policy_reflection, forced_context


def _generates_after_first_yes(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> int:
    yes_position = next((end for _, _, accepted, end in _successful_judges(text) if accepted is True), None)
    if yes_position is None:
        return 0
    return sum(1 for start, _, call in calls if start > yes_position and _tool_name(call) == "generate_image")


def _extract_plan_lines(text: str) -> list[str]:
    prose = _assistant_prose_lines(text)
    header = re.search(r"\bPlan\s*:", prose, re.IGNORECASE)
    body = prose[header.end() :] if header else prose
    return [
        line
        for match in re.finditer(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+(.+)$", body)
        if len(_tokens(line := match.group(1).strip())) >= 4
    ]


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

    steps = _as_sequence(ground_truth.get("reference_steps"))
    reference = " ".join(str(step.get("reflection") or "") for step in steps if isinstance(step, dict)).strip()
    if not reference and judges:
        feedback = []
        for _, _, _, end in judges:
            window = text[max(0, end - 1400) : end]
            feedback.extend(
                match.group(1).strip()
                for match in re.finditer(r"(?im)^\s*(?:findings|suggested_fixes)\s*:\s*(.*)$", window)
            )
        reference = " ".join(item for item in feedback if item.lower() not in {"", "none", "n/a"}).strip()
    if not reference:
        return quality
    return 0.5 * quality + 0.5 * _coverage(_reflection_text(text), reference)


def _plan_reward(text: str, ground_truth: dict[str, Any]) -> float:
    references = [
        str(item).strip() for item in _as_sequence(ground_truth.get("reference_subtasks")) if str(item).strip()
    ]
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
) -> float:
    raw_blocks = len(_TOOL_CALL_RE.findall(text))
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
    last_generate_end = max((end for _, end, call in calls if _tool_name(call) == "generate_image"), default=-1)
    checks = [
        raw_blocks > 0 and len(calls) == raw_blocks,
        successful_generates >= 1,
        bool(verdicts) and verdicts[-1][3] > last_generate_end,
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
    raw = ground_truth.get("task_type")
    if raw is None:
        raw = extra_info.get("task_type")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    task_type = str(raw).strip()
    if task_type not in {"reflect", "plan"}:
        return None
    return task_type


def _protocol_reward(
    *,
    raw_blocks: int,
    parsed_calls: int,
    judge_ok: int,
    judge_failed: int,
    blocked: bool,
) -> float:
    """Parse-level protocol validity: every call block and every verdict parsed, no hard block.

    Deliberately disjoint from ``R_format``, which owns the *structural* checks
    (a generate happened, the verdict lands after the images it judged, the turn
    ended in the terminal token). This dim is the "nothing silently failed" half:
    an unparsed ``<tool_call>`` block, an ``agentic_judge ok=0``, or a
    ``blocked_after_*`` stop all cost it. A transcript that is entirely empty is
    not rewarded for having nothing to fail -- with no call at all there is no
    protocol to be valid.
    """
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
    """Image quality of the **final** pass plus the gain a rewrite bought.

    Scores the last trusted verdict (mean of its C/A) and adds ``max(0, final -
    first)`` when the episode ran more than one judged pass, capped at 1. A single
    pass is scored on its own quality -- never penalised for not having a rewrite
    to improve on -- and a rewrite that made the image worse earns no bonus, it
    keeps only the (lower) final quality. Zero when the transcript carries no
    trusted verdict, or when the episode hard-blocked or rewrote after a YES.
    """
    if successful_generates < 1 or blocked or rewrite_after_yes > 0:
        return 0.0
    qualities = [0.5 * (correctness + aesthetics) for correctness, aesthetics, _, _ in _successful_judges(text)]
    if not qualities:
        return 0.0
    final = qualities[-1]
    improvement = max(0.0, final - qualities[0]) if len(qualities) > 1 else 0.0
    return min(1.0, final + improvement)


def _active_weights(
    ground_truth: dict[str, Any], extra_info: dict[str, Any], *, task_type: str
) -> dict[str, float] | None:
    """Return positive active-set weights, or None if a ``w_*`` value is garbage.

    ``plan`` is active only for plan rows, mirroring the existing "a dim that does
    not apply is not scored" rule. Every other dim -- including the composite
    ``image``/``done``/``protocol`` -- is scored whenever this function is reached,
    which already requires a confirmed ``generate_image`` (a transcript without one
    returns a hard zero before the dims are computed). An absent ``w_<dim>`` falls
    back to ``DEFAULT_DIM_WEIGHTS``.
    """
    weights = {}
    for dim in DIMS:
        if dim == "plan" and task_type != "plan":
            continue
        raw = ground_truth.get(f"w_{dim}")
        if raw is None:
            raw = extra_info.get(f"w_{dim}")
        if raw is None:
            value = DEFAULT_DIM_WEIGHTS[dim]
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
            if value < 0:
                return None
        if value > 0:
            weights[dim] = value
    return weights


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
    gt = _as_dict(ground_truth)
    metadata = dict(extra_info or {})
    version = str(metadata.get("corl_reward_version", "v1"))
    if version not in ("v1", "v2"):
        raise ValueError("corl_reward_version must be v1 or v2")
    trusted = _trusted_v2_reward(metadata) if version == "v2" else None
    task_type = _require_task_type(gt, metadata)
    if task_type is None:
        return _zero_result(method="agentic_multidim_missing_task_type")

    try:
        expected = max(1, int(gt.get("expected_num_images", metadata.get("expected_num_images", 1))))
    except (TypeError, ValueError):
        expected = 1

    text = _resolve_solution_text(solution_str, kwargs=kwargs, extra_info=metadata)
    kwargs.pop("solution_image", None)
    if not text.strip():
        result = _zero_result(method="agentic_multidim_empty")
        result.update(task_type=task_type, expected_num_images=expected)
        return result
    # One canonical text for every helper below. The Bagel Co-RL loop renders the same
    # facts in a different vocabulary; rewriting it here (rather than teaching each dim
    # about two dialects) is what keeps ``_extract_tool_calls`` spans aligned with the
    # ``_JUDGE_OK_RE`` positions they are gated against.
    text = _normalize_trajectory_dialect(text)

    calls = _extract_tool_calls(text)
    prompts = _generate_prompts(calls)
    names = [_tool_name(call) for _, _, call in calls]
    judge_ok, judge_failed, judge_rate = _judge_parse_stats(text, calls)
    successful_generates = _count_successful_generates(text)
    terminal_done, policy_reflection, forced_context = _terminal_decision(text)
    blocked = bool(
        re.search(
            r"\b(?:blocked_after_yes|blocked_after_max_passes)=1\b|generate_image blocked:",
            text,
            re.IGNORECASE,
        )
    )
    rewrites_after_yes = _generates_after_first_yes(text, calls)
    # Resolved *after* the trajectory is in hand: a row with no generate, or one whose
    # calls all failed to parse, returns its structural zero above, so a bad ``w_*`` must
    # not turn that zero into a different zero with a misleading ``method``.
    weights = _active_weights(gt, metadata, task_type=task_type)
    if weights is None:
        return _zero_result(method="agentic_multidim_bad_weights")
    tool_reward = float(successful_generates >= 1 and judge_ok >= 1)

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
        if trusted is not None:
            result.update(trusted)
        return result

    valid_terminal_context = judge_ok > 0 and not blocked and rewrites_after_yes == 0
    closed = valid_terminal_context and terminal_done and (policy_reflection or forced_context)
    protocol_reward = _protocol_reward(
        raw_blocks=len(_TOOL_CALL_RE.findall(text)),
        parsed_calls=len(calls),
        judge_ok=judge_ok,
        judge_failed=judge_failed,
        blocked=blocked,
    )
    rewards = {
        "reflect": _reflection_reward(text, gt),
        "plan": _plan_reward(text, gt),
        "format": _format_reward(
            text,
            task_type=task_type,
            successful_generates=successful_generates,
            forced_context=forced_context,
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
        rewards["done"] = float(trusted["autonomous_done"])
        result.update(trusted)
        result["method"] = "agentic_multidim_corl_v2"
    weight_sum = sum(weights.values())
    score = sum(weights[dim] * rewards[dim] for dim in weights) / weight_sum if weight_sum else 0.0
    result.update(
        score=float(min(1.0, score)),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_tool_call=float(bool(calls)),
        protocol_ok=int(protocol_reward == 1.0),
        rollout_valid=1,
    )
    return result
