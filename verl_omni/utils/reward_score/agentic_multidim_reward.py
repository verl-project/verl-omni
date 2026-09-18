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

The active reward set is ``{reflect, format, tool, result, improve}``. ``improve``
scores the judge-outcome lift across the rewrite chain. ``done`` and ``tool_call``
reproduce the PR1 closed-loop indicators for metrics, but are not additional score
dimensions. Invalid rollouts (no parsed ``generate_image`` call or no successful PNG)
receive score zero and ``rollout_valid=0``.

Every data source is scored by this one formula. The reflect and plan corpora run the
same loop and the same weights and differ only in their system prompt, so their reward
curves are comparable; a row's ``task_type`` is carried for monitoring only and never
selects a dimension. ``task_type`` is still required as a data contract.

Judge C/A is trusted only after a parsed ``judge_image`` ``<tool_call>`` and the
tool observation header. Coverage is token F1 (not recall-only), so dumping
reference words into a long blob does not max ``R_reflect``.
Rewrite-after-YES zeros ``R_result`` as well as the Done indicator. ``R_result``
requires a terminal trusted YES. ``R_tool`` needs a successful
PNG generate plus a trusted judge (not merely a parsed tool call).
"""

from __future__ import annotations

import json
import re
from typing import Any

from verl_omni.utils.agentic.judge_delta import judge_delta_reward

DIMS = ("reflect", "format", "tool", "result", "improve")
#: Dims renamed after parquet was already built under the old weight key.
_LEGACY_WEIGHT_ALIASES: dict[str, tuple[str, ...]] = {"improve": ("w_novelty",)}
# Names consumed by AgenticMetricsAgentLoopManager when PR1 and PR3 are
# composed. Keeping them here lets this independent PR specify that contract.
REWARD_COMPONENTS = tuple(f"reward_{name}" for name in (*DIMS, "done", "tool_call"))

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_TOOL_OBS_LINE_RE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool(?:_dropped)?|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
_REFLECTION_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    return {
        "score": 0.0,
        **{f"reward_{dim}": 0.0 for dim in DIMS},
        "reward_done": 0.0,
        "reward_tool_call": 0.0,
        "num_hermes_tool_calls": 0,
        "num_generate_image_prompts": 0,
        "num_generate_image_prompts_requested": 0,
        "num_generate_image_prompts_dropped": 0,
        "num_judge_image_calls": 0,
        "num_judge_image_calls_requested": 0,
        "num_judge_image_calls_dropped": 0,
        "judge_parse_ok": 0,
        "judge_parse_fail": 0,
        "judge_parse_ok_rate": 0.0,
        "judge_delta": 0.0,
        "num_prompt_rewrites": 0,
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
    """Strip protocol payloads while preserving line boundaries."""
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


def _count_executed_generates(text: str) -> int:
    """Count ``generate_image`` responses the harness actually ran.

    Every executed generate leaves one ``agentic_tool ok=`` marker: ``ok=1`` with an
    image path, or ``ok=0`` when ``max_generate_image_passes`` refuses the call.
    Judges emit ``agentic_judge``, the Reflection cue emits ``agentic_reflect`` and the
    drop notice emits ``agentic_tool_dropped`` (no ``ok=``), so this counts generates
    only and cannot be inflated by the text's ``<tool_call>`` blocks.

    Args:
        text: Decoded trajectory text.

    Returns:
        Number of executed ``generate_image`` calls.
    """
    return len(re.findall(r"\bagentic_tool\s+ok=[01]\b", text or "", re.IGNORECASE))


def _count_successful_generates(text: str) -> int:
    return sum(
        1
        for line in (text or "").splitlines()
        if re.search(r"\bagentic_tool\s+ok=1\b", line, re.IGNORECASE)
        and any(path.lower().endswith(".png") for path in re.findall(r"\bpath=([^\s'\"]+)", line, re.IGNORECASE))
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
    return hits


def _terminal_decision(text: str) -> tuple[bool, bool, bool]:
    judges = _successful_judges(text)
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


def _generates_after_first_yes(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> int:
    yes_position = next((end for _, _, accepted, end in _successful_judges(text) if accepted is True), None)
    if yes_position is None:
        return 0
    return sum(1 for start, _, call in calls if start > yes_position and _tool_name(call) == "generate_image")


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
            feedback.extend(
                match.group(1).strip()
                for match in re.finditer(r"(?im)^\s*(?:findings|suggested_fixes)\s*:\s*(.*)$", window)
            )
        reference = " ".join(item for item in feedback if item.lower() not in {"", "none", "n/a"}).strip()
    if not reference:
        return quality
    return 0.5 * quality + 0.5 * _coverage(_reflection_text(text), reference)


def _format_reward(
    text: str,
    *,
    successful_generates: int,
    forced_context: bool = False,
) -> float:
    raw_blocks = len(_TOOL_CALL_RE.findall(text))
    calls = _extract_tool_calls(text)
    names = [_tool_name(call) for _, _, call in calls]
    generates = [index for index, name in enumerate(names) if name == "generate_image"]
    judges = [index for index, name in enumerate(names) if name == "judge_image"]
    terminal_done, _, _ = _terminal_decision(text)
    checks = [
        raw_blocks > 0 and len(calls) == raw_blocks,
        successful_generates >= 1,
        bool(judges) and (not generates or max(judges) > max(generates)),
        terminal_done,
        # #409 force-injects Reflection (stripped from prose) then policy Done.
        # Count forced_context so the default curriculum can still saturate format.
        bool(_REFLECTION_RE.search(_assistant_prose(text))) or forced_context,
    ]
    return sum(checks) / len(checks)


def _expected_num_images(ground_truth: dict[str, Any], extra_info: dict[str, Any]) -> int | None:
    """Return the reference image budget for a row, or ``None`` when there is none.

    ``expected_num_images`` is *derived*, not a curated label: reflect rows take it
    from the image count of the reference trajectory and plan rows from the number of
    plan slots. Rows with no reference trajectory at all (the ``No breakdown
    needed.`` sentinel) therefore have no budget to report, and the field is absent or
    ``None``. Coercing those to ``1`` would invent a one-shot cap.

    Args:
        ground_truth: ``reward_model.ground_truth`` mapping.
        extra_info: ``extra_info`` fallback mapping.

    Returns:
        Positive budget, or ``None`` when the row carries no reference budget.
    """
    for source in (ground_truth, extra_info):
        if source is None or "expected_num_images" not in source:
            continue
        raw = source["expected_num_images"]
        if raw is None:
            continue
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return None
    return None


def _report_expected(expected: int | None) -> int:
    """Project the budget into the reward dict as an int.

    ``0`` means "this row has no reference budget" (no reference trajectory), which is
    distinct from a positive budget. Kept numeric so the field stays safe to average
    and log.
    """
    return 0 if expected is None else int(expected)


def _result_reward(
    text: str,
    *,
    successful_generates: int,
    terminal_done: bool,
    blocked: bool,
    rewrite_after_yes: int,
) -> float:
    """Score the rollout's own stopping decision, for every data source.

    ``expected_num_images`` is deliberately NOT enforced. It is the reference
    trajectory's image count, and the protocol tells the agent to rewrite until the
    judge is satisfied while hiding this field from it, so capping at the reference's
    budget punishes exactly the iteration the task asks for. Nothing is lost by dropping
    it: ``rewrite_after_yes`` already blocks generating after a YES (so a rollout cannot
    farm result points by looping), and ``agentic_image_gen.max_generate_image_passes``
    already bounds total generates at the tool. Enforcing cost belongs there, not here,
    because a per-dim reward cannot separate "iterated usefully" from "iterated". Fail
    closed on a terminal NO: early-stop alone is not a free result point.
    """
    if blocked or not terminal_done or successful_generates < 1 or rewrite_after_yes > 0:
        return 0.0
    judges = _successful_judges(text)
    final_yes = bool(judges) and judges[-1][2] is True
    return 1.0 if final_yes else 0.0


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


def _active_weights(ground_truth: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, float] | None:
    """Return positive active-set weights, or None if a ``w_*`` value is garbage."""
    weights = {}
    for dim in DIMS:
        raw = ground_truth.get(f"w_{dim}")
        if raw is None:
            raw = extra_info.get(f"w_{dim}")
        if raw is None:
            # ``improve`` replaced the ``novelty`` dim, and parquet built before the
            # swap baked ``w_novelty``. Honour the old key so a dataset rebuild is not
            # required to keep ``RPCO_W_NOVELTY`` meaningful.
            for alias in _LEGACY_WEIGHT_ALIASES.get(dim, ()):
                raw = ground_truth.get(alias)
                if raw is None:
                    raw = extra_info.get(alias)
                if raw is not None:
                    break
        if raw is None:
            value = 1.0
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
            optional references and ``w_*`` weights. ``task_type`` is carried into the
            metrics for monitoring; it does not select a dimension.
        extra_info: Fallback for ``task_type`` / weights; optional tokenizer.
        **kwargs: May include ``responses`` + tokenizer, or ``solution_image``
            (rejected).

    Returns:
        Dict with ``score``, per-dim ``reward_*``, and metric schema fields.
    """
    del data_source
    gt = _as_dict(ground_truth)
    metadata = dict(extra_info or {})
    task_type = _require_task_type(gt, metadata)
    if task_type is None:
        return _zero_result(method="agentic_multidim_missing_task_type")
    weights = _active_weights(gt, metadata)
    if weights is None:
        return _zero_result(method="agentic_multidim_bad_weights")

    expected = _expected_num_images(gt, metadata)

    text = _resolve_solution_text(solution_str, kwargs=kwargs, extra_info=metadata)
    kwargs.pop("solution_image", None)
    if not text.strip():
        result = _zero_result(method="agentic_multidim_empty")
        result.update(task_type=task_type, expected_num_images=_report_expected(expected))
        return result

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
    with_judge_reward = float(successful_generates >= 1 and judge_ok >= 1)

    # ``judge_image`` calls the model emitted, from the text alone.
    judge_calls_requested = sum(name == "judge_image" for name in names)
    # Calls the harness actually ran: every executed judge leaves an
    # ``agentic_judge ok=`` marker (ok=1 parsed, ok=0 parse-failed). With
    # ``multi_turn.max_parallel_calls=1`` a judge emitted in the same assistant
    # turn as a generate is dropped, so ``requested`` over-counts and must not be
    # what the metrics report as "calls".
    judge_calls_executed = judge_ok + judge_failed
    # Same requested/executed split for generates, so ``num_generate_image_prompts``
    # is comparable to ``num_judge_image_calls`` and cannot report 16 prompts for a
    # rollout that produced three images.
    generate_prompts_requested = len(prompts)
    generate_prompts_executed = _count_executed_generates(text)
    # Judge-outcome lift across the rewrite chain: did iterating actually help?
    # ``judge_delta_reward`` documents why this replaced an n-gram novelty measure.
    judges = _successful_judges(text)
    judge_delta_score, judge_delta_value = judge_delta_reward(
        [(correctness, aesthetics, good_enough) for correctness, aesthetics, good_enough, _ in judges]
    )

    result = _zero_result(method="agentic_multidim")
    result.update(
        num_hermes_tool_calls=len(calls),
        num_generate_image_prompts=generate_prompts_executed,
        num_generate_image_prompts_requested=generate_prompts_requested,
        num_generate_image_prompts_dropped=max(0, generate_prompts_requested - generate_prompts_executed),
        num_judge_image_calls=judge_calls_executed,
        num_judge_image_calls_requested=judge_calls_requested,
        num_judge_image_calls_dropped=max(0, judge_calls_requested - judge_calls_executed),
        judge_parse_ok=judge_ok,
        judge_parse_fail=judge_failed,
        judge_parse_ok_rate=float(judge_rate),
        judge_delta=judge_delta_value,
        num_prompt_rewrites=max(0, len(prompts) - 1),
        terminal_done=int(terminal_done),
        terminal_policy_reflection=int(policy_reflection),
        forced_reflection_context=int(forced_context),
        n_successful_generates=successful_generates,
        expected_num_images=_report_expected(expected),
        task_type=task_type,
        rewrite_after_yes=rewrites_after_yes,
        reward_tool_call=float(bool(calls)),
        reward_tool=with_judge_reward,
    )
    if not prompts or successful_generates == 0:
        return result

    valid_terminal_context = judge_ok > 0 and not blocked and rewrites_after_yes == 0
    closed = valid_terminal_context and terminal_done and (policy_reflection or forced_context)
    rewards = {
        "reflect": _reflection_reward(text, gt),
        "format": _format_reward(
            text,
            successful_generates=successful_generates,
            forced_context=forced_context,
        ),
        "tool": with_judge_reward,
        "improve": judge_delta_score,
        "result": _result_reward(
            text,
            successful_generates=successful_generates,
            terminal_done=terminal_done,
            blocked=blocked,
            rewrite_after_yes=rewrites_after_yes,
        ),
    }
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
