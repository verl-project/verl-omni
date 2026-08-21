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
"""Scalar reward: ``generate_image`` + actor self-reflection protocol.

Protocol (gated):
  generate_image → (image obs attached)
  actor writes a short reflection, then either ``Done.`` OR rewrite +
  ``generate_image`` in the **same** assistant turn.

Frozen Qwen3-VL judge serves dual role: (1) in-turn ``judge_image`` agent tool
(structured VL feedback the agent reads before deciding Done / rewrite), and
(2) reward C/A for ``reward_correctness`` / ``reward_aesthetics``. Reward prefers
scores from the first ``good_enough=YES`` ``agentic_judge ok=1`` observation
(protocol: YES → Done); otherwise the last successful judge. This blocks
rewrite-after-YES roulette from replacing a good C/A with a failed last image.
If absent, it falls back to ``call_reflect_vlm`` via ``AGENTIC_VLLM_URL``
(OpenAI chat) **only** for a PNG under the rollout images root (never an
arbitrary ``path=`` the policy wrote). There is no legacy ``/reflect`` path.
C/A and closed-protocol credit require a real ``judge_image`` ``<tool_call>``;
forged ``agentic_judge ok=1`` prose without that call does not count.

Scalar mix terms (enter ``score`` via weighted mix):
  ``reward_tool_call``, ``reward_correctness`` (gated), ``reward_aesthetics``
  (gated), ``reward_done``.
Additive multiturn term (also enters ``score``):
  ``reward_delta_c`` = C lift after first ``good_enough=NO`` → rewrite → closed;
  applied as ``score += w_delta_c * f_delta_c`` (default ``w_delta_c=0.15``).
  Zero when first judge is not NO, trajectory is not closed, or rewrite-after-YES.
  ``reward_rewrite_yes`` = 1 when closed after first-NO → ≤2 gens → YES (the
  preferred overfit path); applied as ``score += w_rewrite_yes * f_rewrite_yes``
  (default ``w_rewrite_yes=0.12``). Max-pass ``Done`` without YES is discounted.

Final score (before ΔC):
  score = base + scale * mix
  mix   = (w_tool_call * f_tool_call
         + w_correctness * f_correctness_mix
         + w_aesthetics * f_aesthetics_mix
         + w_done * f_done) / w_sum
  then score = min(1, score + w_delta_c * f_delta_c)

``base``/``scale`` are a protocol tier: Qwen3-VL often returns C/A ≈ 0.9 even on
mediocre images. Open loops use a tiny ``scale`` so high VL C/A cannot plateau the
mean reward without learning Done.

Tiers (mix ∈ [0, 1]; open-loop C/A enter mix at 5%):
  no generate_image / no successful PNG     → score = 0
  gen, no Reflection:     base=0.02 scale=0.05 → ≈0.02–0.07
  Reflection:, open       base=0.04 scale=0.30 → ≈0.04–0.34
  closed, weak C/A        base=0.05 scale=0.65 → ≈0.05–0.70  (protocol_ok=1)
  closed, C/A ≥ 0.70      base=0.10 scale=0.90 → ≈0.10–1.00  (protocol_ok=1)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from verl_omni.utils.reward_score.agentic_image_judge_client import call_reflect_vlm

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FUNCTION_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)</function>", re.IGNORECASE | re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>\s*(.*?)\s*</parameter>", re.IGNORECASE | re.DOTALL)
_TOOL_OK = re.compile(r"agentic_tool\s+ok=1", re.IGNORECASE)
_PATH_RE = re.compile(r"path=([^\s'\"]+)", re.IGNORECASE)
_FORCED_REFLECTION_MARKER_RE = re.compile(r"\bagentic_forced_reflection=1\b", re.IGNORECASE)
_BLOCKED_GENERATE_RE = re.compile(
    r"\b(?:blocked_after_yes|blocked_after_max_passes)=1\b|generate_image blocked:",
    re.IGNORECASE,
)
_JUDGE_OBS_HEADER = "VL judge on the last generated image"
_TOOL_OBS_LINE = re.compile(
    r"(?im)^(?!.*\bReflection\s*:).*\b("
    r"agentic_tool|agentic_reflect|agentic_judge|"
    r"VL judge on the last generated image|"
    r"image_vis=|Frozen (?:diffusion|Qwen)|Image reflection vs user request"
    r")\b.*$"
)
_AGENT_REFLECTION_MARKER_RE = re.compile(r"\bReflection\s*:", re.IGNORECASE)
_CORRECTNESS_DIMENSIONS = (
    "subject_entities",
    "attributes",
    "relations_layout",
    "scene_context",
    "completeness",
)
_AESTHETICS_DIMENSIONS = (
    "composition",
    "lighting",
    "color",
    "fidelity",
    "appeal",
)


def _as_dict(ground_truth: Any) -> dict[str, Any]:
    if ground_truth is None:
        return {}
    if isinstance(ground_truth, dict):
        return dict(ground_truth)
    if isinstance(ground_truth, str):
        raw = ground_truth.strip()
        if raw.startswith("{"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        return {"user_request": raw}
    return {}


def _extract_tool_calls(text: str) -> list[tuple[int, int, dict[str, Any]]]:
    """Parse Hermes JSON and/or Qwen3.5 XML tool calls inside ``<tool_call>`` blocks."""
    out: list[tuple[int, int, dict[str, Any]]] = []
    for match in _TOOL_CALL_RE.finditer(text or ""):
        body = (match.group(1) or "").strip()
        call = _parse_tool_call_body(body)
        if call is not None:
            out.append((match.start(), match.end(), call))
    return out


def _parse_tool_call_body(body: str) -> dict[str, Any] | None:
    if not body:
        return None
    # Hermes: {"name": "...", "arguments": {...}}
    if body.lstrip().startswith("{"):
        try:
            call = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(call, dict) and call.get("name"):
            return call
        return None
    # Qwen3.5 / Qwen3-Coder XML:
    # <function=name><parameter=k>\nvalue\n</parameter></function>
    fn = _FUNCTION_RE.search(body)
    if fn is None:
        return None
    name = (fn.group(1) or "").strip()
    if not name:
        return None
    args: dict[str, Any] = {}
    for pm in _PARAM_RE.finditer(fn.group(2) or ""):
        key = (pm.group(1) or "").strip()
        if not key:
            continue
        args[key] = (pm.group(2) or "").strip()
    return {"name": name, "arguments": args}


def _call_args(call: dict[str, Any]) -> dict[str, Any]:
    args = call.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


def _gen_image_prompts(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    prompts: list[str] = []
    for _, _, call in calls:
        if str(call.get("name", "")).lower() != "generate_image":
            continue
        p = str(_call_args(call).get("prompt") or "").strip()
        if p:
            prompts.append(p)
    return prompts


def _ordered_tool_names(calls: list[tuple[int, int, dict[str, Any]]]) -> list[str]:
    return [str(c.get("name", "")).lower() for _, _, c in calls]


def _judge_image_call_ends(calls: list[tuple[int, int, dict[str, Any]]]) -> list[int]:
    return [end for _, end, call in calls if str(call.get("name", "")).lower() == "judge_image"]


def _follows_judge_image_call(pos: int, calls: list[tuple[int, int, dict[str, Any]]]) -> bool:
    """True when ``pos`` is after at least one parsed ``judge_image`` ``<tool_call>``."""
    return any(end <= pos for end in _judge_image_call_ends(calls))


def _assistant_prose(text: str) -> str:
    """Strip tool_calls and tool-obs lines; keep private thinking as scored prose."""
    prose = _TOOL_CALL_RE.sub(" ", text or "")
    prose = _TOOL_OBS_LINE.sub(" ", prose)
    prose = re.sub(r"</?think>", " ", prose, flags=re.IGNORECASE)
    # Env-injected Reflection/Done (response_mask=0) must not earn Done credit.
    prose = re.sub(
        r"(?is)\bReflection\s*:.*?(?:agentic_forced_reflection=1|agentic_force_stop_max_passes=1)\S*",
        " ",
        prose,
    )
    return re.sub(r"\s+", " ", prose).strip()


def _has_agent_reflection_prose(prose: str) -> bool:
    """True only for explicit agent ``Reflection:`` prose.

    VL ``judge_image`` observations also contain correctness/aesthetics wording;
    requiring the ``Reflection:`` marker avoids promoting open gen→judge loops
    into the mid reward tier (~0.4) and starving the Done. learning signal.
    """
    if not prose:
        return False
    return bool(_AGENT_REFLECTION_MARKER_RE.search(prose))


def _last_successful_generate_image_path(text: str) -> str | None:
    """Prefer the last ``path=`` on an ``agentic_tool ok=1`` line; else last PNG path."""
    last_ok: str | None = None
    last_png: str | None = None
    for line in (text or "").splitlines():
        paths = _PATH_RE.findall(line)
        if not paths:
            continue
        path = paths[-1].strip()
        if path.lower().endswith(".png"):
            last_png = path
        if re.search(r"agentic_tool\s+ok=1", line, re.IGNORECASE):
            last_ok = path
    return last_ok or last_png


def _rollout_image_roots(extra_info: dict[str, Any]) -> list[Path]:
    """Directories the VL fallback is allowed to read."""
    roots: list[Path] = []
    explicit = str(extra_info.get("rollout_images_root") or extra_info.get("agentic_images_root") or "").strip()
    relpath = str(extra_info.get("trajectory_relpath") or "").strip()
    if explicit:
        base = Path(explicit).expanduser()
        roots.append(base)
        if relpath:
            roots.append(base / relpath)
    try:
        from verl_omni.agent_loop.agentic_trajectory_context import resolve_rollout_images_root

        env_root = resolve_rollout_images_root()
        roots.append(env_root)
        if relpath:
            roots.append(env_root / relpath)
    except Exception:  # noqa: BLE001
        pass
    return roots


def _path_if_under_rollout_root(path: str, extra_info: dict[str, Any]) -> str | None:
    """Return the resolved PNG path if it lives under a known rollout images root."""
    if not path or not str(path).lower().endswith(".png"):
        return None
    try:
        resolved = Path(path).expanduser().resolve()
    except OSError:
        return None
    for root in _rollout_image_roots(extra_info):
        try:
            root_resolved = root.expanduser().resolve()
            resolved.relative_to(root_resolved)
        except (OSError, ValueError):
            continue
        return str(resolved)
    return None


def _confined_generate_image_path(text: str, extra_info: dict[str, Any]) -> str | None:
    """Last successful generate PNG that is inside the rollout dump root."""
    candidates: list[str] = []
    for line in (text or "").splitlines():
        if not _TOOL_OK.search(line):
            continue
        for raw in _PATH_RE.findall(line):
            path = raw.strip()
            if path.lower().endswith(".png"):
                candidates.append(path)
    if not candidates:
        fallback = _last_successful_generate_image_path(text)
        if fallback:
            candidates.append(fallback)
    for path in reversed(candidates):
        confined = _path_if_under_rollout_root(path, extra_info)
        if confined:
            return confined
    return None


def _has_successful_generated_image(text: str) -> bool:
    """Accept live tool lines regardless of whether ``path`` precedes ``ok=1``."""
    for line in (text or "").splitlines():
        if _TOOL_OK.search(line) and any(path.lower().endswith(".png") for path in _PATH_RE.findall(line)):
            return True
    return False


def _user_request_from_gt(gt: dict[str, Any], extra_info: dict[str, Any]) -> str:
    for key in ("user_request", "raw_prompt"):
        val = extra_info.get(key) or gt.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


_CORRECTNESS_EQ_RE = re.compile(r"\bcorrectness\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_AESTHETICS_EQ_RE = re.compile(r"\baesthetics\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_GOOD_ENOUGH_EQ_RE = re.compile(r"\bgood_enough\s*=\s*(YES|NO|1|0|true|false)\b", re.IGNORECASE)
_AGENTIC_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_AGENTIC_JUDGE_PARSE_FAIL_RE = re.compile(r"\bagentic_judge\s+ok=0\b|\bparse_ok\s*=\s*0\b", re.IGNORECASE)


def _judge_parse_stats(text: str, calls: list[tuple[int, int, dict[str, Any]]] | None = None) -> tuple[int, int, float]:
    """Return ``(n_ok, n_fail, parse_ok_rate)`` from trajectory judge observations.

    Markers that are not after a parsed ``judge_image`` ``<tool_call>`` are ignored
    so assistant prose cannot mint parse-ok counts.
    """
    blob = text or ""
    parsed_calls = calls if calls is not None else _extract_tool_calls(blob)
    n_ok = 0
    for match in _AGENTIC_JUDGE_OK_RE.finditer(blob):
        if _follows_judge_image_call(match.start(), parsed_calls):
            n_ok += 1
    n_fail = 0
    for match in re.finditer(r"\bagentic_judge\s+ok=0\b", blob, flags=re.IGNORECASE):
        if _follows_judge_image_call(match.start(), parsed_calls):
            n_fail += 1
    if n_fail == 0:
        for match in _AGENTIC_JUDGE_PARSE_FAIL_RE.finditer(blob):
            if _follows_judge_image_call(match.start(), parsed_calls):
                n_fail += 1
    n_attempts = n_ok + n_fail
    rate = float(n_ok) / float(n_attempts) if n_attempts else 1.0
    return n_ok, n_fail, rate


def _good_enough_from_window(window: str) -> bool | None:
    """Parse ``good_enough`` from a judge window.

    Prefer the *last* match so a prior NO in the same lookback does not shadow
    a later YES (C/A already use the last hit in the window).
    """
    matches = list(_GOOD_ENOUGH_EQ_RE.finditer(window or ""))
    if not matches:
        return None
    tok = matches[-1].group(1).strip().lower()
    if tok in {"yes", "1", "true"}:
        return True
    if tok in {"no", "0", "false"}:
        return False
    return None


def _iter_successful_judge_scores(text: str) -> list[tuple[float, float, bool | None, int]]:
    """Yield ``(c, a, good_enough, end_pos)`` for each successful judge obs.

    A hit must follow a parsed ``judge_image`` ``<tool_call>`` and the tool's
    ``VL judge on the last generated image`` header so policy prose cannot mint C/A.
    """
    blob = text or ""
    calls = _extract_tool_calls(blob)
    out: list[tuple[float, float, bool | None, int]] = []
    for match in _AGENTIC_JUDGE_OK_RE.finditer(blob):
        if not _follows_judge_image_call(match.start(), calls):
            continue
        window = blob[max(0, match.start() - 1400) : match.end()]
        if _JUDGE_OBS_HEADER not in window:
            continue
        if re.search(r"\bparse_ok\s*=\s*0\b", window, re.IGNORECASE):
            continue
        c_hits = list(_CORRECTNESS_EQ_RE.finditer(window))
        a_hits = list(_AESTHETICS_EQ_RE.finditer(window))
        if not c_hits or not a_hits:
            continue
        try:
            c = max(0.0, min(1.0, float(c_hits[-1].group(1))))
            a = max(0.0, min(1.0, float(a_hits[-1].group(1))))
        except (TypeError, ValueError):
            continue
        out.append((c, a, _good_enough_from_window(window), match.end()))
    return out


def _policy_terminal_decision(text: str) -> tuple[bool, bool, bool]:
    """Return ``(terminal_done, policy_reflection, forced_reflection_context)``.

    A credited Done must be a sampled terminal answer after the latest successful
    judge.  Bare planning prose such as ``I'll stop when Done.`` is deliberately
    rejected.  A policy may emit either ``Reflection: ... Done.`` directly or
    exactly ``Done.`` after an injected, masked Reflection stop cue.
    """
    blob = text or ""
    judges = _iter_successful_judge_scores(blob)
    if not judges:
        return False, False, False

    judge_end = judges[-1][3]
    line_end = blob.find("\n", judge_end)
    anchor = len(blob) if line_end < 0 else line_end + 1
    forced_context = False
    for marker in _FORCED_REFLECTION_MARKER_RE.finditer(blob, anchor):
        anchor = marker.end()
        forced_context = True

    suffix = blob[anchor:]
    suffix = _assistant_prose(suffix)
    suffix = re.sub(r"<\|[^>]+\|>|</?tool_response>|</?assistant>", " ", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"^\s*(?:assistant|user)\s+", "", suffix, flags=re.IGNORECASE)
    suffix = re.sub(r"\s+", " ", suffix).strip()
    policy_reflection = bool(_AGENT_REFLECTION_MARKER_RE.search(suffix))

    if policy_reflection:
        terminal_done = bool(re.search(r"\bDone\.\s*$", suffix, re.IGNORECASE))
    else:
        terminal_done = bool(re.fullmatch(r"Done\.", suffix, re.IGNORECASE))
    return terminal_done, policy_reflection, forced_context


def _parse_last_agentic_judge_scores(
    text: str,
) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
    """Reuse C/A from judge obs: first ``good_enough=YES``, else last ok=1.

    Only trusts windows that follow a ``judge_image`` ``<tool_call>`` and end in
    ``agentic_judge ok=1`` (written by our tool). Bare ``correctness=`` markers
    the policy might hallucinate never contribute C/A. Parse failures
    (``parse_ok=0`` / unparseable) also never contribute.
    """
    hits = _iter_successful_judge_scores(text)
    if not hits:
        return None, None, {}, {}
    for c, a, good_enough, _ in hits:
        if good_enough is True:
            return c, a, {}, {}
    c, a, _, _ = hits[-1]
    return c, a, {}, {}


def _num_generate_after_first_yes(text: str, calls: list[tuple[int, int, dict[str, Any]]]) -> int:
    """Count ``generate_image`` calls after the first ``good_enough=YES`` judge."""
    yes_pos: int | None = None
    for _, _, good_enough, end_pos in _iter_successful_judge_scores(text):
        if good_enough is True:
            yes_pos = end_pos
            break
    if yes_pos is None:
        return 0
    n = 0
    for start, _, call in calls:
        if start <= yes_pos:
            continue
        if str(call.get("name", "")).lower() == "generate_image":
            n += 1
    return n


def _delta_c_bonus(text: str, preferred_c: float) -> tuple[float, float | None, bool]:
    """Bonus for lifting C after a first-pass failure (``good_enough=NO``).

    Returns ``(f_delta_c in [0,1], first_c or None, first_was_no)``.
    """
    hits = _iter_successful_judge_scores(text)
    if not hits:
        return 0.0, None, False
    first_c, _, first_ge, _ = hits[0]
    first_was_no = first_ge is False
    if not first_was_no:
        return 0.0, float(first_c), False
    delta = max(0.0, min(1.0, float(preferred_c) - float(first_c)))
    return delta, float(first_c), True


def _rewrite_then_yes(text: str, *, n_generate: int) -> bool:
    """True for the preferred path: first judge NO → exactly 2 gens → later YES."""
    if n_generate != 2:
        return False
    hits = _iter_successful_judge_scores(text)
    if len(hits) < 2:
        return False
    if hits[0][2] is not False:
        return False
    return any(ge is True for _, _, ge, _ in hits[1:])


def _closed_via_max_pass_without_yes(text: str) -> bool:
    """Max-pass stop cue + no YES anywhere — weaker than rewrite→YES."""
    blob = text or ""
    if "agentic_force_stop_max_passes=1" not in blob:
        return False
    return not any(ge is True for _, _, ge, _ in _iter_successful_judge_scores(blob))


def _vl_judge_correctness_aesthetics(
    text: str,
    *,
    user_request: str,
    image_prompt: str,
    extra_info: dict[str, Any] | None = None,
) -> tuple[float | None, float | None, dict[str, float], dict[str, float]]:
    """Resolve C/A from trajectory judge obs, else re-call frozen VL on last PNG.

    Fallback only reads a PNG under the rollout images root. Returns
    ``(None, None, {}, {})`` when neither source works — callers must treat C/A
    as 0.0 (no heuristic fallback for reward).
    """
    parsed = _parse_last_agentic_judge_scores(text)
    if parsed[0] is not None and parsed[1] is not None:
        return parsed

    image_path = _confined_generate_image_path(text, extra_info or {})
    if not image_path:
        return None, None, {}, {}
    scored = call_reflect_vlm(
        user_request=user_request or "",
        image_prompt=image_prompt or "",
        notes="",
        image_path=image_path,
    )
    if scored is None:
        return None, None, {}, {}
    return (
        float(scored["correctness"]),
        float(scored["aesthetics"]),
        dict(scored.get("correctness_scores") or {}),
        dict(scored.get("aesthetics_scores") or {}),
    )


def _zero_result(*, method: str) -> dict[str, float | str | int | None]:
    result: dict[str, float | str | int | None] = {
        "score": 0.0,
        "reward_tool_call": 0.0,
        "reward_correctness": 0.0,
        "reward_aesthetics": 0.0,
        "reward_done": 0.0,
        "num_hermes_tool_calls": 0,
        "num_generate_image_prompts": 0,
        "num_judge_image_calls": 0,
        "judge_parse_ok": 0,
        "judge_parse_fail": 0,
        "judge_parse_ok_rate": 1.0,
        "protocol_ok": 0,
        "rewrite_after_yes": 0,
        "reward_delta_c": 0.0,
        "reward_rewrite_yes": 0.0,
        "first_correctness": 0.0,
        "first_judge_no": 0,
        "rollout_valid": 0,
        "method": method,
    }
    result.update({f"reward_correctness_{key}": 0.0 for key in _CORRECTNESS_DIMENSIONS})
    result.update({f"reward_aesthetics_{key}": 0.0 for key in _AESTHETICS_DIMENSIONS})
    return result


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Score an agentic image-generation trajectory for GRPO."""
    del data_source, kwargs
    extra_info = dict(extra_info or {})
    gt = _as_dict(ground_truth)

    blob = solution_str or ""
    if not blob.strip():
        return _zero_result(method="agentic_hermes_tool_calls")

    calls = _extract_tool_calls(blob)
    prompts = _gen_image_prompts(calls)
    names = _ordered_tool_names(calls)
    # Count judge_image tool calls alongside generate_image calls.
    n_reflect = sum(1 for n in names if n == "judge_image")
    f_tool_call = 1.0 if calls else 0.0

    if not prompts:
        # No generate_image → invalid rollout for GRPO (masked out of the update).
        out = _zero_result(method="agentic_no_generate")
        out["reward_tool_call"] = float(f_tool_call)
        out["num_hermes_tool_calls"] = int(len(calls))
        out["num_judge_image_calls"] = int(n_reflect)
        out["rollout_valid"] = 0
        out["score"] = 0.0
        return out
    if not _has_successful_generated_image(blob):
        # A parseable tool call is not enough: tool failures, blocked calls and
        # stale-latch/no-PNG trajectories must never mint protocol or Done credit.
        out = _zero_result(method="agentic_no_successful_image")
        out["reward_tool_call"] = float(f_tool_call)
        out["num_hermes_tool_calls"] = int(len(calls))
        out["num_generate_image_prompts"] = int(len(prompts))
        out["num_judge_image_calls"] = int(n_reflect)
        out["rollout_valid"] = 0
        return out

    user_request = _user_request_from_gt(gt, extra_info)
    last_c, last_a, correctness_scores, aesthetics_scores = _vl_judge_correctness_aesthetics(
        blob,
        user_request=user_request,
        image_prompt=prompts[-1] if prompts else "",
        extra_info=extra_info,
    )
    n_judge_ok, n_judge_fail, judge_parse_rate = _judge_parse_stats(blob, calls)
    # No successful parse anywhere → keep C/A at 0 (do not invent scores).
    if last_c is None and last_a is None and n_judge_fail > 0 and n_judge_ok == 0:
        last_c, last_a = 0.0, 0.0
        correctness_scores, aesthetics_scores = {}, {}

    w_tool_call = float(extra_info.get("w_tool_call", gt.get("w_tool_call", 0.10)))
    w_correctness = float(extra_info.get("w_correctness", gt.get("w_correctness", 0.35)))
    w_aesthetics = float(extra_info.get("w_aesthetics", gt.get("w_aesthetics", 0.35)))
    # Closed-loop Done. must dominate open gen→judge loops (was only 0.10 and
    # got swamped by already-high C/A from the frozen judge).
    w_done = float(extra_info.get("w_done", gt.get("w_done", 0.20)))
    # Multiturn headroom: reward C lift after a failed first judge (NO → rewrite).
    w_delta_c = float(extra_info.get("w_delta_c", gt.get("w_delta_c", 0.15)))
    w_rewrite_yes = float(extra_info.get("w_rewrite_yes", gt.get("w_rewrite_yes", 0.12)))
    w_sum = w_tool_call + w_correctness + w_aesthetics + w_done
    if w_sum <= 0:
        w_tool_call, w_correctness, w_aesthetics, w_done, w_sum = 0.10, 0.35, 0.35, 0.20, 1.0

    prose = _assistant_prose(blob)
    has_refl = _has_agent_reflection_prose(prose)
    terminal_done, terminal_policy_reflection, forced_reflection_context = _policy_terminal_decision(blob)
    n_rewrite_after_yes = _num_generate_after_first_yes(blob, calls)
    valid_terminal_context = bool(
        n_reflect > 0 and n_judge_ok > 0 and not _BLOCKED_GENERATE_RE.search(blob) and n_rewrite_after_yes == 0
    )
    closed = bool(
        valid_terminal_context and terminal_done and (terminal_policy_reflection or forced_reflection_context)
    )
    distinct = len(prompts) >= 2 and prompts[0].lower().strip() != prompts[-1].lower().strip()
    f_correctness = float(last_c if last_c is not None else 0.0)
    f_aesthetics = float(last_a if last_a is not None else 0.0)
    # Mix terms: VL quality only fully counts after Reflection+Done. Open loops
    # keep a tiny fraction so C/A still appears in logs / weak ranking, but the
    # mean score cannot plateau near ~0.45 without learning Done.
    ca_mix_scale = 1.0 if closed else 0.05
    f_correctness_mix = ca_mix_scale * f_correctness
    f_aesthetics_mix = ca_mix_scale * f_aesthetics
    # No partial reward for incidental prose such as "Stop when Done.".
    f_done = 1.0 if closed else 0.0
    ca_ok = last_c is not None and last_a is not None and last_c >= 0.70 and last_a >= 0.70
    f_delta_c, first_c, first_judge_no = _delta_c_bonus(blob, f_correctness)
    if not closed:
        # ΔC is a multiturn bonus on top of a closed protocol.
        f_delta_c = 0.0
    f_rewrite_yes = 1.0 if closed and _rewrite_then_yes(blob, n_generate=len(prompts)) else 0.0

    # High tier only for protocol_ok. Gen without Reflection: is starved.
    # closed already requires policy Reflection: or forced-reflection context + Done.
    if closed and (len(prompts) == 1 or distinct):
        protocol_ok = 1
        if ca_ok:
            base, scale = 0.10, 0.90
        else:
            # Closed loop but weak/missing C/A: protocol alone cannot dominate.
            base, scale = 0.05, 0.65
    elif has_refl and len(prompts) >= 1:
        # Reflection without Done — keep below closed tier.
        base, scale = 0.04, 0.30
        protocol_ok = 0
    else:
        # generate_image / judge without agent Reflection: — starve this plateau.
        base, scale = 0.02, 0.05
        protocol_ok = 0

    # good_enough=YES means stop. Extra generate_image after YES is protocol break
    # and was the main overfit failure mode (score/C/A drift via rewrite roulette).
    if n_rewrite_after_yes > 0:
        protocol_ok = 0
        base, scale = min(base, 0.05), min(scale, 0.35)
        f_done = 0.0
        # No ΔC / rewrite-YES credit for gambling after YES.
        f_delta_c = 0.0
        f_rewrite_yes = 0.0

    # Prefer NO→one rewrite→YES over surviving to max-pass Done with all NO.
    if closed and _closed_via_max_pass_without_yes(blob):
        scale = min(scale, 0.55)
        f_rewrite_yes = 0.0

    # Scalar: tool_call gate + (gated) VL C/A + closed-loop Done + multiturn ΔC.
    total = base + scale * (
        (
            w_tool_call * f_tool_call
            + w_correctness * f_correctness_mix
            + w_aesthetics * f_aesthetics_mix
            + w_done * f_done
        )
        / w_sum
    )
    total = float(min(1.0, total + w_delta_c * f_delta_c + w_rewrite_yes * f_rewrite_yes))

    result: dict[str, float | str | int | None] = {
        "score": float(total),
        "reward_tool_call": float(f_tool_call),
        # Log raw VL C/A (pre-gate) so WandB tracks image quality separately.
        "reward_correctness": f_correctness,
        "reward_aesthetics": f_aesthetics,
        "reward_done": float(f_done),
        "reward_delta_c": float(f_delta_c),
        "reward_rewrite_yes": float(f_rewrite_yes),
        "first_correctness": float(first_c if first_c is not None else 0.0),
        "first_judge_no": int(bool(first_judge_no)),
        "num_hermes_tool_calls": int(len(calls)),
        "num_generate_image_prompts": int(len(prompts)),
        "num_judge_image_calls": int(n_reflect),
        "judge_parse_ok": int(n_judge_ok),
        "judge_parse_fail": int(n_judge_fail),
        "judge_parse_ok_rate": float(judge_parse_rate),
        "protocol_ok": int(protocol_ok),
        "rewrite_after_yes": int(n_rewrite_after_yes),
        "rollout_valid": 1,
        "method": "agentic_hermes_tool_calls",
    }
    result.update(
        {f"reward_correctness_{key}": float(correctness_scores.get(key, 0.0)) for key in _CORRECTNESS_DIMENSIONS}
    )
    result.update(
        {f"reward_aesthetics_{key}": float(aesthetics_scores.get(key, 0.0)) for key in _AESTHETICS_DIMENSIONS}
    )
    # Always emit the full schema so Ray reward workers never KeyError on a
    # missing key when batching reward_extra_info (verl takes keys from sample 0).
    schema = _zero_result(method=str(result.get("method") or "agentic_hermes_tool_calls"))
    schema.update(result)
    return schema
