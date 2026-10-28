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

"""BAGEL Co-RL judge observations and forced reflection turn protocol.

Response-mask contract
----------------------
* Forced Hermes ``<tool_call>`` teacher tokens → ``response_mask=1`` (policy
  format the actor must learn; GRPO may credit them).
* Injected Reflection prose from ``build_forced_reflection`` → ``response_mask=0``
  (environment context only; reward strips force markers).
* Terminal ``Done.`` sampled by the policy after a stop cue → ``response_mask=1``
  (GRPO needs a sampled terminal action).
* Compact tool observations (``path=…`` / judge text) → ``response_mask=0``.
"""

from __future__ import annotations

import re

__all__ = [
    "build_forced_reflection",
    "derive_good_enough_from_scores",
    "format_rm_scores_as_judge_text",
]

_JUDGE_OK_RE = re.compile(r"\bagentic_judge\s+ok=1\b", re.IGNORECASE)
_CORRECTNESS_RE = re.compile(r"\bcorrectness\s*=\s*([0-9.]+)", re.IGNORECASE)
_AESTHETICS_RE = re.compile(r"\baesthetics\s*=\s*([0-9.]+)", re.IGNORECASE)
_GOOD_ENOUGH_RE = re.compile(r"\bgood_enough\s*=\s*(YES|NO)", re.IGNORECASE)
_RUBBER_STAMP_RE = re.compile(r"\brubber_stamp\s*=\s*(True|False|1|0|YES|NO)", re.IGNORECASE)
_FINDINGS_RE = re.compile(
    r"\bfindings:\s*(.+?)(?:\n\s*suggested_fixes:|\n\s*agentic_judge\b)", re.IGNORECASE | re.DOTALL
)
_FIXES_RE = re.compile(r"\bsuggested_fixes:\s*(.+?)(?:\n\s*agentic_judge\b|\n\n|\Z)", re.IGNORECASE | re.DOTALL)


def _parse_rubber_stamp(tool_text: str) -> bool:
    match = _RUBBER_STAMP_RE.search(tool_text or "")
    if not match:
        return False
    return match.group(1).strip().lower() in {"true", "1", "yes"}


def format_rm_scores_as_judge_text(
    *,
    correctness: float,
    aesthetics: float,
    good_enough: bool,
    findings: str = "",
    suggested_fixes: str = "",
    rubber_stamp: bool = False,
    similarity: float | None = None,
) -> str:
    """Render RM scores into judge observations consumed by forced reflection."""
    findings_text = (findings or "").strip() or "see VL facet scores above"
    if similarity is not None and "similarity" not in findings_text.lower():
        findings_text = f"{findings_text}; unicot_similarity={float(similarity):.4f}"
    fixes_text = (suggested_fixes or "").strip() or "none"
    ge = "YES" if good_enough else "NO"
    return (
        f"agentic_judge ok=1 correctness={float(correctness):.4f} "
        f"aesthetics={float(aesthetics):.4f} good_enough={ge} "
        f"rubber_stamp={bool(rubber_stamp)}\n"
        f"findings: {findings_text}\n"
        f"suggested_fixes: {fixes_text}\n"
        f"agentic_judge"
    )


def build_forced_reflection(
    tool_text: str,
    *,
    force_done: bool = False,
    generate_pass: int = 0,
    max_passes: int = 3,
) -> tuple[str, bool] | None:
    """Build ``(assistant_text, stop_required)`` from a successful judge observation.

    Forced text is context-only (``response_mask=0``). Even when the judge says YES
    or the pass cap is reached, leave ``Done.`` to one subsequent policy decode
    so GRPO has a sampled terminal action to reinforce.
    """
    if not _JUDGE_OK_RE.search(tool_text or ""):
        return None
    c_m = _CORRECTNESS_RE.search(tool_text)
    a_m = _AESTHETICS_RE.search(tool_text)
    g_m = _GOOD_ENOUGH_RE.search(tool_text)
    f_m = _FINDINGS_RE.search(tool_text)
    x_m = _FIXES_RE.search(tool_text)
    correctness = c_m.group(1) if c_m else "?"
    aesthetics = a_m.group(1) if a_m else "?"
    good_enough = (g_m.group(1).upper() == "YES") if g_m else False
    rubber_stamp = _parse_rubber_stamp(tool_text)
    findings = re.sub(r"\s+", " ", (f_m.group(1) if f_m else "").strip())[:220]
    fixes = re.sub(r"\s+", " ", (x_m.group(1) if x_m else "").strip())[:160]
    if not findings:
        findings = "see VL facet scores above"
    stamp = f"rubber_stamp={rubber_stamp}"

    if good_enough:
        text = (
            f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
            f"good_enough=YES. {stamp}. {findings} Stop now; do not call another tool. "
            f"Your next and only action must be exactly Done. agentic_stop_decision_required=1"
        )
        return text, True
    if force_done:
        text = (
            f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
            f"good_enough=NO after generate_image pass {generate_pass}/{max_passes}. {stamp}. "
            f"{findings} {max_passes}-pass max reached; stop now and do not call another tool. "
            f"Your next and only action must be exactly Done. "
            f"agentic_force_stop_max_passes=1 agentic_stop_decision_required=1"
        )
        return text, True
    fix_note = f" Suggested fixes: {fixes}." if fixes and fixes.lower() != "none" else ""
    text = (
        f"Reflection: VL judge reports correctness={correctness}, aesthetics={aesthetics}, "
        f"good_enough=NO. {stamp}. {findings}.{fix_note} Rewriting the diffusion prompt next."
    )
    return text, False


def derive_good_enough_from_scores(
    *,
    correctness: float | None = None,
    aesthetics: float | None = None,
    similarity: float | None = None,
    threshold: float = 0.7,
) -> bool:
    """Reduce RM facet scores to a stop/continue bit for the Hermes loop."""
    facets = [v for v in (correctness, aesthetics, similarity) if v is not None]
    if not facets:
        return False
    return float(sum(facets) / len(facets)) >= float(threshold)
