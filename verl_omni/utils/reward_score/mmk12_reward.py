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

"""MMK12 (image-in / text-out math) reward — aligned with MM-EUREKA,
adapted for Qwen3-Omni native thinking mode.

Design:

1. **Correctness judgement splits by answer type**:

   * **Choice questions** (ground truth is a single letter A-E) are judged by
     a pure string match on the model output - no sympy, no subprocess, no
     timeout. If the model output a letter, match it directly; if it output a
     value instead, fall back to comparing that value against the correct
     option's content (``extra_info["options"]``) for 0.5 partial credit.
   * **Numeric / LaTeX answers** reuse verl's ``math_verify.compute_score``
     (``ExprExtractionConfig`` + ``LatexExtractionConfig``), which is
     subprocess-isolated and timed.

   Choices need no sympy, so they skip the subprocess timeout that sympy
   requires - no separate ``StringExtractionConfig`` pool is maintained.

2. **Answer extraction** prefers the ``<answer>…</answer>`` tag.  If the tag
   is present, only its inner text is fed to math_verify.  This avoids
   mis-extraction when the model writes intermediate numbers inside
   ``<think>``.

3. **Format reward is progressive** (two equal-weighted checks, aligned
   with MM-EUREKA 32B — no explicit ``<think>`` required):

   * **answer_tag**: exactly one ``<answer>…</answer>`` pair.
   * **boxed**: ``\\boxed{…}`` inside ``<answer>``.

   Reward ladder (with default ``format_score=0.3``): 0.0 / 0.15 / 0.30.

4. **Total reward** is additive, normalized to [0, 1]:
   ``score = (accuracy_reward + format_reward) / (1 + format_score)``.
"""

import json
import re

from verl.utils.reward_score.math_verify import compute_score as _math_verify_score

_ANGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|\^\\circ|\\circ)")
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{")
_CHOICE_RE = re.compile(r"(?<![A-Za-z])([A-E])(?![A-Za-z])")


def _normalize_angle(text: str) -> str:
    """Strip angle symbols so math_verify sees plain numbers."""
    return _ANGLE_RE.sub(r"\1", text)


def _extract_answer_tag(text: str) -> str | None:
    """Return the inner text of the first ``<answer>…</answer>`` tag, or None."""
    m = _ANSWER_TAG_RE.search(text or "")
    return m.group(1).strip() if m else None


def _is_choice_gt(gt: str) -> bool:
    """True when the ground truth is a single multiple-choice letter (A-E)."""
    return len(gt) == 1 and gt.upper() in "ABCDE"


def _extract_choice_letter(content: str) -> str | None:
    """Return the last standalone A-E letter in ``content``, or None.

    Matches ``\\boxed{B}``, ``answer: B`` and a bare ``B`` uniformly. Returns
    None when the model output no choice letter (e.g. it wrote a value).
    """
    letters = _CHOICE_RE.findall(content or "")
    return letters[-1].upper() if letters else None


def _compute_format_score(
    text: str,
    answer_inner: str | None,
    format_score: float = 0.3,
) -> float:
    """Progressive format reward: two independent checks, equal-weighted.

    * **answer_tag**: exactly one ``<answer>…</answer>`` pair.
    * **boxed**: ``\\boxed{…}`` inside ``<answer>``.
    """
    text = text or ""
    has_answer = answer_inner is not None and (text.count("<answer>") == 1 and text.count("</answer>") == 1)
    has_boxed = answer_inner is not None and _BOXED_RE.search(answer_inner) is not None
    return format_score * (has_answer + has_boxed) / 2.0


def compute_score(
    solution_str: str,
    ground_truth: str,
    format_score: float = 0.3,
    math_verify_timeout: float = 20.0,
    **kwargs,
) -> dict:
    """MMK12 reward entrypoint.

    Reward formula (additive, normalized to [0, 1]):
        score = (accuracy_reward + format_reward) / (1 + format_score)

    * ``accuracy_reward`` ∈ {0, 1} - choice: string match; numeric/LaTeX: verl math_verify.
    * ``format_reward`` ∈ [0, ``format_score``] — progressive (answer_tag + boxed).

    Config:
        reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12_reward.py
        reward.custom_reward_function.name=compute_score
    """
    gt = _normalize_angle(str(ground_truth).strip())
    solution_str = _normalize_angle(solution_str or "")

    # 1. Extract <answer> inner text (used for correctness; keep solution_str for format).
    answer_inner = _extract_answer_tag(solution_str)
    content = answer_inner if answer_inner is not None else solution_str

    # 2. Correctness judgement — dispatch by (gt type, pred type).
    if _is_choice_gt(gt):
        pred_letter = _extract_choice_letter(content)
        if pred_letter is not None:
            # gt is a letter and the model also output a letter -> pure string
            # match. No sympy, no subprocess, no timeout needed.
            accuracy_reward = 1.0 if pred_letter == gt.upper() else 0.0
        else:
            # gt is a letter but the model output a value (not a letter) ->
            # partial credit (0.5) if the value matches the correct option's
            # content. Reuses verl's math_verify (subprocess-isolated, timed).
            options_raw = (kwargs.get("extra_info") or {}).get("options") or "{}"
            try:
                options = json.loads(options_raw)
            except (ValueError, TypeError):
                options = {}
            option_content = options.get(gt.upper())
            if option_content:
                accuracy_reward = _math_verify_score(content, option_content, timeout=math_verify_timeout) * 0.5
            else:
                accuracy_reward = 0.0
    else:
        # Numeric / LaTeX answer -> reuse verl's math_verify (subprocess-isolated,
        # with timeout).
        accuracy_reward = _math_verify_score(content, gt, timeout=math_verify_timeout)

    # 3. Format reward (progressive: answer_tag + boxed).
    format_reward = _compute_format_score(solution_str, answer_inner, format_score)

    # 4. Total reward (additive, normalized).
    score = (accuracy_reward + format_reward) / (1.0 + format_score)

    return {
        "score": float(score),
        "format_reward": float(format_reward),
        "accuracy": float(accuracy_reward),
    }
