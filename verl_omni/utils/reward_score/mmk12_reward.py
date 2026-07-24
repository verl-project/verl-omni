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

1. **Unified correctness judgement via math_verify** with all three
   extraction configs (``StringExtractionConfig`` + ``LatexExtractionConfig``
   + ``ExprExtractionConfig``), matching MM-EUREKA's ``accuracy_reward_func``.
   This handles both single-letter choices (A/B/C/D/E) and numeric/LaTeX
   answers uniformly — no separate choice-regex branch.

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

The math_verify parse can call ``signal.alarm()``, which raises in worker
threads.  We isolate it in a ``ProcessPoolExecutor`` (spawn context), the
same trick verl uses in ``verl/utils/reward_score/math_verify.py``.
"""

import json
import re

from verl_omni.utils.reward_score.reward_utils import math_verify_score as _math_verify_score

# Angle-like expressions: ``60°``, ``60^\circ``, ``60\circ``.
# Normalized before any parser sees the text.
_ANGLE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:°|\^\\circ|\\circ)")

# Extract content inside <answer>…</answer> (non-greedy, DOTALL).
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# \boxed{…} presence check (only checks for \boxed{ prefix; brace balancing
# is handled by math_verify anyway).
_BOXED_RE = re.compile(r"\\boxed\{")


def _normalize_angle(text: str) -> str:
    """Strip angle symbols so math_verify sees plain numbers."""
    return _ANGLE_RE.sub(r"\1", text)


def _extract_answer_tag(text: str) -> str | None:
    """Return the inner text of the first ``<answer>…</answer>`` tag, or None."""
    m = _ANSWER_TAG_RE.search(text or "")
    return m.group(1).strip() if m else None


def _compute_format_score(
    text: str,
    answer_inner: str | None,
    format_score: float = 0.3,
) -> float:
    """Progressive format reward: two independent checks, equal-weighted.

    * **answer_tag**: exactly one ``<answer>…</answer>`` pair.
    * **boxed**: ``\\boxed{…}`` inside ``<answer>``.

    Reward = ``format_score`` × passed / 2.  Ladder (default 0.3):
    0/2 → 0.000, 1/2 → 0.150, 2/2 → 0.300.
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

    * ``accuracy_reward`` ∈ {0, 1} — math_verify with String+Latex+Expr configs.
    * ``format_reward`` ∈ [0, ``format_score``] — progressive (answer_tag + boxed).

    Config:
        reward.custom_reward_function.path=verl_omni/utils/reward_score/mmk12.py
        reward.custom_reward_function.name=compute_score
    """
    gt = _normalize_angle(str(ground_truth).strip())
    solution_str = _normalize_angle(solution_str or "")

    # 1. Extract <answer> inner text (used for correctness; keep solution_str for format).
    answer_inner = _extract_answer_tag(solution_str)
    content = answer_inner if answer_inner is not None else solution_str

    # 2. Correctness judgement — unified math_verify path.
    accuracy_reward = _math_verify_score(content, gt, timeout=math_verify_timeout)

    # 2b. Multi-choice content fallback: if gt is a single letter and math_verify
    # failed, but ``extra_info["options"]`` provides the option content, retry
    # against the correct option's content.  Partial credit (0.5) since the
    # model gave the right value but didn't follow the "output letter" instruction.
    if accuracy_reward < 1.0 and len(gt) == 1 and gt.upper() in "ABCDE":
        options_raw = (kwargs.get("extra_info") or {}).get("options") or "{}"
        try:
            options = json.loads(options_raw)
        except (ValueError, TypeError):
            options = {}
        option_content = options.get(gt.upper())
        if option_content:
            accuracy_reward = _math_verify_score(content, option_content, timeout=math_verify_timeout) * 0.5

    # 3. Format reward (progressive: answer_tag + boxed).
    format_reward = _compute_format_score(solution_str, answer_inner, format_score)

    # 4. Total reward (additive, normalized).
    score = (accuracy_reward + format_reward) / (1.0 + format_score)

    return {
        "score": float(score),
        "format_reward": float(format_reward),
        "accuracy": float(accuracy_reward),
    }
