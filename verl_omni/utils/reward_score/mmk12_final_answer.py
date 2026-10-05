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
"""Conservative MMK12 evaluation of final choice/numeric literals, without format bonuses."""

import re
from fractions import Fraction

_TAG = re.compile(r"<answer>(.*?)</answer>\s*$", re.DOTALL)
_PREFIX = re.compile(r"^(?:(?:the |final |correct )*(?:answer|option|choice)(?:\s+is)?\s*:?\s*)", re.IGNORECASE)
_LITERAL = re.compile(r"(?:[A-E]|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?:/[+-]?\d+)?)", re.IGNORECASE)


def _final_literal(text: str) -> str | None:
    if "<answer>" in text or "</answer>" in text:
        match = _TAG.search(text)
        if match is None or text.count("<answer>") != 1 or text.count("</answer>") != 1:
            return None
        text = match.group(1)
    elif "<think>" in text and "</think>" not in text:
        return None
    elif "</think>" in text:
        text = text.rsplit("</think>", 1)[1]

    lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() not in {"\\[", "\\]", "$$"}]
    if not lines:
        return None
    last = lines[-1].removesuffix(".").strip()
    if last.startswith("**") and last.endswith("**"):
        last = last[2:-2].strip()
    if last.startswith("\\[") and last.endswith("\\]"):
        last = last[2:-2].strip()
    last = last.strip(" $")
    start = last.rfind("\\boxed{")
    if start >= 0:
        depth = 1
        end = start + len("\\boxed{")
        while end < len(last) and depth:
            depth += (last[end] == "{") - (last[end] == "}")
            end += 1
        if depth or last[end:].strip(" $."):
            return None
        last = last[start + len("\\boxed{") : end - 1].strip()
    last = _PREFIX.sub("", last).strip(" $").removesuffix(".")
    if last.startswith("(") and last.endswith(")"):
        last = last[1:-1].strip()
    return last if _LITERAL.fullmatch(last) else None


def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None, **kwargs) -> dict:
    """Score only a supported final literal; unsupported or token-limited outputs receive zero.

    This is not a general symbolic-math grader. It reports answer coverage separately
    and deliberately rejects responses at the token budget, even if they contain an answer.
    """
    at_limit = bool((extra_info or {}).get("response_at_token_limit", False))
    answer = None if at_limit else _final_literal(solution_str or "")
    gt = str(ground_truth).strip().strip("$")
    accuracy = 0.0
    if answer is not None:
        if gt.upper() in "ABCDE" and len(gt) == 1:
            accuracy = float(answer.upper() == gt.upper())
        else:
            try:
                accuracy = float(Fraction(answer) == Fraction(gt))
            except (ValueError, ZeroDivisionError):
                pass
    return {
        "score": accuracy,
        "accuracy": accuracy,
        "answer_present": float(answer is not None),
        "at_token_limit": float(at_limit),
    }
