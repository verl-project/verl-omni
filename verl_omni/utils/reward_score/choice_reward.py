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
"""Rule reward for multiple-choice multimodal questions."""

import re
from typing import Any

_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def extract_answer(text: Any) -> str:
    """Return the first answer-tag payload, matching Relax's AVQA reward.

    Tag matching is case-insensitive: the AVQA converter preserves
    upper-case ``<ANSWER>`` ground truths alongside lower-case ones.
    """
    match = _ANSWER_TAG_RE.search(str(text or ""))
    return match.group(1).strip() if match else ""


def compute_score(
    solution_str: str,
    ground_truth: str,
    **kwargs,
) -> dict[str, float]:
    """Return binary exact-match reward for content inside ``<answer>``.

    An untagged response (no ``<answer>`` payload) always scores 0; an
    untagged ``ground_truth`` falls back to its raw text so a bare label
    like ``"B"`` still matches a properly tagged prediction.  Without the
    fallback, both ``extract_answer`` calls return ``""`` and an empty or
    malformed response was silently scored as correct (issue #643).
    """
    del kwargs
    prediction = extract_answer(solution_str)
    target = extract_answer(ground_truth)
    if not target:
        target = str(ground_truth or "").strip()
    accuracy = float(prediction == target) if prediction else 0.0
    return {
        "score": accuracy,
        "accuracy": accuracy,
    }
