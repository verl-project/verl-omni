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

"""GSM8K reward — math_verify numeric match, no format component.

Design:

1. **Answer extraction** prefers the ``<answer>…</answer>`` tag (the MiniCPM-o
   checkpoint emits these tags natively, and ``minicpm_naive`` keeps them in the
   decoded string). Without the tag the full response is scored, so ``\\boxed{}``
   and plain final answers both work.
2. **Correctness** reuses verl's ``math_verify.compute_score``
   (``ExprExtractionConfig`` + ``LatexExtractionConfig``), which is
   subprocess-isolated and timed.
3. **No format reward**: ``score == accuracy``, so the initial reward equals raw
   accuracy — the property that makes GSM8K the OPD verification task.

Config:
    reward.custom_reward_function.path=verl_omni/utils/reward_score/gsm8k_reward.py
    reward.custom_reward_function.name=compute_score
"""

import re

from verl.utils.reward_score.math_verify import compute_score as _math_verify_score

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _extract_answer_tag(text: str) -> str | None:
    """Return the inner text of the first ``<answer>…</answer>`` tag, or None."""
    match = _ANSWER_TAG_RE.search(text or "")
    return match.group(1).strip() if match else None


def compute_score(
    solution_str: str,
    ground_truth: str,
    math_verify_timeout: float = 20.0,
    **kwargs,
) -> dict:
    """GSM8K reward entrypoint.

    Returns ``{"score", "accuracy"}`` with ``score == accuracy`` ∈ {0, 1} from
    verl's math_verify numeric compare.
    """
    gt = str(ground_truth).strip()
    answer_inner = _extract_answer_tag(solution_str)
    # An empty tag is a degenerate answer, not "use the whole response": scoring
    # the full text would credit stray numbers in the reasoning.
    if answer_inner is None:
        content = solution_str or ""
    else:
        content = answer_inner
    accuracy = _math_verify_score(content, gt, timeout=math_verify_timeout)
    return {"score": float(accuracy), "accuracy": float(accuracy)}
