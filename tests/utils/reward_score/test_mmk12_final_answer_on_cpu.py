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

import math

import pytest

from verl_omni.utils.reward_score.mmk12_final_answer import compute_score


@pytest.mark.parametrize(
    ("text", "truth"),
    [
        ("<think>Perhaps A.</think><answer>B</answer>", "B"),
        (r"Final answer: $\boxed{C}$", "C"),
        ("The correct answer is B.", "B"),
        ("Option (D)", "D"),
        ("**E**", "E"),
        ("<think>Intermediate 14.</think>\n12", "12"),
        (r"<answer>$\boxed{12}$</answer>", "12"),
        ("\\[\n.5\n\\]", "0.5"),
        ("-3", "-3"),
        ("1/2", "0.5"),
    ],
)
def test_scores_only_supported_final_answers(text, truth):
    result = compute_score(solution_str=text, ground_truth=truth, data_source="mmk12")
    assert result["score"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["answer_present"] == 1.0
    assert all(math.isfinite(value) for value in result.values())


@pytest.mark.parametrize(
    ("text", "truth"),
    [
        ("Intermediate 14.\nThe final answer is 12.", "14"),
        ("<think>Intermediate 14.</think>\n12", "14"),
        ("<think>Coordinates (l/2, 3).", "3"),
        ("We considered A, B, C and D but have not decided.", "D"),
        ("<answer>B</answer>\nA", "B"),
        ("<answer>A</answer><answer>B</answer>", "B"),
        (r"\boxed{B", "B"),
        (".5", "5"),
        ("1**2", "12"),
        ("", "A"),
    ],
)
def test_does_not_credit_intermediate_or_ambiguous_answers(text, truth):
    assert compute_score(text, truth)["score"] == 0.0


def test_token_limited_answers_are_unscored():
    result = compute_score("<answer>B</answer>", "B", extra_info={"response_at_token_limit": True})
    assert result == {"score": 0.0, "accuracy": 0.0, "answer_present": 0.0, "at_token_limit": 1.0}
