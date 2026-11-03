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

"""CPU tests for the GSM8K reward: extraction precision and the no-format contract."""

import pytest

from verl_omni.utils.reward_score.gsm8k_reward import compute_score


def test_gsm8k_scores_boxed_answer():
    result = compute_score("Six times twelve is seventy-two. $\\boxed{72}$", "72")
    assert result["score"] == 1.0
    assert result["accuracy"] == 1.0


def test_gsm8k_scores_wrong_answer_zero():
    result = compute_score("The answer is $\\boxed{71}$.", "72")
    assert result["score"] == 0.0


def test_gsm8k_prefers_answer_tag_over_earlier_numbers():
    # The reasoning mentions the ground truth's neighbors; only the <answer>
    # tag content decides, so extraction noise cannot flip the score.
    response = "Maybe 71? Or 73? <answer>72</answer>"
    assert compute_score(response, "72")["score"] == 1.0
    assert compute_score(response, "71")["score"] == 0.0


def test_gsm8k_empty_answer_tag_scores_zero():
    # A degenerate empty tag must not fall back to whole-response scoring,
    # which would credit stray numbers in the reasoning.
    assert compute_score("Twice 36 is 72. <answer></answer>", "72")["score"] == 0.0


def test_gsm8k_accepts_manager_call_convention():
    # The naive reward manager calls with data_source / extra_info kwargs.
    result = compute_score("$\\boxed{1000}$", "1000", data_source="openai/gsm8k", extra_info={})
    assert result["score"] == 1.0


@pytest.mark.parametrize("ground_truth", ["72", "3.5", "1000"])
def test_gsm8k_matches_plain_numeric_ground_truths(ground_truth):
    assert compute_score(f"The answer is $\\boxed{{{ground_truth}}}$.", ground_truth)["score"] == 1.0
