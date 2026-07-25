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

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/choice_reward.py"
    spec = importlib.util.spec_from_file_location("choice_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


choice_reward = _load_module()
compute_score = choice_reward.compute_score
extract_answer = choice_reward.extract_answer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>B</answer>", "B"),
        ("reasoning... <answer> C </answer>", "C"),
        ("<answer>\\boxed{D}</answer>", "\\boxed{D}"),
        ("The final answer is D.", ""),
    ],
)
def test_extract_answer_matches_relax_behavior(text, expected):
    assert extract_answer(text) == expected


def test_compute_score_rewards_exact_tagged_answer():
    result = compute_score("reasoning <answer>B</answer>", "<answer>B</answer>")
    assert result == {"score": 1.0, "accuracy": 1.0}


@pytest.mark.parametrize("response", ["<answer>A</answer>", "The final answer is B.", "<answer>b</answer>"])
def test_compute_score_rejects_non_exact_answer(response):
    result = compute_score(response, "<answer>B</answer>")
    assert result == {"score": 0.0, "accuracy": 0.0}


def test_compute_score_uses_first_answer_tag_like_relax():
    result = compute_score("<answer>A</answer><answer>B</answer>", "<answer>A</answer>")
    assert result == {"score": 1.0, "accuracy": 1.0}
