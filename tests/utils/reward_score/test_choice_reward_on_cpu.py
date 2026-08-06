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


@pytest.mark.parametrize(
    ("response", "ground_truth", "expected"),
    [
        ("reasoning <answer>B</answer>", "<answer>B</answer>", 1.0),
        ("<answer>A</answer>", "<answer>B</answer>", 0.0),
        ("The final answer is B.", "<answer>B</answer>", 0.0),
        ("<answer>b</answer>", "<answer>B</answer>", 0.0),
        ("<answer>A</answer><answer>B</answer>", "<answer>A</answer>", 1.0),
    ],
)
def test_compute_score_uses_first_tag_and_exact_match(response, ground_truth, expected):
    assert compute_score(response, ground_truth) == {"score": expected, "accuracy": expected}
