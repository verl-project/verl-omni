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

import pytest

from verl_omni.utils.reward_score.gsm8k import compute_score


@pytest.mark.parametrize(
    ("response", "ground_truth", "expected"),
    [
        ("reasoning\n#### 42", "42", 1.0),
        ("reasoning\n#### 41", "42", 0.0),
        ("<answer>42</answer>", "42", 0.0),
        ("no answer", "42", 0.0),
    ],
)
def test_gsm8k_scores_hash_answer(response, ground_truth, expected):
    result = compute_score(response, ground_truth, data_source="openai/gsm8k", extra_info={})
    assert result == {"score": expected, "accuracy": expected}
