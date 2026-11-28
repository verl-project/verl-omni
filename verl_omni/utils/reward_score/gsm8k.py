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
"""GSM8K reward that accepts the reward-loop call signature.

The loop always passes ``data_source`` and ``extra_info``. verl's scorer only
takes the solution text and the ground-truth number, and returns a bare float.
"""

from verl.utils.reward_score.gsm8k import compute_score as _gsm8k_score


def compute_score(solution_str: str, ground_truth: str, **kwargs) -> dict[str, float]:
    """Score a GSM8K response by the number after ``####``.

    ``ground_truth`` is the extracted answer (for example ``"42"``), as written
    by verl's GSM8K preprocessor. A missing or wrong ``####`` answer scores 0.
    """
    del kwargs
    accuracy = float(_gsm8k_score(solution_str, ground_truth))
    return {"score": accuracy, "accuracy": accuracy}
