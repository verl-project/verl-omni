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
"""CPU tests for MMK12 reward parsing."""

import pytest

from verl_omni.utils.reward_score.mmk12_reward import compute_score


def test_mmk12_rejects_malformed_present_options():
    with pytest.raises(ValueError, match="valid JSON"):
        compute_score("42", "A", extra_info={"options": "not-json"})
