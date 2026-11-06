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

"""Compatibility wrapper for the UnifiedReward 2.0 scorer.

New code should import
:func:`verl_omni.utils.reward_score.vlm_as_a_judge.compute_score_unified_reward`.
This module stays importable for existing ``pkg://`` configs and file-path loads.
"""

from verl_omni.utils.reward_score.vlm_as_a_judge import (
    DEFAULT_UNIFIED_REWARD_MODEL_PATH,
    UNIFIED_REWARD_SCORE_PATTERN,
    _aggregate_unified_reward_scores,
    _build_unified_reward_prompt,
    _parse_unified_reward_scores,
    _prepare_solution_frames,
    compute_score_unified_reward,
)

__all__ = [
    "DEFAULT_UNIFIED_REWARD_MODEL_PATH",
    "UNIFIED_REWARD_SCORE_PATTERN",
    "compute_score_unified_reward",
    # Re-exported for callers that imported these helpers from this module.
    "_aggregate_unified_reward_scores",
    "_build_unified_reward_prompt",
    "_parse_unified_reward_scores",
    "_prepare_solution_frames",
]
