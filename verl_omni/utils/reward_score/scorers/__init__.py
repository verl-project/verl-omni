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
"""Class-based reward scorers under ``utils/reward_score/scorers/``.

Importing this package registers every bundled scorer (registration is an
import side effect, mirroring ``verl_omni/pipelines/__init__.py``). The flat
modules one directory up remain the frozen home of the free-function scorers
and are never moved.
"""

from verl_omni.utils.reward_score.scorers.base import RewardScorer

# Importing a scorer module registers it; keep in sync with the scorer modules above.
from verl_omni.utils.reward_score.scorers.pickscore import PickScoreScorer
from verl_omni.utils.reward_score.scorers.registry import get_scorer_cls, peek_registered_scorers, register_scorer

__all__ = [
    "PickScoreScorer",
    "RewardScorer",
    "get_scorer_cls",
    "peek_registered_scorers",
    "register_scorer",
]
