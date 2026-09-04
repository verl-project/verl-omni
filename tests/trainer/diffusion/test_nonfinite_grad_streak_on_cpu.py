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
"""CPU tests for the non-finite grad_norm streak tracker (verl-project/verl-omni#388, item A3)."""

import math

import pytest

from verl_omni.trainer.diffusion.diffusion_trainer_utils import track_nonfinite_grad_streak


class TestTrackNonfiniteGradStreak:
    def test_finite_grad_norm_resets_streak(self):
        assert track_nonfinite_grad_streak(streak=3, grad_norm=1.5, max_consecutive=5) == 0

    def test_none_grad_norm_resets_streak(self):
        assert track_nonfinite_grad_streak(streak=3, grad_norm=None, max_consecutive=5) == 0

    def test_non_finite_grad_norm_increments_streak(self):
        assert track_nonfinite_grad_streak(streak=0, grad_norm=float("nan"), max_consecutive=5) == 1
        assert track_nonfinite_grad_streak(streak=1, grad_norm=float("inf"), max_consecutive=5) == 2

    def test_disabled_threshold_never_raises(self):
        streak = 0
        for _ in range(100):
            streak = track_nonfinite_grad_streak(streak, math.nan, max_consecutive=0)
        assert streak == 100

    def test_raises_once_streak_exceeds_threshold(self):
        streak = 0
        for _ in range(3):
            streak = track_nonfinite_grad_streak(streak, math.nan, max_consecutive=3)
        assert streak == 3
        with pytest.raises(RuntimeError, match="non-finite"):
            track_nonfinite_grad_streak(streak, math.nan, max_consecutive=3)

    def test_intervening_finite_step_prevents_abort(self):
        streak = 0
        for _ in range(3):
            streak = track_nonfinite_grad_streak(streak, math.nan, max_consecutive=3)
        streak = track_nonfinite_grad_streak(streak, 2.0, max_consecutive=3)
        assert streak == 0
        for _ in range(3):
            streak = track_nonfinite_grad_streak(streak, math.nan, max_consecutive=3)
        assert streak == 3
