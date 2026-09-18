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
"""CPU tests for the judge-outcome lift primitives."""

from __future__ import annotations

import pytest

from verl_omni.utils.agentic.judge_delta import judge_delta_reward, preferred_judge


def test_judge_delta_matches_the_live_cn_poster_chain():
    """The rollout that motivated the dim: two appends that the judge rewarded.

    ``sample_9003`` re-sent its recipe with a trailing legibility clause each pass and
    the judge moved ``correctness 0.00 -> 0.36 -> 0.56``, ``aesthetics 0.44 -> 0.60 ->
    0.80``. The chain never reached ``good_enough=YES``, so it is scored on where it
    finished: mean(0.56 - 0.00, 0.80 - 0.44) = 0.46.
    """
    judges = [(0.00, 0.44, False), (0.36, 0.60, False), (0.56, 0.80, False)]

    assert preferred_judge(judges) == (0.56, 0.80, False)
    assert judge_delta_reward(judges) == (pytest.approx(0.46), pytest.approx(0.46))


def test_judge_delta_is_zero_for_a_single_judge():
    """No chain means no lift to measure."""
    assert judge_delta_reward([(0.2, 0.3, False)]) == (0.0, 0.0)
    assert judge_delta_reward([]) == (0.0, 0.0)


def test_judge_delta_ignores_a_chain_whose_first_judge_passed():
    """A first-pass YES means the protocol asked for a stop, not a recovery."""
    judges = [(0.80, 0.80, True), (1.00, 1.00, True)]

    assert judge_delta_reward(judges) == (0.0, 0.0)


def test_judge_delta_requires_an_explicit_first_pass_failure():
    """``good_enough=None`` is a missing marker, not a NO, so it cannot bank a lift."""
    assert judge_delta_reward([(0.1, 0.1, None), (0.9, 0.9, True)]) == (0.0, 0.0)


def test_judge_delta_prefers_the_satisfying_judge_over_a_later_score():
    """The lift is measured against the image the rollout settled on.

    ``preferred_judge`` returns the first ``good_enough=YES``, matching
    ``agentic_multidim_reward._reflection_reward``, so a later pass that scored higher
    but no longer satisfied the judge cannot inflate the lift.
    """
    judges = [(0.0, 0.0, False), (0.50, 0.50, True), (0.90, 0.90, False)]

    assert preferred_judge(judges) == (0.50, 0.50, True)
    assert judge_delta_reward(judges) == (pytest.approx(0.50), pytest.approx(0.50))


def test_judge_delta_reports_a_regression_but_pays_nothing():
    """The raw value stays signed for diagnosis; the reward term is floored at zero."""
    score, raw_delta = judge_delta_reward([(0.60, 0.70, False), (0.10, 0.20, True)])

    assert raw_delta == pytest.approx(-0.50)
    assert score == 0.0


def test_judge_delta_is_clamped_to_one():
    """Facet scores are already in ``[0, 1]``; the mean lift cannot exceed it."""
    score, raw_delta = judge_delta_reward([(0.0, 0.0, False), (1.0, 1.0, True)])

    assert raw_delta == pytest.approx(1.0)
    assert score == pytest.approx(1.0)
