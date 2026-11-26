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
import math
from pathlib import Path

import pytest


def _load_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/talker_length_control_reward.py"
    spec = importlib.util.spec_from_file_location("talker_length_control_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _load_module()
compute_score = reward.compute_score


def _frames(n, value=1234):
    return " ".join([str(value)] * n)


def _gaussian(diff, sigma):
    return math.exp(-((diff / sigma) ** 2))


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (70, 1.0),  # at the target
        (115, _gaussian(45, 45)),  # natural length of the untrained policy
        (100, _gaussian(30, 45)),  # above the target uses the wide window
    ],
)
def test_score_peaks_at_target_length(frames, expected):
    assert compute_score("x", _frames(frames))["score"] == pytest.approx(expected)


def test_natural_length_has_headroom():
    # The untrained policy emits ~115 frames; the score there must sit well
    # below 1 so the reward can climb as the policy learns to stop earlier.
    score = compute_score("x", _frames(115))["score"]
    assert 0.2 < score < 0.6


def test_overshoot_is_punished_harder_than_the_approach():
    # 40 frames above the target vs 40 below: the short side uses the narrow
    # window, so undershooting into near-empty speech scores far worse.
    above = compute_score("x", _frames(110))["score"]
    below = compute_score("x", _frames(30))["score"]
    assert below < above
    assert below < 0.2


def test_near_empty_speech_scores_near_zero():
    assert compute_score("x", _frames(10))["score"] < 0.05
    assert compute_score("x", _frames(20))["score"] < 0.1


def test_empty_response_is_scored_not_crashed():
    # 0 frames falls back to a 1-frame estimate; it just has to stay bounded.
    score = compute_score("x", "")["score"]
    assert 0.0 <= score <= 1.0


def test_non_numeric_fallback_counts_decoded_chars():
    # If decode produced something other than bare codec ids, fall back to a
    # char-based estimate (4 chars per frame).
    score = compute_score("x", "a" * 280)["score"]
    assert score == pytest.approx(1.0)  # 280 chars ~ 70 frames


def test_score_is_bounded():
    for n in (0, 1, 10, 70, 140, 1000):
        assert 0.0 <= compute_score("x", _frames(n))["score"] <= 1.0
