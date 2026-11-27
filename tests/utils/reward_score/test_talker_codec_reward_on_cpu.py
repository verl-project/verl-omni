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
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/talker_codec_reward.py"
    spec = importlib.util.spec_from_file_location("talker_codec_reward", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


reward = _load_module()
compute_score = reward.compute_score
count_frames = reward.count_frames


def _frames(n, value=1234):
    return " ".join([str(value)] * n)


# ---------------------------------------------------------------------------
# Frame counting
# ---------------------------------------------------------------------------


def test_count_frames_reads_codec_ids():
    assert count_frames("356 4221 18") == 3


def test_count_frames_falls_back_for_non_numeric_decode():
    assert count_frames("a" * 240) == 60
    assert count_frames("") == 1


# ---------------------------------------------------------------------------
# Verification preset (target_frames=None)
# ---------------------------------------------------------------------------


def test_verification_preset_saturates_at_full_frames():
    assert compute_score("x", _frames(100))["score"] == 1.0
    assert compute_score("x", _frames(115))["score"] == 1.0  # untrained talker


def test_verification_preset_ramps_from_floor():
    assert compute_score("x", _frames(25))["score"] == 0.0
    mid = compute_score("x", _frames(62))["score"]
    assert 0.4 < mid < 0.6


def test_verification_preset_is_bounded():
    for n in (0, 1, 10, 60, 120, 1000):
        assert 0.0 <= compute_score("x", _frames(n))["score"] <= 1.0


# ---------------------------------------------------------------------------
# Length-control preset (target_frames=<int>)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frames", "expected"),
    [
        (70, 1.0),  # at the target
        (115, math.exp(-((45 / 45) ** 2))),  # natural length of the untrained policy
        (100, math.exp(-((30 / 45) ** 2))),  # above the target uses the wide window
    ],
)
def test_length_control_peaks_at_target(frames, expected):
    assert compute_score("x", _frames(frames), target_frames=70)["score"] == pytest.approx(expected)


def test_length_control_has_headroom_at_natural_length():
    score = compute_score("x", _frames(115), target_frames=70)["score"]
    assert 0.2 < score < 0.6


def test_length_control_overshoot_punished_harder_than_approach():
    above = compute_score("x", _frames(110), target_frames=70)["score"]
    below = compute_score("x", _frames(30), target_frames=70)["score"]
    assert below < above
    assert below < 0.2


def test_length_control_near_empty_speech_scores_near_zero():
    assert compute_score("x", _frames(10), target_frames=70)["score"] < 0.05
    assert compute_score("x", _frames(20), target_frames=70)["score"] < 0.1


# ---------------------------------------------------------------------------
# Config-key contract
# ---------------------------------------------------------------------------


def test_forwarded_config_keys_do_not_crash():
    result = compute_score(
        "x",
        _frames(70),
        target_frames=70,
        sigma_above=45,
        sigma_below=30,
        some_unrelated_forwarded_key="ignored",
    )
    assert result == {"score": 1.0}


def test_custom_sigmas_change_the_window():
    tight = compute_score("x", _frames(90), target_frames=70, sigma_above=10)["score"]
    wide = compute_score("x", _frames(90), target_frames=70, sigma_above=45)["score"]
    assert tight < wide
