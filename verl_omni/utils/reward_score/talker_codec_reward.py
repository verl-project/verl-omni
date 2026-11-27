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
"""Length-shaped reward for the MiniCPM-o talker stage.

The talker rollout's response tokens are speech codec frames — the response
decodes to a whitespace-separated string of integer codec ids — so text
scoring is meaningless. The frame count is the one signal every talker
experiment can use, and this scorer shapes it in two presets:

* **Verification preset** (``target_frames=None``, the default): a
  saturating length floor. The score grows linearly from ``floor_frames``
  to 1.0 at ``full_frames``. An untrained talker already emits ~115 frames,
  so an E2E smoke run starts at ~1.0 and stays flat — exactly what a
  pipeline-verification run wants (shorter rollouts still spread below 1.0,
  keeping GRPO advantages non-degenerate without needing a task reward).
* **Length-control preset** (``target_frames=<int>``): an asymmetric
  Gaussian window that peaks at the target. Wide above the target (smooth
  approach from the natural length), steep below (an overshoot into
  near-empty speech is punished hard). The untrained talker emits ~115
  frames per turn (measured on the verification set), so ``target_frames=70``
  starts around 0.3 with headroom to climb — the policy must learn to emit
  its stop token earlier to raise the score.

Parameters arrive as keyword arguments, following the repo's scorer
contract (``.agents/rules/reward.md``): the multi-reward managers forward
every non-reserved ``reward_functions.<key>.*`` config entry, so
``reward.reward_functions.talker.target_frames=70`` selects the
length-control preset from the run config. Single-reward runs
(``reward.custom_reward_function.path``) get the defaults.
"""

from __future__ import annotations

import math

__all__ = ["compute_score", "count_frames"]

# Verification preset: score ramps from this many frames to 1.0 at `full_frames`.
DEFAULT_FLOOR_FRAMES = 25
DEFAULT_FULL_FRAMES = 100

# Length-control preset: Gaussian half-widths in frames around `target_frames`.
DEFAULT_SIGMA_ABOVE = 45
DEFAULT_SIGMA_BELOW = 30


def count_frames(solution_str) -> int:
    """Count the codec frames in a decoded talker response.

    The response decodes to whitespace-separated integer codec ids; anything
    else (or an empty string) falls back to a 4-chars-per-frame estimate.

    Args:
        solution_str: The decoded response string.

    Returns:
        The estimated frame count, at least 1.
    """
    parts = (solution_str or "").split()
    if parts and all(p.lstrip("-").isdigit() for p in parts):
        return len(parts)
    return max(len(solution_str or "") // 4, 1)


def compute_score(
    data_source,
    solution_str,
    ground_truth=None,
    extra_info=None,
    *,
    target_frames: int | None = None,
    floor_frames: int = DEFAULT_FLOOR_FRAMES,
    full_frames: int = DEFAULT_FULL_FRAMES,
    sigma_above: float = DEFAULT_SIGMA_ABOVE,
    sigma_below: float = DEFAULT_SIGMA_BELOW,
    **kwargs,
) -> dict:
    """Score a talker response from its codec frame count.

    Args:
        data_source: The dataset source (unused, part of the scorer contract).
        solution_str: The decoded response — whitespace-separated codec ids.
        ground_truth: Unused, part of the scorer contract.
        extra_info: Manager-injected metadata (unused here).
        target_frames: ``None`` selects the saturating verification preset;
            an integer selects the length-control window peaked there.
        floor_frames: Verification preset — score 0 at this length.
        full_frames: Verification preset — score saturates at 1.0 here.
        sigma_above: Length-control preset — window width above the target.
        sigma_below: Length-control preset — window width below the target.
        **kwargs: Absorbs forwarded config keys the scorer does not use.

    Returns:
        ``{"score": float}`` — the shaped length score in [0, 1].
    """
    frames = count_frames(solution_str)
    if target_frames is None:
        score = min(max((frames - floor_frames) / (full_frames - floor_frames), 0.0), 1.0)
    else:
        sigma = sigma_above if frames >= target_frames else sigma_below
        score = math.exp(-(((frames - target_frames) / sigma) ** 2))
    return {"score": float(score)}
