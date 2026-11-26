# Copyright 2026 Bytedance Ltd. and/or its affiliates.
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
"""Length-control reward for the MiniCPM-o talker stage.

The talker rollout's response tokens are speech codec frames, not text, so
text-scoring rewards are meaningless. This reward demonstrates policy learning:
the untrained talker emits ~115 codec frames per turn (measured on the
verification set), and the score peaks at a shorter target length, so the
policy must learn to emit EOS earlier to raise the score.

The window is asymmetric — wider above the target, steeper below — so the
policy can approach the target smoothly from its natural length while an
overshoot into near-empty speech is punished hard.

The response decodes to a whitespace-separated string of integer codec ids,
so the frame count is the number of decoded items.
"""

from __future__ import annotations

import math

__all__ = ["compute_score"]

# Score peaks at this many codec frames; the base policy's natural length is
# ~115, so the initial score sits well below 1 and has room to climb.
TARGET_FRAMES = 70
# Gaussian half-width in frames: wide on the long side (smooth approach),
# narrow on the short side (steep wall against overshoot).
SIGMA_LONG = 45
SIGMA_SHORT = 30


def _count_frames(solution_str):
    parts = (solution_str or "").split()
    if parts and all(p.lstrip("-").isdigit() for p in parts):
        return len(parts)
    # Fallback if decode produced something other than bare codec ids.
    return max(len(solution_str or "") // 4, 1)


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    frames = _count_frames(solution_str)
    sigma = SIGMA_LONG if frames >= TARGET_FRAMES else SIGMA_SHORT
    score = math.exp(-(((frames - TARGET_FRAMES) / sigma) ** 2))
    return {"score": float(score)}
