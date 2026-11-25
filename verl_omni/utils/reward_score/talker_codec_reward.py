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
"""Smoke reward for MiniCPM-o talker-stage rollouts.

The talker rollout's response tokens are speech codec frames, not text, so
text-scoring rewards are meaningless. This reward drives the E2E verification:
a length-shaped score that varies per sample (so GRPO advantages are
non-degenerate) and stays within [0, 1].

Not intended as a task reward — see the talker OPD recipe for the distillation
objective.
"""

from __future__ import annotations

__all__ = ["compute_score"]


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None, **kwargs):
    length = len(solution_str or "")
    # Saturating length shaping: ~0 at 25 chars, ~1 at 225 chars.
    score = min(max((length - 25.0) / 200.0, 0.0), 1.0)
    return {"score": float(score)}
