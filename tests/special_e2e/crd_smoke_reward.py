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
"""Bounded image reward for the offline-built CRD smoke fixture, not OCR evaluation."""

import math

from verl_omni.utils.reward_score.jpeg_compressibility import compute_score as jpeg_score


def compute_score(solution_image, extra_info=None, **kwargs):
    """Map the nonpositive JPEG reward to (0, 1] without an external judge."""
    score = jpeg_score(solution_image, extra_info)
    if not math.isfinite(score) or score > 0:
        raise ValueError("Expected a finite nonpositive JPEG score.")
    return {"score": 1.0 / (1.0 - score)}
