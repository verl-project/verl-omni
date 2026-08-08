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
"""Deterministic length-heuristic reward for the one-step agentic GRPO smoke.

TODO (fred): include multi-dimensional RPCO rewards (reflection / plan /
image quality) with the multi-step e2e in
https://github.com/verl-project/verl-omni/issues/303.
"""

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict | None = None,
    **kwargs: Any,
) -> dict[str, float | str]:
    """Length heuristic so one-step agentic GRPO has non-zero reward variance."""
    del data_source, ground_truth, extra_info, kwargs
    text = (solution_str or "").strip()
    score = 0.0 if not text else min(1.0, len(text) / 256.0)
    return {"score": float(score), "method": "response_length_heuristic"}
