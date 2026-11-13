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
"""RPCO reward for Hermes image-generation trajectories.

Use the naive reward manager: the image-gen loop supplies runtime-owned
``trajectory_messages`` via ``extra_info`` (or a keyword argument). Tool results
are matched to calls and judge results to the latest generated image. Decoded
text alone cannot authenticate tool outcomes; ``solution_image`` is unsupported.

The weighted dimensions are reflect, format, tool, result, and plan (plan rows
only). Done and tool-call indicators are metrics. Task type is required; a
rollout without a parsed generation call or successful PNG scores zero.
Trusted judge scores must be finite and in [0, 1]. Reflection and plan coverage
use token F1; a block or rewrite after YES disables result and Done credit.
"""

from __future__ import annotations

from typing import Any

from verl_omni.utils.reward_score import agentic_reward_core as core

DIMS = ("reflect", "plan", "format", "tool", "result")
#: Names consumed by ``AgenticMetricsAgentLoopManager`` when PR1 and PR3 are composed.
REWARD_COMPONENTS = tuple(f"reward_{name}" for name in (*DIMS, "done", "tool_call"))


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Any = None,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float | str | int | None]:
    """Compute the RFC #302 stage-3 reward and its complete metric schema.

    Args:
        data_source: Unused; kept for the verl ``compute_score`` signature.
        solution_str: Decoded trajectory text (NaiveRewardManager).
        ground_truth: Must include ``task_type`` (``reflect`` / ``plan``) plus
            optional references and ``w_*`` weights.
        extra_info: Fallback for task type / weights, tokenizer, and runtime
            ``trajectory_messages`` with role and tool-call identity preserved.
        **kwargs: May include runtime ``trajectory_messages``, decoded
            ``responses`` + tokenizer, or ``solution_image`` (rejected).

    Returns:
        Dict with ``score``, per-dim ``reward_*``, and metric schema fields.
    """
    del data_source
    gt = core.as_dict(ground_truth)
    metadata = dict(extra_info or {})
    task_type = core.require_task_type(gt, metadata)
    if task_type is None:
        return core.zero_result(method="agentic_multidim_missing_task_type", components=REWARD_COMPONENTS)
    weights = core.active_weights(gt, metadata, task_type=task_type, dims=DIMS)
    if weights is None:
        return core.zero_result(method="agentic_multidim_bad_weights", components=REWARD_COMPONENTS)
    expected = core.expected_num_images(gt, metadata)

    text = core.resolve_solution_text(
        solution_str, kwargs=kwargs, extra_info=metadata, scorer="agentic_multidim_reward"
    )
    kwargs.pop("solution_image", None)
    if not text.strip() and "trajectory_messages" not in kwargs and "trajectory_messages" not in metadata:
        result = core.zero_result(method="agentic_multidim_empty", components=REWARD_COMPONENTS)
        result.update(task_type=task_type, expected_num_images=expected)
        return result

    assessment = core.TrajectoryAssessment.from_messages(
        text, kwargs.get("trajectory_messages", metadata.get("trajectory_messages"))
    )
    result = core.zero_result(method="agentic_multidim", components=REWARD_COMPONENTS)
    result.update(assessment.metrics(), expected_num_images=expected, task_type=task_type)
    if not assessment.valid:
        return result

    rewards = assessment.dimension_scores(task_type=task_type, expected=expected, ground_truth=gt)
    result.update(
        score=core.weighted_score(rewards, weights),
        **{f"reward_{dim}": float(rewards[dim]) for dim in DIMS},
        reward_done=float(assessment.closed),
        protocol_ok=int(rewards["format"] == 1.0),
        rollout_valid=1,
    )
    return result
