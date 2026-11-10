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
"""V2 planning regression from real-run feedback; synthetic tool framing is explicit."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def module(path):
    spec = importlib.util.spec_from_file_location("isolated_initial_plan", ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def real_feedback():
    # Provenance: real v0s9x7t2 step-4 assistant-turn crop, not a synthetic plan.
    return json.loads(
        (Path(__file__).parent / "fixtures/bagel_corl_v0s9x7t2_post_call_feedback.json").read_text(encoding="utf8")
    )


def envelope():
    # Synthetic harness quality needed to exercise the V2 composite path only.
    return {
        "origin": "bagel_corl_tool_rm",
        "version": "v2",
        "num_gen_calls": 1,
        "call_seed_scores": [[0.8]],
        "autonomous_done": False,
        "forced_reflection": True,
        "image_reward_first": 0.8,
        "image_reward_final": 0.8,
        "image_reward_best": 0.8,
        "image_reward_gain": 0.0,
        "image_reward_objective": 0.8,
    }


def transcript(prefix=""):
    crop = real_feedback()
    # Same ordering as real trace: JSON call, feedback in first assistant turn,
    # environment image, then feedback in second assistant turn. Image path and
    # trusted reflection below are synthetic scaffolding, never real-run evidence.
    return "\n".join(
        [
            prefix,
            '{"name":"generate_image","arguments":{"prompt":"fixture"}}',
            crop["feedback_blocks"][0],
            "agentic_tool ok=1 path=fixture.png",
            crop["feedback_blocks"][1],
            "Reflection: VL judge reports correctness=0.8, aesthetics=0.8, good_enough=YES. "
            "Stop now; do not call another tool. Your next and only action must be exactly Done. "
            "agentic_stop_decision_required=1",
            "Done.",
        ]
    )


def test_real_feedback_earns_legacy_plan_credit_but_no_v2_initial_plan_or_format_credit():
    scorer = module("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py")
    gt = {"task_type": "plan", "reference_subtasks": real_feedback()["reference_subtasks"]}
    legacy = scorer.compute_score(solution_str=transcript(), ground_truth=gt)
    updated = scorer.compute_score(
        solution_str=transcript(),
        ground_truth=gt,
        extra_info={"corl_reward_version": "v2", "bagel_corl_reward": envelope()},
    )
    assert legacy["reward_plan"] == pytest.approx(0.19641649763353616)
    assert updated["reward_plan"] == 0.0
    assert updated["reward_format"] == pytest.approx(legacy["reward_format"] - 1 / 6)
    assert legacy["score"] == pytest.approx(0.8013083164300203)


def test_v2_valid_initial_plan_retains_full_plan_and_format_credit():
    scorer = module("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py")
    text = transcript("Plan:\n1. Draw a red fox outdoors")
    result = scorer.compute_score(
        solution_str=text,
        ground_truth={"task_type": "plan", "reference_subtasks": ["Draw a red fox outdoors"]},
        extra_info={"corl_reward_version": "v2", "bagel_corl_reward": envelope()},
    )
    assert result["reward_plan"] == 1
    assert result["reward_format"] == 1


@pytest.mark.parametrize(
    "call",
    [
        '<tools>{"name":"generate_image","arguments":{"prompt":"fixture"}}</tools>',
        '<tool_call>{"name":"judge_image","arguments":{}}</tool_call>',
        '<|start of ToolCall|>[{"name":"generate_image","arguments":{"prompt":"fixture"}}]<|end of ToolCall|>',
        '{"name":"generate_image","arguments":{"prompt":"fixture"}}',
    ],
)
def test_initial_plan_still_matches_and_later_reflection_is_excluded_across_call_dialects(call):
    scorer = module("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py")
    text = "Plan:\n1. Draw a red fox outdoors\n" + call + "\nReflection:\n- Paint blue ocean waves"
    assert hasattr(scorer, "_extract_initial_plan_lines"), "V2 must select only pre-call plan prose"
    assert scorer._extract_initial_plan_lines(text) == ["Draw a red fox outdoors"]
