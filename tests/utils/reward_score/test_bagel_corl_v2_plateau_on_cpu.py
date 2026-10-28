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
"""V2 reward dimensions and the plan prompt's format example."""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def scorer():
    return module("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py", "isolated_v2_plateau_reward")


def envelope(num_gen_calls=1):
    scores = [[0.7]] * num_gen_calls
    return {
        "origin": "bagel_corl_tool_rm",
        "version": "v2",
        "num_gen_calls": num_gen_calls,
        "call_seed_scores": scores,
        "autonomous_done": False,
        "forced_reflection": True,
        "image_reward_first": 0.7,
        "image_reward_final": 0.7,
        "image_reward_best": 0.7,
        "image_reward_gain": 0.0,
        "image_reward_objective": 0.7 - 0.01 * max(0, num_gen_calls - 1),
    }


def forced_transcript(num_generates=1):
    calls = [f'{{"name":"generate_image","arguments":{{"prompt":"subtask {i}"}}}}' for i in range(num_generates)]
    return "\n".join(
        [
            *[f"{call}\nagentic_tool ok=1 path=fixture{i}.png" for i, call in enumerate(calls)],
            "Reflection: VL judge reports correctness=0.8, aesthetics=0.8, good_enough=YES. "
            "Stop now; do not call another tool. Your next and only action must be exactly Done. "
            "agentic_stop_decision_required=1",
            "Done.",
        ]
    )


def v2(text, gt, num_gen_calls=1):
    return scorer().compute_score(
        solution_str=text,
        ground_truth=gt,
        extra_info={"corl_reward_version": "v2", "bagel_corl_reward": envelope(num_gen_calls)},
    )


def test_v2_score_ignores_reflect_when_the_harness_wrote_the_reflection():
    gt = {"task_type": "reflect"}
    light = v2(forced_transcript(), {**gt, "w_reflect": 0.0})
    heavy = v2(forced_transcript(), {**gt, "w_reflect": 5.0})
    assert light["forced_reflection_context"] == 1
    assert light["terminal_policy_reflection"] == 0
    assert heavy["reward_reflect"] > 0
    assert heavy["score"] == pytest.approx(light["score"])


def test_v1_score_still_weights_the_forced_reflect_dim():
    module_ = scorer()
    gt = {"task_type": "reflect"}
    light = module_.compute_score(solution_str=forced_transcript(), ground_truth={**gt, "w_reflect": 0.0})
    heavy = module_.compute_score(solution_str=forced_transcript(), ground_truth={**gt, "w_reflect": 5.0})
    assert heavy["score"] != pytest.approx(light["score"])


@pytest.mark.parametrize(("generates", "expected_result"), [(1, 1 / 3), (2, 2 / 3), (3, 1.0)])
def test_v2_grades_plan_result_by_images_generated(generates, expected_result):
    gt = {"task_type": "plan", "expected_num_images": 3, "reference_subtasks": ["a", "b", "c"]}
    result = v2(forced_transcript(generates), gt, num_gen_calls=generates)
    legacy = scorer().compute_score(solution_str=forced_transcript(generates), ground_truth=gt)
    assert result["n_successful_generates"] == generates
    assert result["reward_result"] == pytest.approx(expected_result)
    assert legacy["reward_result"] == (1.0 if generates == 3 else 0.0)


def test_v2_plan_result_is_not_zeroed_by_a_rewrite_after_a_no_verdict():
    gt = {"task_type": "plan", "expected_num_images": 2, "reference_subtasks": ["a", "b"]}
    calls = [f'{{"name":"generate_image","arguments":{{"prompt":"subtask {i}"}}}}' for i in range(3)]
    text = "\n".join(
        [
            *[f"{call}\nagentic_tool ok=1 path=fixture{i}.png" for i, call in enumerate(calls[:2])],
            "Reflection: VL judge reports correctness=0.4, aesthetics=0.5, good_enough=NO. "
            "rubber_stamp=False. hands are wrong. Rewriting the diffusion prompt next.",
            f"{calls[2]}\nagentic_tool ok=1 path=fixture2.png",
            "Reflection: VL judge reports correctness=0.8, aesthetics=0.8, good_enough=YES. "
            "Stop now; do not call another tool. Your next and only action must be exactly Done. "
            "agentic_stop_decision_required=1",
            "Done.",
        ]
    )
    rewritten = v2(text, gt, num_gen_calls=3)
    over_planned = v2(forced_transcript(3), gt, num_gen_calls=3)
    assert rewritten["n_successful_generates"] == 3
    assert rewritten["reward_result"] == pytest.approx(1.0)
    assert over_planned["reward_result"] == 0.0


def test_v2_accepts_a_terminal_done_without_the_period():
    gt = {"task_type": "reflect"}
    spaced = v2(forced_transcript(), gt)
    unpunctuated = v2(forced_transcript().removesuffix("Done.") + "Done", gt)
    assert spaced["terminal_done"] == 1
    assert unpunctuated["terminal_done"] == 1
    assert unpunctuated["reward_result"] == pytest.approx(spaced["reward_result"])
    assert unpunctuated["reward_done"] == pytest.approx(spaced["reward_done"])


def test_v2_plan_image_charges_only_calls_past_the_plan():
    gt = {"task_type": "plan", "expected_num_images": 3, "reference_subtasks": ["a", "b", "c"]}
    assert v2(forced_transcript(3), gt, num_gen_calls=3)["reward_image"] == pytest.approx(0.7)
    reflect = v2(forced_transcript(3), {"task_type": "reflect"}, num_gen_calls=3)
    assert reflect["reward_image"] == pytest.approx(0.68)


def test_v2_extra_fields_stack_across_episodes_with_different_call_counts():
    import numpy as np

    gt = {"task_type": "reflect"}
    infos = [v2(forced_transcript(k), gt, num_gen_calls=k) for k in (1, 3)]
    no_image = envelope(0)
    no_image.update({key: None for key in no_image if key.startswith("image_reward_")})
    infos.append(
        scorer().compute_score(
            solution_str="Done.",
            ground_truth=gt,
            extra_info={"corl_reward_version": "v2", "bagel_corl_reward": no_image},
        )
    )
    for key in infos[0]:
        values = [info[key] for info in infos]
        np.array(values)
        if not isinstance(values[0], str):
            assert None not in values, key
            float(np.mean(values))


def test_plan_system_prompt_example_parses_as_plan_and_generate_call():
    prompts = module("verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/prompts.py", "isolated_v2_plateau_prompts")
    example = prompts.BAGEL_PLAN_SYSTEM_PROMPT.split("Start your reply like this", 1)[1].split("\n", 1)[1]
    filled = re.sub(r"<complete image prompt for subtask (\d)>", r"a red fox standing in snowy woods \1", example)
    module_ = scorer()
    assert module_._extract_initial_plan_lines(filled) == [
        "a red fox standing in snowy woods 1",
        "a red fox standing in snowy woods 2",
    ]
    calls = module_._extract_tool_calls(module_._normalize_trajectory_dialect(filled))
    assert [module_.tool_name(call) for _, _, call in calls] == ["generate_image"]
