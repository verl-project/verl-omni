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
"""CPU tests for the self-contained RPCO multi-dimensional reward."""

import json

import pytest

from verl_omni.utils.reward_score.agentic_multidim_reward import (
    DIMS,
    REWARD_COMPONENTS,
    compute_score,
)


def _call(name: str, **arguments: str) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments})}\n</tool_call>"


def _generate(prompt: str, path: str, *, ok: bool = True) -> str:
    return "\n".join(
        (
            _call("generate_image", prompt=prompt),
            f"agentic_tool ok={int(ok)} images={int(ok)} path={path}",
        )
    )


def _judge(
    path: str,
    *,
    correctness: float = 0.8,
    aesthetics: float = 0.8,
    accepted: bool = True,
    findings: str = "headline legible and composition balanced",
) -> str:
    return "\n".join(
        (
            _call("judge_image", user_request="same as user message", image_prompt="last"),
            "VL judge on the last generated image:",
            f"path={path}",
            f"correctness={correctness}",
            f"aesthetics={aesthetics}",
            f"good_enough={'YES' if accepted else 'NO'}",
            f"findings: {findings}",
            "suggested_fixes: none",
            "agentic_judge ok=1 parse_ok=1 stub=0",
        )
    )


def _reflect_trajectory(
    *,
    correctness: float = 0.8,
    aesthetics: float = 0.8,
    accepted: bool = True,
) -> str:
    return "\n".join(
        (
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_00.png"),
            _judge(
                "/tmp/image_00.png",
                correctness=correctness,
                aesthetics=aesthetics,
                accepted=accepted,
            ),
            "Reflection: The headline is legible and the composition is balanced. Done.",
        )
    )


def _ground_truth(task_type: str = "reflect", expected: int = 1, **extra) -> dict:
    result = {
        "user_request": "A vertical cafe poster with a bold headline.",
        "task_type": task_type,
        "expected_num_images": expected,
    }
    result.update(extra)
    return result


def _plan_trajectory(lines: list[str], generated: int | None = None) -> str:
    count = len(lines) if generated is None else generated
    plan = [f"{index}. {line}" for index, line in enumerate(lines, start=1)]
    parts = ["Plan:", *plan]
    # Plan mode sends the whole numbered list as one prompt, so every call carries all of
    # it — not one item per call.
    for index in range(count):
        parts.append(_generate("\n".join(plan), f"/tmp/image_{index:02d}.png"))
    parts.extend(
        (
            _judge(f"/tmp/image_{max(0, count - 1):02d}.png"),
            "Reflection: The planned subtask images satisfy the request. Done.",
        )
    )
    return "\n".join(parts)


def test_reflect_reward_blends_judge_quality_and_reference_coverage():
    reference = "The headline is legible and the composition is balanced."
    output = compute_score(
        solution_str=_reflect_trajectory(correctness=0.8, aesthetics=0.6),
        ground_truth=_ground_truth(reference_steps=[{"reflection": reference, "action": "stop"}]),
    )

    assert output["rollout_valid"] == 1
    assert output["reward_reflect"] == pytest.approx(0.85)
    assert output["reward_done"] == 1.0


def test_task_type_labels_the_row_without_changing_the_score():
    """The plan corpus has no dimension of its own.

    Plan rows are graded by the same ``reflect`` dimension as the reflect corpus, and
    ``task_type`` only reaches the output as a monitoring label. The two calls below take
    the *same* trajectory and differ only in ``task_type`` (and in the reference field, so
    that neither has a reflect reference to lean on), so every metric but the label must
    be identical — and the dimension must still be scored, not left at zero.
    """
    subtasks = ["A snowy market with wooden stalls and warm string lights."]
    text = _plan_trajectory(subtasks)
    plan = compute_score(
        solution_str=text,
        ground_truth=_ground_truth(task_type="plan", expected=1, reference_subtasks=subtasks),
    )
    reflect = compute_score(
        solution_str=text,
        ground_truth=_ground_truth(task_type="reflect", expected=1),
    )

    assert plan["task_type"] == "plan"
    assert reflect["task_type"] == "reflect"
    assert set(plan) == set(reflect)
    assert {key: value for key, value in plan.items() if key != "task_type"} == {
        key: value for key, value in reflect.items() if key != "task_type"
    }
    assert plan["rollout_valid"] == 1
    assert plan["reward_reflect"] > 0.0


def test_reflect_reward_falls_back_to_live_judge_findings():
    output = compute_score(
        solution_str=_reflect_trajectory(correctness=0.8, aesthetics=0.6),
        ground_truth=_ground_truth(),
    )

    # Quality is 0.7; findings tokens are a subset of the longer policy reflection
    # so F1 coverage is 5/6, not 1.0 (recall-only used to report 0.85).
    assert output["reward_reflect"] == pytest.approx(0.5 * 0.7 + 0.5 * (10 / 12))


def test_forced_reflection_context_counts_toward_format():
    subtasks = ["A snowy market with wooden stalls and warm string lights."]
    parts = ["Plan:", f"1. {subtasks[0]}", _generate(subtasks[0], "/tmp/image_00.png"), _judge("/tmp/image_00.png")]
    parts.extend(("Reflection: injected stop cue agentic_forced_reflection=1", "Done."))
    output = compute_score(
        solution_str="\n".join(parts),
        ground_truth=_ground_truth(task_type="plan", expected=1, reference_subtasks=subtasks),
    )
    assert output["forced_reflection_context"] == 1
    assert output["terminal_policy_reflection"] == 0
    assert output["reward_format"] == 1.0
    assert output["protocol_ok"] == 1


def test_format_reward_is_structural_check_ratio():
    complete = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())
    open_loop = compute_score(
        solution_str="\n".join(
            (
                _generate("A cafe poster.", "/tmp/image.png"),
                _judge("/tmp/image.png"),
            )
        ),
        ground_truth=_ground_truth(),
    )

    assert complete["reward_format"] == 1.0
    assert 0.0 < open_loop["reward_format"] < 1.0
    assert open_loop["protocol_ok"] == 0


def test_tool_reward_requires_successful_generate_and_trusted_judge():
    output = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())

    assert output["reward_tool"] == 1.0
    assert output["reward_tool_call"] == 1.0

    open_loop = compute_score(
        solution_str=_generate("A poster.", "/tmp/image.png"),
        ground_truth=_ground_truth(),
    )
    assert open_loop["rollout_valid"] == 1
    assert open_loop["reward_tool"] == 0.0
    assert open_loop["reward_tool_call"] == 1.0
    assert open_loop["judge_parse_ok_rate"] == 0.0

    malformed = compute_score(
        solution_str="<tool_call>{bad json}</tool_call>\nagentic_tool ok=1 path=/tmp/image.png",
        ground_truth=_ground_truth(),
    )
    assert malformed["reward_tool"] == 0.0
    assert malformed["reward_tool_call"] == 0.0
    assert malformed["rollout_valid"] == 0


def test_result_follows_the_terminal_judge_not_the_declared_image_count():
    """The result is the judge's verdict, not the declared image budget.

    ``expected_num_images`` is the reference trajectory's image count. Gating on it would
    reward iterating exactly that many times and punish stopping as soon as the judge is
    satisfied, which is the opposite of what the protocol asks for.
    """
    subtasks = [
        "A snowy market with wooden stalls and warm lights.",
        "A decorated carousel in the same winter market.",
    ]
    accepted_after_one = compute_score(
        solution_str=_plan_trajectory(subtasks, generated=1),
        ground_truth=_ground_truth(task_type="plan", expected=2, reference_subtasks=subtasks),
    )
    accepted_after_two = compute_score(
        solution_str=_plan_trajectory(subtasks, generated=2),
        ground_truth=_ground_truth(task_type="plan", expected=2, reference_subtasks=subtasks),
    )
    no_budget_declared = compute_score(
        solution_str=_plan_trajectory(subtasks),
        ground_truth={"task_type": "plan", "reference_subtasks": subtasks},
    )
    rejected = compute_score(
        solution_str="\n".join(
            (
                "Plan:",
                *(f"{index}. {line}" for index, line in enumerate(subtasks, start=1)),
                _generate("\n".join(subtasks), "/tmp/image_00.png"),
                _judge("/tmp/image_00.png", accepted=False),
                "Reflection: The image still misses the requested lights. Done.",
            )
        ),
        ground_truth=_ground_truth(task_type="plan", expected=2, reference_subtasks=subtasks),
    )

    assert accepted_after_one["reward_result"] == 1.0
    assert accepted_after_two["reward_result"] == 1.0
    # The budget is reported, not enforced, so its absence is not a failure either.
    assert no_budget_declared["reward_result"] == 1.0
    # Fail closed on a terminal NO: stopping early is not a free result point.
    assert rejected["reward_result"] == 0.0


def test_reflect_result_requires_terminal_yes_and_ignores_the_reference_budget():
    """Reflect scores on the judge's verdict, not on the reference image count.

    ``expected_num_images`` is the reference trajectory's image count. Enforcing it would
    punish iterating, which is what the reflect protocol asks the agent to do, so the
    budget is logged but never gates ``reward_result``.
    """
    early_yes = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(expected=3))
    early_no = compute_score(
        solution_str=_reflect_trajectory(accepted=False),
        ground_truth=_ground_truth(expected=3),
    )
    over = "\n".join(
        (
            _generate("version one", "/tmp/one.png"),
            _generate("version two", "/tmp/two.png"),
            _judge("/tmp/two.png", accepted=True),
            "Reflection: The second version resolved the findings. Done.",
        )
    )
    over_output = compute_score(solution_str=over, ground_truth=_ground_truth(expected=1))
    blocked = compute_score(
        solution_str="\n".join((over, "agentic_tool ok=0 path=none blocked_after_max_passes=1")),
        ground_truth=_ground_truth(expected=1),
    )

    # A YES is a YES regardless of how far below the reference budget it is...
    assert early_yes["reward_result"] == 1.0
    # ...and iterating past the reference budget is no longer penalised.
    assert over_output["n_successful_generates"] == 2
    assert over_output["reward_result"] == 1.0
    # A terminal NO still fails closed.
    assert early_no["reward_result"] == 0.0
    # And so does a blocked rollout, which is the tool-level dithering guard.
    assert blocked["reward_result"] == 0.0


def test_weighted_total_scores_every_dimension_for_every_data_source():
    """``task_type`` labels a row; it does not select a dimension.

    There is no per-task active set any more, so a ``w_plan`` key left in an older
    parquet is ignored by both corpora and the same weights apply to reflect and plan rows.
    """
    text = _reflect_trajectory(correctness=0.8, aesthetics=0.6)
    ground_truth = _ground_truth(
        reference_steps=[{"reflection": "unrelated reference tokens", "action": "stop"}],
        w_reflect=2.0,
        w_format=1.0,
        w_tool=1.0,
        w_result=1.0,
    )
    output = compute_score(solution_str=text, ground_truth=ground_truth)
    active = {"reflect": 2.0, "format": 1.0, "tool": 1.0, "result": 1.0, "improve": 1.0}
    expected = sum(weight * output[f"reward_{dim}"] for dim, weight in active.items()) / sum(active.values())

    assert output["score"] == pytest.approx(expected)
    # A stale ``w_plan`` is inert: it is not a dimension, so it cannot shift the total.
    with_plan_weight = compute_score(
        solution_str=text,
        ground_truth={**ground_truth, "w_plan": 99.0},
    )
    assert with_plan_weight["score"] == pytest.approx(output["score"])


def test_zero_weights_keep_valid_rollout_but_zero_score():
    ground_truth = _ground_truth(**{f"w_{dim}": 0.0 for dim in DIMS})
    output = compute_score(solution_str=_reflect_trajectory(), ground_truth=ground_truth)

    assert output["rollout_valid"] == 1
    assert output["score"] == 0.0


def test_garbage_weights_fail_closed():
    garbage = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(w_reflect="not-a-float"))
    assert garbage["method"] == "agentic_multidim_bad_weights"
    assert garbage["rollout_valid"] == 0
    assert garbage["score"] == 0.0

    negative = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(w_format=-1.0))
    assert negative["method"] == "agentic_multidim_bad_weights"


def test_done_indicator_requires_successful_judge_and_terminal_decision():
    open_output = compute_score(
        solution_str="\n".join((_generate("A poster.", "/tmp/image.png"), _judge("/tmp/image.png"))),
        ground_truth=_ground_truth(),
    )
    closed_output = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())

    assert open_output["reward_done"] == 0.0
    assert closed_output["reward_done"] == 1.0


def test_rewrite_after_first_yes_breaks_done_indicator():
    text = "\n".join(
        (
            _generate("version one", "/tmp/one.png"),
            _judge("/tmp/one.png", accepted=True),
            _generate("version two", "/tmp/two.png"),
            _judge("/tmp/two.png", accepted=False),
            "Reflection: The unnecessary rewrite is worse. Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["rewrite_after_yes"] == 1
    assert output["reward_done"] == 0.0
    assert output["reward_result"] == 0.0


def test_forced_reflection_text_does_not_count_as_policy_reflection():
    text = "\n".join(
        (
            _generate("A poster.", "/tmp/image.png"),
            _judge("/tmp/image.png"),
            "Reflection: injected stop cue agentic_forced_reflection=1",
            "Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["forced_reflection_context"] == 1
    assert output["terminal_policy_reflection"] == 0
    assert output["terminal_done"] == 1
    assert output["reward_done"] == 1.0
    assert output["reward_format"] == 1.0
    assert output["protocol_ok"] == 1


def test_forced_reflection_text_does_not_inflate_reference_coverage():
    injected = "The headline is legible and the composition is balanced."
    text = "\n".join(
        (
            _generate("A poster.", "/tmp/image.png"),
            _judge("/tmp/image.png", correctness=0.8, aesthetics=0.6),
            f"Reflection: {injected} agentic_forced_reflection=1",
            "Done.",
        )
    )
    output = compute_score(
        solution_str=text,
        ground_truth=_ground_truth(reference_steps=[{"reflection": injected, "action": "stop"}]),
    )

    # Injected text contributes no coverage: only half of the 0.7 judge quality.
    assert output["reward_reflect"] == pytest.approx(0.35)


def test_missing_or_invalid_task_type_fails_closed():
    missing = compute_score(solution_str=_reflect_trajectory(), ground_truth={"user_request": "x"})
    assert missing["method"] == "agentic_multidim_missing_task_type"
    assert missing["rollout_valid"] == 0
    assert missing["score"] == 0.0

    bad = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth(task_type="other"))
    assert bad["method"] == "agentic_multidim_missing_task_type"
    assert bad["score"] == 0.0

    from_extra = compute_score(
        solution_str=_reflect_trajectory(),
        ground_truth={"user_request": "x", "expected_num_images": 1},
        extra_info={"task_type": "reflect"},
    )
    assert from_extra["rollout_valid"] == 1
    assert from_extra["task_type"] == "reflect"


def test_solution_image_without_text_raises():
    with pytest.raises(ValueError, match="solution_str"):
        compute_score(ground_truth=_ground_truth(), solution_image=object())


def test_qwen_xml_tool_calls_are_supported():
    generate = "<tool_call><function=generate_image><parameter=prompt>A cafe poster</parameter></function></tool_call>"
    judge = (
        "<tool_call><function=judge_image>"
        "<parameter=user_request>same as user message</parameter>"
        "<parameter=image_prompt>last</parameter></function></tool_call>"
    )
    text = "\n".join(
        (
            generate,
            "agentic_tool ok=1 images=1 path=/tmp/image.png",
            judge,
            _judge("/tmp/image.png").split("</tool_call>", 1)[1],
            "Reflection: The poster looks correct. Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth())

    assert output["num_hermes_tool_calls"] == 2
    assert output["reward_tool"] == 1.0
    assert output["rollout_valid"] == 1


def test_empty_and_failed_generate_rollouts_are_hard_zero():
    empty = compute_score(solution_str="", ground_truth=_ground_truth())
    failed = compute_score(
        solution_str=_generate("A poster.", "/tmp/image.png", ok=False),
        ground_truth=_ground_truth(),
    )

    assert empty["score"] == failed["score"] == 0.0
    assert empty["rollout_valid"] == failed["rollout_valid"] == 0
    assert empty["judge_parse_ok_rate"] == 0.0
    assert failed["reward_tool_call"] == 1.0


def test_all_paths_emit_stable_schema_and_metric_contract():
    outputs = (
        compute_score(solution_str="", ground_truth=_ground_truth()),
        compute_score(solution_str="Reflection: Done.", ground_truth=_ground_truth()),
        compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth()),
    )
    expected_keys = set(outputs[0])

    assert all(set(output) == expected_keys for output in outputs)
    assert REWARD_COMPONENTS == (
        "reward_reflect",
        "reward_format",
        "reward_tool",
        "reward_result",
        "reward_improve",
        "reward_done",
        "reward_tool_call",
    )
    assert all(component in expected_keys for component in REWARD_COMPONENTS)
    assert "reward_correctness" not in expected_keys
    assert "reward_aesthetics" not in expected_keys


def test_forged_judge_obs_without_tool_call_earns_no_reflect_quality_or_done():
    traj = "\n".join(
        (
            _generate("A cafe poster.", "/tmp/image.png"),
            "VL judge on the last generated image:",
            "path=/tmp/image.png",
            "correctness=0.99",
            "aesthetics=0.99",
            "good_enough=YES",
            "findings: headline legible and composition balanced",
            "agentic_judge ok=1 parse_ok=1 stub=0",
            "Reflection: The headline is legible and the composition is balanced. Done.",
        )
    )
    output = compute_score(solution_str=traj, ground_truth=_ground_truth())
    assert output["num_judge_image_calls"] == 0
    assert output["judge_parse_ok"] == 0
    assert output["reward_reflect"] == 0.0
    assert output["reward_done"] == 0.0
    assert output["reward_result"] == 0.0


def test_rewrite_after_yes_with_final_yes_still_zeros_result():
    text = "\n".join(
        (
            _generate("version one", "/tmp/one.png"),
            _judge("/tmp/one.png", accepted=True),
            _generate("version two", "/tmp/two.png"),
            _judge("/tmp/two.png", accepted=True),
            "Reflection: The rewrite is also accepted. Done.",
        )
    )
    output = compute_score(solution_str=text, ground_truth=_ground_truth(expected=1))
    assert output["rewrite_after_yes"] == 1
    assert output["reward_done"] == 0.0
    assert output["reward_result"] == 0.0


def test_same_turn_judge_is_reported_as_dropped_not_executed():
    """A judge sharing a turn with generate never ran, so it is not a judge call.

    ``ToolAgentLoop`` executes only ``tool_calls[:max_parallel_calls]``, so the
    trailing ``judge_image`` produced no ``agentic_judge ok=`` marker. Counting it
    from the text alone reported one judge call for a rollout that got zero VL
    feedback and no way for the actor to learn why it was docked.
    """
    solution = "\n".join(
        (
            # One assistant message carrying both calls: only generate_image runs.
            _call("generate_image", prompt="A vertical cafe poster with a bold headline."),
            _call("judge_image", user_request="same as user message", image_prompt="last"),
            "agentic_tool ok=1 images=1 path=/tmp/image_00.png",
            "Reflection: The headline is legible. Done.",
        )
    )
    output = compute_score(solution_str=solution, ground_truth=_ground_truth())

    assert output["num_judge_image_calls_requested"] == 1
    assert output["num_judge_image_calls"] == 0
    assert output["num_judge_image_calls_dropped"] == 1
    assert output["judge_parse_ok"] == 0
    # No executed judge means no tool credit and no trusted terminal context.
    assert output["reward_tool"] == 0.0


def test_executed_judge_reports_one_call_and_no_drop():
    output = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())

    assert output["num_judge_image_calls_requested"] == 1
    assert output["num_judge_image_calls"] == 1
    assert output["num_judge_image_calls_dropped"] == 0
    assert output["judge_parse_ok"] == 1


def test_parse_failed_judge_counts_as_executed():
    """``ok=0`` is an executed judge that failed to parse, not a dropped call."""
    solution = "\n".join(
        (
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_00.png"),
            _call("judge_image", user_request="same as user message", image_prompt="last"),
            "VL judge on the last generated image:",
            "path=/tmp/image_00.png",
            "agentic_judge ok=0 stub=0 backend=vllm parse_retries=1",
        )
    )
    output = compute_score(solution_str=solution, ground_truth=_ground_truth())

    assert output["num_judge_image_calls_requested"] == 1
    assert output["num_judge_image_calls"] == 1
    assert output["num_judge_image_calls_dropped"] == 0
    assert output["judge_parse_fail"] == 1


def test_generate_counters_separate_requested_executed_and_dropped():
    """``num_generate_image_prompts`` must count executed calls, not text prompts.

    Mirrors a rollout that re-emits the same prompt every turn: after the third
    successful image the pass cap refuses further calls (``ok=0``), so the text
    holds many prompts while only a few corresponds to real work.
    """
    solution = "\n".join(
        (
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_00.png"),
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_01.png"),
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_02.png"),
            _call("generate_image", prompt="A vertical cafe poster with a bold headline."),
            "generate_image blocked: already completed 3/3 successful generate_image passes.",
            "agentic_tool ok=0 stub=0 images=0 backend=blocked_after_max_passes prompt='cafe'",
            _call("generate_image", prompt="A vertical cafe poster with a bold headline."),
        )
    )
    output = compute_score(solution_str=solution, ground_truth=_ground_truth())

    assert output["num_generate_image_prompts_requested"] == 5
    assert output["num_generate_image_prompts"] == 4  # 3 ok=1 + 1 blocked but executed
    assert output["num_generate_image_prompts_dropped"] == 1
    assert output["n_successful_generates"] == 3


def test_dropped_tool_notice_is_context_only():
    """The drop notice must not leak into assistant prose.

    The notice is an environment observation carrying an explanatory sentence, so it
    has to be stripped before prose-based scoring or it could earn reflection credit.
    """
    solution = "\n".join(
        (
            _reflect_trajectory(),
            "Ignored: at most 1 tool call(s) run per turn. judge_image in this turn was not executed. "
            "Emit one tool call per turn and wait for its <tool_response> before the next call. "
            "agentic_tool_dropped n=1 names=judge_image",
        )
    )
    clean = compute_score(solution_str=solution, ground_truth=_ground_truth())
    baseline = compute_score(solution_str=_reflect_trajectory(), ground_truth=_ground_truth())

    assert clean["reward_reflect"] == baseline["reward_reflect"]
    assert clean["score"] == baseline["score"]


def test_drop_notice_does_not_inflate_executed_generate_count():
    """``agentic_tool_dropped`` has no ``ok=`` marker, so it is not a generate."""
    solution = "\n".join(
        (
            _generate("A vertical cafe poster with a bold headline.", "/tmp/image_00.png"),
            "Ignored: at most 1 tool call(s) run per turn. judge_image in this turn was not executed. "
            "agentic_tool_dropped n=1 names=judge_image",
        )
    )
    output = compute_score(solution_str=solution, ground_truth=_ground_truth())

    assert output["num_generate_image_prompts"] == 1
    assert output["num_generate_image_prompts_requested"] == 1


def _rewrite_trajectory(first: str, second: str) -> str:
    """A two-pass reflect rollout that rewrites ``first`` into ``second``."""
    return "\n".join(
        (
            _generate(first, "/tmp/image_00.png"),
            _judge("/tmp/image_00.png", accepted=False),
            _generate(second, "/tmp/image_01.png"),
            _judge("/tmp/image_01.png", accepted=True),
            "Reflection: The rewrite resolved the findings. Done.",
        )
    )


def _rewrite_trajectory_scored(
    first: str,
    second: str,
    *,
    first_correctness: float,
    first_aesthetics: float,
    last_correctness: float,
    last_aesthetics: float,
    first_accepted: bool = False,
) -> str:
    """A two-pass reflect rollout whose judges report explicit facet scores."""
    return "\n".join(
        (
            _generate(first, "/tmp/image_00.png"),
            _judge(
                "/tmp/image_00.png",
                correctness=first_correctness,
                aesthetics=first_aesthetics,
                accepted=first_accepted,
            ),
            _generate(second, "/tmp/image_01.png"),
            _judge(
                "/tmp/image_01.png",
                correctness=last_correctness,
                aesthetics=last_aesthetics,
                accepted=True,
            ),
            "Reflection: The rewrite resolved the findings. Done.",
        )
    )


def test_improve_scores_the_judge_lift_not_the_text_change():
    """This dim scores the outcome, not the wording — the reverse of its predecessor.

    The replaced measure scored text distance, so it read the live CN-poster chain as
    "append-only" and paid ``0.097`` for it. The judge scores for that same chain went
    ``correctness 0.00 -> 0.56`` and ``aesthetics 0.44 -> 0.80``, so the appends were
    working and the rollout was still climbing when the pass cap stopped it. Scoring the
    lift pays for that, and pays nothing for a text change the judge did not reward.
    """
    base = "A vertical cafe poster with a ceramic coffee cup on a rustic table."
    appended = f"{base} The text is rendered in a clear, legible sans-serif font."
    ground_truth = _ground_truth(expected=2)

    lifted = compute_score(
        solution_str=_rewrite_trajectory_scored(
            base,
            appended,
            first_correctness=0.0,
            first_aesthetics=0.44,
            last_correctness=0.56,
            last_aesthetics=0.80,
        ),
        ground_truth=ground_truth,
    )
    stalled = compute_score(
        solution_str=_rewrite_trajectory_scored(
            base,
            appended,
            first_correctness=0.30,
            first_aesthetics=0.50,
            last_correctness=0.30,
            last_aesthetics=0.50,
        ),
        ground_truth=ground_truth,
    )

    assert lifted["num_prompt_rewrites"] == 1
    # mean(0.56 - 0.00, 0.80 - 0.44) = 0.46
    assert lifted["judge_delta"] == pytest.approx(0.46)
    assert lifted["reward_improve"] == pytest.approx(0.46)
    # The same text change, judged as leaving the image where it was, earns nothing.
    assert stalled["judge_delta"] == pytest.approx(0.0)
    assert stalled["reward_improve"] == pytest.approx(0.0)


def test_improve_reports_a_regression_as_negative_without_paying():
    """A rewrite that makes the image worse must not bank a positive delta.

    The unclamped value stays visible in the metrics so a regression is diagnosable,
    while the reward term contributes nothing.
    """
    base = "A vertical cafe poster with a ceramic coffee cup on a rustic table."
    worse = f"{base} Oversaturated neon colors, cluttered composition, illegible text."
    ground_truth = _ground_truth(expected=2)

    output = compute_score(
        solution_str=_rewrite_trajectory_scored(
            base,
            worse,
            first_correctness=0.60,
            first_aesthetics=0.70,
            last_correctness=0.10,
            last_aesthetics=0.20,
        ),
        ground_truth=ground_truth,
    )

    # mean(0.10 - 0.60, 0.20 - 0.70) = -0.50
    assert output["judge_delta"] == pytest.approx(-0.50)
    assert output["reward_improve"] == pytest.approx(0.0)


def test_improve_is_zero_for_a_single_generate():
    """No chain means no lift to measure, so the dim stays out of single-pass rows.

    The predecessor scored these rows against the raw request. A judge-lift measure has
    nothing to compare, and the absolute image quality they do produce is already scored
    by ``reward_reflect``, which reads the preferred judge directly.
    """
    output = compute_score(
        solution_str=_reflect_trajectory(correctness=0.9, aesthetics=0.9),
        ground_truth=_ground_truth(expected=1),
    )

    assert output["num_prompt_rewrites"] == 0
    assert output["judge_delta"] == 0.0
    assert output["reward_improve"] == 0.0


def test_improve_ignores_a_chain_whose_first_judge_already_passed():
    """Nothing to recover from: the protocol asked the policy to stop, and it did.

    ``agentic_reward._delta_c_bonus`` gates on a first-pass NO for the same reason, and
    ``rewrite_after_yes`` is what penalises rewriting once the judge is satisfied.
    """
    output = compute_score(
        solution_str=_rewrite_trajectory_scored(
            "A vertical cafe poster with a warm cup illustration.",
            "A vertical cafe poster with a warm cup illustration, refined lighting.",
            first_correctness=0.80,
            first_aesthetics=0.80,
            last_correctness=1.00,
            last_aesthetics=1.00,
            first_accepted=True,
        ),
        ground_truth=_ground_truth(expected=2),
    )

    assert output["num_prompt_rewrites"] == 1
    assert output["judge_delta"] == 0.0
    assert output["reward_improve"] == 0.0


def test_legacy_novelty_weight_aliases_the_improve_dim():
    """Parquet built before the swap baked ``w_novelty``; it must still take effect.

    Without the alias, an operator's ``RPCO_W_NOVELTY=0`` would silently stop working
    and the dim would return to weight 1.0 on a dataset rebuild.
    """
    text = _rewrite_trajectory_scored(
        "A vertical cafe poster with a warm cup illustration.",
        "Risograph poster, flat ink layers, geometric cup on cobalt, condensed caps headline.",
        first_correctness=0.0,
        first_aesthetics=0.40,
        last_correctness=0.50,
        last_aesthetics=0.80,
    )
    ground_truth = _ground_truth(expected=2, w_novelty=0.0)
    output = compute_score(solution_str=text, ground_truth=ground_truth)

    assert output["reward_improve"] > 0.0
    # The dim is still reported, but contributes nothing to the scalar.
    active = {"reflect": 1.0, "format": 1.0, "tool": 1.0, "result": 1.0}
    expected = sum(weight * output[f"reward_{dim}"] for dim, weight in active.items()) / sum(active.values())
    assert output["score"] == pytest.approx(expected)


def test_explicit_improve_weight_beats_the_legacy_alias():
    """An explicit ``w_improve`` must not be overridden by a stale ``w_novelty``."""
    text = _rewrite_trajectory_scored(
        "A vertical cafe poster with a warm cup illustration.",
        "Risograph poster, flat ink layers, geometric cup on cobalt, condensed caps headline.",
        first_correctness=0.0,
        first_aesthetics=0.40,
        last_correctness=0.50,
        last_aesthetics=0.80,
    )
    ground_truth = _ground_truth(expected=2, w_improve=0.0, w_novelty=1.0)
    output = compute_score(solution_str=text, ground_truth=ground_truth)

    assert output["reward_improve"] > 0.0
    active = {"reflect": 1.0, "format": 1.0, "tool": 1.0, "result": 1.0}
    expected = sum(weight * output[f"reward_{dim}"] for dim, weight in active.items()) / sum(active.values())
    assert output["score"] == pytest.approx(expected)


def test_reference_budget_never_caps_reflect_iteration():
    """Whatever the budget says — absent, fabricated, or derived — reflect may iterate.

    ``UniCoT-Breakdown-3K``'s ``No breakdown needed.`` sentinel has no reference
    trajectory, so its budget is ``None``; the derived rows carry 1/2/3. None of these
    may gate ``reward_result``, or the 3906 reflect rows whose reference generated once
    would forbid the rewrite loop the system prompt asks for.
    """
    text = _rewrite_trajectory(
        "A vertical cafe poster with a warm cup illustration.",
        "Risograph poster, flat ink layers, geometric cup on cobalt, condensed caps headline.",
    )
    budgets = {
        "absent": {"task_type": "reflect"},
        "null": {"task_type": "reflect", "expected_num_images": None},
        "derived one-shot": {"task_type": "reflect", "expected_num_images": 1},
        "derived multi": {"task_type": "reflect", "expected_num_images": 3},
    }
    for label, ground_truth in budgets.items():
        output = compute_score(solution_str=text, ground_truth=ground_truth)
        assert output["n_successful_generates"] == 2, label
        assert output["reward_result"] == 1.0, label

    # The field stays numeric for logging; 0 signals "no reference budget".
    assert compute_score(solution_str=text, ground_truth=budgets["null"])["expected_num_images"] == 0
    assert compute_score(solution_str=text, ground_truth=budgets["derived one-shot"])["expected_num_images"] == 1
