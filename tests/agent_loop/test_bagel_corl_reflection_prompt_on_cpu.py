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

"""CPU tests for the Bagel Co-RL system-turn rewrite."""

import pytest

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.episode import (
    REFLECTION_SYSTEM_PROMPT,
    resolve_task_type,
    with_reflection_system_prompt,
)

# Mirrors the dataset builder (build_unicot_agentic_rl.PLAN_SYSTEM_PROMPT).
_PLAN_SYSTEM_PROMPT = (
    "You are a visual creation agent with two tools:\n"
    "1. Write a short numbered plan of at most three complete subtask image prompts.\n"
)


def _plan_messages() -> list[dict]:
    return [
        {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": "Draw a cat riding a bicycle through a night market."},
    ]


def test_plan_row_keeps_the_dataset_system_turn():
    messages = _plan_messages()
    rendered = with_reflection_system_prompt(messages, task_type="plan")
    assert rendered is messages
    assert rendered[0]["content"] == _PLAN_SYSTEM_PROMPT
    assert "numbered plan" in rendered[0]["content"]


@pytest.mark.parametrize("label", ["plan", "PLAN", "Plan", " plan ", "\tplan\n"])
def test_plan_label_is_matched_case_and_whitespace_insensitively(label):
    messages = _plan_messages()
    assert with_reflection_system_prompt(messages, task_type=label) is messages


@pytest.mark.parametrize("label", ["reflect", "REFLECT", None, "", "breakdown"])
def test_non_plan_rows_get_the_reflection_protocol(label):
    messages = _plan_messages()
    rendered = with_reflection_system_prompt(messages, task_type=label)
    assert rendered[0]["content"] == REFLECTION_SYSTEM_PROMPT
    # The caller's messages are copied, not mutated in place.
    assert messages[0]["content"] == _PLAN_SYSTEM_PROMPT


def test_system_turn_is_prepended_when_the_row_has_none():
    messages = [{"role": "user", "content": "hi"}]
    rendered = with_reflection_system_prompt(messages, task_type="reflect")
    assert rendered[0] == {"role": "system", "content": REFLECTION_SYSTEM_PROMPT}
    assert rendered[1] == {"role": "user", "content": "hi"}


def test_plan_row_without_a_system_turn_is_untouched():
    messages = [{"role": "user", "content": "hi"}]
    assert with_reflection_system_prompt(messages, task_type="plan") is messages


@pytest.mark.parametrize("prompt", ["plain text", None, [], ()])
def test_non_message_prompts_are_returned_unchanged(prompt):
    assert with_reflection_system_prompt(prompt, task_type="reflect") == prompt
    assert with_reflection_system_prompt(prompt, task_type="plan") == prompt


def test_resolve_task_type_reads_extra_info():
    assert resolve_task_type(extra_info={"task_type": "plan"}) == "plan"


def test_resolve_task_type_falls_back_to_ground_truth():
    assert resolve_task_type(ground_truth={"task_type": "reflect"}) == "reflect"


def test_resolve_task_type_prefers_extra_info():
    resolved = resolve_task_type(extra_info={"task_type": "plan"}, ground_truth={"task_type": "reflect"})
    assert resolved == "plan"


def test_resolve_task_type_skips_empty_labels():
    resolved = resolve_task_type(extra_info={"task_type": ""}, ground_truth={"task_type": "reflect"})
    assert resolved == "reflect"


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"extra_info": None, "ground_truth": None},
        {"extra_info": "plan", "ground_truth": ["plan"]},
        {"extra_info": {}, "ground_truth": {}},
    ],
)
def test_resolve_task_type_returns_none_when_absent(kwargs):
    assert resolve_task_type(**kwargs) is None


def test_plan_label_resolution_keeps_the_plan_prompt():
    """The two helpers are wired together at the Bagel Co-RL call site."""
    messages = _plan_messages()
    task_type = resolve_task_type(extra_info={"task_type": "plan"})
    assert with_reflection_system_prompt(messages, task_type=task_type) is messages
