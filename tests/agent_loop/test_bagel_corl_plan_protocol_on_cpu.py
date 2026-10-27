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
"""Plan rows: the environment walks the policy through its subtasks before the judge."""

from __future__ import annotations

import asyncio

import pytest

from verl_omni.agent_loop.bagel_corl_lib import BagelGenerateImageTool, parse_und_tool_call, run_serial_episode

GENERATE = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "subtask"}}\n</tool_call>'
JUDGE = (
    '<tool_call>\n{"name": "judge_image", "arguments": {"user_request": "same", "image_prompt": "last"}}\n</tool_call>'
)
VERDICT = "agentic_judge ok=1 correctness=0.8000 aesthetics=0.8000 good_enough=YES rubber_stamp=False\nfindings: fine"


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) + 1000 for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(chr(int(t) - 1000) for t in token_ids)


async def _gen(**kwargs):
    return [{"valid": True, "image_path": "/tmp/plan.png"} for _ in kwargs["seeds"]]


def _episode(script, verdict=VERDICT, **kwargs):
    turns = iter([*script, "Done."])

    async def und_decode(**_):
        return {"token_ids": [7], "text": next(turns)}

    async def _run():
        return await run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            prompt_ids=[1],
            und_decode=und_decode,
            generate_tool=BagelGenerateImageTool(gen_samples_per_call=1, max_generate_passes=3, generate_fn=_gen),
            build_judge_text_fn=lambda samples: verdict,
            tokenizer=_FakeTokenizer(),
            max_und_turns=8,
            **kwargs,
        )

    return asyncio.run(_run())


def _observations(episode):
    return [r["text"] for r in episode.turn_trace if r["record"] == "env_obs"]


def test_plan_episode_is_cued_and_an_early_judge_is_declined_once():
    episode = _episode([GENERATE, JUDGE, GENERATE, GENERATE, JUDGE], task_type="plan", expected_num_images=3)
    observations = _observations(episode)
    assert episode.num_gen_calls == 3
    assert "Plan step 1/3 rendered" in observations[0]
    assert observations[1].startswith("agentic_judge ok=0")
    assert "subtask 2 first" in observations[1]
    assert "Plan step 2/3 rendered" in observations[2]
    assert not any("Plan step 3/3" in text for text in observations)
    assert sum(text.startswith("agentic_judge ok=0") for text in observations) == 1
    assert any(r["record"] == "reflection" for r in episode.turn_trace)


def test_a_second_early_judge_on_a_plan_row_is_judged():
    episode = _episode([GENERATE, JUDGE, JUDGE], task_type="plan", expected_num_images=3)
    observations = _observations(episode)
    assert episode.num_gen_calls == 1
    assert sum(text.startswith("agentic_judge ok=0") for text in observations) == 1
    assert observations[-1].startswith("VL judge on the last generated image")


PLAN = "Plan:\n1. a red fox in snowy woods at dawn\n2. the same fox leaping over a frozen stream\n\n"
CALL = '{\n  "name": "generate_image",\n  "arguments": {\n    "prompt": "a red fox in snowy woods"\n  }\n}'


@pytest.mark.parametrize("wrapped", [CALL, f"《《《《《{CALL}》》》》》", f"{CALL}\n"])
def test_a_call_that_ends_the_turn_after_a_plan_is_parsed(wrapped):
    call = parse_und_tool_call(f"{PLAN}{wrapped}<|im_end|>")
    assert call["name"] == "generate_image"
    assert call["arguments"]["prompt"] == "a red fox in snowy woods"


def test_a_turn_ending_in_two_calls_runs_the_first():
    judge = '{"name": "judge_image", "arguments": {"user_request": "same", "image_prompt": "last"}}'
    assert parse_und_tool_call(f"{PLAN}{CALL}\n\n{judge}<|im_end|>")["name"] == "generate_image"


@pytest.mark.parametrize(
    "text",
    [
        f"{PLAN}For example {CALL} would draw the fox, but I will describe it first.",
        f'{PLAN}{{"name": "generate_image", "parameters": {{"prompt": {{"type": "string"}}}}}}',
    ],
)
def test_a_quoted_example_or_schema_echo_after_a_plan_is_not_a_call(text):
    assert parse_und_tool_call(text) is None


NO_VERDICT = VERDICT.replace("good_enough=YES", "good_enough=NO").replace("0.8000", "0.4000")


@pytest.mark.parametrize("terminal", ["Done", "Done.", "Done<|im_end|>", "Done.<|im_end|>"])
def test_a_done_without_the_period_still_ends_the_episode(terminal):
    """The checkpoint writes ``Done`` as often as ``Done.``; a missing period is not a continuation."""
    from verl_omni.agent_loop.bagel_corl_lib import und_turn_kind

    assert und_turn_kind(terminal) == "done"


def test_a_silent_turn_after_a_no_verdict_is_not_declined():
    """A bare ``<|im_end|>`` is not a Done the policy wrote; re-sampling it repeated prompt fragments."""
    episode = _episode([GENERATE, JUDGE, "<|im_end|>"], verdict=NO_VERDICT, task_type="reflect", expected_num_images=1)
    assert not any(text.startswith("agentic_stop ok=0") for text in _observations(episode))
    assert episode.turn_trace[-1]["text"] == "<|im_end|>"


@pytest.mark.parametrize("task_type", ["reflect", "plan"])
def test_done_after_a_no_verdict_is_declined_until_the_last_generate(task_type):
    script = [GENERATE, JUDGE, "Done.", GENERATE, JUDGE, "Done.", GENERATE, JUDGE]
    episode = _episode(script, verdict=NO_VERDICT, task_type=task_type, expected_num_images=1)
    declined = [text for text in _observations(episode) if text.startswith("agentic_stop ok=0")]
    assert episode.num_gen_calls == 3
    assert len(declined) == 2
    assert "2 of 3 generate_image calls remain" in declined[0]
    assert "rewrite the diffusion prompt" in declined[0]
    assert episode.turn_trace[-1]["text"] == "Done."


def test_done_before_judging_the_new_image_asks_for_the_judge():
    episode = _episode([GENERATE, "Done.", JUDGE], task_type="reflect", expected_num_images=1)
    declined = [text for text in _observations(episode) if text.startswith("agentic_stop ok=0")]
    assert len(declined) == 1
    assert "call judge_image on the new image" in declined[0]


def test_reflect_rows_keep_the_judge_after_every_image():
    episode = _episode([GENERATE, JUDGE], task_type="reflect", expected_num_images=3)
    observations = _observations(episode)
    assert not any("Plan step" in text for text in observations)
    assert observations[-1].startswith("VL judge on the last generated image")
