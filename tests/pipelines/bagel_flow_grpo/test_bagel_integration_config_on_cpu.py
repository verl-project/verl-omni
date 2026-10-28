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
"""BAGEL-specific settings extend component APIs without replacing them."""

import pytest
from omegaconf import OmegaConf

from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.config import bagel_scorer_knobs_from_config
from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.prompts import (
    BAGEL_PLAN_SYSTEM_PROMPT,
    prepare_bagel_prompt,
)


def test_bagel_scorer_overrides_survive_the_shared_component_boundary():
    config = OmegaConf.create(
        {
            "agentic_image_gen": {
                "score_backend": "pickscore",
                "corl_reward_version": "v2",
                "unified_good_enough_threshold": 0.71,
                "pickscore_good_enough_threshold": 0.91,
                "vllm_url": "http://judge",
                "good_enough_threshold": 0.83,
            }
        }
    )
    knobs = bagel_scorer_knobs_from_config(config)
    assert knobs["score_backend"] == "pickscore"
    assert knobs["corl_reward_version"] == "v2"
    assert knobs["unified_good_enough_threshold"] == 0.71
    assert knobs["pickscore_good_enough_threshold"] == 0.91
    assert knobs["vllm_url"] == "http://judge"
    assert knobs["good_enough_threshold"] == 0.83


def test_bagel_scorer_defaults_and_missing_config():
    knobs = bagel_scorer_knobs_from_config(OmegaConf.create({"agentic_image_gen": {}}))
    assert knobs["score_backend"] == "unified_reward"
    assert knobs["corl_reward_version"] == "v1"
    assert knobs["unified_good_enough_threshold"] == 0.62
    assert knobs["pickscore_good_enough_threshold"] == 1.0
    with pytest.raises(ValueError):
        bagel_scorer_knobs_from_config(None)


def test_known_unicot_plans_keep_the_verified_bagel_protocol_without_changing_the_builder():
    from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl as builder

    messages = [{"role": "system", "content": builder.PLAN_SYSTEM_PROMPT}, {"role": "user", "content": "draw foxes"}]
    result = prepare_bagel_prompt(
        messages, task_type="plan", extra_info={"unicot_source": "Fr0zencr4nE/UniCoT-Breakdown-3K"}
    )
    assert result[0]["content"] == BAGEL_PLAN_SYSTEM_PROMPT
    assert "Start your reply like this" in result[0]["content"]
    assert "good_enough=NO" in result[0]["content"]
    assert messages[0]["content"] == builder.PLAN_SYSTEM_PROMPT
    assert result[1] == messages[1]


@pytest.mark.parametrize("task,info", [("plan", {}), ("reflect", {"unicot_source": "Fr0zencr4nE/UniCoT-Breakdown-3K"})])
def test_custom_tasks_and_poster_prompts_are_preserved(task, info):
    messages = [{"role": "system", "content": "custom protocol"}, {"role": "user", "content": "custom request"}]
    assert prepare_bagel_prompt(messages, task_type=task, extra_info=info) is messages


def test_known_plan_retokenizes_instead_of_using_stale_dataset_ids(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen import agent_loop

    class PromptPrepared(Exception):
        pass

    class StopBeforeRollout:
        @property
        def actor_rollout_ref(self):
            raise PromptPrepared

    rendered = []

    def tokenize(messages, **kwargs):
        rendered.append(messages)
        return [7, 8]

    monkeypatch.setattr(agent_loop, "_inject_tools_schema_enabled", lambda: False)
    loop = SimpleNamespace(tokenizer=SimpleNamespace(apply_chat_template=tokenize), config=StopBeforeRollout())
    with pytest.raises(PromptPrepared):
        asyncio.run(
            agent_loop.BagelMultiturnAgentLoop.run(
                loop,
                {},
                raw_prompt=[{"role": "user", "content": "draw foxes"}],
                prompt_ids=[99],
                extra_info={"task_type": "plan", "unicot_source": "Fr0zencr4nE/UniCoT-Breakdown-3K"},
            )
        )
    assert rendered[0][0]["content"] == BAGEL_PLAN_SYSTEM_PROMPT
