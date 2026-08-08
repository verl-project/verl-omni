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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_reward_module():
    path = Path(__file__).parents[3] / "verl_omni/utils/reward_score/omnivideo_qi.py"
    spec = importlib.util.spec_from_file_location("omnivideo_qi_reward", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


omnivideo_qi = _load_reward_module()

VALID_RESPONSE = """<time>1.0-2.5</time><caption>A person opens the door.</caption>
<time>4.0-5.5</time><caption>The person enters the room.</caption>
<thinking>Let me think. The two events show an arrival.</thinking>
<answer>B</answer>"""


def test_parse_response_accepts_strict_ordered_groundings():
    parsed = omnivideo_qi.parse_response(VALID_RESPONSE)

    assert parsed.format_valid
    assert [(pair.start, pair.end) for pair in parsed.pairs] == [(1.0, 2.5), (4.0, 5.5)]
    assert parsed.answer == "B"


@pytest.mark.parametrize(
    "response",
    [
        VALID_RESPONSE.replace("4.0-5.5", "2.0-5.5"),
        VALID_RESPONSE.replace("1.0-2.5", "1-2.5"),
        VALID_RESPONSE.replace(
            "<thinking>Let me think. The two events show an arrival.</thinking>", "<thinking></thinking>"
        ),
        f"prefix {VALID_RESPONSE}",
    ],
)
def test_parse_response_rejects_non_strict_formats(response):
    assert not omnivideo_qi.parse_response(response).format_valid


@pytest.mark.asyncio
async def test_choice_answer_still_receives_outcome_reward_without_valid_grounding(monkeypatch):
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_URL", raising=False)

    result = await omnivideo_qi.compute_score(
        solution_str="<answer>C</answer>",
        ground_truth="C. kitchen",
        extra_info={"is_multiple_choice": True},
    )

    assert result["score"] == 1.0
    assert result["format"] == 0.0
    assert result["answer"] == 1.0
    assert result["intent"] == 0.0


@pytest.mark.asyncio
async def test_compute_score_implements_qi_equation(monkeypatch):
    config = omnivideo_qi.JudgeConfig(url="http://judge/v1/chat/completions", model="judge")
    monkeypatch.setattr(omnivideo_qi, "_judge_config", lambda *args: config)
    monkeypatch.setattr(omnivideo_qi, "_load_segment_frames", lambda *args: ["data:image/png;base64,frame"])

    async def fake_chat_complete(session, config, content, system_prompt):
        del session, config, content
        if system_prompt == omnivideo_qi.CONSISTENCY_SYSTEM_PROMPT:
            return 0.8, "consistent"
        if system_prompt == omnivideo_qi.COMPLETENESS_SYSTEM_PROMPT:
            return 0.6, "complete"
        raise AssertionError("multiple-choice answer should not call the answer judge")

    monkeypatch.setattr(omnivideo_qi, "_chat_complete", fake_chat_complete)

    result = await omnivideo_qi.compute_score(
        solution_str=VALID_RESPONSE,
        ground_truth="B. entering the room",
        extra_info={
            "is_multiple_choice": True,
            "question": "What happens?",
            "video_path": "/data/example.mp4",
        },
    )

    assert result["format"] == 1.0
    assert result["answer"] == 1.0
    assert result["consistency"] == pytest.approx(0.8)
    assert result["completeness"] == pytest.approx(0.6)
    assert result["intent"] == pytest.approx(0.7)
    assert result["score"] == pytest.approx(2.7)


def test_judge_config_normalizes_openai_base_url(monkeypatch):
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_URL", "http://judge:8000/v1")
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_MODEL", "local-judge")

    config = omnivideo_qi._judge_config(reward_model_tokenizer=SimpleNamespace(name_or_path="ignored"))

    assert config.url == "http://judge:8000/v1/chat/completions"
    assert config.model == "local-judge"


def test_judge_config_uses_resource_efficient_default(monkeypatch):
    monkeypatch.setenv("OMNIVIDEO_QI_JUDGE_URL", "http://judge:8000/v1")
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_MODEL", raising=False)

    config = omnivideo_qi._judge_config()

    assert config.model == "Qwen/Qwen3-VL-30B-A3B-Instruct"


def test_judge_config_uses_colocated_reward_router(monkeypatch):
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_URL", raising=False)
    monkeypatch.delenv("OMNIVIDEO_QI_JUDGE_MODEL", raising=False)

    config = omnivideo_qi._judge_config(
        reward_router_address="reward-router:9000",
        reward_model_tokenizer=SimpleNamespace(name_or_path="/models/qwen3-vl-30b"),
    )

    assert config.url == "http://reward-router:9000/v1/chat/completions"
    assert config.model == "/models/qwen3-vl-30b"
