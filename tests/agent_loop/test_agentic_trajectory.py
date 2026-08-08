# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import torch
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.tools.function_tool import get_function_tool

from verl_omni.agent_loop.agentic_trajectory import (
    AgenticMetadata,
    AgenticTrajectory,
    AgenticTurn,
    ToolCall,
    ToolOutput,
    agentic_trajectory_from_dict,
    agentic_trajectory_to_dict,
)
from verl_omni.agent_loop.diffusion_tool import DIFFUSION_TOOL_SCHEMA, generate_image


def _make_turn(idx, tokens, logprobs, text, decision, tc=None, to=None):
    return AgenticTurn(idx, tokens, logprobs, text, tc, to, decision)


class TestTrajectory:
    def test_round_trip_full(self):
        t0 = _make_turn(
            0,
            [1, 2, 3],
            [0.1, 0.2, 0.3],
            "text0",
            "continue",
            ToolCall("t2i", {"prompt": "p0"}),
            ToolOutput("image", torch.zeros(3, 64, 64), is_stub=True),
        )
        t1 = _make_turn(1, [4, 5], [0.4, 0.5], "text1", "stop")
        trajectory = AgenticTrajectory(
            "test",
            [t0, t1],
            0.8,
            AgenticMetadata(2, True, "agent_stop", tool_stubbed=True),
        )

        restored = agentic_trajectory_from_dict(agentic_trajectory_to_dict(trajectory))

        assert restored.prompt == "test"
        assert restored.reward_score == 0.8
        assert restored.metadata.num_turns == 2
        assert restored.metadata.tool_stubbed is True
        assert restored.turns[0].tool_call.params["prompt"] == "p0"
        assert restored.turns[0].tool_output.is_stub is True

    def test_round_trip_minimal(self):
        trajectory = AgenticTrajectory("minimal", [_make_turn(0, [1], [0.1], "text", "stop")])
        restored = agentic_trajectory_from_dict(agentic_trajectory_to_dict(trajectory))
        assert restored.prompt == "minimal"
        assert len(restored.turns) == 1

    def test_prompt_rewriting_captured(self):
        t0 = _make_turn(
            0,
            [1],
            [0.1],
            "t0",
            "continue",
            ToolCall("t2i", {"prompt": "original"}),
            ToolOutput("image", torch.zeros(3, 8, 8)),
        )
        t1 = _make_turn(
            1,
            [2],
            [0.2],
            "t1",
            "stop",
            ToolCall("t2i", {"prompt": "rewritten"}),
        )
        trajectory = AgenticTrajectory("p", [t0, t1])
        assert trajectory.turns[0].tool_call.params["prompt"] == "original"
        assert trajectory.turns[1].tool_call.params["prompt"] == "rewritten"


class TestTrainMaskContract:
    def test_all_agent_turns_train_and_tool_observations_do_not(self):
        # Stock ToolAgentLoop contract:
        # assistant turn 0 | tool observation | assistant turn 1
        response_mask = [1, 1, 1] + [0, 0] + [1, 1]
        assert response_mask == [1, 1, 1, 0, 0, 1, 1]


class TestStockToolAgentWiring:
    def test_recipe_uses_stock_tool_agent(self):
        root = Path(__file__).resolve().parents[2]
        recipe = (root / "tests/special_e2e/run_agentic_grpo_lance.sh").read_text()
        assert "default_agent_loop=tool_agent" in recipe
        assert "agent_loop_config_path=null" in recipe
        assert "multi_turn.enable=true" in recipe
        assert "function_tool_path=verl_omni/agent_loop/diffusion_tool.py" in recipe

    def test_tool_agent_is_upstream_class(self):
        assert ToolAgentLoop.__module__ == "verl.experimental.agent_loop.tool_agent_loop"

    def test_diffusion_function_tool_registered(self):
        tool = get_function_tool("generate_image")
        assert tool.fn is generate_image
        assert tool.tool_schema.function.name == DIFFUSION_TOOL_SCHEMA["function"]["name"]

    def test_diffusion_tool_stub_without_endpoint(self, monkeypatch, tmp_path):
        monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
        monkeypatch.delenv("AGENTIC_LANCE_SERVER_URL", raising=False)
        monkeypatch.setenv("AGENTIC_DIFFUSION_IMAGE_DIR", str(tmp_path / "rollout_images"))
        response, reward, metrics = generate_image("a blue hat")
        assert response.image is None
        assert "stub diffusion result" in response.text
        assert reward == 0.0
        assert metrics["tool_stubbed"] is True
        assert metrics["diffusion_backend"] == "stub"
        assert metrics["num_images"] == 0
        assert metrics["image_paths"]
        assert Path(metrics["image_paths"][0]).name == "STUB_NO_IMAGE.txt"
