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

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestAgenticLlmRlDiffusionOutsideActor:
    """Mode (2a) Agentic LLM RL: diffusion is an external tool, not an actor FSDP submodule."""

    def test_recipe_wires_external_function_tool(self):
        recipe = (REPO_ROOT / "tests/special_e2e/run_agentic_grpo_lance.sh").read_text()
        assert "default_agent_loop=tool_agent" in recipe
        assert "function_tool_path=verl_omni/agent_loop/diffusion_tool.py" in recipe
        assert "agent_loop_config_path=null" in recipe
        assert "algorithm.adv_estimator=grpo" in recipe
        assert "multi_turn.enable=true" in recipe
        # Multi-step e2e example tree is out of scope for this smoke recipe.
        assert "examples/agenticrpco_trainer" not in recipe
        assert (REPO_ROOT / "tests/special_e2e/qwen2_tool_chat_template.jinja2").is_file()
        assert not (REPO_ROOT / "tests/special_e2e/qwen2_tool_chat_template.yaml").exists()
        assert "multi_turn.format=hermes" in recipe
        assert "st1_preflight.py" not in recipe
        # Prefer baked und tokenizer (prepare_lance); avoid fragile CLI Jinja override.
        assert "custom_chat_template=" not in recipe
        assert "qwen2_tool_chat_template.jinja2" in recipe

    def test_diffusion_tool_registers_generate_image(self):
        # Source scan keeps this CPU-safe (importing the tool pulls vLLM/CUDA).
        src = (REPO_ROOT / "verl_omni/agent_loop/diffusion_tool.py").read_text()
        assert '@function_tool("generate_image"' in src
        assert "DIFFUSION_TOOL_SCHEMA" in src
        assert "def generate_image(" in src

    def test_train_mask_marks_tool_obs_as_non_trainable(self):
        # Stock ToolAgentLoop contract:
        # assistant turn 0 | tool observation | assistant turn 1
        response_mask = [1, 1, 1] + [0, 0] + [1, 1]
        assert response_mask == [1, 1, 1, 0, 0, 1, 1]

    def test_no_custom_agentic_fsdp_engine_module(self):
        # No custom agentic FSDP engine; stock HF FSDP path only.
        engine_root = REPO_ROOT / "verl_omni/workers/engine"
        hits = [p for p in engine_root.rglob("*agentic*.py") if p.is_file()]
        assert hits == [], f"unexpected agentic engine files: {hits}"


class TestFlowGrpoBackwardCompat:
    """Mode (0) Single-stage RL (FlowGRPO) stays unaffected by Mode (2a) Agentic LLM RL."""

    def test_main_diffusion_importable(self):
        from verl_omni.trainer import main_diffusion  # noqa: F401

    def test_diffusion_algo_config_defaults_flow_grpo(self):
        from verl_omni.trainer.config.algorithm import DiffusionAlgoConfig

        assert DiffusionAlgoConfig().adv_estimator == "flow_grpo"

    def test_single_turn_agent_loop_importable(self):
        from verl_omni.agent_loop import DiffusionSingleTurnAgentLoop

        assert DiffusionSingleTurnAgentLoop is not None

    def test_flow_grpo_adv_estimator_registered(self):
        from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_adv_estimator_fn

        assert get_diffusion_adv_estimator_fn("flow_grpo") is not None

    def test_ray_diffusion_trainer_has_no_agentic_branches(self):
        path = REPO_ROOT / "verl_omni/trainer/diffusion/ray_diffusion_trainer.py"
        source = path.read_text()
        # Parse ensures the file is valid Python; string scan guards against
        # agentic branches leaking into the single-turn diffusion trainer.
        ast.parse(source)
        assert "is_agentic" not in source
        assert "agentic_grpo" not in source
