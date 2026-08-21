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
"""CPU tests for the frozen agentic function tools (generate_image / judge_image).

The tool module is loaded from the worker-resolved path so the same test runs
against whichever checkout the worker binds (verl_omni/tools/image_gen.py).
"""

import base64
import importlib.util
import io

import pytest
from PIL import Image

from verl_omni.agent_loop import agentic_trajectory_context as _traj_ctx
from verl_omni.agent_loop.agentic_metrics_manager import AgenticAgentLoopWorker
from verl_omni.agent_loop.agentic_trajectory_context import (
    clear_latest_tool_image_for_active_rollout,
    count_live_generate_artifacts_for_active_rollout,
    register_tool_artifact,
    reset_active_trajectory_relpath,
    reset_active_user_prompt,
    set_active_trajectory_relpath,
    set_active_user_prompt,
)

_TOOL_PATH = AgenticAgentLoopWorker._AGENTIC_FUNCTION_TOOLS
if not _TOOL_PATH.is_file():
    raise FileNotFoundError(f"agentic function tools not found at {_TOOL_PATH}")


def _load_tools_module():
    spec = importlib.util.spec_from_file_location("agentic_tools_under_test", _TOOL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load tool module from {_TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # registers generate_image / judge_image
    return module


tools = _load_tools_module()


def _clear_all_tool_artifacts() -> None:
    """Test-only: wipe the process-global generate_image registry between cases."""
    with _traj_ctx._artifact_registry_lock:
        _traj_ctx._artifact_registry.clear()
        _traj_ctx._artifact_by_id.clear()
        _traj_ctx._latest_image_by_rollout.clear()
    _traj_ctx.set_latest_tool_image_path(None)


@pytest.fixture(autouse=True)
def _isolate_tool_artifact_registry():
    """Prior tests share ``step_*/sample_*`` keys; wipe leftover registry rows."""
    _clear_all_tool_artifacts()
    yield
    _clear_all_tool_artifacts()


def _png_b64(color=(12, 34, 56)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_tool_schemas_declare_both_functions():
    gen = tools.DIFFUSION_TOOL_SCHEMA["function"]
    judge = tools.JUDGE_TOOL_SCHEMA["function"]
    assert gen["name"] == "generate_image"
    assert gen["parameters"]["required"] == ["prompt"]
    assert judge["name"] == "judge_image"
    assert judge["parameters"]["required"] == ["user_request", "image_prompt"]


def test_generate_image_stub_without_service(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTIC_VLLM_OMNI_URL", raising=False)
    monkeypatch.delenv("AGENTIC_QWEN_IMAGE_URL", raising=False)
    monkeypatch.delenv("AGENTIC_DIFFUSION_TOOL_URL", raising=False)
    monkeypatch.setenv("AGENTIC_E2E_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENTIC_E2E_RUN_NAME", "cpu_test")

    response, reward, metrics = tools.generate_image("a cafe poster")

    assert "[stub diffusion result]" in response.text
    assert reward == 0.0
    assert metrics["tool_stubbed"] is True


def test_generate_image_blocked_after_yes(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_E2E_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENTIC_E2E_RUN_NAME", "cpu_test")
    yes_token = tools.set_good_enough_yes_reached(True)
    try:
        response, reward, metrics = tools.generate_image("a cafe poster")
    finally:
        tools.set_good_enough_yes_reached(False)
        del yes_token
    assert "generate_image blocked" in response.text
    assert "good_enough=YES" in response.text
    assert "agentic_block_generate_after_yes=1" in response.text
    assert "agentic_tool ok=0" in response.text
    assert metrics["blocked_after_yes"] == 1


def test_generate_image_blocked_after_max_passes(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTIC_E2E_ROOT", str(tmp_path))
    monkeypatch.setenv("AGENTIC_E2E_RUN_NAME", "cpu_test")
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "1")
    png = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(png)

    tokens = set_active_trajectory_relpath("step_000001/sample_0.00")
    try:
        register_tool_artifact(prompt="a cafe poster", paths=[str(png)], backend="qwen_image")
        response, reward, metrics = tools.generate_image("a cafe poster")
    finally:
        reset_active_trajectory_relpath(tokens)
    assert "generate_image blocked" in response.text
    assert "already completed 1/1 successful generate_image passes" in response.text
    assert "agentic_block_generate_after_max_passes=1" in response.text
    assert "agentic_tool ok=0" in response.text
    assert metrics["generate_passes"] == 1


def test_judge_image_stub_without_vllm(monkeypatch):
    monkeypatch.delenv("AGENTIC_VLLM_URL", raising=False)
    response, reward, metrics = tools.judge_image("same as user message", "last")
    assert "AGENTIC_VLLM_URL unset" in response.text
    assert metrics["judge_stub"] is True


def test_expand_judge_user_request_placeholders():
    bound = "A vertical artistic cafe poster."
    token = set_active_user_prompt(bound)
    try:
        assert tools._expand_judge_user_request("same as user message") == bound
        assert tools._expand_judge_user_request("last") == bound
        assert tools._expand_judge_user_request("some other task text") == "some other task text"
    finally:
        reset_active_user_prompt(token)
    assert tools._expand_judge_user_request("raw without binding") == "raw without binding"


def test_expand_judge_image_prompt_to_latest_live_prompt(tmp_path):
    png = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (4, 5, 6)).save(png)
    tokens = set_active_trajectory_relpath("step_000001/sample_0.00")
    try:
        register_tool_artifact(prompt="latest diffusion prompt", paths=[str(png)], backend="qwen_image")
        assert tools._expand_judge_image_prompt("last") == "latest diffusion prompt"
        assert tools._expand_judge_image_prompt("") == "latest diffusion prompt"
    finally:
        reset_active_trajectory_relpath(tokens)


def test_clear_latest_image_prunes_only_active_rollout_registry(tmp_path):
    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(png_a)
    Image.new("RGB", (1, 1), (4, 5, 6)).save(png_b)

    tokens_a = set_active_trajectory_relpath("step_000001/sample_0.00")
    register_tool_artifact(prompt="prompt a", paths=[str(png_a)], backend="qwen_image")
    assert count_live_generate_artifacts_for_active_rollout() == 1
    reset_active_trajectory_relpath(tokens_a)

    tokens_b = set_active_trajectory_relpath("step_000001/sample_1.00")
    register_tool_artifact(prompt="prompt b", paths=[str(png_b)], backend="qwen_image")
    assert count_live_generate_artifacts_for_active_rollout() == 1
    reset_active_trajectory_relpath(tokens_b)

    tokens_a = set_active_trajectory_relpath("step_000001/sample_0.00")
    try:
        clear_latest_tool_image_for_active_rollout()
        assert count_live_generate_artifacts_for_active_rollout() == 0
    finally:
        reset_active_trajectory_relpath(tokens_a)

    tokens_b = set_active_trajectory_relpath("step_000001/sample_1.00")
    try:
        assert count_live_generate_artifacts_for_active_rollout() == 1
        clear_latest_tool_image_for_active_rollout()
        assert count_live_generate_artifacts_for_active_rollout() == 0
    finally:
        reset_active_trajectory_relpath(tokens_b)


def test_decode_images_base64():
    images = tools._decode_images({"images_base64": [_png_b64()]})
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert tools._decode_images({}) == []
    assert tools._decode_images({"images_base64": ["not-base64!!"]}) == []
