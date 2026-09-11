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
"""CPU tests for the frozen agentic function tools (generate_image / judge_image)."""

from __future__ import annotations

import base64
import io

import pytest
from omegaconf import OmegaConf
from PIL import Image
from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY

import verl_omni.tools.image_gen as image_gen
from verl_omni.tools import trajectory
from verl_omni.tools.trajectory import artifacts


def _bind_tool_cfg(*, e2e_root=None, run_name="cpu_test", **overrides):
    node = {
        "vllm_omni_url": "",
        "qwen_image_url": "",
        "diffusion_tool_url": "",
        "vllm_url": "",
        "block_generate_after_yes": True,
        "block_generate_after_max_passes": True,
        "max_generate_image_passes": 3,
        **overrides,
    }
    if e2e_root is not None:
        node["e2e_root"] = str(e2e_root)
    cfg = OmegaConf.create(
        {
            "trainer": {"experiment_name": run_name},
            "agentic_image_gen": node,
        }
    )
    trajectory.bind_agentic_image_gen(cfg)
    trajectory.bind_run_artifacts(cfg)


def _clear_all_tool_artifacts() -> None:
    """Test-only: wipe the process-global generate_image registry between cases."""
    with artifacts._artifact_registry_lock:
        artifacts._artifact_registry.clear()
        artifacts._artifact_by_id.clear()
        artifacts._latest_image_by_rollout.clear()
    artifacts.set_latest_tool_image_path(None)


@pytest.fixture(autouse=True)
def _isolate_tool_artifact_registry():
    """Prior tests share ``step_*/sample_*`` keys; wipe leftover registry rows."""
    _clear_all_tool_artifacts()
    trajectory.clear_agentic_image_gen()
    trajectory.clear_run_artifacts()
    yield
    _clear_all_tool_artifacts()
    trajectory.clear_agentic_image_gen()
    trajectory.clear_run_artifacts()


def _png_b64(color=(12, 34, 56)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_tool_schemas_declare_both_functions():
    gen = FUNCTION_TOOL_REGISTRY["generate_image"].tool_schema.function
    judge = FUNCTION_TOOL_REGISTRY["judge_image"].tool_schema.function
    assert gen.name == "generate_image"
    assert gen.parameters.required == ["prompt"]
    assert judge.name == "judge_image"
    assert judge.parameters.required == ["user_request", "image_prompt"]


def test_generate_image_stub_without_service(tmp_path):
    _bind_tool_cfg(e2e_root=tmp_path, run_name="cpu_test")
    response, reward, metrics = image_gen.generate_image("a cafe poster")
    assert "[stub diffusion result]" in response.text
    assert reward == 0.0
    assert metrics["tool_stubbed"] is True


def test_generate_image_blocked_after_yes(tmp_path):
    _bind_tool_cfg(e2e_root=tmp_path, run_name="cpu_test")
    yes_token = image_gen.set_good_enough_yes_reached(True)
    try:
        response, reward, metrics = image_gen.generate_image("a cafe poster")
    finally:
        image_gen.set_good_enough_yes_reached(False)
        del yes_token
    assert "generate_image blocked" in response.text
    assert "good_enough=YES" in response.text
    assert "agentic_block_generate_after_yes=1" in response.text
    assert "agentic_tool ok=0" in response.text
    assert metrics["blocked_after_yes"] == 1


def test_generate_image_blocked_after_max_passes(tmp_path):
    _bind_tool_cfg(e2e_root=tmp_path, run_name="cpu_test", max_generate_image_passes=1)
    png = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(png)

    tokens = trajectory.set_active_trajectory_relpath("step_000001/sample_0.00")
    try:
        trajectory.register_tool_artifact(prompt="a cafe poster", paths=[str(png)], backend="qwen_image")
        response, reward, metrics = image_gen.generate_image("a cafe poster")
    finally:
        trajectory.reset_active_trajectory_relpath(tokens)
    assert "generate_image blocked" in response.text
    assert "already completed 1/1 successful generate_image passes" in response.text
    assert "agentic_block_generate_after_max_passes=1" in response.text
    assert "agentic_tool ok=0" in response.text
    assert metrics["generate_passes"] == 1


def test_judge_image_stub_without_vllm():
    _bind_tool_cfg(vllm_url="")
    response, reward, metrics = image_gen.judge_image("same as user message", "last")
    assert "agentic_image_gen.vllm_url unset" in response.text
    assert metrics["judge_stub"] is True


def test_expand_judge_user_request_placeholders():
    bound = "A vertical artistic cafe poster."
    token = trajectory.active_user_prompt.set(bound)
    try:
        assert image_gen._expand_judge_user_request("same as user message") == bound
        assert image_gen._expand_judge_user_request("last") == bound
        assert image_gen._expand_judge_user_request("some other task text") == "some other task text"
    finally:
        trajectory.active_user_prompt.reset(token)
    assert image_gen._expand_judge_user_request("raw without binding") == "raw without binding"


def test_expand_judge_image_prompt_to_latest_live_prompt(tmp_path):
    png = tmp_path / "image.png"
    Image.new("RGB", (1, 1), (4, 5, 6)).save(png)
    tokens = trajectory.set_active_trajectory_relpath("step_000001/sample_0.00")
    try:
        trajectory.register_tool_artifact(prompt="latest diffusion prompt", paths=[str(png)], backend="qwen_image")
        assert image_gen._expand_judge_image_prompt("last") == "latest diffusion prompt"
        assert image_gen._expand_judge_image_prompt("") == "latest diffusion prompt"
    finally:
        trajectory.reset_active_trajectory_relpath(tokens)


def test_clear_latest_image_prunes_only_active_rollout_registry(tmp_path):
    png_a = tmp_path / "a.png"
    png_b = tmp_path / "b.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(png_a)
    Image.new("RGB", (1, 1), (4, 5, 6)).save(png_b)

    tokens_a = trajectory.set_active_trajectory_relpath("step_000001/sample_0.00")
    trajectory.register_tool_artifact(prompt="prompt a", paths=[str(png_a)], backend="qwen_image")
    assert trajectory.count_live_generate_artifacts_for_active_rollout() == 1
    trajectory.reset_active_trajectory_relpath(tokens_a)

    tokens_b = trajectory.set_active_trajectory_relpath("step_000001/sample_1.00")
    trajectory.register_tool_artifact(prompt="prompt b", paths=[str(png_b)], backend="qwen_image")
    assert trajectory.count_live_generate_artifacts_for_active_rollout() == 1
    trajectory.reset_active_trajectory_relpath(tokens_b)

    tokens_a = trajectory.set_active_trajectory_relpath("step_000001/sample_0.00")
    try:
        trajectory.clear_latest_tool_image_for_active_rollout()
        assert trajectory.count_live_generate_artifacts_for_active_rollout() == 0
    finally:
        trajectory.reset_active_trajectory_relpath(tokens_a)

    tokens_b = trajectory.set_active_trajectory_relpath("step_000001/sample_1.00")
    try:
        assert trajectory.count_live_generate_artifacts_for_active_rollout() == 1
        trajectory.clear_latest_tool_image_for_active_rollout()
        assert trajectory.count_live_generate_artifacts_for_active_rollout() == 0
    finally:
        trajectory.reset_active_trajectory_relpath(tokens_b)


def test_decode_images_base64():
    images = image_gen._decode_images({"images_base64": [_png_b64()]})
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert image_gen._decode_images({}) == []
    assert image_gen._decode_images({"images_base64": ["not-base64!!"]}) == []


def test_qwen_image_geometry_and_seed_from_hydra(tmp_path):
    _bind_tool_cfg(
        e2e_root=tmp_path,
        qwen_image_seed=7,
        qwen_image_diversify_seed=False,
        qwen_image_height=256,
        qwen_image_width=384,
        qwen_image_steps=8,
        qwen_image_true_cfg_scale=2.5,
    )
    assert image_gen._qwen_image_seed("a cafe poster") == 7
    assert image_gen._require_hydra_int("qwen_image_height", 512) == 256
    assert image_gen._require_hydra_int("qwen_image_width", 512) == 384
    assert image_gen._require_hydra_int("qwen_image_steps", 20) == 8
    assert image_gen._require_hydra_float("qwen_image_true_cfg_scale", 4.0) == 2.5
    _bind_tool_cfg(e2e_root=tmp_path, qwen_image_height="garbage")
    with pytest.raises(ValueError, match="qwen_image_height"):
        image_gen._require_hydra_int("qwen_image_height", 512)


def test_count_live_generate_artifacts_unbound_rid_is_zero(tmp_path):
    """Missing active rollout id must not count every registry row."""
    from verl_omni.tools import trajectory

    _clear_all_tool_artifacts()
    png = tmp_path / "orphan.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    # Register with an explicit rid while nothing is active on this task.
    trajectory.register_tool_artifact(
        prompt="orphan",
        paths=[str(png)],
        backend="qwen_image",
        rollout_id="some_other_rollout",
    )
    assert trajectory.get_active_rollout_id() is None
    assert trajectory.count_live_generate_artifacts_for_active_rollout() == 0
    _clear_all_tool_artifacts()
