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
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from omegaconf import OmegaConf
from PIL import Image
from verl.tools.function_tool import FUNCTION_TOOL_REGISTRY

import verl_omni.tools.image_gen as image_gen
from verl_omni.tools import trajectory
from verl_omni.tools.trajectory import artifacts
from verl_omni.utils.agentic import image_gen_rollout_dump as dump_mod


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


def test_expand_judge_user_request_always_defers_to_the_bound_task():
    """The evaluation target is the bound request, whatever the model supplies.

    A rollout that pastes its own rewritten diffusion prompt must not be able to pick
    the text its ``correctness`` is scored against. The old resolver only canonicalised
    placeholders and truncated pastes, so a longer elaborated rewrite passed straight
    through and became the judge's "User request"; this asserts that hole is closed.
    """
    bound = "A vertical artistic cafe poster."
    token = trajectory.active_user_prompt.set(bound)
    try:
        for supplied in (
            "same as user message",
            "last",
            "some other task text",
            # The observed 9001 shape: the model's own rewrite, longer than the task.
            bound + " The headline reads ARTISAN ROAST in bold, heavy, elegant serif font.",
        ):
            assert image_gen._expand_judge_user_request(supplied) == bound
    finally:
        trajectory.active_user_prompt.reset(token)
    # Nothing bound: the argument is all there is to grade against.
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


def test_save_images_serializes_index_and_meta_under_concurrency(tmp_path):
    """Concurrent ``generate_image`` calls must not share an ``image_NN`` index.

    Regression: parallel plan subtasks inside one rollout both read the same
    "next" index and interleaved the ``meta.json`` read-modify-write, producing
    duplicate ``image_NN`` files and concatenated (invalid) JSON.
    """
    _bind_tool_cfg(e2e_root=tmp_path, run_name="cpu_test")
    relpath = "step_000001/sample_7.00"
    calls = 24

    def _worker(index):
        # ``asyncio.to_thread`` copies the caller context; mirror that per thread.
        trajectory.set_active_trajectory_relpath(relpath)
        trajectory.active_user_prompt.set(f"poster {index}")
        image_gen._save_images(
            [Image.new("RGB", (1, 1), (index % 255, 0, 0))],
            f"prompt {index}",
            backend="vllm_omni",
            tool_stubbed=False,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_worker, range(calls)))

    traj_dir = tmp_path / "cpu_test" / "rollout_images" / relpath
    names = sorted(path.name for path in traj_dir.glob("image_*.png"))
    assert [int(name.split("_")[1]) for name in names] == list(range(calls))
    meta = json.loads((traj_dir / "meta.json").read_text())
    assert meta["num_images"] == calls
    assert len(meta["calls"]) == calls
    assert sorted(call["index"] for call in meta["calls"]) == list(range(calls))


def test_live_write_and_materialize_share_the_traj_dir_lock(tmp_path):
    """The live tool and post-processing must serialise on one folder lock.

    Regression: ``materialize_rollout_images`` rewrote ``meta.json`` without the
    lock that ``_save_images`` takes, so a post-processing rewrite could land
    between the live tool's read and write and drop or corrupt its call rows.
    """
    _bind_tool_cfg(e2e_root=tmp_path, run_name="cpu_test")
    relpath = "step_000001/sample_9.00"
    calls = 16
    image_gen._save_images([Image.new("RGB", (1, 1))], "seed", backend="vllm_omni", tool_stubbed=False)

    def _live(index):
        trajectory.set_active_trajectory_relpath(relpath)
        trajectory.active_user_prompt.set(f"poster {index}")
        image_gen._save_images([Image.new("RGB", (1, 1))], f"prompt {index}", backend="vllm_omni", tool_stubbed=False)

    def _materialize(index):
        dump_mod.materialize_rollout_images(
            decoded_response=f"prompt {index}",
            run_dir=tmp_path / "cpu_test",
            relpath=relpath,
            user_prompt=f"poster {index}",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: _live(i) if i % 2 else _materialize(i), range(calls)))

    traj_dir = tmp_path / "cpu_test" / "rollout_images" / relpath
    # A single valid document: the lock + atomic publish never interleave writers.
    meta = json.loads((traj_dir / "meta.json").read_text())
    assert meta["trajectory_relpath"] == relpath
    assert meta["source"] == "direct_tool_write"
    assert meta["num_images"] == len(meta["calls"]) == len(list(traj_dir.glob("image_*.png")))
    assert sorted(call["index"] for call in meta["calls"]) == list(range(meta["num_images"]))
    assert not list(traj_dir.glob("*.tmp"))
