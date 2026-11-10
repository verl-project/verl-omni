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

"""Episode lifetime and generation/judgment ownership across concurrent tools."""

import asyncio

import pytest

from verl_omni.tools import trajectory


def episode(identity):
    assert hasattr(trajectory, "EpisodeState"), "episode-owned trajectory state is missing"
    return trajectory.EpisodeState(episode_id=identity, artifact_relpath="same/output/path", user_prompt=identity)


def test_nested_failure_restores_the_outer_episode_and_releases_inner_state(tmp_path):
    outer, inner = episode("outer"), episode("inner")
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    with trajectory.episode_scope(outer):
        outer.record_generation(call_id="outer-call", prompt="outer", paths=[str(image)])
        with pytest.raises(RuntimeError, match="tool failed"):
            with trajectory.episode_scope(inner):
                inner.record_generation(call_id="inner-call", prompt="inner", paths=[str(image)])
                raise RuntimeError("tool failed")
        assert trajectory.current_episode() is outer
        assert outer.latest_generation.call_id == "outer-call"
        assert inner.closed and not inner.generations
    assert trajectory.current_episode() is None
    assert outer.closed and not outer.generations
    assert image.exists()


def test_parallel_siblings_and_threaded_judges_do_not_share_state(tmp_path):
    async def run(identity):
        state = episode(identity)
        image = tmp_path / f"{identity}.png"
        image.write_bytes(identity.encode())
        with trajectory.episode_scope(state):
            call = state.record_generation(
                call_id="call", prompt=identity, paths=[str(image)], artifact_ids=["same-id"]
            )
            await asyncio.sleep(0)

            def judge():
                active = trajectory.current_episode()
                assert active is state
                active.record_judgment(call.call_id, good_enough=identity == "a")

            await asyncio.to_thread(judge)
            return state.resolve_image(artifact_id="same-id"), state.good_enough

    async def gather():
        return await asyncio.gather(run("a"), run("b"))

    assert asyncio.run(gather()) == [(str(tmp_path / "a.png"), True), (str(tmp_path / "b.png"), False)]


def test_artifact_ids_are_scoped_and_unknown_judgments_cannot_stop_an_episode(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    with trajectory.episode_scope(episode("one")) as first:
        first.record_generation(call_id="one", prompt="p", paths=[str(image)], artifact_ids=["private"])
        with trajectory.episode_scope(episode("two")) as second:
            second.record_generation(call_id="two", prompt="p", paths=[str(image)], artifact_ids=["other"])
            assert second.resolve_image(artifact_id="private") is None
            with pytest.raises(ValueError, match="generation"):
                second.record_judgment("one", good_enough=True)
            assert not second.good_enough


def test_one_generation_call_can_own_multiple_images(tmp_path):
    images = [tmp_path / "one.png", tmp_path / "two.png"]
    for image in images:
        image.write_bytes(b"pixels")
    with trajectory.episode_scope(episode("seeds")) as state:
        generated = state.record_generation(
            call_id="seed-group", prompt="scene", paths=[str(p) for p in images], artifact_ids=["one", "two"]
        )
        assert generated.paths == tuple(str(p) for p in images)
        assert len(state.generations) == 1
        assert state.resolve_image(artifact_id="two") == str(images[1])
        state.record_judgment("seed-group", good_enough=True)
        assert state.judgments[-1].call_id == "seed-group"
        assert state.good_enough


def test_cancelled_scope_rejects_late_tool_results():
    state = episode("cancelled")

    async def run():
        with trajectory.episode_scope(state):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert state.closed
    with pytest.raises(RuntimeError, match="closed"):
        state.record_generation(call_id="late", prompt="p", paths=[])


def test_nested_binding_of_the_same_episode_does_not_close_its_owner():
    state = episode("shared-with-tool")
    with trajectory.episode_scope(state):
        with trajectory.episode_scope(state):
            state.record_generation(call_id="first", prompt="p", paths=[])
        assert not state.closed and len(state.generations) == 1
    assert state.closed


def test_existing_tool_helpers_share_the_scoped_owner_across_thread_hops(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")

    async def run():
        with trajectory.episode_scope(episode("callbacks")) as state:
            trajectory.register_tool_artifact(prompt="scene", paths=[str(image)], artifact_id="image")
            assert state.latest_generation.prompt == "scene"
            await asyncio.to_thread(trajectory.set_good_enough_yes_reached, True)
            assert state.good_enough and trajectory.get_good_enough_yes_reached()
            assert trajectory.resolve_tool_image_path(artifact_id="image") == str(image)
        assert trajectory.resolve_tool_image_path(artifact_id="image") is None

    asyncio.run(run())


def test_frozen_judge_feedback_is_associated_with_its_generated_image(tmp_path, monkeypatch):
    import verl_omni.tools.image_gen as tools

    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    with trajectory.episode_scope(episode("judge")) as state:
        state.record_generation(call_id="generated", prompt="p", paths=[str(image)])
        monkeypatch.setattr(
            tools,
            "_call_judge_vlm",
            lambda *args: ("good_enough=YES", {"good_enough": True, "parse_ok": 1, "image_path": str(image)}),
        )
        tools.judge_image("request", "p")
        assert state.judgments[-1].call_id == "generated"
        assert state.good_enough


@pytest.mark.parametrize("reached", [False, True])
def test_compatibility_latch_setter_preserves_real_judgment_history(reached):
    with trajectory.episode_scope(episode("latch")) as state:
        state.record_generation(call_id="generated", prompt="p", paths=[])
        state.record_judgment("generated", good_enough=True)
        history = tuple(state.judgments)

        trajectory.set_good_enough_yes_reached(reached)

        assert state.good_enough is reached
        assert tuple(state.judgments) == history


def test_legacy_prompt_binding_stays_local_to_a_copied_thread_context():
    with trajectory.episode_scope(episode("outer")) as state:
        asyncio.run(asyncio.to_thread(trajectory.set_active_user_prompt, "thread prompt"))
        assert trajectory.get_active_user_prompt() == "outer"
        assert trajectory.current_episode() is state


def test_overlapping_legacy_prompt_bindings_restore_each_task_and_parent():
    async def child(prompt):
        token = trajectory.set_active_user_prompt(prompt)
        try:
            await asyncio.sleep(0)
            assert trajectory.get_active_user_prompt() == prompt
        finally:
            trajectory.reset_active_user_prompt(token)
        assert trajectory.get_active_user_prompt() == "outer"

    async def run():
        await asyncio.gather(child("a"), child("b"))

    with trajectory.episode_scope(episode("outer")):
        asyncio.run(run())
        assert trajectory.get_active_user_prompt() == "outer"


@pytest.mark.parametrize("nested", [False, True])
def test_legacy_direct_path_binding_updates_identity_and_restores_owner(tmp_path, nested):
    from contextlib import nullcontext

    from verl_omni.utils.agentic.artifacts import rollout_id_from_relpath

    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")
    parent = episode("outer") if nested else None
    with trajectory.episode_scope(parent) if parent else nullcontext():
        relpath = "step_000001/sample_0.00"
        token = trajectory.set_active_trajectory_relpath(relpath)
        child = trajectory.current_episode()
        try:
            assert trajectory.get_active_rollout_id() == rollout_id_from_relpath(relpath)
            assert child is not parent
            trajectory.register_tool_artifact(prompt="p", paths=[str(image)])
            assert trajectory.count_live_generate_artifacts_for_active_rollout() == 1
        finally:
            trajectory.reset_active_trajectory_relpath(token)
        assert child.closed
        assert trajectory.current_episode() is parent


def test_prompt_binding_from_unbound_context_releases_its_scratch_owner():
    """Prompt reset restores no owner and cannot leak artifacts or a YES latch."""
    from contextvars import Context

    def run():
        assert trajectory.current_episode() is None
        outer = trajectory.set_active_user_prompt("outer")
        state = trajectory.current_episode()
        try:
            state.record_generation(call_id="generated", prompt="p", paths=[])
            trajectory.set_good_enough_yes_reached(True)
            inner = trajectory.set_active_user_prompt("inner")
            trajectory.reset_active_user_prompt(inner)
            assert trajectory.current_episode() is state and not state.closed
            assert trajectory.get_active_user_prompt() == "outer"
        finally:
            trajectory.reset_active_user_prompt(outer)
        assert trajectory.current_episode() is None
        assert trajectory.get_active_user_prompt() is None
        assert not trajectory.get_good_enough_yes_reached()
        assert state.closed and not state.generations

    Context().run(run)
