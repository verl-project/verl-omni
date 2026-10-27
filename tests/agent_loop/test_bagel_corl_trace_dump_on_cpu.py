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
"""CPU tests for the per-episode trajectory dump of the Bagel Co-RL UND lane."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from verl_omni.agent_loop import bagel_corl as loop_mod
from verl_omni.agent_loop.bagel_corl import (
    dump_episode_hermes_action,
    dump_episode_images,
    dump_episode_trace,
    episode_artifact_keys,
    episode_seed_index,
)
from verl_omni.agent_loop.bagel_corl_lib import (
    BagelGenerateImageTool,
    EpisodeRollout,
    run_serial_episode,
)

# What the AR engine actually returned for the stalled episodes: the role label repeated
# until the 1024-token budget ran out. ``und_turn_kind`` is right to call this ``continue``.
DEGENERATE_TEXT = "assistant\n" * 128


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) + 1000 for ch in text]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        _ = skip_special_tokens
        return "".join(chr(int(t) - 1000) for t in token_ids)


async def _noop_gen(**kwargs):
    return [{"valid": True, "image_path": "/tmp/x.png"} for _ in kwargs["seeds"]]


def _run_episode(und_decode, *, samples, generate_fn, **kwargs):
    async def run():
        tool = BagelGenerateImageTool(gen_samples_per_call=samples, generate_fn=generate_fn)
        return await run_serial_episode(
            dataset_task_uid="task", policy_version=1, und_decode=und_decode, generate_tool=tool, **kwargs
        )

    return asyncio.run(run())


def _degenerate_episode(max_und_turns: int = 3) -> EpisodeRollout:
    """Drive a real episode whose every UND turn is a repetition loop (K stays 0)."""

    async def und_decode(**kwargs):
        _ = kwargs
        return {"token_ids": [11] * 1024, "text": DEGENERATE_TEXT}

    return _run_episode(
        und_decode,
        samples=2,
        generate_fn=_noop_gen,
        prompt_ids=[1, 2, 3],
        tokenizer=_FakeTokenizer(),
        max_und_turns=max_und_turns,
    )


def test_degenerate_turns_are_recorded_with_their_raw_text():
    """The K=0 episode must leave a per-turn record -- that is the whole point of the dump."""
    episode = _degenerate_episode(max_und_turns=3)

    assert episode.num_gen_calls == 0
    turns = [r for r in episode.turn_trace if r["record"] == "und_turn"]
    assert len(turns) == 3
    assert {r["kind"] for r in turns} == {"continue"}
    assert all(r["text"] == DEGENERATE_TEXT for r in turns)
    assert [r["turn"] for r in turns] == [1, 2, 3]
    assert all(r["out_tokens"] == 1024 for r in turns)
    # No GEN call happened, so no gen_call record may be fabricated.
    assert [r for r in episode.turn_trace if r["record"] == "gen_call"] == []


def test_a_generate_image_call_is_recorded_as_its_own_trace_entry():
    """The other half of the trace: when GEN *does* fire, the prompt/paths are captured."""

    async def und_decode(**kwargs):
        return {
            "token_ids": [7],
            "text": '{"name": "generate_image", "arguments": {"prompt": "a cat"}}',
        }

    episode = _run_episode(
        und_decode, samples=2, generate_fn=_noop_gen, prompt_ids=[1], tokenizer=_FakeTokenizer(), max_und_turns=2
    )

    calls = [r for r in episode.turn_trace if r["record"] == "gen_call"]
    assert len(calls) == 1
    assert calls[0]["prompt"] == "a cat"
    assert calls[0]["num_samples"] == 2
    assert calls[0]["num_valid"] == 2


def test_dump_writes_the_k0_signature_and_the_raw_turn_text(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "resolve_run_dir", lambda: tmp_path)
    episode = _degenerate_episode(max_und_turns=2)
    tok = _FakeTokenizer()

    written = dump_episode_trace(
        episode=episode,
        relpath="step_000007/sample_12.01",
        step=7,
        sample_index=12,
        user_prompt="draw a cat",
        prompt_ids=tok.encode("hello"),
        tokenizer=tok,
    )

    assert written is not None
    target = tmp_path / "rollout_trajectories" / "step_000007"
    assert (target / "sample_12.01.json").is_file()
    assert (target / "sample_12.01.txt").is_file()

    payload = json.loads((target / "sample_12.01.json").read_text())
    assert payload["gen_lane_skipped"] is True
    assert payload["num_gen_calls"] == 0
    assert payload["turn_kind_counts"] == {"continue": 2}
    assert payload["turn_trace"][0]["text"] == DEGENERATE_TEXT
    assert payload["user_prompt"] == "draw a cat"
    # The prompt is decoded too, so the dump shows what the UND lane was actually asked.
    assert payload["prompt_text"] == "hello"

    text = (target / "sample_12.01.txt").read_text()
    assert "gen_lane_skipped=True" in text
    assert "turn=1 kind=continue" in text
    assert "assistant" in text


def test_dump_records_the_gen_call_block_when_the_lane_ran(tmp_path, monkeypatch):
    monkeypatch.setattr(loop_mod, "resolve_run_dir", lambda: tmp_path)

    async def und_decode(**kwargs):
        return {"token_ids": [7], "text": '{"name": "generate_image", "arguments": {"prompt": "a dog"}}'}

    episode = _run_episode(und_decode, samples=1, generate_fn=_noop_gen, prompt_ids=[1], max_und_turns=1)
    dump_episode_trace(episode=episode, relpath="step_000001/sample_1.01", step=1, sample_index=1)

    payload = json.loads((tmp_path / "rollout_trajectories" / "step_000001" / "sample_1.01.json").read_text())
    assert payload["gen_lane_skipped"] is False
    assert payload["gen_calls"][0]["prompt"] == "a dog"


def test_dump_can_be_opted_out(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_mod, "resolve_run_dir", lambda: tmp_path)
    monkeypatch.setenv("BAGEL_CORL_TRACE_DUMP", "0")
    episode = _degenerate_episode(max_und_turns=1)

    assert dump_episode_trace(episode=episode, relpath="step_000001/sample_1.01") is None
    assert not (tmp_path / "rollout_trajectories").exists()


def test_dump_failure_never_propagates(monkeypatch):
    """Diagnostics must not be able to fail a rollout the trainer could otherwise use."""

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(loop_mod, "resolve_run_dir", _boom)
    episode = _degenerate_episode(max_und_turns=1)

    assert dump_episode_trace(episode=episode, relpath="step_000001/sample_1.01") is None


# hermes_actions: the action-level index the bagel TQ worker never wrote
def _hermes_rows(tmp_path, step_tag="step_000004"):
    path = tmp_path / "hermes_actions" / f"{step_tag}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _bind_hermes(tmp_path, monkeypatch):
    """Point the shared run root at the test tree."""
    from verl_omni.tools.trajectory import paths as traj_paths

    monkeypatch.setattr(traj_paths, "_diffusion_image_dir", tmp_path / "rollout_images")


def test_hermes_rows_append_to_one_step_file(tmp_path, monkeypatch):
    """Every episode of a step appends: the file is an index, so it must not be clobbered."""
    _bind_hermes(tmp_path, monkeypatch)
    episode = _degenerate_episode(max_und_turns=1)

    dump_episode_hermes_action(episode=episode, relpath="step_000004/sample_0.01", step=4, sample_index=0)
    dump_episode_hermes_action(episode=episode, relpath="step_000004/sample_1.01", step=4, sample_index=1)

    rows = _hermes_rows(tmp_path)
    assert [r["trajectory_relpath"] for r in rows] == [
        "step_000004/sample_0.01",
        "step_000004/sample_1.01",
    ]
    assert [r["sample_index"] for r in rows] == [0, 1]


def test_hermes_row_records_the_k0_signature(tmp_path, monkeypatch):
    """A K=0 episode still gets a row: "GEN never asked" is the first thing to check."""
    _bind_hermes(tmp_path, monkeypatch)
    episode = _degenerate_episode(max_und_turns=1)

    dump_episode_hermes_action(episode=episode, relpath="step_000004/sample_0.01", step=4, sample_index=0)

    row = _hermes_rows(tmp_path)[0]
    assert row["gen_lane_skipped"] is True
    assert row["num_gen_calls"] == 0
    assert row["turn_kind_counts"] == {"continue": 1}
    assert row["num_tool_calls_executed"] == 0
    assert row["image_paths"] == []


def test_hermes_row_points_at_the_materialized_image_not_the_scratch_file(tmp_path, monkeypatch):
    """The row must name the ``rollout_images/`` copy, matching the sibling agentic lane."""
    _bind_hermes(tmp_path, monkeypatch)
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    async def gen_fn(**kwargs):
        return [{"valid": True, "image_path": str(scratch / f"gen_{i}.png")} for i, _ in enumerate(kwargs["seeds"])]

    async def und_decode(**kwargs):
        _ = kwargs
        return {"token_ids": [7], "text": '{"name": "generate_image", "arguments": {"prompt": "a dog"}}'}

    episode = _run_episode(und_decode, samples=2, generate_fn=gen_fn, prompt_ids=[1], max_und_turns=1)
    dump_episode_hermes_action(
        episode=episode,
        relpath="step_000002/sample_3.01",
        step=2,
        sample_index=3,
        user_prompt="draw a dog",
    )

    row = _hermes_rows(tmp_path, "step_000002")[0]
    assert row["gen_lane_skipped"] is False
    assert row["num_gen_calls"] == 1
    assert row["num_tool_calls_executed"] == 1
    assert row["image_dir"].endswith("rollout_images/step_000002/sample_3.01")
    assert len(row["image_paths"]) == 2
    assert all("/rollout_images/" in p and p.endswith(".png") for p in row["image_paths"])
    # Scratch provenance is kept, so a reviewer can still find the original file.
    assert all("scratch" in p for p in row["source_image_paths"])
    assert row["gen_calls"][0]["prompt"] == "a dog"
    assert row["user_prompt"] == "draw a dog"


def test_hermes_dump_can_be_opted_out(monkeypatch, tmp_path):
    monkeypatch.setattr(loop_mod, "resolve_run_dir", lambda: tmp_path)
    monkeypatch.setenv("BAGEL_CORL_TRACE_DUMP", "0")
    episode = _degenerate_episode(max_und_turns=1)

    assert dump_episode_hermes_action(episode=episode, relpath="step_000001/sample_1.01") is None
    assert not (tmp_path / "hermes_actions").exists()


def test_hermes_dump_failure_never_propagates(monkeypatch):
    """Same guarantee as the trajectory dump: diagnostics cannot fail a rollout."""

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(loop_mod, "resolve_run_dir", _boom)
    episode = _degenerate_episode(max_und_turns=1)

    assert dump_episode_hermes_action(episode=episode, relpath="step_000001/sample_1.01") is None


def _gen_episode(scratch):
    """One episode that really fires ``generate_image`` with two GEN seeds."""
    scratch.mkdir(parents=True, exist_ok=True)

    async def gen_fn(**kwargs):
        rows = []
        for i, _ in enumerate(kwargs["seeds"]):
            path = scratch / f"gen_seed{i}.png"
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + bytes([i]))
            rows.append({"valid": True, "image_path": str(path)})
        return rows

    async def und_decode(**kwargs):
        _ = kwargs
        return {"token_ids": [7], "text": '{"name": "generate_image", "arguments": {"prompt": "a cat"}}'}

    return _run_episode(und_decode, samples=2, generate_fn=gen_fn, prompt_ids=[1], max_und_turns=1)


def test_episode_path_produces_all_three_artifact_trees(tmp_path, monkeypatch):
    """The recipe's only live hook must yield traj + hermes_actions + rollout_images."""
    _bind_hermes(tmp_path, monkeypatch)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    episode = _gen_episode(scratch)
    relpath = "step_000009/sample_2.01"
    tok = _FakeTokenizer()
    step = 9

    dump_episode_trace(
        episode=episode,
        relpath=relpath,
        step=step,
        sample_index=2,
        user_prompt="draw a cat",
        prompt_ids=tok.encode("hi"),
        tokenizer=tok,
    )
    dump_episode_hermes_action(episode=episode, relpath=relpath, step=step, sample_index=2, user_prompt="draw a cat")
    copied = dump_episode_images(episode=episode, relpath=relpath, step=step)

    run_root = tmp_path
    # ``relpath`` ends in ``.01`` (the rollout index), so append ``.json`` rather than
    # using ``with_suffix``, which would rewrite that ``.01`` instead of extending it.
    assert (run_root / "rollout_trajectories" / f"{relpath}.json").is_file()
    assert (run_root / "hermes_actions" / "step_000009.jsonl").is_file()
    images_dir = run_root / "rollout_images" / relpath
    assert images_dir.is_dir(), "rollout_images/<relpath>/ must exist after the episode dump"
    assert len(copied) == 2
    assert sorted(p.name for p in images_dir.glob("*.png")) == ["gen_seed0.png", "gen_seed1.png"]
    assert (images_dir / "meta.json").is_file()

    # The index and the copied files have to agree, or the index misleads a reviewer.
    row = _hermes_rows(tmp_path, "step_000009")[0]
    assert sorted(Path(p).name for p in row["image_paths"]) == ["gen_seed0.png", "gen_seed1.png"]
    assert all((run_root / "rollout_images" / relpath / Path(p).name).is_file() for p in row["image_paths"])


def _bind_project(tmp_path, monkeypatch):
    """Make ``tmp_path`` the project root, with the run tree under it."""
    _bind_hermes(tmp_path, monkeypatch)
    monkeypatch.setattr(loop_mod, "resolve_run_dir", lambda: tmp_path)
    monkeypatch.setenv("VERLOMNI_ROOT", str(tmp_path))


def test_trajectory_names_the_project_relative_copy_not_the_scratch_file(tmp_path, monkeypatch):
    """The trajectory JSON/TXT must point at the durable ``rollout_images/`` copy."""
    _bind_project(tmp_path, monkeypatch)
    episode = _gen_episode(tmp_path / "scratch")
    relpath = "step_000009/sample_2.01"

    dump_episode_trace(episode=episode, relpath=relpath, step=9, sample_index=2)
    dump_episode_images(episode=episode, relpath=relpath, step=9)

    payload = json.loads((tmp_path / "rollout_trajectories" / f"{relpath}.json").read_text())
    expected = [f"rollout_images/{relpath}/gen_seed0.png", f"rollout_images/{relpath}/gen_seed1.png"]
    assert payload["gen_calls"][0]["image_paths"] == expected
    trace_call = next(r for r in payload["turn_trace"] if r["record"] == "gen_call")
    assert trace_call["image_paths"] == expected
    assert all((tmp_path / p).is_file() for p in expected), "the named copies must exist"
    assert all("scratch" in p for p in trace_call["source_image_paths"])

    text = (tmp_path / "rollout_trajectories" / f"{relpath}.txt").read_text()
    assert f"image: rollout_images/{relpath}/gen_seed0.png" in text
    image_lines = [line for line in text.splitlines() if line.strip().startswith("image:")]
    assert len(image_lines) == 2 and not any("scratch" in line for line in image_lines)
    # Observation text is what the policy was conditioned on, so it stays verbatim.
    obs = next(r for r in payload["turn_trace"] if r["record"] == "env_obs")
    assert "scratch" in obs["text"]
    # The episode itself keeps the scratch paths: the GEN lane still reads those files.
    live = next(r for r in episode.turn_trace if r["record"] == "gen_call")
    assert all("scratch" in p for p in live["image_paths"])


def test_hermes_row_and_meta_are_project_relative(tmp_path, monkeypatch):
    _bind_project(tmp_path, monkeypatch)
    episode = _gen_episode(tmp_path / "scratch")
    relpath = "step_000009/sample_2.01"

    dump_episode_hermes_action(episode=episode, relpath=relpath, step=9, sample_index=2)
    dump_episode_images(episode=episode, relpath=relpath, step=9)

    row = _hermes_rows(tmp_path, "step_000009")[0]
    assert row["image_dir"] == f"rollout_images/{relpath}"
    assert row["image_paths"] == [f"rollout_images/{relpath}/gen_seed{i}.png" for i in range(2)]
    meta = json.loads((tmp_path / "rollout_images" / relpath / "meta.json").read_text())
    assert meta["image_paths"] == row["image_paths"]


def test_validation_paths_follow_the_val_prefix_the_images_are_copied_under(tmp_path, monkeypatch):
    """Validation PNGs land under ``rollout_images/val/<relpath>/``, so every record must say so."""
    _bind_project(tmp_path, monkeypatch)
    episode = _gen_episode(tmp_path / "scratch")
    relpath = "step_000030/sample_5.01"
    image_relpath = f"val/{relpath}"

    dump_episode_trace(episode=episode, relpath=relpath, step=30, image_relpath=image_relpath)
    dump_episode_hermes_action(episode=episode, relpath=relpath, step=30, image_relpath=image_relpath)
    dump_episode_images(episode=episode, relpath=image_relpath, step=30)

    payload = json.loads((tmp_path / "rollout_trajectories" / f"{relpath}.json").read_text())
    row = _hermes_rows(tmp_path, "step_000030")[0]
    for paths in (payload["gen_calls"][0]["image_paths"], row["image_paths"]):
        assert all(p.startswith(f"rollout_images/{image_relpath}/") for p in paths)
        assert all((tmp_path / p).is_file() for p in paths)


def test_project_relpath_keeps_paths_outside_the_project_absolute(tmp_path, monkeypatch):
    from verl_omni.tools.trajectory.paths import project_relpath

    monkeypatch.setenv("VERLOMNI_ROOT", str(tmp_path / "repo"))

    assert project_relpath(tmp_path / "repo" / "outputs" / "a.png") == "outputs/a.png"
    assert project_relpath("/tmp/bagel_corl_gen/x.png") == "/tmp/bagel_corl_gen/x.png"


def test_episode_image_dump_is_a_noop_for_a_k0_episode(tmp_path, monkeypatch):
    """K=0 wrote no image, so no folder may be manufactured for it."""
    _bind_hermes(tmp_path, monkeypatch)
    episode = _degenerate_episode(max_und_turns=1)

    copied = dump_episode_images(episode=episode, relpath="step_000003/sample_0.01", step=3)

    assert copied == []
    assert not (tmp_path / "rollout_images" / "step_000003" / "sample_0.01").exists()


def test_episode_image_dump_failure_never_propagates(monkeypatch, tmp_path):
    """A failing copy must be swallowed, and must not touch a real output tree."""
    _bind_hermes(tmp_path, monkeypatch)
    import verl_omni.utils.agentic.image_gen_rollout_dump as dump_mod

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(dump_mod, "resolve_rollout_images_root", _boom)
    episode = _gen_episode(tmp_path / "scratch")

    assert dump_episode_images(episode=episode, relpath="step_000001/sample_1.01") == []


def test_two_tasks_at_the_same_sibling_index_do_not_collide():
    """``session_id`` is the rollout *sibling* index, not the sample identity."""
    a_path, a_seed = episode_artifact_keys(dataset_task_uid="task_a", session_id=0, global_steps=60)
    b_path, b_seed = episode_artifact_keys(dataset_task_uid="task_b", session_id=0, global_steps=60)
    sibling_path, sibling_seed = episode_artifact_keys(dataset_task_uid="task_a", session_id=1, global_steps=60)

    assert a_path != b_path, "two tasks at sibling 0 must not share a trajectory folder"
    assert a_seed != b_seed, "two tasks at sibling 0 must not share a noise draw"
    assert a_path != sibling_path and a_seed != sibling_seed
    # ``sample_index`` carries the dataset sample, ``session_id`` goes to ``rollout_n``.
    assert a_path == "step_000060/sample_task_a.01"
    assert sibling_path == "step_000060/sample_task_a.02"


def test_episode_seed_index_is_process_stable_not_salted():
    """Replay has to redraw the same noise, so the index cannot use salted ``hash``."""
    import zlib

    assert episode_seed_index("uid:0") == episode_seed_index("uid:0")
    assert episode_seed_index("uid:0") == zlib.crc32(b"uid:0")
    assert episode_seed_index("uid:0") != episode_seed_index("uid:1")
