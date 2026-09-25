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
"""CPU tests for the e2e agentic artifact layout.

Pins the baseline hierarchy that the PR #409/#411/#412 rebase regressed:

    <e2e_root>/<experiment_name>/            (single level, no double nesting)
    <run>/hermes_actions/step_XXXXXX.{jsonl,txt}
    <run>/rollout_trajectories/step_XXXXXX/sample_<index>[.<nn>].{json,txt}
    <run>/rollout_images/step_XXXXXX/sample_<index>[.<nn>]/
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

import verl_omni  # noqa: F401
from verl_omni.agent_loop import omni_agent_loop
from verl_omni.agent_loop.omni_agent_loop import OmniAgentLoopManager
from verl_omni.tools import trajectory
from verl_omni.tools.trajectory import build_trajectory_relpath, resolve_run_dir
from verl_omni.tools.trajectory import locking as traj_locking
from verl_omni.utils.agentic.image_gen_rollout_dump import dump_raw_rollouts, dump_rollout_artifacts
from verl_omni.utils.agentic_val_viz import resolve_agentic_val_viz_provider


class _Tok:
    """Minimal tokenizer: ``split_rollout_turns`` only needs ``decode``."""

    pad_token_id = 0

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return " ".join(str(int(token)) for token in ids)


def _bind(root, run_name="layout_test"):
    cfg = OmegaConf.create(
        {
            "trainer": {"experiment_name": run_name},
            "agentic_image_gen": {"e2e_root": str(root)},
        }
    )
    trajectory.bind_run_artifacts(cfg)


class _PieceTok:
    """Token id -> text piece, so a dumped turn decodes into real tool-call JSON."""

    pad_token_id = 0

    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids, skip_special_tokens=False):  # noqa: ARG002
        return "".join(self.pieces.get(int(token), "") for token in ids)


def _output(response_ids):
    return SimpleNamespace(
        prompt_ids=[1, 2, 3],
        response_ids=list(response_ids),
        response_mask=[1] * len(response_ids),
        reward_score=0.5,
        extra_fields={"reward_extra_info": {"num_generate_image_prompts": 2, "reward_tool_call": 0.25}},
    )


def test_train_and_val_relpaths_match_baseline_layout():
    assert build_trajectory_relpath(step=5, sample_index=42, rollout_n=0) == "step_000005/sample_42.00"
    # Val omits the group suffix so it can never collide with a train ``.00`` dir.
    assert build_trajectory_relpath(step=5, sample_index=9001, rollout_n=0, validate=True) == "step_000005/sample_9001"
    assert build_trajectory_relpath(step=None, sample_index=None, rollout_n=0) == "step_unknown/sample_unknown.00"


def test_run_dir_is_single_level_under_e2e_root(tmp_path):
    """``e2e_root`` is the parent: paths.py appends ``experiment_name`` itself."""
    _bind(tmp_path, run_name="agentic_rpco_demo")
    assert resolve_run_dir() == tmp_path.resolve() / "agentic_rpco_demo"


def test_dump_rollout_artifacts_writes_trajectory_and_appends_monitor(tmp_path):
    _bind(tmp_path)
    dump_rollout_artifacts(
        tokenizer=_Tok(),
        step=0,
        relpath="step_000000/sample_9001",
        sample_index=9001,
        raw_prompt=[{"role": "user", "content": "a cafe poster"}],
        outputs=_output([10, 11, 12]),
    )

    step_dir = tmp_path / "layout_test" / "rollout_trajectories" / "step_000000"
    payload = json.loads((step_dir / "sample_9001.json").read_text())
    assert payload["trajectory_relpath"] == "step_000000/sample_9001"
    assert payload["user_prompt"] == "a cafe poster"
    assert (step_dir / "sample_9001.txt").is_file()

    jsonl = tmp_path / "layout_test" / "hermes_actions" / "step_000000.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["sample_index"] == 9001
    assert rows[0]["reward_metrics"]["score"] == 0.5
    assert rows[0]["reward_metrics"]["num_generate_image_prompts"] == 2
    assert (tmp_path / "layout_test" / "hermes_actions" / "step_000000.txt").is_file()


def test_concurrent_appends_keep_each_jsonl_row_parseable(tmp_path):
    """V1 workers append to one step monitor; no row may be spliced."""
    _bind(tmp_path)

    def _dump(index):
        dump_rollout_artifacts(
            tokenizer=_Tok(),
            step=3,
            relpath=f"step_000003/sample_{index}.00",
            sample_index=index,
            raw_prompt=[{"role": "user", "content": f"poster {index}"}],
            outputs=_output([1, 2]),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(_dump, range(24)))

    lines = [
        line
        for line in (tmp_path / "layout_test" / "hermes_actions" / "step_000003.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(lines) == 24
    assert sorted(json.loads(line)["sample_index"] for line in lines) == list(range(24))
    assert len(list((tmp_path / "layout_test" / "rollout_trajectories" / "step_000003").glob("*.json"))) == 24


def test_dump_rollout_artifacts_preserves_live_image_meta(tmp_path):
    """Live tool PNGs are indexed into the trajectory payload, not overwritten."""
    _bind(tmp_path)
    relpath = "step_000000/sample_9002"
    image_dir = tmp_path / "layout_test" / "rollout_images" / relpath
    image_dir.mkdir(parents=True)
    (image_dir / "image_00_abcdef012345.png").write_bytes(b"not-a-real-png")

    dump_rollout_artifacts(
        tokenizer=_Tok(),
        step=0,
        relpath=relpath,
        sample_index=9002,
        raw_prompt=[{"role": "user", "content": "poster"}],
        outputs=_output([7, 8]),
    )

    payload = json.loads(
        (tmp_path / "layout_test" / "rollout_trajectories" / "step_000000" / "sample_9002.json").read_text()
    )
    assert payload["image_dir"].endswith(relpath.replace("/", str(Path("/"))))
    assert len(payload["image_paths"]) == 1
    meta = json.loads((image_dir / "meta.json").read_text())
    assert meta["trajectory_relpath"] == relpath
    assert meta["source"] == "direct_tool_write"


def test_dumped_turns_expose_the_rewritten_diffusion_prompt(tmp_path):
    """``tool_prompt`` makes a rewrite chain readable without unescaping ``decode``.

    ``turn_prompt`` is the whole chat template (identical on every turn) and
    ``turn_obs`` only the judge observation, so the rewritten diffusion prompt that
    the harness accepted existed only inside the escaped JSON tool call.
    """
    _bind(tmp_path)
    gen = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "PROMPT_V1"}}\n</tool_call>'
    judge_call = (
        '<tool_call>\n{"name": "judge_image", "arguments": '
        '{"user_request": "same as user message", "image_prompt": "PROMPT_V1"}}\n</tool_call>'
    )
    judge_obs = "<tool_response>\nagentic_judge ok=1 good_enough =NO\n</tool_response>"
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "PROMPT_V2"}}\n</tool_call>'
    tokenizer = _PieceTok({10: gen, 11: judge_obs, 12: judge_call, 13: judge_obs, 14: rewrite})

    dump_rollout_artifacts(
        tokenizer=tokenizer,
        step=0,
        relpath="step_000000/sample_9003",
        sample_index=9003,
        raw_prompt=[{"role": "user", "content": "a poster"}],
        outputs=SimpleNamespace(
            prompt_ids=[1, 2, 3],
            response_ids=[10, 11, 12, 13, 14],
            response_mask=[1, 0, 1, 0, 1],
            reward_score=0.1,
            extra_fields={},
        ),
    )

    run_dir = tmp_path / "layout_test"
    payload = json.loads((run_dir / "rollout_trajectories" / "step_000000" / "sample_9003.json").read_text())
    assert [(turn["tool_name"], turn["tool_prompt"]) for turn in payload["rollout_turns"]] == [
        ("generate_image", "PROMPT_V1"),
        ("judge_image", "PROMPT_V1"),
        ("generate_image", "PROMPT_V2"),
    ]
    # The judge echoes the prompt it inspected, so the pair reads as
    # submit v1 -> echo v1 -> submit v2 without unescaping ``decode``.

    row = json.loads((run_dir / "hermes_actions" / "step_000000.jsonl").read_text().splitlines()[0])
    assert [turn["tool_prompt"] for turn in row["rollout_turns"]] == ["PROMPT_V1", "PROMPT_V1", "PROMPT_V2"]
    text_block = (run_dir / "rollout_trajectories" / "step_000000" / "sample_9003.txt").read_text()
    assert "turn_3_tool_prompt:" in text_block
    assert "PROMPT_V2" in text_block


def test_dumped_turns_expose_the_token_wise_advantage(tmp_path):
    """The dump reports the mask the optimizer used, per turn and per rollout.

    ``turn_kind`` labels a protocol stage; ``turn_advantage`` is the trainer's view of
    who owns the tokens. Without it a reader cannot tell a policy rewrite from the
    harness-injected Reflection sitting in the same turn.
    """
    _bind(tmp_path)
    gen = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "V1"}}\n</tool_call>'
    # The judge observation and the injected cue are one contiguous mask=0 run, which
    # is why the splitter has to separate them textually.
    judge_then_cue = (
        "<tool_response>\nVL judge on the last generated image:\n  agentic_judge ok=1 good_enough =NO\n"
        "</tool_response>\nReflection: rewrite next. agentic_forced_reflection=1"
    )
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "V2"}}\n</tool_call>'
    stop_cue = "Reflection: Stop. agentic_stop_decision_required=1 agentic_forced_reflection=1"
    tokenizer = _PieceTok({10: gen, 11: judge_then_cue, 12: rewrite, 13: stop_cue})

    dump_rollout_artifacts(
        tokenizer=tokenizer,
        step=0,
        relpath="step_000000/sample_9004",
        sample_index=9004,
        raw_prompt=[{"role": "user", "content": "a poster"}],
        outputs=SimpleNamespace(
            prompt_ids=[1, 2, 3],
            response_ids=[10, 11, 12, 13],
            response_mask=[1, 0, 1, 0],
            reward_score=0.1,
            extra_fields={},
        ),
    )

    run_dir = tmp_path / "layout_test"
    payload = json.loads((run_dir / "rollout_trajectories" / "step_000000" / "sample_9004.json").read_text())
    turns = payload["rollout_turns"]
    assert [(turn["turn_advantage"], turn["policy_tokens"], turn["env_tokens"]) for turn in turns] == [
        ("policy", 1, 0),
        ("policy+cue", 1, 1),
        ("cue", 0, 1),
    ]
    assert [turn["injected_cue"] for turn in turns] == [False, True, True]
    # The labels agree with the mask: the rewrite turn was trainable and the trailing
    # cue turn was not, so the cue gets the cue label and the rewrite does not.
    assert [turn["turn_kind"] for turn in turns] == [
        "call_generate_image",
        "agent_rewrite_after_forced_reflection_then_call_generate_image",
        "forced_reflection_stop_cue",
    ]
    assert payload["num_policy_tokens"] == 2
    assert payload["num_injected_cue_turns"] == 2
    # The mask partition holds end to end: every response token is accounted for.
    assert sum(turn["policy_tokens"] + turn["env_tokens"] for turn in turns) == 4

    text_block = (run_dir / "rollout_trajectories" / "step_000000" / "sample_9004.txt").read_text()
    # The text transcript names the mask on both bodies, so a cue is not read as a
    # model response and a policy span is not read as harness scaffolding.
    assert "advantage=policy+cue" in text_block
    assert "turn_3_response_masked0_injected_cue:" in text_block
    assert "decode_masked1_policy:" in text_block

    row = json.loads((run_dir / "hermes_actions" / "step_000000.jsonl").read_text().splitlines()[0])
    assert [turn["turn_advantage"] for turn in row["rollout_turns"]] == ["policy", "policy+cue", "cue"]


def _dump_raw_output(*, response_pieces, non_tensor):
    """Minimal DataProto stand-in for ``dump_raw_rollouts`` (list-backed rows)."""

    class _Row:
        def __init__(self, values):
            self._values = list(values)

        def tolist(self):
            return list(self._values)

        def __iter__(self):
            return iter(self._values)

    ids = list(response_pieces)
    return SimpleNamespace(
        batch={"responses": [_Row(ids)], "response_mask": [_Row([1] * len(ids))]},
        non_tensor_batch=non_tensor,
    )


def test_raw_dump_threads_task_type_into_the_turn_labels(tmp_path):
    """``task_type`` rides into the dump as a label, not as a labelling rule.

    The dump must report the row's ``task_type`` so a reader knows which corpus a
    trajectory came from, while ``turn_kind`` classifies both corpora identically: a turn
    whose decode carries a numbered plan is labelled as one whichever row it came from,
    because that is what the loop's reopen checks.
    """
    _bind(tmp_path)
    plan = "1. A librarian floats in an underwater cave library with fish nearby.\n"
    gen = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "V1"}}\n</tool_call>'
    judge = '<tool_call>\n{"name": "judge_image", "arguments": {"user_request": "x"}}\n</tool_call>'
    tokenizer = _PieceTok({10: plan, 11: gen, 12: judge})
    ids = [10, 11, 12]

    def _dump(task_type):
        dump_raw_rollouts(
            tokenizer=tokenizer,
            output=_dump_raw_output(
                response_pieces=ids,
                non_tensor={
                    "raw_prompt": [{"role": "user", "content": "a poster"}],
                    "trajectory_relpath": ["step_000000/sample_9004"],
                    "agentic_task_type": [task_type],
                },
            ),
            step=0,
            validate=True,
            write_monitor=False,
        )
        return json.loads(
            (tmp_path / "layout_test" / "rollout_trajectories" / "step_000000" / "sample_9004.json").read_text()
        )

    plan_payload = _dump("plan")
    assert plan_payload["task_type"] == "plan"
    # The literal streams ride along in the JSON, so the payload is self-contained.
    assert plan_payload["raw_response"] == f"{plan}{gen}{judge}"
    labelled = ["plan_then_call_generate_image_then_call_judge_image"]

    reflect_payload = _dump("reflect")
    assert reflect_payload["task_type"] == "reflect"
    # Same decode, same labels: the row's label does not change how the turn is read.
    assert [turn["turn_kind"] for turn in plan_payload["rollout_turns"]] == labelled
    assert [turn["turn_kind"] for turn in reflect_payload["rollout_turns"]] == labelled

    # The text dump leads with the raw episode and labels the annotated section.
    text_block = (tmp_path / "layout_test" / "rollout_trajectories" / "step_000000" / "sample_9004.txt").read_text()
    raw_start = text_block.index("raw_response (decoded response tokens")
    assert raw_start < text_block.index("annotated_turns:")
    raw_section = text_block[raw_start : text_block.index("annotated_turns:")]
    # The plan and the two calls are all still in the literal response, in order.
    assert (
        raw_section.index("A librarian floats")
        < raw_section.index('{"name": "generate_image"')
        < raw_section.index('{"name": "judge_image"')
    )


def test_raw_dump_prefers_the_relpath_index_over_the_batch_slot(tmp_path):
    """``sample_index`` must name the sample, not the row's position in the batch.

    ``output.non_tensor_batch["index"]`` disappears whenever the agent reward loop is
    enabled — the parent forwards the input non-tensor batch only when
    ``reward_loop_worker_handles`` is None — so the dump fell back to ``np.arange`` and
    wrote ``sample_index`` 0..3 beside folders named 9001..9004.
    """
    _bind(tmp_path)
    tokenizer = _PieceTok({10: "Done."})

    for extra, expected in (({"index": [3]}, 9004), ({}, 9004), ({"index": [9004]}, 9004)):
        non_tensor = {
            "raw_prompt": [{"role": "user", "content": "a poster"}],
            "trajectory_relpath": ["step_000000/sample_9004"],
            "agentic_task_type": ["plan"],
            **extra,
        }
        dump_raw_rollouts(
            tokenizer=tokenizer,
            output=_dump_raw_output(response_pieces=[10], non_tensor=non_tensor),
            step=0,
            validate=True,
            write_monitor=False,
        )
        payload = json.loads(
            (tmp_path / "layout_test" / "rollout_trajectories" / "step_000000" / "sample_9004.json").read_text()
        )
        assert payload["sample_index"] == expected


def test_raw_dump_leads_with_the_literal_episode(tmp_path):
    """``sample_*.txt`` opens with the raw token streams, not a paraphrase.

    The dump is the only record of what the trainer optimised, so the literal prompt and
    response tokens come first and the mask-annotated breakdown follows. The full
    chat-templated input is deliberately not repeated per turn: it is the system prompt
    plus every earlier turn, so printing it each time buried the episode under the same
    ~40 lines per turn.
    """
    _bind(tmp_path)
    gen = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "V1"}}\n</tool_call>'
    obs = "<tool_response>\npath=/tmp/a.png agentic_tool ok=1 images=1 backend=vllm_omni\n</tool_response>"
    judge = '<tool_call>\n{"name": "judge_image", "arguments": {"user_request": "x"}}\n</tool_call>'
    tokenizer = _PieceTok({10: gen, 11: obs, 12: judge})

    dump_rollout_artifacts(
        tokenizer=tokenizer,
        step=0,
        relpath="step_000000/sample_9004",
        sample_index=9004,
        raw_prompt=[{"role": "user", "content": "a poster"}],
        outputs=SimpleNamespace(
            prompt_ids=[10, 11, 12],
            response_ids=[10, 11, 12],
            response_mask=[1, 0, 1],
            reward_score=0.1,
            extra_fields={"agentic_task_type": "plan"},
        ),
    )

    run_dir = tmp_path / "layout_test"
    text_block = (run_dir / "rollout_trajectories" / "step_000000" / "sample_9004.txt").read_text()
    assert "task_type=plan sample_index=9004 rollout_n=0" in text_block
    assert "raw_prompt (decoded prompt tokens: the exact input):" in text_block
    assert "raw_response (decoded response tokens: the exact episode output):" in text_block
    assert "annotated_turns:" in text_block
    # Inside the raw response, the call and its observation sit in literal order. The
    # lines are indented by the renderer, so match single-line fragments.
    raw_section = text_block[
        text_block.index("raw_response (decoded response tokens") : text_block.index("annotated_turns:")
    ]
    gen_line = '{"name": "generate_image"'
    obs_line = "path=/tmp/a.png agentic_tool ok=1 images=1 backend=vllm_omni"
    judge_line = '{"name": "judge_image"'
    assert raw_section.index(gen_line) < raw_section.index(obs_line) < raw_section.index(judge_line)
    # The raw episode precedes the mask-annotated reading of it.
    assert text_block.index("raw_response (decoded response tokens") < text_block.index("annotated_turns:")
    # The turn block carries the observation delta, not the whole chat template.
    assert "turn_1_obs:" in text_block
    assert "turn_1_prompt:" not in text_block

    payload = json.loads((run_dir / "rollout_trajectories" / "step_000000" / "sample_9004.json").read_text())
    assert payload["task_type"] == "plan"
    assert payload["raw_response"] == f"{gen}{obs}{judge}"
    # The full templated input stays machine-readable, per turn, in the JSON.
    assert payload["rollout_turns"][0]["turn_prompt"]


def test_val_set_rows_never_touch_rollout_trajectories_or_monitor(monkeypatch):
    """A val-set batch is neither dumped nor mask-discarded (baseline contract).

    Only the fixed 9001-9004 holdout populates a validation step, so a val batch
    of hundreds of rows must not append to ``hermes_actions`` or create
    ``sample_<index>`` trajectory files.
    """
    dumped: list = []
    discarded: list = []
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: dumped.append(kwargs))
    monkeypatch.setattr(omni_agent_loop, "discard_invalid_rollouts", lambda output: discarded.append(output))
    monkeypatch.setattr(omni_agent_loop, "_stamp_scorer_knobs", lambda batch, config: None)
    monkeypatch.setattr(omni_agent_loop, "AgenticRewardMetrics", SimpleNamespace(aggregate=lambda ntb: {}))
    monkeypatch.setattr(
        omni_agent_loop.OmniAgentLoopManager, "_maybe_run_val_viz", lambda self, step, *, dedicated_manager: None
    )
    monkeypatch.setattr(
        omni_agent_loop.AgentLoopManager,
        "generate_sequences",
        lambda self, batch: SimpleNamespace(non_tensor_batch={}, meta_info={}),
    )

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager.config = OmegaConf.create({})
    manager._monitor_tokenizer = _Tok()

    def _run(validate):
        batch = SimpleNamespace(meta_info={"validate": validate, "global_steps": 7})
        OmniAgentLoopManager.generate_sequences(manager, batch)

    _run(False)
    assert len(dumped) == 1
    assert len(discarded) == 1

    _run(True)
    # Unchanged: the val pass skipped both, the train pass already ran them.
    assert len(dumped) == 1
    assert len(discarded) == 1


def test_tq_worker_skips_dump_for_validation(monkeypatch):
    """``rollout_valid``/val workers must not write the step monitor."""
    import asyncio

    dumped: list = []
    monkeypatch.setattr(omni_agent_loop, "dump_rollout_artifacts", lambda **kwargs: dumped.append(kwargs))

    async def _noop_postprocess(self, output, validate, **kwargs):
        return None

    monkeypatch.setattr(omni_agent_loop._AgentLoopWorkerTQImpl, "_agent_loop_postprocess", _noop_postprocess)

    worker = object.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    worker.tokenizer = _Tok()
    kwargs = {
        "_agentic_trajectory_relpath": "step_000070/sample_12",
        "global_steps": 70,
        "index": 12,
        "raw_prompt": [{"role": "user", "content": "hi"}],
    }

    asyncio.run(omni_agent_loop.OmniAgentLoopWorkerTQImpl._agent_loop_postprocess(worker, "out", True, **kwargs))
    assert dumped == []

    asyncio.run(omni_agent_loop.OmniAgentLoopWorkerTQImpl._agent_loop_postprocess(worker, "out", False, **kwargs))
    assert len(dumped) == 1
    assert dumped[0]["relpath"] == "step_000070/sample_12"
    assert dumped[0]["step"] == 70


def test_val_viz_provider_is_env_gated_and_uses_holdout_indices(monkeypatch):
    monkeypatch.delenv("AGENTIC_VAL_VIZ", raising=False)
    assert resolve_agentic_val_viz_provider() is None

    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")
    provider = resolve_agentic_val_viz_provider()
    assert provider is not None
    batch = provider.build_batch(0, eos_token_id=2, pad_token_id=0)
    assert list(batch.non_tensor_batch["index"]) == [9001, 9002, 9003, 9004]
    assert set(batch.non_tensor_batch["data_source"]) == {"agentic_val_viz"}
    assert batch.meta_info["validate"] is True


def test_traj_dir_lock_is_shared_across_modules(tmp_path):
    """``image_gen`` and the dump module must serialise on one lock object.

    ``locking`` is the single lock domain: importing it from either module
    yields the same reentrant lock, so the live tool cannot allocate an
    ``image_NN`` index (or rewrite ``meta.json``) mid-way through a
    post-processing rewrite of the same trajectory folder.
    """
    import verl_omni.tools.image_gen as image_gen

    traj_dir = tmp_path / "rollout_images" / "step_000001" / "sample_3.00"
    assert traj_locking.traj_dir_lock(traj_dir) is traj_locking.traj_dir_lock(traj_dir)
    # ``image_gen`` no longer defines its own lock domain.
    assert not hasattr(image_gen, "_traj_dir_lock")
    assert not hasattr(image_gen, "_traj_dir_exclusive")


def test_traj_dir_exclusive_blocks_a_second_writer(tmp_path):
    """A writer holding the folder lock keeps another writer out until it exits."""
    traj_dir = tmp_path / "rollout_images" / "step_000001" / "sample_4.00"
    traj_dir.mkdir(parents=True)
    order: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    def _holder():
        with traj_locking.traj_dir_exclusive(traj_dir):
            order.append("first-in")
            entered.set()
            release.wait(timeout=5)
            order.append("first-out")

    def _waiter():
        entered.wait(timeout=5)
        with traj_locking.traj_dir_exclusive(traj_dir):
            order.append("second-in")

    first = threading.Thread(target=_holder)
    second = threading.Thread(target=_waiter)
    first.start()
    second.start()
    assert entered.wait(timeout=5)
    # The second writer must still be blocked while the first holds the lock.
    time.sleep(0.2)
    assert "second-in" not in order
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)
    assert order == ["first-in", "first-out", "second-in"]


def test_manager_runs_val_holdout_once_per_step(monkeypatch):
    """Holdouts must not repeat within a step, and never touch the val partition."""
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: None)
    monkeypatch.setattr(omni_agent_loop, "_stamp_scorer_knobs", lambda batch, config: None)
    dispatched: list = []
    monkeypatch.setattr(
        omni_agent_loop.AgentLoopManager, "generate_sequences", lambda self, batch: dispatched.append(batch)
    )

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager._val_viz_provider = SimpleNamespace(build_batch=lambda *args, **kwargs: SimpleNamespace())
    manager._val_viz_logged_steps = set()
    manager._val_viz_manager = SimpleNamespace()
    manager._monitor_tokenizer = SimpleNamespace(eos_token_id=2, pad_token_id=0)
    manager.config = OmegaConf.create({})

    OmniAgentLoopManager._maybe_run_val_viz(manager, 5, dedicated_manager=False)
    OmniAgentLoopManager._maybe_run_val_viz(manager, 5, dedicated_manager=False)
    OmniAgentLoopManager._maybe_run_val_viz(manager, 6, dedicated_manager=False)

    assert len(dispatched) == 2
    assert manager._val_viz_logged_steps == {5, 6}


def test_val_holdouts_split_reflect_from_plan(monkeypatch):
    """9001/9003 must run reflect and 9002/9004 plan, each with a usable reference.

    The holdout labels were always right, but the loop never read ``task_type``, so the
    plan cases ran the reflect machinery. The plan references must also be per-part —
    plan mode sends the whole list as one prompt, so an item is a part of that
    description rather than a render that restates the ones before it — and
    ``expected_num_images`` still records the source slot count.
    """
    monkeypatch.setenv("AGENTIC_VAL_VIZ", "1")
    provider = resolve_agentic_val_viz_provider()
    batch = provider.build_batch(0, eos_token_id=2, pad_token_id=0)

    from verl_omni.utils.dataset.visual_reflection import build_unicot_agentic_rl

    expected_types = {9001: "reflect", 9002: "plan", 9003: "reflect", 9004: "plan"}
    seen: dict[int, str] = {}
    for index, reward_model, raw_prompt in zip(
        batch.non_tensor_batch["index"],
        batch.non_tensor_batch["reward_model"],
        batch.non_tensor_batch["raw_prompt"],
        strict=True,
    ):
        ground_truth = dict(reward_model)["ground_truth"]
        system_prompt = list(raw_prompt)[0]["content"]
        index = int(index)
        task_type = ground_truth["task_type"]
        seen[index] = task_type
        if task_type == "plan":
            references = list(ground_truth["reference_subtasks"])
            assert len(references) >= 2, index
            assert ground_truth["expected_num_images"] == len(references), index
            # Per-part: no reference repeats an earlier one.
            assert references[0] not in references[-1], index
            assert system_prompt == build_unicot_agentic_rl.PLAN_SYSTEM_PROMPT, index
        else:
            assert "reference_subtasks" not in ground_truth, index
            assert ground_truth["expected_num_images"] == 1, index
            assert system_prompt == build_unicot_agentic_rl.REFLECT_SYSTEM_PROMPT, index

    assert seen == expected_types
