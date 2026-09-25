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
"""CPU tests for OmniAgentLoop wiring and the dump helpers it delegates to."""

from __future__ import annotations

import asyncio
import re
from types import SimpleNamespace

import numpy as np
import pytest
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

import verl_omni  # noqa: F401
from verl_omni.agent_loop import omni_agent_loop
from verl_omni.agent_loop.omni_agent_loop import OmniAgentLoopManager, OmniAgentLoopWorker
from verl_omni.tools.trajectory import (
    active_trajectory_relpath,
    active_user_prompt,
    reset_active_trajectory_relpath,
    set_active_trajectory_relpath,
)
from verl_omni.utils.agentic.image_gen_rollout_dump import discard_invalid_rollouts
from verl_omni.utils.agentic.image_gen_rollout_parse import (
    ADVANTAGE_CUE,
    ADVANTAGE_ENV,
    ADVANTAGE_POLICY,
    ADVANTAGE_POLICY_AND_CUE,
    advantage_class,
    extract_generate_image_prompts,
    extract_tool_calls,
    split_env_blob,
    split_rollout_turns,
    tool_call_order,
    turn_kind,
    turn_record,
)


def test_worker_stamps_rollout_kwargs_and_resets_context(monkeypatch):
    captured: dict = {"kwargs": None}

    async def _parent_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        del sampling_params, trajectory, agent_name, trace
        captured["kwargs"] = dict(kwargs)
        captured["relpath_during_run"] = active_trajectory_relpath.get()
        captured["user_prompt_during_run"] = active_user_prompt.get()
        return "ok"

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", _parent_run)
    worker = OmniAgentLoopWorker.__new__(OmniAgentLoopWorker)
    from verl_omni.pipelines.agentllm_grpo.agent_loop import _IMAGE_GEN_FUNCTION_TOOLS

    assert _IMAGE_GEN_FUNCTION_TOOLS.is_file()

    prior_path = set_active_trajectory_relpath("prior/path")
    prior_prompt = active_user_prompt.set("prior prompt")

    async def _run_then_read_context():
        result = await OmniAgentLoopWorker._run_agent_loop(
            worker,
            {},
            {"step": 7, "sample_index": 3, "rollout_n": 1, "validate": False},
            agent_name="image_gen_tool_agent",
            raw_prompt=[{"role": "user", "content": "draw a cafe poster"}],
        )
        return result, active_trajectory_relpath.get(), active_user_prompt.get()

    try:
        result, path_after, prompt_after = asyncio.run(_run_then_read_context())
    finally:
        active_user_prompt.reset(prior_prompt)
        reset_active_trajectory_relpath(prior_path)

    assert result == "ok"
    assert captured["kwargs"]["_agentic_step"] == 7
    assert captured["kwargs"]["_agentic_validate"] is False
    assert captured["kwargs"]["_agentic_trajectory_relpath"] == "step_000007/sample_3.01"
    assert captured["relpath_during_run"] == "step_000007/sample_3.01"
    assert captured["user_prompt_during_run"] == "draw a cafe poster"
    assert path_after == "prior/path"
    assert prompt_after == "prior prompt"


def _binder_config(*, loop_name, tool_format=None, tool_path=None):
    """Minimal config carrying only the keys a recipe config binder reads."""
    from omegaconf import OmegaConf

    multi_turn = {}
    if tool_format is not None:
        multi_turn["format"] = tool_format
    if tool_path is not None:
        multi_turn["function_tool_path"] = tool_path
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "agent": {"default_agent_loop": loop_name},
                    "multi_turn": multi_turn,
                }
            }
        }
    )


def test_recipe_config_binders_are_resolved_by_loop_name():
    """The generic Omni worker must not name any one recipe's tools itself.

    Regression: ``OmniAgentLoopMixin`` compared ``default_agent_loop`` against the
    literal ``"image_gen_tool_agent"`` and set ``function_tool_path`` and
    ``multi_turn.format`` inline, which put the image-gen recipe inside the shared
    worker. The pipeline owns that binder now and the worker only looks it up.
    """
    import inspect

    from verl_omni.agent_loop.agent_loop_config import AGENT_LOOP_CONFIG_BINDERS
    from verl_omni.pipelines.agentllm_grpo.agent_loop import (
        _IMAGE_GEN_FUNCTION_TOOLS,
        _IMAGE_GEN_TOOL_FORMAT,
        bind_image_gen_tool_agent,
    )

    assert AGENT_LOOP_CONFIG_BINDERS["image_gen_tool_agent"] is bind_image_gen_tool_agent

    # Unset keys get this recipe's tools and tool-call format.
    config = _binder_config(loop_name="image_gen_tool_agent")
    bind_image_gen_tool_agent(config)
    multi_turn = config.actor_rollout_ref.rollout.multi_turn
    assert multi_turn.function_tool_path == str(_IMAGE_GEN_FUNCTION_TOOLS)
    assert multi_turn.format == _IMAGE_GEN_TOOL_FORMAT

    # An explicit Hydra override still wins.
    config = _binder_config(loop_name="image_gen_tool_agent", tool_format="custom", tool_path="/tmp/other_tools.py")
    bind_image_gen_tool_agent(config)
    multi_turn = config.actor_rollout_ref.rollout.multi_turn
    assert multi_turn.function_tool_path == "/tmp/other_tools.py"
    assert multi_turn.format == "custom"

    # The generic worker holds no recipe name, so nothing else can be specialised.
    source = inspect.getsource(omni_agent_loop.OmniAgentLoopMixin)
    assert "image_gen_tool_agent" not in source
    assert "_AGENTIC_FUNCTION_TOOLS" not in source


def test_manager_dumps_before_discarding_invalid_rollouts(monkeypatch):
    from omegaconf import OmegaConf

    order: list[str] = []
    monkeypatch.setattr(AgentLoopManager, "generate_sequences", lambda self, prompts: prompts.output)
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: order.append("dump") or kwargs)
    monkeypatch.setattr(omni_agent_loop, "discard_invalid_rollouts", lambda output: order.append("discard") or output)
    monkeypatch.setattr(omni_agent_loop, "AgenticRewardMetrics", SimpleNamespace(aggregate=lambda batch: {}))

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager._monitor_tokenizer = object()
    manager.config = OmegaConf.create({"agentic_image_gen": {"vllm_url": "http://cli", "good_enough_threshold": 0.55}})
    output = SimpleNamespace(non_tensor_batch={"extra_info": np.array([{"w_tool_call": 0.1}], dtype=object)})
    prompts = SimpleNamespace(
        meta_info={"global_steps": 4},
        non_tensor_batch={"extra_info": np.array([{"w_tool_call": 0.1}], dtype=object)},
        output=output,
    )
    assert OmniAgentLoopManager.generate_sequences(manager, prompts) is output
    assert order == ["dump", "discard"]
    extra = output.non_tensor_batch["extra_info"][0]
    assert extra["vllm_url"] == "http://cli"
    assert extra["good_enough_threshold"] == 0.55
    assert extra["w_tool_call"] == 0.1


def test_inbound_stamp_reaches_compute_score_kwargs(monkeypatch):
    """Pinned ``AgentLoopWorker._compute_score`` reads extra_info from inbound kwargs.

    Default RayPPOTrainer enables ``agent_reward_loop`` with no RM, so scoring
    runs during worker generate — not on the concatenated manager output.
    Streaming ``_postprocess`` does not copy ``input_non_tensor_batch``.
    """
    from omegaconf import OmegaConf

    captured: dict = {}

    def _parent_generate(self, prompts):
        del self
        # Same construction as pinned AgentLoopWorker.generate_sequences (fefb0802).
        kwargs = {k: v[0] for k, v in prompts.non_tensor_batch.items() if k != "__do_sample__"}
        captured["kwargs"] = kwargs
        captured["remote_extra"] = np.array([kwargs["extra_info"]])[0]
        # Simulate streaming postprocess: no copy of input extra_info.
        return SimpleNamespace(non_tensor_batch={"__num_turns__": np.array([1])}, meta_info={})

    monkeypatch.setattr(AgentLoopManager, "generate_sequences", _parent_generate)
    monkeypatch.setattr(omni_agent_loop, "dump_raw_rollouts", lambda **kwargs: None)
    monkeypatch.setattr(omni_agent_loop, "discard_invalid_rollouts", lambda output: output)
    monkeypatch.setattr(omni_agent_loop, "AgenticRewardMetrics", SimpleNamespace(aggregate=lambda batch: {}))

    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager._monitor_tokenizer = object()
    manager.config = OmegaConf.create({"agentic_image_gen": {"vllm_url": "http://cli", "good_enough_threshold": 0.55}})
    prompts = SimpleNamespace(
        meta_info={"global_steps": 4},
        non_tensor_batch={"extra_info": np.array([{"w_tool_call": 0.1}], dtype=object)},
    )
    output = OmniAgentLoopManager.generate_sequences(manager, prompts)
    extra = captured["kwargs"]["extra_info"]
    assert extra["vllm_url"] == "http://cli"
    assert extra["good_enough_threshold"] == 0.55
    assert extra["w_tool_call"] == 0.1
    assert captured["remote_extra"]["vllm_url"] == "http://cli"
    assert captured["remote_extra"]["good_enough_threshold"] == 0.55
    # Output stamp still fills knobs after streaming postprocess dropped extra_info.
    out_extra = output.non_tensor_batch["extra_info"][0]
    assert out_extra["vllm_url"] == "http://cli"
    assert out_extra["good_enough_threshold"] == 0.55


def test_manager_v1_tensordict_dispatches_without_meta_info(monkeypatch):
    import torch
    from tensordict import TensorDict

    chunks = []

    class _Remote:
        def remote(self, chunk):
            chunks.append(chunk)
            return "ref"

    class _Worker:
        generate_sequences = _Remote()

    monkeypatch.setattr(omni_agent_loop.ray, "get", lambda refs: refs)
    manager = OmniAgentLoopManager.__new__(OmniAgentLoopManager)
    manager.agent_loop_workers = [_Worker()]
    batch = TensorDict({"input_ids": torch.zeros(2, 1, dtype=torch.int64)}, batch_size=[2])
    assert OmniAgentLoopManager.generate_sequences(manager, batch) is None
    assert len(chunks) == 1
    assert chunks[0].batch_size == torch.Size([2])


def test_stamp_scorer_knobs_does_not_override_row_values():
    from omegaconf import OmegaConf

    output = SimpleNamespace(
        non_tensor_batch={"extra_info": np.array([{"vllm_url": "http://row", "w_done": 0.2}], dtype=object)}
    )
    omni_agent_loop._stamp_scorer_knobs(
        output,
        OmegaConf.create({"agentic_image_gen": {"vllm_url": "http://cli", "good_enough_threshold": 0.55}}),
    )
    extra = output.non_tensor_batch["extra_info"][0]
    assert extra["vllm_url"] == "http://row"
    assert extra["good_enough_threshold"] == 0.55
    assert extra["w_done"] == 0.2


def test_turn_kind_stop_rewrite_and_continue():
    judge_no = "VL judge on the last generated image:\n  good_enough =NO\n  agentic_judge ok=1"
    judge_yes = judge_no.replace("good_enough =NO", "good_enough =YES")
    continue_cue = "Reflection: rewrite next. agentic_forced_reflection=1"
    stop_cue = "Reflection: Stop. agentic_stop_decision_required=1 agentic_forced_reflection=1"
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "lion"}}\n</tool_call>'
    done = "Reflection: The image meets the original request. Done.<|im_end|>"
    assert turn_kind(done, judge_yes, stop_cue) == "agent_done_after_forced_reflection"
    assert turn_kind(rewrite, judge_no, continue_cue) == (
        "agent_rewrite_after_forced_reflection_then_call_generate_image"
    )
    assert turn_kind("", judge_yes, stop_cue) == "forced_reflection_stop_cue"
    assert turn_kind(done, judge_no, "") == "agent_reflection_done"


def test_advantage_class_names_the_mask_composition():
    """The class is the trainer's ``response_mask`` view of a turn, not a text read."""
    assert advantage_class(policy_tokens=None, injected_cue=False) == ""
    assert advantage_class(policy_tokens=None, injected_cue=True) == ""
    assert advantage_class(policy_tokens=7, injected_cue=False) == ADVANTAGE_POLICY
    assert advantage_class(policy_tokens=7, injected_cue=True) == ADVANTAGE_POLICY_AND_CUE
    assert advantage_class(policy_tokens=0, injected_cue=True) == ADVANTAGE_CUE
    assert advantage_class(policy_tokens=0, injected_cue=False) == ADVANTAGE_ENV


def test_turn_kind_checks_its_claim_against_the_mask():
    """A label that claims mask=0 tokens must not be put on a mask=1 turn.

    The cue labels describe harness text the optimizer never saw and the call labels
    describe policy tokens it did see, so the same transcript text has to be labelled
    differently depending on the mask, and the mask wins over the text.
    """
    cue = "Reflection: rewrite next. agentic_forced_reflection=1"
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "v2"}}\n</tool_call>'

    # Injected cue plus a policy rewrite: the label records both.
    assert turn_kind(rewrite, "obs", cue, advantage=ADVANTAGE_POLICY_AND_CUE) == (
        "agent_rewrite_after_forced_reflection_then_call_generate_image"
    )
    # The identical text on a policy-only turn must not claim an injection happened.
    assert turn_kind(rewrite, "obs", cue, advantage=ADVANTAGE_POLICY) == "call_generate_image"
    # A cue-only turn trains nothing: the cue family is what its mask entitles it to.
    assert turn_kind("", "obs", cue, advantage=ADVANTAGE_CUE) == "forced_reflection_continue"

    done = "Reflection: The image meets the original request. Done.<|im_end|>"
    stop_cue = "Reflection: Stop. agentic_stop_decision_required=1 agentic_forced_reflection=1"
    assert turn_kind(done, "VL judge ok=1", stop_cue, advantage=ADVANTAGE_POLICY_AND_CUE) == (
        "agent_done_after_forced_reflection"
    )
    # With no injected cue the same ``Done.`` is the policy ending on its own, and
    # calling it a forced stop would credit the harness with the policy's decision.
    assert turn_kind(done, "VL judge ok=1", "", advantage=ADVANTAGE_POLICY) == "agent_reflection_done"

    # An unknown advantage keeps the legacy text sniffing, so older callers and
    # hand-written transcripts still label.
    assert turn_kind(rewrite, "obs", cue) == "agent_rewrite_after_forced_reflection_then_call_generate_image"
    assert turn_kind("", "obs", cue) == "forced_reflection_continue"


def test_turn_kind_labels_the_first_tool_call():
    """A generate+judge turn is a generate turn with a dropped trailing call.

    Regression: ``turn_kind`` tested ``judge_image`` before ``generate_image``, so a
    turn whose *first* call was ``generate_image`` was labelled ``call_judge_image``
    and read as a judge-first rollout in the dumps.
    """
    gen_only = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "skull"}}\n</tool_call>'
    judge_only = (
        '<tool_call>\n{"name": "judge_image", "arguments": {"user_request": "same as user message"}}\n</tool_call>'
    )
    gen_then_judge = f"{gen_only}\n{judge_only}"
    judge_then_gen = f"{judge_only}\n{gen_only}"

    assert tool_call_order(gen_then_judge) == ["generate_image", "judge_image"]
    assert tool_call_order(judge_then_gen) == ["judge_image", "generate_image"]
    assert tool_call_order("no tools here") == []

    # The first call names the turn; trailing calls are appended, never promoted.
    assert turn_kind(gen_then_judge, "prompt", "") == "call_generate_image_then_call_judge_image"
    assert turn_kind(judge_then_gen, "prompt", "") == "call_judge_image_then_call_generate_image"
    # Single-call labels are unchanged.
    assert turn_kind(gen_only, "prompt", "") == "call_generate_image"
    assert turn_kind(judge_only, "prompt", "") == "call_judge_image"


def test_plan_turn_labels_do_not_depend_on_the_task_type():
    """The label is a claim about the text, and both corpora are labelled the same way.

    Regression: the ``plan``/``plan_then_*`` labels were gated on ``task_type == "plan"``
    so the dump carried a protocol the loop no longer implements. The gate is the
    numbered list now, which is also what the loop's reopen checks — so a plan turn and
    a plan-labelled turn are the same turn by construction.
    """
    plan = "Plan:\n1. A librarian floats in an underwater cave library with fish nearby.\n"
    call = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a poster"}}\n</tool_call>'

    for task_type in ("reflect", "plan", ""):
        assert turn_kind(plan, "prompt", "", task_type=task_type) == "plan"
        assert turn_kind(f"{plan}{call}", "prompt", "", task_type=task_type) == "plan_then_call_generate_image"

    # A turn with no numbered list is never a plan turn, whatever the row says it is.
    assert turn_kind(call, "prompt", "", task_type="plan") == "call_generate_image"


def test_extract_generate_image_prompts_hermes_and_qwen():
    hermes = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat"}}\n</tool_call>'
    qwen = "<tool_call>\n<function=generate_image>\n<parameter=prompt>\na dog\n</parameter>\n</function>\n</tool_call>"
    assert extract_generate_image_prompts(hermes) == ["a cat"]
    assert extract_generate_image_prompts(qwen) == ["a dog"]


def test_extract_tool_calls_parses_hermes_and_qwen_arguments():
    hermes = (
        '<tool_call>\n{"name": "judge_image", "arguments": '
        '{"user_request": "same as user message", "image_prompt": "a cat"}}\n</tool_call>'
    )
    qwen = (
        "<tool_call>\n<function=judge_image>\n"
        "<parameter=user_request>\nsame as user message\n</parameter>\n"
        "<parameter=image_prompt>\na cat\n</parameter>\n</function>\n</tool_call>"
    )
    for decode in (hermes, qwen):
        assert extract_tool_calls(decode) == [
            {"name": "judge_image", "arguments": {"user_request": "same as user message", "image_prompt": "a cat"}}
        ]
    # A body that is not valid JSON is skipped rather than misread.
    assert extract_tool_calls("<tool_call>{not json}</tool_call>") == []
    assert extract_tool_calls("no tools here") == []


def test_turn_record_exposes_the_accepted_tool_prompt():
    """``tool_prompt`` is the prompt string the accepted tool call carries.

    Regression for the rewrite turns: ``turn_prompt`` carries the whole chat
    template (identical on every turn) and ``turn_obs`` only the judge feedback, so
    the rewritten prompt was visible nowhere but the escaped JSON in ``decode``.
    """
    first = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat poster"}}\n</tool_call>'
    rewrite = (
        '<tool_call>\n{"name": "generate_image", "arguments": '
        '{"prompt": "a cat poster, the title is legible and correctly spelled"}}\n</tool_call>'
    )
    judge = (
        '<tool_call>\n{"name": "judge_image", "arguments": '
        '{"user_request": "x", "image_prompt": "ECHO_OF_LAST"}}\n</tool_call>'
    )
    judge_then_gen = f"{judge}\n{first}"

    record = turn_record(turn=1, turn_prompt="<|im_start|>system\nblah", response="", decode=first)
    assert (record["tool_name"], record["tool_prompt"]) == ("generate_image", "a cat poster")

    rewritten = turn_record(turn=3, turn_prompt="<|im_start|>system\nblah", response="", decode=rewrite)
    assert rewritten["turn_prompt"] == record["turn_prompt"]
    assert rewritten["tool_prompt"] != record["tool_prompt"]
    assert rewritten["tool_prompt"].endswith("legible and correctly spelled")

    # Judge turns echo the prompt they judged; ``tool_name`` says which is which.
    judged = turn_record(turn=2, turn_prompt="obs", response="", decode=judge)
    assert (judged["tool_name"], judged["tool_prompt"]) == ("judge_image", "ECHO_OF_LAST")
    # Only the *first* call runs, so a dropped trailing generate must not win.
    assert turn_record(turn=4, turn_prompt="obs", response="", decode=judge_then_gen)["tool_prompt"] == "ECHO_OF_LAST"

    # The judge schema suggests the literal shortcut ``image_prompt="last"``; report
    # the placeholder verbatim so dumps show the model skipped the echo.
    shortcut = '<tool_call>\n{"name": "judge_image", "arguments": {"image_prompt": "last"}}\n</tool_call>'
    assert turn_record(turn=5, turn_prompt="obs", response="", decode=shortcut)["tool_prompt"] == "last"

    # A turn with no call, or a call that carries no prompt, leaves the field empty.
    assert turn_record(turn=6, turn_prompt="obs", response="", decode="Done.")["tool_name"] == ""
    silent = '<tool_call>\n{"name": "judge_image", "arguments": {}}\n</tool_call>'
    assert turn_record(turn=7, turn_prompt="obs", response="", decode=silent)["tool_prompt"] == ""


def test_split_env_blob_and_rollout_turns():
    blob = (
        "<tool_response>\nagentic_tool ok=1 path=/tmp/x.png\n</tool_response>\n"
        "Reflection: rewrite next. agentic_forced_reflection=1"
    )
    prompt, response = split_env_blob(blob)
    assert "agentic_tool ok=1" in prompt
    assert response.startswith("Reflection:")

    class _Tok:
        pad_token_id = 0

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(chr(64 + int(x)) for x in ids)

    turns = split_rollout_turns([1, 2, 3, 4], [1, 1, 0, 0], _Tok())
    assert [turn["decode"] for turn in turns] == ["AB", ""]
    assert turns[1]["turn_prompt"] == "CD"


def test_splitter_reports_the_mask_each_turn_owns():
    """Per-turn counts are the mask itself: policy is mask=1, env is mask=0.

    The counts have to partition the response exactly, because they are what a reader
    uses to see how much of a rollout the optimizer actually trained on.
    """
    pieces = {
        10: '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "v1"}}\n</tool_call>',
        11: " plus a sampled continuation",
        # One contiguous mask=0 run holding the judge observation *and* the injected
        # cue, which is exactly what ``split_env_blob`` has to separate.
        12: (
            "<tool_response>\nVL judge on the last generated image:\n  agentic_judge ok=1 good_enough =NO\n"
            "</tool_response>\nReflection: rewrite next. agentic_forced_reflection=1"
        ),
        13: '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "v2"}}\n</tool_call>',
        14: "Reflection: Stop. agentic_stop_decision_required=1 agentic_forced_reflection=1",
    }

    class _PieceTok:
        pad_token_id = 0

        @staticmethod
        def decode(ids, skip_special_tokens=False):
            del skip_special_tokens
            return "".join(pieces.get(int(token), "") for token in ids)

    ids = [10, 11, 12, 13, 14]
    mask = [1, 1, 0, 1, 0]
    turns = split_rollout_turns(ids, mask, _PieceTok())

    assert [(turn["policy_tokens"], turn["env_tokens"]) for turn in turns] == [(2, 0), (1, 1), (0, 1)]
    assert [turn["turn_advantage"] for turn in turns] == [
        ADVANTAGE_POLICY,
        ADVANTAGE_POLICY_AND_CUE,
        ADVANTAGE_CUE,
    ]
    assert [turn["injected_cue"] for turn in turns] == [False, True, True]
    # The two counts partition the response, so nothing is double-counted or lost.
    assert sum(turn["policy_tokens"] for turn in turns) == sum(mask)
    assert sum(turn["env_tokens"] for turn in turns) == len(mask) - sum(mask)


def test_discard_invalid_rollouts_zeros_mask_but_restores_if_all_invalid():
    class _MaskRow:
        def __init__(self, vals):
            self.vals = list(vals)

        def zero_(self):
            self.vals = [0] * len(self.vals)

        def any(self):
            return any(self.vals)

        def copy_(self, other):
            self.vals = list(other.vals)

    class _Mask:
        def __init__(self, rows):
            self.rows = rows
            self.shape = (len(rows),)

        def __getitem__(self, i):
            return self.rows[i]

        def clone(self):
            return _Mask([_MaskRow(row.vals) for row in self.rows])

        def any(self):
            return any(row.any() for row in self.rows)

        def copy_(self, other):
            for dst, src in zip(self.rows, other.rows, strict=True):
                dst.copy_(src)

    mask = _Mask([_MaskRow([1, 1]), _MaskRow([1, 1])])
    discard_invalid_rollouts(
        SimpleNamespace(batch={"response_mask": mask}, non_tensor_batch={"rollout_valid": np.array([1, 0])})
    )
    assert mask.rows[0].vals == [1, 1]
    assert mask.rows[1].vals == [0, 0]

    all_invalid = _Mask([_MaskRow([1, 1]), _MaskRow([1, 0])])
    discard_invalid_rollouts(
        SimpleNamespace(batch={"response_mask": all_invalid}, non_tensor_batch={"rollout_valid": np.array([0, 0])})
    )
    assert all_invalid.rows[0].vals == [1, 1]
    assert all_invalid.rows[1].vals == [1, 0]


def test_tq_sessions_get_distinct_artifact_relpaths():
    """All ``rollout.n`` sessions of one sample must get their own folder.

    Regression: V1 TransferQueue expands ``rollout.n`` inside the worker
    (``session_id``) *after* ``get_trajectory_info`` has already inferred
    ``rollout_n`` from repeated batch indices. Every sibling therefore reported
    ``rollout_n=0`` and collapsed onto ``sample_<index>.00``, which collapsed one
    trajectory JSON (last writer wins), merged every sibling's PNGs into one
    folder (so ``image_paths`` became the union and ``image_paths_in_obs`` only a
    subset), and gave all siblings one ``rollout_id`` — letting a sibling's image
    reach ``judge_image`` and a sibling's registry clear abort it.
    """
    from verl_omni.tools.trajectory import build_trajectory_relpath

    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    assert omni_agent_loop.OmniAgentLoopWorkerTQImpl._AGENTIC_ROLLOUT_N_FROM_SESSION_ID is True

    # Exactly what ``AgentLoopWorkerTQ._run_prompt`` hands each sibling: same
    # ``trajectory`` dict, differing ``session_id``.
    trajectory = {"step": 1, "sample_index": 2086, "rollout_n": 0, "validate": False}
    relpaths = [
        build_trajectory_relpath(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=omni_agent_loop.OmniAgentLoopWorkerTQImpl._agentic_rollout_n(
                worker, trajectory, {"session_id": session_id}
            ),
            validate=trajectory["validate"],
        )
        for session_id in range(8)
    ]
    assert relpaths == [f"step_000001/sample_2086.{n:02d}" for n in range(8)]
    assert len(set(relpaths)) == 8

    # The per-relpath rollout id is what scopes the artifact registry / judge lookup.
    from verl_omni.tools.trajectory.paths import rollout_id_from_relpath

    assert len({rollout_id_from_relpath(relpath) for relpath in relpaths}) == 8


def test_datapath_rollout_n_wins_over_stray_session_id():
    """DataProto rows are pre-expanded, so ``rollout_n`` must not be overridden."""
    data_proto_worker = OmniAgentLoopWorker.__new__(OmniAgentLoopWorker)
    assert OmniAgentLoopWorker._AGENTIC_ROLLOUT_N_FROM_SESSION_ID is False
    trajectory = {"step": 0, "sample_index": 7, "rollout_n": 3, "validate": False}
    # Even if a batch column happens to be named ``session_id``, rollout_n wins.
    assert OmniAgentLoopWorker._agentic_rollout_n(data_proto_worker, trajectory, {"session_id": 5}) == 3
    assert OmniAgentLoopWorker._agentic_rollout_n(data_proto_worker, trajectory, {}) == 3


def test_tq_rollout_n_falls_back_when_session_id_absent():
    """A missing ``session_id`` must degrade to the old behaviour, not crash."""
    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    trajectory = {"step": 0, "sample_index": 11, "rollout_n": 2, "validate": False}
    assert omni_agent_loop.OmniAgentLoopWorkerTQImpl._agentic_rollout_n(worker, trajectory, {}) == 2


def test_tq_run_agent_loop_wires_session_id_into_relpath(monkeypatch):
    """The TQ ``_run_agent_loop`` must bind ``sample_<index>.<session_id>``."""
    captured: list[str] = []

    async def _parent_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        del sampling_params, trajectory, agent_name, trace
        captured.append(kwargs["_agentic_trajectory_relpath"])
        return "ok"

    monkeypatch.setattr(omni_agent_loop._AgentLoopWorkerTQImpl, "_run_agent_loop", _parent_run)
    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    trajectory = {"step": 4, "sample_index": 2086, "rollout_n": 0, "validate": False}

    async def _run_all():
        for session_id in range(8):
            await omni_agent_loop.OmniAgentLoopWorkerTQImpl._run_agent_loop(
                worker,
                {},
                trajectory,
                agent_name="image_gen_tool_agent",
                session_id=session_id,
                raw_prompt=[{"role": "user", "content": "draw a cafe poster"}],
            )

    asyncio.run(_run_all())
    assert captured == [f"step_000004/sample_2086.{n:02d}" for n in range(8)]


def _set_trace_config(*, backend, max_samples_per_worker):
    """Point the ``RolloutTraceConfig`` singleton at a backend without importing it.

    Bypasses ``init`` on purpose: ``init`` would import weave/mlflow and open a
    network client just to exercise the trace-selection arithmetic.
    """
    from verl.utils.rollout_trace import RolloutTraceConfig

    config = RolloutTraceConfig.get_instance()
    config.backend = backend
    config.max_samples_per_step_per_worker = max_samples_per_worker
    config._initialized = True

    def restore():
        # Drops the instance, so the class-level defaults (backend=None) come back.
        RolloutTraceConfig.reset()

    return restore


def test_tq_trace_selection_mirrors_the_dataproto_rule():
    """``max_samples_per_step_per_worker`` picks a subset of *unique* indices.

    Regression: ``AgentLoopWorkerTQ.generate_sequences`` hardcoded
    ``trace_this_sample = False`` behind a ``TODO(wuxibin)``, so
    ``rollout.trace.backend`` silently produced no traces at all on the
    V1/TransferQueue path even with a backend configured.
    """
    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    restore = _set_trace_config(backend="weave", max_samples_per_worker=2)
    try:
        selected = worker._agentic_select_traced_samples({"index": np.array([5, 5, 6, 7, 8])})
    finally:
        restore()

    # 4 unique indices, cap 2: exactly two distinct indices, all python ints so
    # ``trajectory["sample_index"]`` (a numpy/torch scalar) can match by value.
    assert isinstance(selected, set)
    assert len(selected) == 2
    assert selected <= {5, 6, 7, 8}
    assert all(type(value) is int for value in selected)


@pytest.mark.parametrize(
    ("backend", "max_samples_per_worker", "index"),
    [
        (None, 2, [1, 2, 3]),  # tracing off: nothing to select
        ("weave", None, [1, 2, 3]),  # no cap: parent traces every row already
        ("weave", 3, [1, 2, 3]),  # cap covers every unique index: same as no cap
        ("weave", 9, [1, 2, 3]),
    ],
)
def test_tq_trace_selection_leaves_the_parent_flag_alone(backend, max_samples_per_worker, index):
    """``None`` means "trace every row", so the parent's flag must be untouched."""
    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    restore = _set_trace_config(backend=backend, max_samples_per_worker=max_samples_per_worker)
    try:
        assert worker._agentic_select_traced_samples({"index": np.array(index)}) is None
    finally:
        restore()


def test_tq_run_prompt_threads_the_selection_into_the_trace_flag(monkeypatch):
    """Only the selected sample's rows are traced; its siblings all agree."""
    seen: dict[int, bool] = {}

    async def _parent_run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        del self, prompt, sampling_params
        seen[int(trajectory["sample_index"])] = trace

    monkeypatch.setattr(omni_agent_loop._AgentLoopWorkerTQImpl, "_run_prompt", _parent_run_prompt)
    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    worker._agentic_traced_samples = {3}

    for sample_index in (3, 3, 8, 9):
        coroutine = omni_agent_loop.OmniAgentLoopWorkerTQImpl._run_prompt(
            worker, {}, {}, trajectory={"sample_index": sample_index}
        )
        # The parent wraps this call in ``asyncio.create_task``, so it must stay
        # awaitable while the decision is taken synchronously at call time.
        assert asyncio.iscoroutine(coroutine)
        asyncio.run(coroutine)

    assert seen == {3: True, 8: False, 9: False}


def test_tq_run_prompt_reads_the_selection_synchronously(monkeypatch):
    """The trace decision must not be deferred to when the task first runs.

    ``_run_prompt`` is intentionally a plain function: the parent creates one task
    per row in a tight loop and then clears ``_agentic_traced_samples``, so an
    ``async`` body would read the stash after it was already reset.
    """
    captured: list[bool] = []

    async def _parent_run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        del self, prompt, sampling_params, trajectory
        captured.append(trace)

    monkeypatch.setattr(omni_agent_loop._AgentLoopWorkerTQImpl, "_run_prompt", _parent_run_prompt)
    assert not asyncio.iscoroutinefunction(omni_agent_loop.OmniAgentLoopWorkerTQImpl._run_prompt)

    worker = omni_agent_loop.OmniAgentLoopWorkerTQImpl.__new__(omni_agent_loop.OmniAgentLoopWorkerTQImpl)
    worker._agentic_traced_samples = {4}

    async def _dispatch_then_clear():
        coroutines = [
            omni_agent_loop.OmniAgentLoopWorkerTQImpl._run_prompt(
                worker, {}, {}, trajectory={"sample_index": sample_index}
            )
            for sample_index in (4, 5)
        ]
        worker._agentic_traced_samples = None
        await asyncio.gather(*coroutines)

    asyncio.run(_dispatch_then_clear())
    assert captured == [True, False]


def _drop_notice_setup(monkeypatch, tool_names):
    """Build a loop whose parent executes only the first tool call of a turn."""
    from types import SimpleNamespace

    from verl.experimental.agent_loop.tool_parser import FunctionCall

    from verl_omni.pipelines.agentllm_grpo import agent_loop as mod

    async def _parent(self, agent_data):
        agent_data.messages.append(
            {"role": "tool", "content": "path=/tmp/image_00.png agentic_tool ok=1 images=1 backend=vllm_omni"}
        )
        return mod.AgentState.GENERATING

    async def _merge(self, previous_messages, updated_messages, token_ids, response_mask, *args, **kwargs):
        added = len(updated_messages) - len(previous_messages)
        return (
            SimpleNamespace(token_ids=[*token_ids, *([0] * added)]),
            [*response_mask, *([0] * added)],
            [*(kwargs.get("response_logprobs") or []), *([0.0] * added)],
        )

    monkeypatch.setattr(mod.ToolAgentLoop, "_handle_processing_tools_state", _parent)
    monkeypatch.setattr(mod.ToolAgentLoop, "ct_merge_non_assistant_msg", _merge)
    monkeypatch.setattr(mod, "max_generate_passes", lambda: 99)
    # Hydra knobs are unbound outside a live worker; forced Reflection is not
    # under test here, so take the early return after the drop notice.
    monkeypatch.setattr(mod, "agentic_get_bool", lambda *args, **kwargs: False)

    loop = mod.ImageGenToolAgentLoop.__new__(mod.ImageGenToolAgentLoop)
    loop.max_parallel_calls = 1
    loop.response_length = 4096
    loop.tool_schemas = []
    agent_data = SimpleNamespace(
        tool_calls=[
            FunctionCall(name=name, arguments="{}", tool_call_id=f"call_{index}")
            for index, name in enumerate(tool_names)
        ],
        messages=[],
        prompt_ids=[],
        response_mask=[],
        response_logprobs=[],
        extra_fields={},
    )
    return mod, loop, agent_data


def test_dropped_same_turn_tool_call_gets_a_tool_response_notice(monkeypatch):
    """A same-turn call past ``max_parallel_calls`` must not vanish silently.

    Regression: the parent slices ``tool_calls[:max_parallel_calls]`` and reports
    nothing, so "generate then judge in one message" lost its judge with no tool
    response and no error, while the scorer still docked the rollout.
    """
    mod, loop, agent_data = _drop_notice_setup(monkeypatch, ["generate_image", "judge_image"])

    state = asyncio.run(mod.ImageGenToolAgentLoop._handle_processing_tools_state(loop, agent_data))

    assert state == mod.AgentState.GENERATING
    notices = [
        message
        for message in agent_data.messages
        if message.get("role") == "tool" and "agentic_tool_dropped" in str(message.get("content"))
    ]
    assert len(notices) == 1
    assert notices[0]["tool_call_id"] == "call_1"
    assert "judge_image" in notices[0]["content"]
    assert "not executed" in notices[0]["content"]
    assert agent_data.extra_fields["num_dropped_tool_calls"] == 1
    assert agent_data.extra_fields["dropped_tool_names"] == "judge_image"
    # The notice is an environment observation, so its tokens must not be sampled.
    assert set(agent_data.response_mask) <= {0}


def test_single_tool_call_turn_gets_no_drop_notice(monkeypatch):
    """The common one-call-per-turn rollout must stay byte-identical."""
    mod, loop, agent_data = _drop_notice_setup(monkeypatch, ["generate_image"])

    state = asyncio.run(mod.ImageGenToolAgentLoop._handle_processing_tools_state(loop, agent_data))

    assert state == mod.AgentState.GENERATING
    contents = [str(message.get("content")) for message in agent_data.messages]
    assert not any("agentic_tool_dropped" in content for content in contents)
    assert "num_dropped_tool_calls" not in agent_data.extra_fields


def test_worker_stamps_task_type_from_extra_info(monkeypatch):
    """``task_type`` reaches the loop as a monitoring label.

    ``task_type`` lives in ``extra_info`` and used to stop at the prompt. The loop now
    carries it for the trajectory dump and the reward metrics. It no longer selects a
    protocol: reflect and plan rollouts run the same loop, and only their system prompt
    differs.
    """
    captured: dict = {}

    async def _parent_run(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        del sampling_params, trajectory, agent_name, trace
        captured["kwargs"] = dict(kwargs)
        return "ok"

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", _parent_run)
    worker = OmniAgentLoopWorker.__new__(OmniAgentLoopWorker)

    asyncio.run(
        OmniAgentLoopWorker._run_agent_loop(
            worker,
            {},
            {"step": 0, "sample_index": 9004, "rollout_n": 0, "validate": True},
            agent_name="image_gen_tool_agent",
            raw_prompt=[{"role": "user", "content": "draw a poster"}],
            extra_info={"task_type": "PLAN"},
        )
    )

    assert captured["kwargs"]["_agentic_task_type"] == "plan"
    assert captured["kwargs"]["_agentic_validate"] is True


def _bare_loop(*, task_type: str, validate: bool = False):
    from verl_omni.pipelines.agentllm_grpo import agent_loop as mod

    loop = mod.ImageGenToolAgentLoop.__new__(mod.ImageGenToolAgentLoop)
    loop._agentic_task_type = task_type
    loop._agentic_validate = validate
    loop._agentic_step = 3
    return mod, loop


def test_forced_actions_are_exempt_for_validation_only():
    """Reflect and plan training both get the cues and the substitutions.

    Regression: the gate used to read ``_agentic_protocol() == "plan" or validate``, so
    the two corpora ran different loops — plan was pushed out of the rewrite loop the
    wording of its own prompt assumes, and the two validation reward curves were not
    comparable. Only validation is exempt now: a val rollout must measure the policy
    rather than the curriculum.
    """
    _, reflect_train = _bare_loop(task_type="reflect")
    _, reflect_val = _bare_loop(task_type="reflect", validate=True)
    _, plan_train = _bare_loop(task_type="plan")
    _, plan_val = _bare_loop(task_type="plan", validate=True)
    _, unknown = _bare_loop(task_type="")

    assert reflect_train._agentic_exempt_from_forced_actions() is False
    assert plan_train._agentic_exempt_from_forced_actions() is False
    assert unknown._agentic_exempt_from_forced_actions() is False
    assert reflect_val._agentic_exempt_from_forced_actions() is True
    assert plan_val._agentic_exempt_from_forced_actions() is True


def test_the_loop_never_branches_on_task_type():
    """``task_type`` is a label, not a protocol selector.

    A structural guard rather than a behavioural one: any reintroduced branch would be
    invisible to the behavioural tests until a rollout happened to hit it, which is
    exactly how the plan/reflect split grew back the first time.
    """
    import inspect

    from verl_omni.pipelines.agentllm_grpo import agent_loop as mod

    source = inspect.getsource(mod)
    assert "_agentic_protocol" not in source
    # No comparison of the label against a literal either, however it is spelled.
    assert not re.search(r'task_type["\']?\s*(?:==|!=|in)\s*[\(\{\[]?["\']plan', source)
    assert not re.search(r'["\']plan["\']\s*(?:==|!=)\s*self\._agentic_task_type', source)


def _premature_judge_setup(monkeypatch, *, task_type: str = "reflect", validate: bool = False):
    """Build a loop holding one ``judge_image`` call with no image behind it."""
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    from verl_omni.pipelines.agentllm_grpo import agent_loop as mod

    async def _merge(self, previous_messages, updated_messages, token_ids, response_mask, *args, **kwargs):
        added = len(updated_messages) - len(previous_messages)
        return (
            SimpleNamespace(token_ids=[*token_ids, *([0] * added)]),
            [*response_mask, *([0] * added)],
            [*(kwargs.get("response_logprobs") or []), *([0.0] * added)],
        )

    monkeypatch.setattr(mod.ToolAgentLoop, "ct_merge_non_assistant_msg", _merge)

    mod, loop = _bare_loop(task_type=task_type, validate=validate)
    loop.response_length = 4096
    loop.tool_schemas = []
    request = "一张垂直构图的平面设计海报，背景是纯粹而鲜艳的宝蓝色"
    agent_data = SimpleNamespace(
        tool_calls=[FunctionCall(name="judge_image", arguments="{}", tool_call_id="call_0")],
        messages=[
            {"role": "user", "content": request},
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "judge_image", "arguments": {"user_request": "x"}}\n</tool_call>',
            },
        ],
        prompt_ids=[],
        response_mask=[],
        response_logprobs=[],
        extra_fields={},
    )
    return mod, loop, agent_data


def test_premature_judge_is_refused_with_an_observation(monkeypatch):
    """A judge with no image must be answered, never silently overwritten.

    Regression: the harness replaced the call with
    ``generate_image(prompt=<raw user request>)`` and marked the replacement ``mask=1``.
    The policy's own action vanished, the bare restatement both system prompts forbid was
    trained as the policy's text, and the transcript described the harness rather than
    the model.
    """
    mod, loop, agent_data = _premature_judge_setup(monkeypatch)

    state = asyncio.run(mod.ImageGenToolAgentLoop._agentic_refuse_premature_judge(loop, agent_data))

    assert state == mod.AgentState.GENERATING
    # The policy's call is still in the transcript: nothing was overwritten.
    assistant = [message for message in agent_data.messages if message.get("role") == "assistant"]
    assert len(assistant) == 1
    assert "judge_image" in str(assistant[0].get("content"))
    notices = [
        message
        for message in agent_data.messages
        if message.get("role") == "tool" and "agentic_tool_refused" in str(message.get("content"))
    ]
    assert len(notices) == 1
    assert notices[0]["tool_call_id"] == "call_0"
    assert "no image to judge" in notices[0]["content"]
    assert "generate_image" in notices[0]["content"]
    # The old write-path sent the raw request as the diffusion prompt.
    assert "宝蓝色" not in notices[0]["content"]
    # Nothing ran, so the calls must not reach the processing state.
    assert agent_data.tool_calls == []
    # The notice is environment feedback: its tokens must not enter the advantage.
    assert set(agent_data.response_mask) <= {0}
    assert agent_data.extra_fields["refused_premature_judge"] is True


@pytest.mark.parametrize(("task_type", "validate"), [("reflect", False), ("plan", False), ("reflect", True)])
def test_premature_judge_refusal_is_not_gated_on_protocol_or_validate(monkeypatch, task_type, validate):
    """A refusal is env feedback, so plan and validation rollouts get it too.

    The retired substitution had to be withheld from them because it injected harness
    text as the policy's action. A refusal injects nothing the policy owns, so withholding
    it would only deny those rollouts the recovery a real environment grants.
    """
    mod, loop, agent_data = _premature_judge_setup(monkeypatch, task_type=task_type, validate=validate)

    state = asyncio.run(mod.ImageGenToolAgentLoop._agentic_refuse_premature_judge(loop, agent_data))

    assert state == mod.AgentState.GENERATING
    assert agent_data.extra_fields["refused_premature_judge"] is True


def test_premature_judge_refusal_replaced_the_action_substitution():
    """The call site refuses and the write-path is gone.

    Two properties made the old behavior harmful, and both are asserted here: the refusal
    must not sit behind the forced-action predicate (that would deny plan/val the
    recovery), and the method must not author a replacement action for the policy.
    """
    import inspect

    from verl_omni.pipelines.agentllm_grpo.agent_loop import ImageGenToolAgentLoop

    handler = inspect.getsource(ImageGenToolAgentLoop._handle_generating_state)
    refuse = inspect.getsource(ImageGenToolAgentLoop._agentic_refuse_tool_calls)

    condition_start = handler.index('agentic_get_bool("refuse_premature_judge")')
    condition_end = handler.index("return await self._agentic_refuse_premature_judge", condition_start)
    assert "_agentic_exempt_from_forced_actions" not in handler[condition_start:condition_end]
    # No substitution anywhere on the refusal path: it neither rewrites the assistant
    # span nor builds a call, so it can never train harness text as the policy's own.
    assert "_replace_last_assistant_with_tool_call" not in refuse
    assert "hermes_tool_call" not in refuse
    assert "last_user_text" not in refuse
    assert "_rewrite_premature_judge_to_generate" not in inspect.getsource(ImageGenToolAgentLoop)


#: A first turn that writes the numbered plan and calls nothing.
_PLAN_TURN_TEXT = (
    "Plan:\n"
    "1. A librarian floats in an underwater cave library with fish nearby.\n"
    "2. Add a gold satin gown and filtered light beams from above.\n"
)


def _plan_turn_agent_data(*, tool_calls=(), extra_messages=()):
    messages = [
        {"role": "user", "content": "draw a poster"},
        {"role": "assistant", "content": _PLAN_TURN_TEXT},
        *extra_messages,
    ]
    return SimpleNamespace(
        messages=messages,
        tool_calls=list(tool_calls),
        extra_fields={},
    )


def test_plan_turn_continues_to_the_first_generate():
    """The stock loop ends on a tool-call-free turn; a plan must survive that rule."""
    mod, loop = _bare_loop(task_type="plan")
    agent_data = _plan_turn_agent_data()

    state = loop._agentic_continue_after_plan_turn(agent_data, mod.AgentState.TERMINATED, messages_before=1)

    assert state == mod.AgentState.GENERATING
    # Latched so a policy that only emits prose cannot loop on the reopen.
    assert agent_data.extra_fields["plan_turn_seen"] is True
    assert loop._agentic_continue_after_plan_turn(agent_data, mod.AgentState.TERMINATED, messages_before=2) is None


def test_plan_turn_reopens_for_every_task_type():
    """The reopen is gated by the numbered list, not by the row's label.

    That is what keeps reflect and plan on one code path: it is the system prompt that
    decides which corpus writes prose first, and a reflect rollout calls the tool on turn
    one so this never fires for it.
    """
    mod, loop = _bare_loop(task_type="reflect")
    agent_data = _plan_turn_agent_data()

    state = loop._agentic_continue_after_plan_turn(agent_data, mod.AgentState.TERMINATED, messages_before=1)

    assert state == mod.AgentState.GENERATING


def test_plan_turn_does_not_reopen_without_a_written_plan():
    """A prose-only turn that carries no plan is still a finished rollout."""
    mod, loop = _bare_loop(task_type="plan")
    agent_data = _plan_turn_agent_data()
    agent_data.messages[-1]["content"] = "I am still thinking about the poster."
    assert loop._agentic_continue_after_plan_turn(agent_data, mod.AgentState.TERMINATED, messages_before=1) is None

    # A tool call on the turn means it was never a tool-call-free exit.
    mod, loop = _bare_loop(task_type="plan")
    agent_data = _plan_turn_agent_data(tool_calls=[SimpleNamespace(name="generate_image")])
    assert loop._agentic_continue_after_plan_turn(agent_data, mod.AgentState.TERMINATED, messages_before=1) is None


def test_plan_turn_does_not_reopen_after_an_image_exists():
    """Once generating has started, the stock terminal rules apply again."""
    mod, loop = _bare_loop(task_type="plan")
    agent_data = _plan_turn_agent_data(
        extra_messages=[
            {"role": "tool", "content": "prompt='poster' backend=vllm agentic_tool ok=1 images=1"},
            {"role": "assistant", "content": _PLAN_TURN_TEXT},
        ]
    )

    assert loop._agentic_continue_after_plan_turn(agent_data, mod.AgentState.TERMINATED, messages_before=3) is None


def test_plan_turn_does_not_reopen_when_the_harness_terminated():
    """Only the no-tool-call exit appends a message; budget exits must not be reopened."""
    mod, loop = _bare_loop(task_type="plan")
    agent_data = _plan_turn_agent_data()
    continue_plan = loop._agentic_continue_after_plan_turn

    # No new message appended → a length or turn-budget termination.
    assert continue_plan(agent_data, mod.AgentState.TERMINATED, messages_before=2) is None
    # A non-TERMINATED state is returned untouched.
    assert continue_plan(agent_data, mod.AgentState.PROCESSING_TOOLS, messages_before=1) is None
