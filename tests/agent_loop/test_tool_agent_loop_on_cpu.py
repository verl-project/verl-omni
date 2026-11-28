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
"""CPU tests for the image-gen multi-turn agent loop helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import verl_omni  # noqa: F401
from verl_omni.tools.agent_helper import image_gen_utils as utils
from verl_omni.utils.agentic.config import bind_agentic_image_gen, clear_agentic_image_gen


def _judge_obs(*, correctness=0.80, aesthetics=0.76, good_enough="YES", findings="text is legible", fixes="none"):
    return (
        "VL judge on the last generated image:\n"
        f"  correctness={correctness}\n  aesthetics ={aesthetics}\n  good_enough ={good_enough}\n"
        f"  findings: {findings}\n  suggested_fixes: {fixes}\n"
        "  agentic_judge ok=1 parse_ok=1 backend=vllm parse_retries=0"
    )


def _gen_obs(prompt="a poster", backend="qwen_image"):
    return (
        f"Frozen diffusion produced the image. path=/tmp/x.png agentic_tool ok=1 stub=0 images=1 "
        f"backend={backend} prompt='{prompt}'"
    )


@pytest.mark.parametrize("value", [False, "0", "false", "off", "no"])
def test_force_reflection_after_judge_hydra_gate(value):
    from verl_omni.utils.agentic.config import agentic_get_bool

    clear_agentic_image_gen()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"force_reflection_after_judge": value}}))
    assert agentic_get_bool("force_reflection_after_judge") is False
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {}}))
    assert agentic_get_bool("force_reflection_after_judge") is True  # yaml default on
    clear_agentic_image_gen()


def test_max_generate_passes_hydra():
    u = utils
    clear_agentic_image_gen()
    with pytest.raises(RuntimeError, match="unbound"):
        u.max_generate_passes()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {}}))
    assert u.max_generate_passes() == 3
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"max_generate_image_passes": 5}}))
    assert u.max_generate_passes() == 5
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"max_generate_image_passes": "garbage"}}))
    with pytest.raises(ValueError, match="max_generate_image_passes"):
        u.max_generate_passes()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"max_generate_image_passes": 0}}))
    with pytest.raises(ValueError, match=">= 1"):
        u.max_generate_passes()
    clear_agentic_image_gen()


def test_force_first_generate_probability_schedule():
    u = utils
    clear_agentic_image_gen()
    # force_first_generate defaults False via explicit getter default → 0.0 even unbound
    assert u.force_first_generate_probability(5) == 0.0  # off by default
    assert u.force_first_generate_probability(5, validate=True) == 0.0  # never on val
    bind_agentic_image_gen(
        OmegaConf.create(
            {
                "agentic_image_gen": {
                    "force_first_generate": True,
                    "force_first_warmup_steps": 10,
                    "force_first_end_step": 20,
                }
            }
        )
    )
    assert u.force_first_generate_probability(5) == 1.0  # warmup
    assert u.force_first_generate_probability(None) == 1.0  # missing step → 0
    assert u.force_first_generate_probability(15) == pytest.approx(0.5)  # linear anneal
    assert u.force_first_generate_probability(20) == 0.0  # annealed off
    with pytest.raises(ValueError, match="step must be an int"):
        u.force_first_generate_probability("not-an-int")
    bind_agentic_image_gen(
        OmegaConf.create(
            {
                "agentic_image_gen": {
                    "force_first_generate": True,
                    "force_first_warmup_steps": "garbage",
                    "force_first_end_step": 20,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="force_first_warmup_steps"):
        u.force_first_generate_probability(5)
    clear_agentic_image_gen()


def test_rewrite_judge_before_generate_hydra():
    from verl_omni.utils.agentic.config import agentic_get_bool

    clear_agentic_image_gen()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {}}))
    assert agentic_get_bool("rewrite_judge_before_generate") is True  # yaml default on
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {"rewrite_judge_before_generate": False}}))
    assert agentic_get_bool("rewrite_judge_before_generate") is False
    clear_agentic_image_gen()


def test_tool_calls_are_premature_judge():
    u = utils
    assert u.tool_calls_are_premature_judge([]) is False
    assert u.tool_calls_are_premature_judge(None) is False
    assert u.tool_calls_are_premature_judge([SimpleNamespace(name="judge_image")]) is True
    assert (
        u.tool_calls_are_premature_judge([SimpleNamespace(name="generate_image"), SimpleNamespace(name="judge_image")])
        is False
    )


def test_last_user_text_content_forms():
    u = utils
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
        {"role": "assistant", "content": "noise"},
    ]
    assert u.last_user_text(messages) == "part one part two"
    messages[0]["content"] = "plain string task"
    assert u.last_user_text(messages) == "plain string task"
    assert u.last_user_text([{"role": "assistant", "content": "x"}]) == ""


def test_hermes_tool_call_wire_format():
    text = utils.hermes_tool_call("generate_image", prompt="a cafe poster")
    assert text.startswith("<tool_call>\n") and text.endswith("\n</tool_call>")
    payload = json.loads(text[len("<tool_call>\n") : -len("\n</tool_call>")])
    assert payload == {"name": "generate_image", "arguments": {"prompt": "a cafe poster"}}


def test_messages_after_last_user_excludes_fewshot_demos():
    u = utils
    demo = [
        {"role": "user", "content": "demo task"},
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="demo")},
        {"role": "tool", "content": _gen_obs(backend="fewshot")},
    ]
    live = [
        {"role": "user", "content": "live task"},
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="live")},
        {"role": "tool", "content": _gen_obs(backend="qwen_image")},
    ]
    assert u.messages_after_last_user(demo + live) == live[1:]
    assert u.messages_after_last_user([{"role": "assistant", "content": "x"}]) == [
        {"role": "assistant", "content": "x"}
    ]


def test_count_successful_judges_ignores_fewshot():
    u = utils
    demo = [
        {"role": "user", "content": "demo"},
        {"role": "assistant", "content": u.hermes_tool_call("judge_image", user_request="same", image_prompt="last")},
        {"role": "tool", "content": _judge_obs().replace("backend=vllm", "backend=fewshot")},
    ]
    live = [
        {"role": "user", "content": "live"},
        {"role": "assistant", "content": u.hermes_tool_call("judge_image", user_request="same", image_prompt="last")},
        {"role": "tool", "content": _judge_obs()},
        {"role": "assistant", "content": u.hermes_tool_call("judge_image", user_request="same", image_prompt="last")},
        {"role": "tool", "content": _judge_obs(good_enough="NO")},
    ]
    assert u.count_successful_judges(demo + live) == 2


def test_count_successful_generates_requires_live_backend():
    u = utils
    live = [
        {"role": "user", "content": "live"},
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="p1")},
        {"role": "tool", "content": _gen_obs(backend="qwen_image")},
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="p2")},
        {"role": "tool", "content": _gen_obs(backend="fewshot")},  # demo marker, not live
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="p3")},
        {"role": "tool", "content": "agentic_tool ok=1 images=1 prompt='p3'"},  # no live backend token
    ]
    assert u.count_successful_generates(live) == 1


def test_last_live_generate_prompt_extraction():
    u = utils
    live = [
        {"role": "user", "content": "live"},
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="first")},
        {"role": "tool", "content": _gen_obs(prompt="first", backend="qwen_image")},
        {"role": "assistant", "content": u.hermes_tool_call("generate_image", prompt="rewritten")},
        {"role": "tool", "content": _gen_obs(prompt="rewritten", backend="qwen_image")},
    ]
    assert u.last_live_generate_prompt(live) == "rewritten"
    assert u.last_live_generate_prompt([]) == ""


def test_tool_message_text_forms():
    u = utils
    assert u.tool_message_text({"content": "plain"}) == "plain"
    assert u.tool_message_text({"content": [{"type": "text", "text": "a"}, {"type": "image", "image": "x"}]}) == "a"
    assert u.tool_message_text({"content": None}) == ""


def test_build_forced_reflection_yes_stop_cue():
    text, stop_required = utils.build_forced_reflection(_judge_obs(good_enough="YES"))
    assert stop_required is True
    assert "good_enough=YES" in text
    assert "agentic_stop_decision_required=1" in text
    assert "exactly Done" in text


def test_build_forced_reflection_no_continue_cue():
    u = utils
    text, stop_required = u.build_forced_reflection(_judge_obs(good_enough="NO", fixes="add legible text"))
    assert stop_required is False
    assert "good_enough=NO" in text
    assert "Suggested fixes: add legible text." in text
    assert "Rewriting the diffusion prompt next" in text
    text_no_fix, _ = u.build_forced_reflection(_judge_obs(good_enough="NO", fixes="none"))
    assert "Suggested fixes" not in text_no_fix


def test_build_forced_reflection_max_passes_stop_cue():
    text, stop_required = utils.build_forced_reflection(
        _judge_obs(good_enough="NO"), force_done=True, generate_pass=3, max_passes=3
    )
    assert stop_required is True
    assert "pass 3/3" in text
    assert "agentic_force_stop_max_passes=1" in text
    assert "agentic_stop_decision_required=1" in text


def test_build_forced_reflection_requires_judge_ok():
    assert utils.build_forced_reflection("no judge marker here") is None


def test_fits_response_budget_rejects_overflow():
    u = utils
    assert u.fits_response_budget(mask_len=10, n_new_ids=5, response_length=16) is True
    assert u.fits_response_budget(mask_len=10, n_new_ids=6, response_length=16) is False
    assert u.fits_response_budget(mask_len=10, n_new_ids=0, response_length=16) is False
    assert u.fits_response_budget(mask_len=0, n_new_ids=1, response_length=1) is False


@pytest.mark.parametrize("outcome", ["success", "parent_error", "missing_data"])
def test_run_reuses_worker_episode_until_worker_scope_exits(monkeypatch, tmp_path, outcome):
    import asyncio
    from types import SimpleNamespace

    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    from verl_omni.agent_loop.omni_agent_loop import OmniAgentLoopWorker
    from verl_omni.agent_loop.tool_agent_loop import ImageGenToolAgentLoop
    from verl_omni.tools.trajectory import current_episode

    owners = []
    image = tmp_path / "image.png"
    image.write_bytes(b"pixels")

    async def parent_run(self, sampling_params, **kwargs):
        state = current_episode()
        assert state is owners[-1]
        state.record_generation(call_id="call", prompt="scene", paths=[str(image)])
        await asyncio.to_thread(state.record_judgment, "call", good_enough=True)
        if outcome == "parent_error":
            raise RuntimeError("tool failed")
        if outcome != "missing_data":
            self._agent_data = SimpleNamespace(messages=[{"role": "assistant", "content": "Done."}])
        return SimpleNamespace(extra_fields={})

    monkeypatch.setattr(ToolAgentLoop, "run", parent_run)
    loop = ImageGenToolAgentLoop.__new__(ImageGenToolAgentLoop)

    async def worker_parent(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
        state = current_episode()
        owners.append(state)
        assert state.user_prompt == "request"
        try:
            return await loop.run(sampling_params, **kwargs)
        finally:
            assert current_episode() is state
            assert not state.closed
            assert len(state.generations) == len(state.judgments) == 1
            assert state.good_enough

    monkeypatch.setattr(AgentLoopWorker, "_run_agent_loop", worker_parent)
    worker = OmniAgentLoopWorker.__new__(OmniAgentLoopWorker)

    async def run():
        invocation = worker._run_agent_loop(
            {},
            {"step": 1, "sample_index": 0, "rollout_n": 0, "validate": False},
            agent_name="image_gen_tool_agent",
            raw_prompt=[{"role": "user", "content": "request"}],
        )
        if outcome != "success":
            message = "tool failed" if outcome == "parent_error" else "runtime trajectory messages"
            with pytest.raises(RuntimeError, match=message):
                await invocation
        else:
            output = await invocation
            assert output.extra_fields["trajectory_relpath"] == "step_000001/sample_0.00"
            assert output.extra_fields["trajectory_messages"] == [{"role": "assistant", "content": "Done."}]
        state = owners[-1]
        assert state.closed and not state.generations and not state.judgments
        assert current_episode() is None

    asyncio.run(run())


class _Tokenizer:
    eos_token = "<|im_end|>"

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, ids, skip_special_tokens=True):
        return "Reflection: the poster is fine. Done."


def _bare_loop():
    import asyncio

    from verl_omni.agent_loop.tool_agent_loop import ImageGenToolAgentLoop

    loop = object.__new__(ImageGenToolAgentLoop)
    loop.tokenizer = _Tokenizer()
    loop.loop = asyncio.get_running_loop()
    loop.response_length = 1024
    loop.tools = {}
    return loop


def _agent_data(messages):
    from verl.experimental.agent_loop.tool_agent_loop import AgentData

    return AgentData(
        messages=list(messages),
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs=None,
        metrics={},
        request_id="r",
        tools_kwargs={},
    )


@pytest.mark.asyncio
async def test_teacher_forced_call_is_recorded_as_a_structured_call():
    from verl.experimental.agent_loop.tool_parser import FunctionCall

    loop = _bare_loop()
    agent_data = _agent_data([{"role": "user", "content": "a poster"}])
    tool_call = FunctionCall(name="generate_image", arguments=json.dumps({"prompt": "a poster"}))
    hermes = utils.hermes_tool_call("generate_image", prompt="a poster")
    await loop._replace_last_assistant_with_tool_call(agent_data, hermes, tool_call)

    message = agent_data.messages[-1]
    assert tool_call.tool_call_id
    assert message["tool_calls"] == [
        {
            "type": "function",
            "id": tool_call.tool_call_id,
            "function": {"name": "generate_image", "arguments": tool_call.arguments},
        }
    ]


@pytest.mark.asyncio
async def test_budget_terminated_turn_is_recorded_in_messages(monkeypatch):
    from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

    async def stop_on_budget(self, agent_data, sampling_params, ignore_termination=False):
        agent_data.response_ids = [1, 2, 3]
        agent_data.assistant_turns += 1
        return AgentState.TERMINATED

    monkeypatch.setattr(ToolAgentLoop, "_handle_generating_state", stop_on_budget)
    clear_agentic_image_gen()
    bind_agentic_image_gen(OmegaConf.create({"agentic_image_gen": {}}))
    loop = _bare_loop()
    agent_data = _agent_data([{"role": "user", "content": "a poster"}])
    state = await loop._handle_generating_state(agent_data, {})
    clear_agentic_image_gen()

    assert state == AgentState.TERMINATED
    assert agent_data.messages[-1] == {"role": "assistant", "content": "Reflection: the poster is fine. Done."}


@pytest.mark.parametrize("with_ids", [False, True])
def test_runtime_transcript_binds_tool_results_and_excludes_fewshot(with_ids):
    call = {"function": {"name": "generate_image", "arguments": '{"prompt": "a poster"}'}}
    result = {"role": "tool", "content": [{"type": "text", "text": "path=/tmp/image.png"}]}
    if with_ids:
        call["id"] = result["tool_call_id"] = "generate-1"
    messages = [
        {"role": "assistant", "content": "fewshot"},
        {"role": "user", "content": "a poster"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        result,
        {"role": "assistant", "content": "Done."},
    ]
    exported = utils.runtime_trajectory_messages(messages)
    assert len(exported) == 3
    call_id = exported[0]["tool_calls"][0]["id"]
    assert call_id == ("generate-1" if with_ids else "call_0_0")
    assert exported[1] == {"role": "tool", "content": "path=/tmp/image.png", "tool_call_id": call_id}
    assert exported[2] == {"role": "assistant", "content": "Done."}


def _hermes_rollout_messages(*, with_ids: bool = False) -> list[dict]:
    """Messages as ``ToolAgentLoop`` records a reflect rollout under the Hermes parser."""
    from verl_omni.utils.agentic_image_judge_parse import format_judge_observation

    judge_text, _ = format_judge_observation(
        image_path="/tmp/x.png",
        parsed={"correctness": 0.8, "aesthetics": 0.7, "good_enough": True, "findings": "text is legible"},
        backend="vllm",
    )

    def call(name: str, call_id: str, **arguments: str) -> dict:
        entry = {"type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}
        if with_ids:
            entry["id"] = call_id
        return entry

    def response(content: str, call_id: str) -> dict:
        message = {"role": "tool", "content": content}
        if with_ids:
            message["tool_call_id"] = call_id
        return message

    return [
        {"role": "system", "content": "tools"},
        {"role": "user", "content": "a poster"},
        {"role": "assistant", "content": "", "tool_calls": [call("generate_image", "g", prompt="a poster")]},
        response(_gen_obs(), "g"),
        {"role": "assistant", "content": "", "tool_calls": [call("judge_image", "j", user_request="a poster")]},
        response(judge_text, "j"),
        {"role": "assistant", "content": "Reflection: the poster text is legible. Done."},
    ]


def test_exported_messages_authenticate_a_hermes_rollout_for_the_rpco_reward():
    """The Hermes parser sets no call IDs: the export is what makes outcomes trusted."""
    from verl_omni.agent_loop.utils import messages_to_text
    from verl_omni.utils.reward_score.agentic_multidim_reward import compute_score

    messages = _hermes_rollout_messages()
    kwargs = {
        "solution_str": messages_to_text(messages[2:]),
        "ground_truth": {"task_type": "reflect", "expected_num_images": 1},
    }
    raw = compute_score(**kwargs, extra_info={"trajectory_messages": messages})
    assert raw["n_successful_generates"] == 0
    assert raw["score"] == 0.0

    exported = compute_score(**kwargs, extra_info={"trajectory_messages": utils.runtime_trajectory_messages(messages)})
    assert exported["rollout_valid"] == 1
    assert exported["judge_parse_ok"] == 1
    assert exported["reward_tool"] == exported["reward_result"] == exported["reward_done"] == 1.0


@pytest.mark.asyncio
async def test_run_publishes_runtime_trajectory_messages(monkeypatch):
    from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop

    messages = _hermes_rollout_messages()

    async def fake_pending(self, agent_data, sampling_params):
        return None

    async def fake_run(self, sampling_params, **kwargs):
        agent_data = _agent_data(messages)
        await self._handle_pending_state(agent_data, sampling_params)
        return SimpleNamespace(extra_fields=agent_data.extra_fields)

    monkeypatch.setattr(ToolAgentLoop, "_handle_pending_state", fake_pending)
    monkeypatch.setattr(ToolAgentLoop, "run", fake_run)
    output = await _bare_loop().run({})

    assert output.extra_fields["trajectory_messages"] == utils.runtime_trajectory_messages(messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("include_runtime_messages", [False, True])
async def test_pinned_naive_reward_manager_forwards_runtime_tool_fields(include_runtime_messages):
    import numpy as np
    import torch
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

    from verl_omni.agent_loop.utils import messages_to_text
    from verl_omni.utils.reward_score.agentic_multidim_reward import compute_score

    messages = _hermes_rollout_messages()
    decoded = messages_to_text(messages[2:])
    tool_fields = (
        {"trajectory_messages": utils.runtime_trajectory_messages(messages)} if include_runtime_messages else {}
    )
    data = DataProto.from_dict(
        tensors={"responses": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)},
        non_tensors={
            "data_source": np.array(["agentic"], dtype=object),
            "reward_model": np.array(
                [{"ground_truth": {"task_type": "reflect", "expected_num_images": 1}}], dtype=object
            ),
            "extra_info": np.array([{}], dtype=object),
            "tool_extra_fields": np.array([tool_fields], dtype=object),
        },
    )
    manager = NaiveRewardManager(
        OmegaConf.create({}), SimpleNamespace(decode=lambda *args, **kwargs: decoded), compute_score
    )
    output = await manager.run_single(data)
    metrics = output["reward_extra_info"]

    assert metrics["rollout_valid"] == int(include_runtime_messages)
    assert metrics["n_successful_generates"] == int(include_runtime_messages)
    assert metrics["reward_result"] == float(include_runtime_messages)
    assert (output["reward_score"] > 0) is include_runtime_messages
