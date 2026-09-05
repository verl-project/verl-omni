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
"""CPU tests for the image-gen multi-turn agent loop helpers.

Helpers are loaded from source so collection does not run ``verl_omni/__init__.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_UTILS_PATH = Path(__file__).resolve().parents[2] / "verl_omni" / "agent_loop" / "utils.py"


def _load_utils():
    # Stub parent packages so we never execute verl_omni/__init__.py (CUDA).
    root = _UTILS_PATH.parents[1]
    for name, path in (
        ("verl_omni", root),
        ("verl_omni.agent_loop", root / "agent_loop"),
    ):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]  # type: ignore[attr-defined]
            sys.modules[name] = mod
    if "verl_omni.agent_loop.utils" in sys.modules and hasattr(sys.modules["verl_omni.agent_loop.utils"], "__file__"):
        return sys.modules["verl_omni.agent_loop.utils"]
    spec = importlib.util.spec_from_file_location("verl_omni.agent_loop.utils", _UTILS_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent_loop utils from {_UTILS_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_utils = _load_utils()
build_forced_reflection = _utils.build_forced_reflection
count_successful_generates = _utils.count_successful_generates
count_successful_judges = _utils.count_successful_judges
fits_response_budget = _utils.fits_response_budget
force_enabled = _utils.force_enabled
force_first_generate_probability = _utils.force_first_generate_probability
hermes_tool_call = _utils.hermes_tool_call
last_live_generate_prompt = _utils.last_live_generate_prompt
last_user_text = _utils.last_user_text
max_generate_passes = _utils.max_generate_passes
messages_after_last_user = _utils.messages_after_last_user
rewrite_judge_before_generate = _utils.rewrite_judge_before_generate
tool_calls_are_premature_judge = _utils.tool_calls_are_premature_judge
tool_message_text = _utils.tool_message_text


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


@pytest.mark.parametrize("value", ["0", "false", "off", "no"])
def test_force_enabled_env_gate(monkeypatch, value):
    monkeypatch.setenv("AGENTIC_FORCE_REFLECTION_AFTER_JUDGE", value)
    assert force_enabled() is False
    monkeypatch.delenv("AGENTIC_FORCE_REFLECTION_AFTER_JUDGE")
    assert force_enabled() is True  # default on


def test_max_generate_passes_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", raising=False)
    assert max_generate_passes() == 3
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "5")
    assert max_generate_passes() == 5
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "garbage")
    assert max_generate_passes() == 3
    monkeypatch.setenv("AGENTIC_MAX_GENERATE_IMAGE_PASSES", "0")
    assert max_generate_passes() == 1  # floored at 1


def test_force_first_generate_probability_schedule(monkeypatch):
    monkeypatch.delenv("AGENTIC_FORCE_FIRST_GENERATE", raising=False)
    assert force_first_generate_probability(5) == 0.0  # off by default
    assert force_first_generate_probability(5, validate=True) == 0.0  # never on val
    monkeypatch.setenv("AGENTIC_FORCE_FIRST_GENERATE", "1")
    monkeypatch.setenv("AGENTIC_FORCE_FIRST_WARMUP_STEPS", "10")
    monkeypatch.setenv("AGENTIC_FORCE_FIRST_END_STEP", "20")
    assert force_first_generate_probability(5) == 1.0  # warmup
    assert force_first_generate_probability(15) == pytest.approx(0.5)  # linear anneal
    assert force_first_generate_probability(20) == 0.0  # annealed off
    assert force_first_generate_probability("not-an-int") == 1.0  # step coerced to 0


def test_rewrite_judge_before_generate_env(monkeypatch):
    monkeypatch.delenv("AGENTIC_REWRITE_JUDGE_BEFORE_GENERATE", raising=False)
    assert rewrite_judge_before_generate() is True  # default on
    monkeypatch.setenv("AGENTIC_REWRITE_JUDGE_BEFORE_GENERATE", "0")
    assert rewrite_judge_before_generate() is False


def test_tool_calls_are_premature_judge():
    assert tool_calls_are_premature_judge([]) is False
    assert tool_calls_are_premature_judge(None) is False
    assert tool_calls_are_premature_judge([SimpleNamespace(name="judge_image")]) is True
    assert (
        tool_calls_are_premature_judge([SimpleNamespace(name="generate_image"), SimpleNamespace(name="judge_image")])
        is False
    )


def test_last_user_text_content_forms():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
        {"role": "assistant", "content": "noise"},
    ]
    assert last_user_text(messages) == "part one part two"
    messages[0]["content"] = "plain string task"
    assert last_user_text(messages) == "plain string task"
    assert last_user_text([{"role": "assistant", "content": "x"}]) == ""


def test_hermes_tool_call_wire_format():
    text = hermes_tool_call("generate_image", prompt="a cafe poster")
    assert text.startswith("<tool_call>\n") and text.endswith("\n</tool_call>")
    payload = json.loads(text[len("<tool_call>\n") : -len("\n</tool_call>")])
    assert payload == {"name": "generate_image", "arguments": {"prompt": "a cafe poster"}}


def test_messages_after_last_user_excludes_fewshot_demos():
    demo = [
        {"role": "user", "content": "demo task"},
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="demo")},
        {"role": "tool", "content": _gen_obs(backend="fewshot")},
    ]
    live = [
        {"role": "user", "content": "live task"},
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="live")},
        {"role": "tool", "content": _gen_obs(backend="qwen_image")},
    ]
    assert messages_after_last_user(demo + live) == live[1:]
    assert messages_after_last_user([{"role": "assistant", "content": "x"}]) == [{"role": "assistant", "content": "x"}]


def test_count_successful_judges_ignores_fewshot():
    demo = [
        {"role": "user", "content": "demo"},
        {"role": "assistant", "content": hermes_tool_call("judge_image", user_request="same", image_prompt="last")},
        {"role": "tool", "content": _judge_obs().replace("backend=vllm", "backend=fewshot")},
    ]
    live = [
        {"role": "user", "content": "live"},
        {"role": "assistant", "content": hermes_tool_call("judge_image", user_request="same", image_prompt="last")},
        {"role": "tool", "content": _judge_obs()},
        {"role": "assistant", "content": hermes_tool_call("judge_image", user_request="same", image_prompt="last")},
        {"role": "tool", "content": _judge_obs(good_enough="NO")},
    ]
    assert count_successful_judges(demo + live) == 2


def test_count_successful_generates_requires_live_backend():
    live = [
        {"role": "user", "content": "live"},
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="p1")},
        {"role": "tool", "content": _gen_obs(backend="qwen_image")},
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="p2")},
        {"role": "tool", "content": _gen_obs(backend="fewshot")},  # demo marker, not live
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="p3")},
        {"role": "tool", "content": "agentic_tool ok=1 images=1 prompt='p3'"},  # no live backend token
    ]
    assert count_successful_generates(live) == 1


def test_last_live_generate_prompt_extraction():
    live = [
        {"role": "user", "content": "live"},
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="first")},
        {"role": "tool", "content": _gen_obs(prompt="first", backend="qwen_image")},
        {"role": "assistant", "content": hermes_tool_call("generate_image", prompt="rewritten")},
        {"role": "tool", "content": _gen_obs(prompt="rewritten", backend="qwen_image")},
    ]
    assert last_live_generate_prompt(live) == "rewritten"
    assert last_live_generate_prompt([]) == ""


def test_tool_message_text_forms():
    assert tool_message_text({"content": "plain"}) == "plain"
    assert tool_message_text({"content": [{"type": "text", "text": "a"}, {"type": "image", "image": "x"}]}) == "a"
    assert tool_message_text({"content": None}) == ""


def test_build_forced_reflection_yes_stop_cue():
    text, stop_required = build_forced_reflection(_judge_obs(good_enough="YES"))
    assert stop_required is True
    assert "good_enough=YES" in text
    assert "agentic_stop_decision_required=1" in text
    assert "exactly Done" in text


def test_build_forced_reflection_no_continue_cue():
    text, stop_required = build_forced_reflection(_judge_obs(good_enough="NO", fixes="add legible text"))
    assert stop_required is False
    assert "good_enough=NO" in text
    assert "Suggested fixes: add legible text." in text
    assert "Rewriting the diffusion prompt next" in text
    text_no_fix, _ = build_forced_reflection(_judge_obs(good_enough="NO", fixes="none"))
    assert "Suggested fixes" not in text_no_fix


def test_build_forced_reflection_max_passes_stop_cue():
    text, stop_required = build_forced_reflection(
        _judge_obs(good_enough="NO"), force_done=True, generate_pass=3, max_passes=3
    )
    assert stop_required is True
    assert "pass 3/3" in text
    assert "agentic_force_stop_max_passes=1" in text
    assert "agentic_stop_decision_required=1" in text


def test_build_forced_reflection_requires_judge_ok():
    assert build_forced_reflection("no judge marker here") is None


def test_fits_response_budget_rejects_overflow():
    assert fits_response_budget(mask_len=10, n_new_ids=5, response_length=16) is True
    assert fits_response_budget(mask_len=10, n_new_ids=6, response_length=16) is False
    assert fits_response_budget(mask_len=10, n_new_ids=0, response_length=16) is False
    assert fits_response_budget(mask_len=0, n_new_ids=1, response_length=1) is False
