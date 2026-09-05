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
"""CPU tests for OmniAgentLoop wiring and the dump helpers it delegates to.

Helpers are loaded from source so this file stays runnable when the full
``verl_omni`` package import chain is unavailable (heavy pipeline / CUDA deps).
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_VERL_OMNI = Path(__file__).resolve().parents[2] / "verl_omni"
_AGENT_LOOP = _VERL_OMNI / "agent_loop"
_TOOLS = _VERL_OMNI / "tools"
_AGENTIC_UTILS = _VERL_OMNI / "utils" / "agentic"


def _ensure_package(name: str, path: Path) -> types.ModuleType:
    mod = sys.modules.get(name)
    if mod is not None:
        return mod
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _load(modname: str, path: Path, *, package_dir: Path | None = None):
    _ensure_package("verl_omni", _VERL_OMNI)
    _ensure_package("verl_omni.agent_loop", _AGENT_LOOP)
    _ensure_package("verl_omni.tools", _TOOLS)
    _ensure_package("verl_omni.utils", _VERL_OMNI / "utils")
    _ensure_package("verl_omni.utils.agentic", _AGENTIC_UTILS)
    if modname in sys.modules and hasattr(sys.modules[modname], "__file__"):
        return sys.modules[modname]
    search = [str(package_dir)] if package_dir is not None else None
    spec = importlib.util.spec_from_file_location(modname, path, submodule_search_locations=search)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    if package_dir is not None:
        mod.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load(
    "verl_omni.utils.metrics_utils",
    _VERL_OMNI / "utils" / "metrics_utils.py",
)
rollout_parse = _load(
    "verl_omni.utils.agentic.image_gen_rollout_parse",
    _AGENTIC_UTILS / "image_gen_rollout_parse.py",
)
_TRAJECTORY = _TOOLS / "trajectory"
traj_ctx = _load(
    "verl_omni.tools.trajectory",
    _TRAJECTORY / "__init__.py",
    package_dir=_TRAJECTORY,
)
rollout_dump = _load(
    "verl_omni.utils.agentic.image_gen_rollout_dump",
    _AGENTIC_UTILS / "image_gen_rollout_dump.py",
)

turn_kind = rollout_parse.turn_kind
extract_generate_image_prompts = rollout_parse.extract_generate_image_prompts
split_env_blob = rollout_parse.split_env_blob
split_rollout_turns = rollout_parse.split_rollout_turns
discard_invalid_rollouts = rollout_dump.discard_invalid_rollouts


def _load_omni_agent_loop():
    """Load omni_agent_loop.py against stub AgentLoop* bases (no Ray)."""
    captured: dict = {"kwargs": None}

    class _AgentLoopWorker:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name, trace=True, **kwargs):
            del sampling_params, trajectory, agent_name, trace
            captured["kwargs"] = dict(kwargs)
            captured["relpath_during_run"] = traj_ctx.get_active_trajectory_relpath()
            captured["user_prompt_during_run"] = traj_ctx.get_active_user_prompt()
            return "ok"

    class _AgentLoopManager:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.model_config = {"path": "stub", "trust_remote_code": False}

        def generate_sequences(self, prompts):
            return prompts.output

    agent_loop_pkg = types.ModuleType("verl.experimental.agent_loop")
    agent_loop_mod = types.ModuleType("verl.experimental.agent_loop.agent_loop")
    tool_loop_mod = types.ModuleType("verl.experimental.agent_loop.tool_agent_loop")
    parser_mod = types.ModuleType("verl.experimental.agent_loop.tool_parser")
    agent_loop_pkg.AgentLoopManager = _AgentLoopManager
    agent_loop_mod.AgentLoopWorker = _AgentLoopWorker
    agent_loop_mod.AgentLoopOutput = type("AgentLoopOutput", (), {})
    agent_loop_mod.register = lambda name: (lambda cls: cls)
    tool_loop_mod.AgentData = type("AgentData", (), {})
    tool_loop_mod.AgentState = type("AgentState", (), {})
    tool_loop_mod.ToolAgentLoop = type("ToolAgentLoop", (), {})
    parser_mod.FunctionCall = type("FunctionCall", (), {})
    utils_mod = types.ModuleType("verl.utils")
    utils_mod.hf_tokenizer = lambda *args, **kwargs: object()

    overlays = {
        "verl": sys.modules.get("verl") or types.ModuleType("verl"),
        "verl.experimental": sys.modules.get("verl.experimental") or types.ModuleType("verl.experimental"),
        "verl.experimental.agent_loop": agent_loop_pkg,
        "verl.experimental.agent_loop.agent_loop": agent_loop_mod,
        "verl.experimental.agent_loop.tool_agent_loop": tool_loop_mod,
        "verl.experimental.agent_loop.tool_parser": parser_mod,
        "verl.utils": utils_mod,
    }
    saved = {name: sys.modules.get(name) for name in overlays}
    sys.modules.update(overlays)
    sys.modules.pop("verl_omni.agent_loop.omni_agent_loop", None)
    sys.modules.pop("verl_omni.agent_loop.tool_agent_loop", None)
    try:
        omni = _load("verl_omni.agent_loop.omni_agent_loop", _AGENT_LOOP / "omni_agent_loop.py")
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev
    omni._captured = captured
    return omni


def test_worker_stamps_rollout_kwargs_and_resets_context():
    omni = _load_omni_agent_loop()
    worker = omni.OmniAgentLoopWorker.__new__(omni.OmniAgentLoopWorker)
    assert omni.OmniAgentLoopWorker._AGENTIC_FUNCTION_TOOLS.is_file()

    prior_path = traj_ctx.set_active_trajectory_relpath("prior/path")
    prior_prompt = traj_ctx.set_active_user_prompt("prior prompt")

    async def _run_then_read_context():
        result = await omni.OmniAgentLoopWorker._run_agent_loop(
            worker,
            {},
            {"step": 7, "sample_index": 3, "rollout_n": 1, "validate": False},
            agent_name="image_gen_tool_agent",
            raw_prompt=[{"role": "user", "content": "draw a cafe poster"}],
        )
        return result, traj_ctx.get_active_trajectory_relpath(), traj_ctx.get_active_user_prompt()

    try:
        result, path_after, prompt_after = asyncio.run(_run_then_read_context())
    finally:
        traj_ctx.reset_active_user_prompt(prior_prompt)
        traj_ctx.reset_active_trajectory_relpath(prior_path)

    captured = omni._captured["kwargs"]
    assert result == "ok"
    assert captured["_agentic_step"] == 7
    assert captured["_agentic_validate"] is False
    assert captured["_agentic_trajectory_relpath"] == "step_000007/sample_3.01"
    assert omni._captured["relpath_during_run"] == "step_000007/sample_3.01"
    assert omni._captured["user_prompt_during_run"] == "draw a cafe poster"
    assert path_after == "prior/path"
    assert prompt_after == "prior prompt"


def test_manager_dumps_before_discarding_invalid_rollouts():
    omni = _load_omni_agent_loop()
    order: list[str] = []
    omni.dump_raw_rollouts = lambda **kwargs: order.append("dump") or kwargs
    omni.discard_invalid_rollouts = lambda output: order.append("discard") or output
    omni.AgenticRewardMetrics = types.SimpleNamespace(aggregate=lambda batch: {})

    manager = omni.OmniAgentLoopManager.__new__(omni.OmniAgentLoopManager)
    manager._monitor_tokenizer = object()
    output = types.SimpleNamespace(non_tensor_batch={})
    prompts = types.SimpleNamespace(meta_info={"global_steps": 4}, output=output)
    assert omni.OmniAgentLoopManager.generate_sequences(manager, prompts) is output
    assert order == ["dump", "discard"]


def test_turn_kind_stop_rewrite_and_continue():
    judge_no = "VL judge on the last generated image:\n  good_enough =NO\n  agentic_judge ok=1"
    judge_yes = judge_no.replace("good_enough =NO", "good_enough =YES")
    continue_cue = "Reflection: rewrite next. agentic_forced_reflection=1"
    stop_cue = "Reflection: Stop. agentic_stop_decision_required=1 agentic_forced_reflection=1"
    rewrite = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "lion"}}\n</tool_call>'
    done = "Reflection: The image meets the original request. Done.<|im_end|>"
    assert turn_kind(done, judge_yes, stop_cue) == "agent_done_after_forced_reflection"
    assert turn_kind(rewrite, judge_no, continue_cue) == "agent_rewrite_after_forced_reflection"
    assert turn_kind("", judge_yes, stop_cue) == "forced_reflection_stop_cue"
    assert turn_kind(done, judge_no, "") == "agent_reflection_done"


def test_extract_generate_image_prompts_hermes_and_qwen():
    hermes = '<tool_call>\n{"name": "generate_image", "arguments": {"prompt": "a cat"}}\n</tool_call>'
    qwen = "<tool_call>\n<function=generate_image>\n<parameter=prompt>\na dog\n</parameter>\n</function>\n</tool_call>"
    assert extract_generate_image_prompts(hermes) == ["a cat"]
    assert extract_generate_image_prompts(qwen) == ["a dog"]


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
        types.SimpleNamespace(batch={"response_mask": mask}, non_tensor_batch={"rollout_valid": np.array([1, 0])})
    )
    assert mask.rows[0].vals == [1, 1]
    assert mask.rows[1].vals == [0, 0]

    all_invalid = _Mask([_MaskRow([1, 1]), _MaskRow([1, 0])])
    discard_invalid_rollouts(
        types.SimpleNamespace(
            batch={"response_mask": all_invalid}, non_tensor_batch={"rollout_valid": np.array([0, 0])}
        )
    )
    assert all_invalid.rows[0].vals == [1, 1]
    assert all_invalid.rows[1].vals == [1, 0]
