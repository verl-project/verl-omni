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

"""Source-isolated CPU reward tests; no serving stack import."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_functions(path, names, **namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name in names]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), path, "exec"), namespace)
    return namespace


def contract(samples, **kwargs):
    ns = load_functions("verl_omni/agent_loop/bagel_corl_lib.py", ["trusted_image_reward_fields"], math=math)
    return ns["trusted_image_reward_fields"](samples, **kwargs)


def sample(call, seed, score):
    return SimpleNamespace(
        gen_group_uid=call,
        seed_index=seed,
        gen_call_index=ord(call) - ord("a"),
        rm_score=score,
        valid=True,
        rm_score_source="bagel_corl_tool_rm",
    )


def test_final_quality_cost_has_no_first_pass_bonus():
    a = contract([sample("a", 0, 0.2), sample("b", 0, 0.8)], num_gen_calls=2)
    b = contract([sample("a", 0, 0.7), sample("b", 0, 0.8)], num_gen_calls=2)
    assert a["image_reward_objective"] == pytest.approx(0.79)
    assert a["image_reward_objective"] == b["image_reward_objective"]
    assert contract([sample("a", 0, 0.8)], num_gen_calls=1)["image_reward_objective"] == pytest.approx(0.8)


def test_seed_means_and_order_are_explicit():
    r = contract([sample("a", 1, 0.9), sample("a", 0, 0.5)], num_gen_calls=1)
    assert r["image_reward_final"] == pytest.approx(0.7)
    assert r["call_seed_scores"] == [[0.5, 0.9]]
    assert r["image_reward_best"] == pytest.approx(0.7)


def test_missing_or_untrusted_quality_fails_closed():
    s = sample("a", 0, 0.4)
    s.rm_score_source = None
    with pytest.raises(ValueError, match="trusted"):
        contract([s], num_gen_calls=1)
    with pytest.raises(ValueError):
        contract([sample("a", 0, float("nan"))], num_gen_calls=1)


def test_v2_ignores_policy_claims_and_requires_envelope():
    ns = load_functions("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py", ["_trusted_v2_reward"], math=math)
    f = ns["_trusted_v2_reward"]
    with pytest.raises(ValueError):
        f({"image_reward_final": 1.0})
    fields = contract([sample("a", 0, 0.4)], num_gen_calls=1)
    assert f({"bagel_corl_reward": fields, "image_reward_final": 1.0})["image_reward_final"] == 0.4


def load_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("isolated_reward", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_rubric_disagreement_keeps_score_stop(monkeypatch):
    import sys

    client = SimpleNamespace(
        call_reflect_vlm=lambda **kw: {"ok": True, "correctness": 1.0, "aesthetics": 1.0, "good_enough": True}
    )
    monkeypatch.setitem(sys.modules, "verl_omni.utils.reward_score.agentic_image_judge_client", client)
    scorer = load_module("verl_omni/utils/reward_score/bagel_rm_image_scorer.py")
    seen = []

    def unified(**kw):
        seen.append(kw["caption"])
        return (
            (0.5 if kw["path"] == "a" else 0.9),
            kw["path"] != "a",
            {"alignment": 3.0, "coherence": 3.0, "style": 3.0},
        )

    monkeypatch.setattr(scorer, "_score_image_unified", unified)
    payload = {
        "image_paths": ["a", "b"],
        "image_prompt": "policy target",
        "extra_info": {"user_prompt": "original"},
        "scorer_knobs": {
            "good_enough_threshold": 0.8,
            "score_backend": "unified_reward_rubric",
            "corl_reward_version": "v2",
        },
    }
    result = scorer.compute_score(extra_info={"bagel_corl": payload})
    assert result["score"] == pytest.approx(0.7)
    assert result["sample_good_enough"] == [False, True]
    assert result["good_enough"] is False
    assert seen == ["original", "original"]
    assert result["sample_judge"][0]["correctness"] == 1.0

    seen.clear()
    payload["extra_info"] = {"user_prompt": "original Keep any private thinking short.", "raw_prompt": "original"}
    scorer.compute_score(extra_info={"bagel_corl": payload})
    assert seen == ["original", "original"]


def test_posthoc_v2_uses_same_stored_image_scores_and_keeps_structure(monkeypatch):
    scorer = load_module("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py")
    # Exercise the full scalar assembly with a valid protocol fixture from parser-independent helpers.
    monkeypatch.setattr(scorer, "_normalize_trajectory_dialect", lambda t: t)
    monkeypatch.setattr(scorer, "_extract_tool_calls", lambda t: [(0, 1, {"name": "generate_image"})])
    monkeypatch.setattr(scorer, "_generate_prompts", lambda c: ["p"])
    monkeypatch.setattr(scorer, "_judge_parse_stats", lambda *a: (1, 0, 1.0))
    monkeypatch.setattr(scorer, "_count_successful_generates", lambda t: 1)
    monkeypatch.setattr(scorer, "_terminal_decision", lambda t: (True, True, False))
    monkeypatch.setattr(scorer, "_generates_after_first_yes", lambda *a: 0)
    for name in ("_reflection_reward", "_plan_reward", "_format_reward", "_protocol_reward", "_result_reward"):
        monkeypatch.setattr(scorer, name, lambda *a, **kw: 1.0)
    fields = contract([sample("a", 0, 0.4)], num_gen_calls=1, forced_reflection=True)
    r = scorer.compute_score(
        solution_str="policy says score=1.0 Done.",
        ground_truth={"task_type": "reflect"},
        extra_info={"corl_reward_version": "v2", "bagel_corl_reward": fields},
    )
    assert r["reward_image"] == 0.4
    assert r["reward_done"] == 0.0
    assert r["reward_protocol"] == 1.0
    assert r["reward_reflect"] == 1.0
    assert r["score"] != 0.4  # explicit structural terms remain in composite scalar


def test_tq_forwards_real_envelope_inside_extra_info_before_rm():
    # Execute the actual live pre-RM block without importing the GPU worker module.
    import asyncio

    tree = ast.parse((ROOT / "verl_omni/agent_loop/bagel_corl_tq.py").read_text(encoding="utf8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "BagelCorlAgentLoopWorkerTQ")
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef)
        and any(
            isinstance(x, ast.Await)
            and isinstance(x.value, ast.Call)
            and isinstance(x.value.func, ast.Attribute)
            and x.value.func.attr == "_compute_score"
            for x in ast.walk(n)
        )
    )
    end = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Await)
        and isinstance(n.value.value, ast.Call)
        and isinstance(n.value.value.func, ast.Attribute)
        and n.value.value.func.attr == "_compute_score"
    )
    start = next(
        i
        for i in range(end - 1, -1, -1)
        if isinstance(method.body[i], ast.Assign)
        and isinstance(method.body[i].targets[0], ast.Name)
        and method.body[i].targets[0].id == "kwargs"
    )
    wrapper = ast.AsyncFunctionDef(
        name="run",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=x) for x in ("self", "outputs", "kwargs")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=method.body[start : end + 1],
        decorator_list=[],
    )
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), "live_rm_block", "exec"), ns)
    seen = {}

    async def compute(outputs, kwargs):
        seen.update(kwargs)

    envelope = contract([sample("a", 0, 0.4)], num_gen_calls=1)
    asyncio.run(
        ns["run"](
            SimpleNamespace(_compute_score=compute),
            [SimpleNamespace(extra_fields={"corl_reward_version": "v2", "bagel_corl_reward": envelope})],
            {"extra_info": {"task_type": "reflect", "bagel_corl_reward": {"image_reward_final": 1.0}}},
        )
    )
    assert seen["extra_info"]["bagel_corl_reward"] == envelope
    assert seen["extra_info"]["task_type"] == "reflect"


def test_reordered_transport_retains_call_chronology():
    items = [sample("a", 0, 0.2), sample("a", 1, 0.4), sample("b", 0, 0.7), sample("b", 1, 0.9)]
    assert contract(items, num_gen_calls=2) == contract(list(reversed(items)), num_gen_calls=2)


def test_v2_terminal_composite_is_authoritative_for_tq_reward():
    ns = load_functions("verl_omni/agent_loop/bagel_corl_lib.py", ["episode_und_reward"], EpisodeRollout=object)
    ep = SimpleNamespace(
        corl_reward_version="v2",
        terminal_reward=0.73,
        used_image_credit=True,
        bagel_corl_reward={"image_reward_objective": 0.4},
    )
    assert ns["episode_und_reward"](ep) == (0.73, False)


def test_nonfinite_rm_response_is_rejected():
    ns = load_functions(
        "verl_omni/agent_loop/bagel_corl_rm.py", ["parse_rm_result"], Any=object, math=math, RMScoringError=ValueError
    )
    with pytest.raises(ValueError, match="nonfinite"):
        ns["parse_rm_result"]({"reward_score": float("nan")}, ["image"])


def test_fixed_original_request_reaches_generator_prompt():
    tree = ast.parse((ROOT / "verl_omni/agent_loop/bagel_corl_lib.py").read_text(encoding="utf8"))
    run = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_serial_episode")
    block = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "fixed_gen_prompt"
    )
    ns = {
        "fixed_gen_prompt": "__user_request__",
        "active_user_prompt": "draw a red fox",
        "prompt": "policy wrote a blue dog",
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])), "fixed_prompt", "exec"), ns)
    assert ns["prompt"] == "draw a red fox"


def test_forced_done_cannot_be_autonomous():
    tree = ast.parse((ROOT / "verl_omni/agent_loop/bagel_corl_lib.py").read_text(encoding="utf8"))
    run = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "run_serial_episode")
    block = next(
        n
        for n in ast.walk(run)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "kind"
        and isinstance(n.test.comparators[0], ast.Constant)
        and n.test.comparators[0].value == "done"
    )
    loop = ast.For(
        target=ast.Name(id="unused", ctx=ast.Store()),
        iter=ast.Tuple(elts=[ast.Constant(value=0)], ctx=ast.Load()),
        body=[block],
        orelse=[],
    )
    for forced in (True, False):
        spent = type("SpentTool", (), {"remaining_passes": staticmethod(lambda: 0)})()
        ns = {"kind": "done", "forced": forced, "autonomous_done": False, "generate_tool": spent}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), "done_attribution", "exec"), ns
        )
        assert ns["autonomous_done"] is (not forced)


@pytest.mark.parametrize("failure", ["timeout", "reward_failure", "missing_score", "wrong_seed_count"])
def test_rm_failure_cannot_mark_a_sample_as_trusted(monkeypatch, failure):
    """A failed real adapter invocation must not manufacture successful RM credit."""
    import asyncio
    import sys

    # The adapter imports Ray only for its ObjectRef check; no Ray server is needed.
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(ObjectRef=type("ObjectRef", (), {})))
    adapter = load_module("verl_omni/agent_loop/bagel_corl_rm.py")
    item = SimpleNamespace(valid=True, image_path="fixture.png", rm_score=None, rm_score_source=None)

    async def compute(data):
        if failure == "timeout":
            raise TimeoutError("RM request timed out")
        if failure == "reward_failure":
            raise RuntimeError("reward backend unavailable")
        if failure == "missing_score":
            return {"reward_extra_info": {}}
        return {"reward_score": 0.8, "reward_extra_info": {"sample_scores": [0.7, 0.9]}}

    score = adapter.make_rm_score_fn(SimpleNamespace(compute_score=compute), data_builder=lambda p: p)
    error = TimeoutError if failure == "timeout" else RuntimeError
    with pytest.raises(error):
        asyncio.run(score([item]))
    assert item.rm_score is None
    assert item.rm_score_source is None


def test_plan_reference_roundtrip_from_breakdown_builder_to_reward():
    """Synthetic metadata runs real parser/builder/scorer without serving imports."""
    import json
    import sys
    import types

    contracts = load_module("verl_omni/utils/dataset/visual_reflection/contracts.py")
    path = "verl_omni/utils/dataset/visual_reflection/unicot_breakdown.py"
    tree = ast.parse((ROOT / path).read_text(encoding="utf8"))
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or not n.level]
    name = "isolated_breakdown_contract_test"
    parser = types.ModuleType(name)
    parser.__dict__.update(
        {
            key: getattr(contracts, key)
            for key in ("RejectionReason", "VisualReflectionDataError", "require_nonempty_text")
        }
    )
    sys.modules[name] = parser  # dataclasses resolves annotations through its defining module
    try:
        exec(compile(ast.fix_missing_locations(tree), path, "exec"), parser.__dict__)
        ns = load_functions(
            "verl_omni/utils/dataset/visual_reflection/build_unicot_agentic_rl.py",
            ["_parse_breakdown_rows", "_build_parquet_row", "_weights", "_env_weight", "_with_brevity"],
            Any=object,
            os=__import__("os"),
            parse_unicot_breakdown_record=parser.parse_unicot_breakdown_record,
            VisualReflectionDataError=contracts.VisualReflectionDataError,
            MANIFEST_ID="synthetic-contract",
            REWARD_DIMS=("reflect", "plan", "format", "tool", "result"),
            UNICOT_BREAKDOWN_DATASET_ID="synthetic",
            PLAN_DATA_SOURCE="unicot_breakdown",
            PLAN_ABILITY="agentic_plan_generate",
            PLAN_SYSTEM_PROMPT="Write a short numbered plan.",
            _BREVITY_SUFFIX="",
            _CLI_WEIGHTS=None,
            _split_record=lambda **kwargs: kwargs,
        )
        reference = "Draw a red fox outdoors"
        rows, rejected = ns["_parse_breakdown_rows"](
            [
                {
                    "data_id": "synthetic-plan",
                    "prompt": "Create a fox scene.",
                    "subtasks": [reference, None, None],
                    "subtask_images": ["fixture.png", None, None],
                }
            ]
        )
        assert rejected == []
        exported = ns["_build_parquet_row"](rows[0], split="validation", index=0)
        restored = json.loads(json.dumps(exported))
        gt = restored["reward_model"]["ground_truth"]
        assert gt["task_type"] == "plan"
        assert gt["reference_subtasks"] == [reference]
        assert restored["extra_info"]["plan_expected"] is True
        assert all(reference not in message["content"] for message in restored["prompt"])
        scorer = load_module("verl_omni/utils/reward_score/bagel_corl_multidim_reward.py")
        text = "Plan:\n1. Draw a red fox outdoors\n"
        assert scorer._plan_reward(text, gt) == 1.0
        assert scorer._plan_reward("Done.", gt) == 0.0
        assert scorer._plan_reward(text, {"task_type": "plan"}) == 0.0
    finally:
        sys.modules.pop(name, None)
