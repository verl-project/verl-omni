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

from __future__ import annotations

import ast
import asyncio
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_lib():
    spec = importlib.util.spec_from_file_location(
        "provenance_protocol", ROOT / "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/turn_protocol.py"
    )
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    path = ROOT / "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/episode.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("verl_omni"))
    ]
    module = ModuleType("serving_provenance_lib")
    sys.modules[module.__name__] = module
    from verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.serving import serving_rollout_provenance
    from verl_omni.tools.trajectory import EpisodeState, current_episode, episode_scope

    latch = [False]
    module.__dict__.update(
        EpisodeState=EpisodeState,
        current_episode=current_episode,
        episode_scope=episode_scope,
        serving_rollout_provenance=serving_rollout_provenance,
        build_forced_reflection=protocol.build_forced_reflection,
        derive_good_enough_from_scores=protocol.derive_good_enough_from_scores,
        format_rm_scores_as_judge_text=protocol.format_rm_scores_as_judge_text,
        derive_rollout_seed=lambda base, index: base + index,
        register_tool_artifact=lambda **kwargs: None,
        get_active_user_prompt=lambda: "a cube",
        clear_good_enough_yes_reached=lambda: latch.__setitem__(0, False),
        get_good_enough_yes_reached=lambda: latch[0],
        set_good_enough_yes_reached=lambda value: latch.__setitem__(0, value),
    )
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def step(text, version=0, replica="und-server"):
    return dict(
        text=text,
        token_ids=[4],
        log_probs=[-0.2],
        rollout_policy_version=version,
        rollout_replica_id=replica,
        serving_provenance_verified=True,
    )


GENERATE = '<tool_call>{"name":"generate_image","arguments":{"prompt":"a cube"}}</tool_call>'
JUDGE = '<tool_call>{"name":"judge_image","arguments":{}}</tool_call>'


def episode(module, steps, gen_version=0):
    iterator = iter(steps)

    async def decode(**kwargs):
        return next(iterator)

    async def generate(**kwargs):
        return [
            dict(
                valid=True,
                all_latents=torch.zeros(2, 1, 2),
                timesteps=torch.tensor([1.0]),
                rollout_log_probs=torch.tensor([-0.2]),
                image_path="/fixture/image.png",
                rm_score=0.8,
                rollout_policy_version=gen_version,
                rollout_replica_id="gen-server",
                serving_provenance_verified=True,
            )
        ]

    tool = module.BagelGenerateImageTool(gen_samples_per_call=1, max_generate_passes=1, generate_fn=generate)

    class Tokenizer:
        def encode(self, text, **kwargs):
            return [7]

    return asyncio.run(
        module.run_serial_episode(
            dataset_task_uid="task",
            policy_version=1,
            episode_uid="ep",
            prompt_ids=[1, 2],
            und_decode=decode,
            generate_tool=tool,
            tokenizer=Tokenizer(),
            max_und_turns=4,
            build_judge_text_fn=lambda samples: "agentic_judge ok=1 good_enough=YES",
        )
    )


def test_request_step_one_records_actual_serving_zero_and_each_lane_identity():
    module = load_lib()
    result = episode(module, [step(GENERATE), step(JUDGE), step("Done.")])
    assert result.policy_version == 0
    assert result.requested_policy_version == 1
    assert result.rollout_replica_id == "und-server"
    assert result.serving_provenance_verified
    assert result.forced_reflection
    assert len(result.und_serving_provenance) == 3
    assert result.gen_samples[0].rollout_policy_version == 0
    assert result.gen_samples[0].rollout_replica_id == "gen-server"
    assert module._as_gen_sample(asdict(result.gen_samples[0])).rollout_replica_id == "gen-server"


@pytest.mark.parametrize(
    "steps,gen_version",
    [
        ([step("continuing", 0), step("Done.", 1)], 0),
        ([step(GENERATE), step(JUDGE), step("Done.", 1)], 0),
        ([step(GENERATE)], 1),
    ],
)
def test_sync_episode_rejects_changed_ordinary_forced_or_gen_versions(steps, gen_version):
    with pytest.raises(ValueError, match="serving.*version|version.*serving"):
        episode(load_lib(), steps, gen_version)


def test_legacy_stub_provenance_is_explicitly_unknown():
    module = load_lib()
    # A Done. before any image is declined, so the stub repeats it until max_und_turns.
    result = episode(module, [dict(text="Done.", token_ids=[4], log_probs=[-0.2])] * 4)
    assert result.policy_version is None
    assert result.requested_policy_version == 1
    assert result.rollout_replica_id is None
    assert not result.serving_provenance_verified


def functions(file, names, namespace):
    tree = ast.parse((ROOT / file).read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and node.name in names
    ]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            file,
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def test_transfer_queue_roundtrip_keeps_observed_versions(monkeypatch):
    import dataclasses

    module = load_lib()
    result = episode(module, [step(GENERATE), step(JUDGE), step("Done.")])
    project = functions(
        "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/agent_loop.py",
        ["episode_to_agent_output"],
        dict(AgentLoopOutput=SimpleNamespace, AgentLoopMetrics=SimpleNamespace),
    )["episode_to_agent_output"]
    output = project(result)
    assert output.extra_fields["policy_version"] == 0
    assert output.extra_fields["requested_policy_version"] == 1
    assert output.extra_fields["rollout_replica_id"] == "und-server"
    path = "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/transfer_queue.py"
    names = [
        "episode_from_agent_extra",
        "pack_dual_lane_episode",
        "DualLanePack",
        "_complete_groups",
        "_sample_has_gen_traj",
        "_count_gen_calls",
        "gate_episode_jk",
        "classify_episode_pattern",
        "gen_tq_key",
    ]
    namespace = dict(module.__dict__, dataclass=dataclasses.dataclass, field=dataclasses.field, torch=torch)
    tq = functions(path, names, namespace)
    rebuilt = tq["episode_from_agent_extra"](
        prompt_ids=result.prompt_ids,
        response_ids=result.response_ids,
        response_mask=result.response_mask,
        extra=output.extra_fields,
        und_group_uid=result.und_group_uid,
        rollout_log_probs=result.rollout_log_probs,
    )
    assert rebuilt.policy_version == 0 and rebuilt.requested_policy_version == 1
    assert rebuilt.rollout_replica_id == "und-server"
    monkeypatch.setitem(
        sys.modules,
        "verl_omni.pipelines.bagel_flow_grpo.agentic_und_gen.batch",
        SimpleNamespace(episode_numeric_id=lambda uid: 42),
    )
    pack = tq["pack_dual_lane_episode"](rebuilt, und_key="und-key", expected_s=1)
    assert pack.und_record["fields"]["policy_version"] == 0
    assert pack.und_record["fields"]["rollout_replica_id"] == "und-server"
    assert pack.gen_records[0]["fields"]["rollout_policy_version"] == 0
    assert pack.gen_records[0]["fields"]["rollout_replica_id"] == "gen-server"


def test_live_und_decode_keeps_observed_output_metadata():
    import functools
    import logging
    import time
    import uuid

    module = load_lib()
    tree = ast.parse(
        (ROOT / "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/agent_loop.py").read_text(encoding="utf-8")
    )
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "_und_decode")

    async def generate(**kwargs):
        return SimpleNamespace(
            token_ids=[4],
            log_probs=[-0.2],
            extra_fields=dict(
                rollout_policy_version=0, rollout_replica_id="real-und", serving_provenance_verified=True
            ),
        )

    namespace = dict(
        module.__dict__,
        self=SimpleNamespace(
            server_manager=SimpleNamespace(generate=generate),
            tokenizer=SimpleNamespace(decode=lambda *args, **kwargs: "Done."),
        ),
        sampling_params={"global_steps": 1},
        timing={"und_decode_s": 0.0},
        time=time,
        partial=functools.partial,
        uuid=uuid,
        max_context_tokens=100,
        max_response_tokens=100,
        _log_engine_asleep_retry=lambda *a, **kw: None,
        und_turn_max_tokens=lambda **kwargs: 10,
        logger=logging.getLogger("fixture"),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<live-und-decode>", "exec"), namespace)
    decoded = asyncio.run(namespace["_und_decode"](prompt_ids=[1], response_ids=[]))
    assert decoded["rollout_policy_version"] == 0
    assert decoded["rollout_replica_id"] == "real-und"


@pytest.mark.parametrize("rewrite", [False, True])
def test_generation_metadata_reads_only_the_current_episode(tmp_path, rewrite):
    """The pipeline owns call metadata while shared state owns generation history."""
    from verl_omni.tools.trajectory import EpisodeState, current_episode, episode_scope

    path = ROOT / "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/episode.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_build_generate_call_meta"
    )
    namespace = {"current_episode": current_episode}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(path), "exec"), namespace)
    image = tmp_path / "previous.png"
    image.write_bytes(b"image")
    with episode_scope(EpisodeState(episode_id="metadata")) as state:
        if rewrite:
            state.record_generation(prompt="previous", paths=[str(image)])
        meta = namespace["_build_generate_call_meta"](prompt="next", user_prompt="request")
        assert meta["call_role"] == ("rewrite" if rewrite else "initial")
        assert meta["controlled_by_reflection"] is rewrite
        assert meta["prev_tool_prompt"] == ("previous" if rewrite else "")
        assert meta["source_image_for_reflection"] == (str(image) if rewrite else "")
        assert meta["tool_prompt"] == meta["llm_prompt"] == "next"
        assert meta["user_prompt"] == "request"
