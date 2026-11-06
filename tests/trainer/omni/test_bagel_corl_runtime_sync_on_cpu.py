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

"""Execute runtime methods without importing unavailable CUDA serving packages.

Ray/CUDA are replaced only at process boundaries. This is not serving parity.
"""

import ast
import logging
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_function(path, name, namespace=None, class_name=None):
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    if class_name:
        nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == class_name).body
    matches = [n for n in nodes if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef) and n.name == name]
    assert matches, f"missing runtime method: {name}"
    node = matches[0]
    node.decorator_list = []
    scope = {"Any": object, **(namespace or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    return scope[name]


def test_requested_policy_version_uses_tq_request_step_only():
    tree = ast.parse(
        (ROOT / "verl_omni/pipelines/bagel_flow_grpo/agentic_und_gen/agent_loop.py").read_text(encoding="utf-8")
    )
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "requested_policy_version" for t in n.targets)
    )
    scope = {"sampling_params": {}, "kwargs": {"global_steps": 26, "policy_version": 4}}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), "policy_version", "exec"),
        scope,
    )
    assert scope["requested_policy_version"] == 26


@pytest.mark.parametrize("stale", [False, True])
def test_publication_verifies_multiple_sentinels_and_sleeps_on_failure(monkeypatch, stale):
    import verl_omni.workers.rollout.vllm_rollout.weight_fingerprint  # noqa: F401

    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=lambda result: result))
    path = ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py"
    flatten = load_function(path, "_flatten_rpc_results")
    publish = load_function(
        path,
        "_publish_und_ar_weights",
        {"_flatten_rpc_results": flatten, "logger": logging.getLogger(__name__)},
        "OmniBagelCoRLTrainerSync",
    )
    events = []
    names = (
        "language_model.model.layers.0.self_attn.o_proj.weight",
        "language_model.lm_head.weight",
    )
    expected = dict(zip(names, (1.25, 2.5), strict=True))

    def rpc(method, kwargs):
        assert method == "policy_param_checksum"
        name = kwargs["name"]
        events.append(name)
        return [[expected[name] + (1 if stale and name == names[1] else 0)]]

    remote = lambda fn: SimpleNamespace(remote=fn)
    server = SimpleNamespace(
        wake_up=remote(lambda **kw: events.append("wake")),
        sleep=remote(lambda: events.append("sleep")),
        collective_rpc=remote(rpc),
    )

    def update(**kwargs):
        assert kwargs["receiver_ranks"] == [0]
        return [{"num_tensors": 50, "checksum": 1.25, "checksums": expected}]

    trainer = SimpleNamespace(
        und_rollout_replicas=[SimpleNamespace(servers=[server], replica_rank=0)],
        _und_stage_devices=lambda _: {0: [0], 1: [1]},
        _und_deploy_config_path=lambda: "deploy",
        UND_AR_POLICY_STAGE_ID=0,
        UND_AR_SYNC_CHECK_PARAM=names[0],
        UND_AR_SYNC_CHECK_PARAMS=names,
        actor_rollout_wg=SimpleNamespace(update_bagel_und_ar_weights=update),
        global_steps=26,
    )
    if stale:
        with pytest.raises(RuntimeError, match="pushed weights"):
            publish(trainer)
    else:
        publish(trainer)
        assert trainer._pending_sync_metrics["bagel_corl/und_sync/num_tensors"] == 50
    assert set(names).issubset(events)
    assert events[-1] == "sleep"


def test_ar_builder_disables_engine_lora_and_routes_to_thinker(tmp_path):
    from omegaconf import OmegaConf

    path = ROOT / "verl_omni/trainer/omni/bagel_corl_trainer.py"
    fixture_path = ROOT / "tests/trainer/omni/test_bagel_corl_und_ar_weights_on_cpu.py"
    config = load_function(fixture_path, "_entry_config", {"OmegaConf": OmegaConf})()
    deploy = tmp_path / "deploy.yaml"
    deploy.write_text(
        "stages:\n  - stage_id: 0\n    devices: '0'\n    max_model_len: 16384\n  - stage_id: 1\n    devices: '1'\n",
        encoding="utf-8",
    )
    builder = load_function(
        path,
        "_build_und_ar_entrypoint_config",
        {
            "_actor_merges_lora": load_function(path, "_actor_merges_lora"),
        },
        "OmniBagelCoRLTrainerSync",
    )
    trainer = SimpleNamespace(
        config=config,
        UND_AR_LOAD_FORMAT="auto",
        UND_AR_POLICY_STAGE_ID=0,
        _und_deploy_config_path=lambda: str(deploy),
        _resolve_und_hf_config_path=lambda *args: "llm.json",
        _und_ar_stage_ids=lambda _: (0, 1),
    )
    max_len = load_function(
        path,
        "_und_stage_max_model_len",
        {"OmegaConf": OmegaConf, "os": os},
        "OmniBagelCoRLTrainerSync",
    )
    trainer._und_stage_max_model_len = max_len
    result = builder(trainer)
    assert result.actor_rollout_ref.model.lora_rank == 0
    assert result.actor_rollout_ref.rollout.engine_kwargs.vllm_omni.weight_sync_stage_ids == [0]
    assert result.actor_rollout_ref.rollout.load_format == "auto"
    assert config.actor_rollout_ref.rollout.load_format == "dummy"


def test_trainer_rollout_gap_preserves_distinct_update_and_reference_metrics():
    # Execute the production metric publication suffix without initializing a trainer.
    tree = ast.parse(
        (Path(__file__).resolve().parents[3] / "verl_omni/trainer/omni/bagel_corl_trainer.py").read_text(
            encoding="utf-8"
        )
    )
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_update_actor")
    start = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "list(metrics.items())"
    )
    end = next(
        i
        for i, node in enumerate(method.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "_divergence_tripwire"
    )
    metrics = {
        "actor/und/update_kl": 0.03,
        "actor/und/ref_kl": 0.7,
        "actor/gen/update_kl": 0.002,
        "rollout_corr/kl": 0.19,
    }
    exec(
        compile(ast.Module(body=method.body[start:end], type_ignores=[]), "trainer_metric_suffix", "exec"),
        {"metrics": metrics, "math": math},
    )
    assert metrics["und/update_kl"] == 0.03
    assert metrics["und/ref_kl"] == 0.7
    assert metrics["gen/update_kl"] == 0.002
    assert metrics["und/rollout_actor_logp_gap"] == 0.19
